"""
Module: src.agent.tools
Chức năng: Cung cấp bộ 5 công cụ tra cứu cốt lõi phục vụ Agent trong hệ thống On-call & Codebase RAG:
1. parse_stack_trace: Dùng Regex bóc tách exception type, file_path, line_number, function_name từ stack trace log.
2. search_codebase_tool: Gọi Hybrid Search trên Qdrant collection `code_chunks` và Cross-Encoder Rerank.
3. get_callers_and_callees_tool: Phân tích AST của các chunk để tìm các hàm được gọi (callees) và các hàm gọi đến nó (callers).
4. search_git_history_tool: Tìm các commit sửa đổi liên quan đến file_path hoặc các commit gần nhất.
5. search_runbooks_tool: Tìm kiếm trong collection `incident_and_docs` với filter `source_type='runbook'` kết hợp Rerank.

Tuân thủ Clean Architecture: Phụ thuộc vào src.domain, src.config, src.retrieval và src.ingestion.
"""
import ast
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import re
import textwrap
from typing import Any

import git

from src.config.settings import get_settings
from src.domain.interfaces import BaseCodeParser, BaseReranker, BaseVectorStore
from src.domain.schemas import CodeNode, IncidentDocNode, RetrievedContext
from src.ingestion.code_parser import TreeSitterCodeParser
from src.ingestion.git_connector import GitHistoryCollector
from src.retrieval.reranker import CrossEncoderReranker, get_reranker
from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

# Danh sách hàm built-in phổ biến của Python để lọc nhiễu khỏi danh sách callees
PYTHON_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes",
    "callable", "chr", "classmethod", "compile", "complex", "delattr", "dict", "dir",
    "divmod", "enumerate", "eval", "exec", "filter", "float", "format", "frozenset",
    "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int",
    "isinstance", "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print",
    "property", "range", "repr", "reversed", "round", "set", "setattr", "slice",
    "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
    "__import__",
}


def _extract_callees_from_python_code(raw_code: str) -> set[str]:
    """
    Trích xuất tên các hàm và phương thức được gọi từ mã nguồn Python bằng AST.
    Có cơ chế dedent chống lỗi thụt lề và fallback regex khi AST gặp cú pháp không hợp lệ.
    """
    callees: set[str] = set()
    dedented = textwrap.dedent(raw_code)
    try:
        tree = ast.parse(dedented)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    callees.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    callees.add(node.func.attr)
    except Exception as e:
        logger.debug(f"Không thể phân tích AST Python cho đoạn mã: {e}. Sử dụng regex fallback.")
        call_pattern = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
        for match in call_pattern.finditer(raw_code):
            func_name = match.group(1)
            callees.add(func_name)

    # Lọc bỏ built-in functions
    return {fn for fn in callees if fn not in PYTHON_BUILTINS}


def _does_code_call_symbol(raw_code: str, target_symbol: str) -> bool:
    """
    Kiểm tra xem một đoạn mã nguồn có gọi tới target_symbol (hoặc method của nó) hay không.
    """
    base_name = target_symbol.split(".")[-1]
    callees = _extract_callees_from_python_code(raw_code)
    if target_symbol in callees or base_name in callees:
        return True

    # Kiểm tra thêm qua regex dự phòng nếu symbol có tiền tố package/module
    pattern = re.compile(rf"\b{re.escape(base_name)}\s*\(")
    return bool(pattern.search(raw_code))


class AgentTools:
    """
    Bộ 5 công cụ phân tích và tra cứu nghiệp vụ dành cho Agent.
    Hỗ trợ Dependency Injection hoàn toàn cho Vector Store, Reranker, Parser và Git.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        reranker: BaseReranker | None = None,
        code_parser: BaseCodeParser | None = None,
        git_collector: GitHistoryCollector | None = None,
        repo_path: str = ".",
    ):
        self.repo_path = repo_path
        self._vector_store = vector_store
        self._reranker = reranker
        self._code_parser = code_parser
        self._git_collector = git_collector

    @property
    def vector_store(self) -> BaseVectorStore:
        """Khởi tạo lười (lazy initialization) cho QdrantStore nếu chưa được inject."""
        if self._vector_store is None:
            self._vector_store = QdrantStore()
        return self._vector_store

    @property
    def reranker(self) -> BaseReranker:
        """Khởi tạo lười cho CrossEncoderReranker nếu chưa được inject."""
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    @property
    def code_parser(self) -> BaseCodeParser:
        """Khởi tạo lười cho TreeSitterCodeParser nếu chưa được inject."""
        if self._code_parser is None:
            self._code_parser = TreeSitterCodeParser()
        return self._code_parser

    @property
    def git_collector(self) -> GitHistoryCollector:
        """Khởi tạo lười cho GitHistoryCollector nếu chưa được inject."""
        if self._git_collector is None:
            self._git_collector = GitHistoryCollector()
        return self._git_collector

    # =========================================================================
    # CÔNG CỤ 1: parse_stack_trace
    # =========================================================================
    def parse_stack_trace(self, error_log: str) -> dict[str, Any]:
        """
        Dùng Regex bóc tách exception type, file_path, line_number, function_name từ stack trace log.
        Hỗ trợ đa ngôn ngữ: Python traceback, Java exceptions, Go panic, JS/TS Node stack traces.

        Args:
            error_log: Chuỗi error log hoặc stack trace thô.

        Returns:
            dict chứa thông tin cấu trúc:
            - exception_type: Loại ngoại lệ (vd: 'ZeroDivisionError', 'NullPointerException')
            - exception_message: Thông điệp lỗi chi tiết
            - frames: Danh sách các frame theo thứ tự từ ngoài vào trong
            - last_frame: Frame cuối cùng (nơi trực tiếp phát sinh lỗi)
            - file_path: Đường dẫn file của frame cuối cùng
            - line_number: Dòng xảy ra lỗi của frame cuối cùng
            - function_name: Tên hàm của frame cuối cùng
            - error_signature: Chữ ký lỗi dạng 'ExceptionType: message'
        """
        result: dict[str, Any] = {
            "exception_type": "UnknownError",
            "exception_message": "",
            "frames": [],
            "last_frame": None,
            "file_path": None,
            "line_number": None,
            "function_name": None,
            "error_signature": "",
        }

        if not error_log or not error_log.strip():
            return result

        clean_log = error_log.strip()
        frames: list[dict[str, Any]] = []
        exception_type = ""
        exception_message = ""

        # ---------------------------------------------------------------------
        # 1. Bóc tách Python Traceback
        # ---------------------------------------------------------------------
        # Pattern frame Python: File "...", line 123, in func_name
        py_frame_pattern = re.compile(
            r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[^\n\r]+))?(?:\r?\n\s+(?P<code>[^\n\r]+))?'
        )
        for match in py_frame_pattern.finditer(clean_log):
            fpath = match.group("file").replace("\\", "/")
            lineno = int(match.group("line"))
            fname = (match.group("func") or "<module>").strip()
            code_line = (match.group("code") or "").strip()
            frames.append({
                "file_path": fpath,
                "line_number": lineno,
                "function_name": fname,
                "code_context": code_line,
            })

        # Pattern exception Python ở dòng cuối: ExceptionType: message
        py_exc_pattern = re.compile(
            r"^([a-zA-Z_][\w\.]*(?:Error|Exception|Fault|Warning|Failure|Interrupt|Exit|Panic))(?::\s*(.*))?$",
            re.MULTILINE,
        )
        py_exc_matches = list(py_exc_pattern.finditer(clean_log))
        if py_exc_matches:
            last_match = py_exc_matches[-1]
            exception_type = last_match.group(1).split(".")[-1]
            exception_message = (last_match.group(2) or "").strip()

        # ---------------------------------------------------------------------
        # 2. Bóc tách Java Stack Trace (nếu chưa có frame Python)
        # ---------------------------------------------------------------------
        if not frames:
            java_frame_pattern = re.compile(
                r"at\s+(?:(?P<pkg>[\w\.$]+)\.)?(?P<class>[\w$]+)\.(?P<func>[\w$<>]+)\((?:(?P<file>[\w\.-]+\.java):(?P<line>\d+)|(?P<native>[^\)]+))\)"
            )
            for match in java_frame_pattern.finditer(clean_log):
                fpath = match.group("file") or f"{match.group('class')}.java"
                fpath = fpath.replace("\\", "/")
                lineno = int(match.group("line")) if match.group("line") else 1
                fname = f"{match.group('class')}.{match.group('func')}"
                frames.append({
                    "file_path": fpath,
                    "line_number": lineno,
                    "function_name": fname,
                    "code_context": "",
                })

            java_exc_pattern = re.compile(
                r"(?:Exception in thread \"[^\"]*\"\s+)?([a-zA-Z_][\w\.]*(?:Exception|Error))(?::\s*(.*))?"
            )
            j_match = java_exc_pattern.search(clean_log)
            if j_match and not exception_type:
                exception_type = j_match.group(1).split(".")[-1]
                exception_message = (j_match.group(2) or "").strip()

        # ---------------------------------------------------------------------
        # 3. Bóc tách Go Panic / Stack Trace (nếu chưa có frame)
        # ---------------------------------------------------------------------
        if not frames:
            go_frame_pattern = re.compile(
                r"(?P<func>[a-zA-Z0-9_\.\/\(\)\*]+)\([^\)]*\)\r?\n\s+(?P<file>[^\s:]+\.go):(?P<line>\d+)"
            )
            for match in go_frame_pattern.finditer(clean_log):
                fpath = match.group("file").replace("\\", "/")
                lineno = int(match.group("line"))
                fname = match.group("func").strip()
                frames.append({
                    "file_path": fpath,
                    "line_number": lineno,
                    "function_name": fname,
                    "code_context": "",
                })

            go_panic_pattern = re.compile(r"panic:\s*(.*)")
            go_p_match = go_panic_pattern.search(clean_log)
            if go_p_match:
                exception_type = "Panic"
                exception_message = go_p_match.group(1).strip()

        # ---------------------------------------------------------------------
        # 4. Bóc tách JS / TS Stack Trace (nếu chưa có frame)
        # ---------------------------------------------------------------------
        if not frames:
            js_frame_pattern = re.compile(
                r"at\s+(?:(?P<func>[\w\.<>$ ]+?)\s+\()?(?P<file>[^:\n\r\(\)]+?\.(?:js|ts|jsx|tsx|mjs|cjs)):(?P<line>\d+):(?P<col>\d+)\)?"
            )
            for match in js_frame_pattern.finditer(clean_log):
                fpath = match.group("file").replace("\\", "/")
                lineno = int(match.group("line"))
                fname = (match.group("func") or "<anonymous>").strip()
                frames.append({
                    "file_path": fpath,
                    "line_number": lineno,
                    "function_name": fname,
                    "code_context": "",
                })

            js_exc_pattern = re.compile(
                r"^([a-zA-Z_]\w*(?:Error|Exception)):\s*(.*)$", re.MULTILINE
            )
            js_m = js_exc_pattern.search(clean_log)
            if js_m and not exception_type:
                exception_type = js_m.group(1)
                exception_message = (js_m.group(2) or "").strip()

        # ---------------------------------------------------------------------
        # 5. Generic File:Line Fallback (nếu vẫn chưa tìm thấy frame)
        # ---------------------------------------------------------------------
        if not frames:
            generic_pattern = re.compile(
                r"(?P<file>[\w\.\-\/\\ ]+\.(?:py|js|ts|go|java|rs|cpp|c|h|rb|php|kt|scala)):(?P<line>\d+)"
            )
            for match in generic_pattern.finditer(clean_log):
                fpath = match.group("file").strip().replace("\\", "/")
                lineno = int(match.group("line"))
                frames.append({
                    "file_path": fpath,
                    "line_number": lineno,
                    "function_name": "<unknown>",
                    "code_context": "",
                })

        # ---------------------------------------------------------------------
        # 6. Fallback tìm Exception Type nếu chưa phát hiện
        # ---------------------------------------------------------------------
        if not exception_type:
            generic_exc = re.search(r"\b([a-zA-Z_]\w*(?:Error|Exception|Panic))\b(?::\s*([^\n\r]+))?", clean_log)
            if generic_exc:
                exception_type = generic_exc.group(1)
                exception_message = (generic_exc.group(2) or "").strip()
            else:
                # Lấy dòng đầu tiên hoặc dòng cuối cùng làm message
                lines = [line.strip() for line in clean_log.splitlines() if line.strip()]
                exception_type = "RuntimeError"
                exception_message = lines[-1] if lines else "Lỗi không xác định"

        last_frame = frames[-1] if frames else None
        error_sig = f"{exception_type}: {exception_message}" if exception_message else exception_type

        result["exception_type"] = exception_type
        result["exception_message"] = exception_message
        result["frames"] = frames
        result["last_frame"] = last_frame
        result["error_signature"] = error_sig
        if last_frame:
            result["file_path"] = last_frame["file_path"]
            result["line_number"] = last_frame["line_number"]
            result["function_name"] = last_frame["function_name"]

        return result

    # =========================================================================
    # CÔNG CỤ 2: search_codebase_tool
    # =========================================================================
    def search_codebase_tool(
        self,
        query: str,
        file_path: str | None = None,
        symbol_name: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """
        Gọi Hybrid Search trên Qdrant collection `code_chunks` và Cross-Encoder Rerank.

        Args:
            query: Câu truy vấn ngôn ngữ tự nhiên hoặc từ khóa mã nguồn.
            file_path: Đường dẫn file cần lọc (tùy chọn).
            symbol_name: Tên định danh hàm/class cần lọc (tùy chọn).
            top_k: Số lượng kết quả ngữ cảnh chắt lọc trả về (mặc định 5).

        Returns:
            Danh sách RetrievedContext được xếp hạng theo mức độ liên quan.
        """
        if not query or not query.strip():
            logger.warning("Truy vấn tìm kiếm mã nguồn rỗng.")
            return []

        filters: dict[str, Any] = {}
        if file_path:
            filters["file_path"] = file_path.replace("\\", "/")
        if symbol_name:
            filters["symbol_name"] = symbol_name

        fetch_k = max(top_k * 3, 10)
        try:
            candidates = self.vector_store.hybrid_search(
                collection="code_chunks",
                query=query,
                filters=filters if filters else None,
                top_k=fetch_k,
            )
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện hybrid_search trên collection 'code_chunks': {e}")
            return []

        if not candidates:
            logger.info(f"Không tìm thấy đoạn mã nguồn nào phù hợp với query='{query}'.")
            return []

        try:
            ranked_results = self.reranker.rerank(
                query=query,
                contexts=candidates,
                top_k=top_k,
            )
            return ranked_results
        except Exception as e:
            logger.warning(f"Lỗi khi thực hiện Cross-Encoder Rerank: {e}. Trả về top kết quả thô.")
            return candidates[:top_k]

    # =========================================================================
    # CÔNG CỤ 3: get_callers_and_callees_tool
    # =========================================================================
    def get_callers_and_callees_tool(
        self,
        symbol_name: str,
        file_path: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Phân tích AST của các chunk để tìm các hàm được gọi (callees) và các hàm gọi đến nó (callers).

        Args:
            symbol_name: Tên hàm hoặc phương thức cần phân tích (vd: 'process_refund').
            file_path: Đường dẫn file chứa hàm nếu đã biết rõ (tùy chọn).

        Returns:
            dict chứa 2 danh sách:
            - 'callers': Danh sách tên hàm/phương thức trong codebase gọi đến symbol_name.
            - 'callees': Danh sách tên hàm/phương thức được symbol_name gọi tới.
        """
        if not symbol_name or not symbol_name.strip():
            return {"callers": [], "callees": []}

        target_name = symbol_name.strip()
        base_target_name = target_name.split(".")[-1]

        callees_set: set[str] = set()
        callers_set: set[str] = set()

        # ---------------------------------------------------------------------
        # 1. TÌM CALLEES: Tìm mã nguồn của symbol_name và phân tích AST
        # ---------------------------------------------------------------------
        target_code_chunks: list[RetrievedContext] = []

        # Thử tìm kiếm trực tiếp trong vector_store
        try:
            filters: dict[str, Any] = {"symbol_name": target_name}
            if file_path:
                filters["file_path"] = file_path.replace("\\", "/")
            target_code_chunks = self.vector_store.hybrid_search(
                collection="code_chunks",
                query=f"def {base_target_name}",
                filters=filters,
                top_k=5,
            )
        except Exception as e:
            logger.debug(f"Không thể truy vấn vector store cho symbol {target_name}: {e}")

        # Nếu chưa tìm thấy và file_path khả dụng trên local filesystem, đọc trực tiếp file
        if not target_code_chunks and file_path:
            full_path = Path(self.repo_path) / file_path if not Path(file_path).is_absolute() else Path(file_path)
            if full_path.is_file():
                try:
                    parsed_nodes = self.code_parser.parse_file(str(full_path), language="python")
                    for node in parsed_nodes:
                        if node.symbol_name == target_name or node.symbol_name.endswith(f".{base_target_name}"):
                            callees_set.update(_extract_callees_from_python_code(node.raw_code))
                except Exception as e:
                    logger.debug(f"Lỗi khi parse file local {full_path}: {e}")

        # Phân tích các chunk tìm được từ vector store
        for chunk in target_code_chunks:
            extracted = _extract_callees_from_python_code(chunk.content)
            callees_set.update(extracted)

        # Loại bỏ trường hợp tự gọi đệ quy hoặc trùng chính nó
        callees_set.discard(target_name)
        callees_set.discard(base_target_name)

        # ---------------------------------------------------------------------
        # 2. TÌM CALLERS: Tìm các chunk gọi tới symbol_name / base_target_name
        # ---------------------------------------------------------------------
        try:
            # Tìm các ứng viên có thể gọi target_name
            candidate_chunks = self.vector_store.hybrid_search(
                collection="code_chunks",
                query=base_target_name,
                top_k=25,
            )
            for chunk in candidate_chunks:
                # Bỏ qua chính định nghĩa của target symbol
                chunk_sym = chunk.symbol_name or ""
                if chunk_sym == target_name or chunk_sym.endswith(f".{base_target_name}"):
                    continue

                if _does_code_call_symbol(chunk.content, base_target_name):
                    caller_ident = chunk_sym if chunk_sym else f"{chunk.file_path}:L{chunk.start_line}"
                    callers_set.add(caller_ident)
        except Exception as e:
            logger.debug(f"Lỗi khi tìm callers qua vector store: {e}")

        # Quét bổ sung local files nếu callers còn ít và thư mục repo tồn tại
        repo_dir = Path(self.repo_path)
        if repo_dir.is_dir() and len(callers_set) < 5:
            try:
                for py_file in repo_dir.glob("**/*.py"):
                    # Bỏ qua các thư mục ẩn, venv, cache
                    parts = py_file.parts
                    if any(p.startswith(".") or p in ("venv", "env", "__pycache__", "build", "dist") for p in parts):
                        continue
                    try:
                        text = py_file.read_text(encoding="utf-8", errors="ignore")
                        if base_target_name in text:
                            nodes = self.code_parser.parse_file(str(py_file), language="python")
                            for node in nodes:
                                if node.symbol_name != target_name and _does_code_call_symbol(node.raw_code, base_target_name):
                                    callers_set.add(node.symbol_name)
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Lỗi khi quét repo tìm callers: {e}")

        return {
            "callers": sorted(list(callers_set)),
            "callees": sorted(list(callees_set)),
        }

    # =========================================================================
    # CÔNG CỤ 4: search_git_history_tool
    # =========================================================================
    def search_git_history_tool(
        self,
        file_path: str | None = None,
        limit: int = 10,
    ) -> list[IncidentDocNode]:
        """
        Tìm các commit sửa đổi liên quan đến file_path hoặc các commit gần nhất trong repository.

        Args:
            file_path: Đường dẫn tương đối của file cần tra cứu lịch sử sửa đổi (tùy chọn).
            limit: Số lượng commit tối đa cần lấy (mặc định 10).

        Returns:
            Danh sách các IncidentDocNode đại diện cho các commit tìm thấy.
        """
        nodes: list[IncidentDocNode] = []

        # 1. Thử lấy trực tiếp từ local Git repository bằng GitPython
        try:
            repo = git.Repo(self.repo_path, search_parent_directories=True)
            if not repo.bare:
                norm_file = file_path.replace("\\", "/").lstrip("/") if file_path else None
                commits_iter = (
                    repo.iter_commits(paths=norm_file, max_count=limit)
                    if norm_file
                    else repo.iter_commits(max_count=limit)
                )

                for c in commits_iter:
                    changed_files: list[str] = []
                    if c.parents:
                        diffs = c.parents[0].diff(c)
                        for d in diffs:
                            if d.b_path:
                                changed_files.append(d.b_path.replace("\\", "/"))
                            elif d.a_path:
                                changed_files.append(d.a_path.replace("\\", "/"))
                    else:
                        changed_files = [f.replace("\\", "/") for f in c.stats.files.keys()]

                    timestamp_iso = datetime.fromtimestamp(c.authored_date, tz=timezone.utc).isoformat()
                    doc_id = hashlib.sha256(f"commit::{c.hexsha}".encode("utf-8")).hexdigest()
                    first_line = c.message.strip().splitlines()[0] if c.message else "Empty commit message"
                    title = f"Commit: {first_line[:100]}"
                    content = (
                        f"Commit SHA: {c.hexsha}\n"
                        f"Tác giả: {c.author.name} <{c.author.email}>\n"
                        f"Thời gian: {timestamp_iso}\n"
                        f"Nội dung thay đổi:\n{c.message}\n"
                        f"Danh sách file ảnh hưởng: {', '.join(changed_files)}"
                    )

                    nodes.append(
                        IncidentDocNode(
                            doc_id=doc_id,
                            source_type="commit",
                            title=title,
                            content=content,
                            related_files=changed_files,
                            commit_sha=c.hexsha,
                            author=c.author.name,
                            timestamp=timestamp_iso,
                            url=None,
                        )
                    )

                if nodes:
                    logger.info(f"Lấy thành công {len(nodes)} commits từ local git cho file_path={file_path}.")
                    return nodes
        except (git.exc.GitError, OSError, ValueError) as e:
            logger.debug(f"Không thể đọc Git history trực tiếp từ local repo ({e}). Thử fallback sang Vector Store.")

        # 2. Fallback: Tìm kiếm trong collection `incident_and_docs` với filter source_type='commit'
        try:
            query_str = file_path if file_path else "commit history"
            filters: dict[str, Any] = {"source_type": "commit"}
            contexts = self.vector_store.hybrid_search(
                collection="incident_and_docs",
                query=query_str,
                filters=filters,
                top_k=limit,
            )

            for ctx in contexts:
                meta = ctx.metadata
                nodes.append(
                    IncidentDocNode(
                        doc_id=ctx.node_id,
                        source_type="commit",
                        title=meta.get("title", f"Commit {ctx.node_id[:7]}"),
                        content=ctx.content,
                        related_files=meta.get("related_files", [ctx.file_path] if ctx.file_path else []),
                        commit_sha=meta.get("commit_sha"),
                        author=meta.get("author"),
                        timestamp=meta.get("timestamp"),
                        url=meta.get("url"),
                    )
                )
        except Exception as e:
            logger.warning(f"Lỗi khi tìm kiếm commit trong vector store: {e}")

        return nodes

    # =========================================================================
    # CÔNG CỤ 5: search_runbooks_tool
    # =========================================================================
    def search_runbooks_tool(
        self,
        error_signature: str,
        top_k: int = 3,
    ) -> list[RetrievedContext]:
        """
        Tìm kiếm trong collection `incident_and_docs` với filter `source_type='runbook'`
        hoặc semantic search trên runbooks/markdown docs, sau đó xếp hạng bằng Cross-Encoder.

        Args:
            error_signature: Tên ngoại lệ, signature lỗi hoặc mô tả triệu chứng sự cố.
            top_k: Số lượng tài liệu Runbook tối đa cần trả về (mặc định 3).

        Returns:
            Danh sách RetrievedContext của các tài liệu Runbook liên quan nhất.
        """
        if not error_signature or not error_signature.strip():
            logger.warning("Error signature rỗng khi tìm kiếm Runbook.")
            return []

        clean_sig = error_signature.strip()
        fetch_k = max(top_k * 3, 10)

        # 1. Tìm kiếm ưu tiên source_type='runbook'
        candidates: list[RetrievedContext] = []
        try:
            candidates = self.vector_store.hybrid_search(
                collection="incident_and_docs",
                query=clean_sig,
                filters={"source_type": "runbook"},
                top_k=fetch_k,
            )
        except Exception as e:
            logger.debug(f"Lỗi khi hybrid search runbooks: {e}")

        # 2. Nếu không tìm thấy, mở rộng tìm kiếm cả tài liệu 'markdown_doc'
        if not candidates:
            try:
                candidates = self.vector_store.hybrid_search(
                    collection="incident_and_docs",
                    query=clean_sig,
                    filters={"source_type": ["runbook", "markdown_doc"]},
                    top_k=fetch_k,
                )
            except Exception as e:
                logger.debug(f"Lỗi khi hybrid search markdown_doc: {e}")

        # 3. Nếu vẫn chưa có, tìm kiếm tự do trên toàn bộ collection
        if not candidates:
            try:
                candidates = self.vector_store.hybrid_search(
                    collection="incident_and_docs",
                    query=f"Runbook hướng dẫn xử lý sự cố: {clean_sig}",
                    filters=None,
                    top_k=fetch_k,
                )
            except Exception as e:
                logger.error(f"Lỗi khi tìm kiếm fallback incident_and_docs: {e}")
                return []

        if not candidates:
            logger.info(f"Không tìm thấy Runbook nào cho error_signature='{clean_sig}'.")
            return []

        # 4. Rerank kết quả bằng Cross-Encoder
        try:
            ranked_results = self.reranker.rerank(
                query=clean_sig,
                contexts=candidates,
                top_k=top_k,
            )
            return ranked_results
        except Exception as e:
            logger.warning(f"Lỗi khi Rerank runbooks: {e}. Trả về top kết quả thô.")
            return candidates[:top_k]


# =============================================================================
# SINGLETON FACTORY VÀ TOOL FUNCTIONS ĐỘC LẬP (MODULE-LEVEL)
# =============================================================================
@lru_cache(maxsize=1)
def get_agent_tools() -> AgentTools:
    """Singleton: Cung cấp đối tượng AgentTools duy nhất trong ứng dụng."""
    return AgentTools()


def parse_stack_trace(error_log: str) -> dict[str, Any]:
    """Hàm độc lập: Dùng Regex bóc tách exception type, file_path, line_number, function_name từ stack trace log."""
    return get_agent_tools().parse_stack_trace(error_log)


def search_codebase_tool(
    query: str,
    file_path: str | None = None,
    symbol_name: str | None = None,
    top_k: int = 5,
) -> list[RetrievedContext]:
    """Hàm độc lập: Gọi Hybrid Search trên Qdrant collection `code_chunks` và Cross-Encoder Rerank."""
    return get_agent_tools().search_codebase_tool(
        query=query,
        file_path=file_path,
        symbol_name=symbol_name,
        top_k=top_k,
    )


def get_callers_and_callees_tool(
    symbol_name: str,
    file_path: str | None = None,
) -> dict[str, list[str]]:
    """Hàm độc lập: Phân tích AST của các chunk để tìm callees và callers."""
    return get_agent_tools().get_callers_and_callees_tool(
        symbol_name=symbol_name,
        file_path=file_path,
    )


def search_git_history_tool(
    file_path: str | None = None,
    limit: int = 10,
) -> list[IncidentDocNode]:
    """Hàm độc lập: Tìm các commit sửa đổi liên quan đến file_path hoặc các commit gần nhất."""
    return get_agent_tools().search_git_history_tool(
        file_path=file_path,
        limit=limit,
    )


def search_runbooks_tool(
    error_signature: str,
    top_k: int = 3,
) -> list[RetrievedContext]:
    """Hàm độc lập: Tìm kiếm trong collection `incident_and_docs` với filter `source_type='runbook'` hoặc semantic search."""
    return get_agent_tools().search_runbooks_tool(
        error_signature=error_signature,
        top_k=top_k,
    )
