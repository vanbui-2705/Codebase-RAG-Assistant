"""
Module: src.ingestion.code_parser
Chức năng: Phân tích cú pháp mã nguồn theo cây AST (Abstract Syntax Tree) bằng Tree-sitter.
Tách hàm, class, method và tiêm Contextual Header trước khi tạo vector embedding.
Tuân thủ Clean Architecture: Chỉ phụ thuộc vào src.domain và src.config.
"""
import logging
from pathlib import Path
from typing import Any
import tree_sitter
from tree_sitter import Language, Parser

from src.domain.interfaces import BaseCodeParser
from src.domain.schemas import CodeNode, compute_code_node_id

logger = logging.getLogger(__name__)


def _load_grammar_safely(lang_module_name: str) -> Any | None:
    """Tải grammar Tree-sitter an toàn, trả về None nếu chưa cài đặt."""
    try:
        mod = __import__(lang_module_name)
        if hasattr(mod, "language"):
            return Language(mod.language())
    except Exception as e:
        logger.debug(f"Không thể nạp Tree-sitter grammar '{lang_module_name}': {e}")
    return None


class TreeSitterCodeParser(BaseCodeParser):
    """
    Bộ phân tích mã nguồn AST đa ngôn ngữ (Python, TypeScript, JavaScript, Go, Java).
    Sử dụng Tree-sitter để phân tách ranh giới class, function, method.
    """

    def __init__(self, repo_name: str = "main_repo"):
        self.repo_name = repo_name
        self._parsers: dict[str, Parser] = {}
        self._init_grammars()

    def _init_grammars(self) -> None:
        """Khởi tạo các Parser cho từng ngôn ngữ được hỗ trợ."""
        grammars = {
            "python": "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "go": "tree_sitter_go",
            "java": "tree_sitter_java",
        }

        for lang, mod_name in grammars.items():
            grammar = _load_grammar_safely(mod_name)
            if grammar is not None:
                try:
                    parser = Parser(grammar)
                    self._parsers[lang] = parser
                except Exception as e:
                    logger.warning(f"Lỗi khởi tạo parser cho {lang}: {e}")

    def parse_file(self, file_path: str, language: str = "python") -> list[CodeNode]:
        """
        Đọc và phân tách 1 file code thành danh sách các CodeNode theo cây AST.

        Args:
            file_path: Đường dẫn tới file code.
            language: Ngôn ngữ lập trình ('python', 'javascript', 'typescript', 'go', 'java').

        Returns:
            Danh sách CodeNode hoàn chỉnh với Contextual Header.
        """
        path_obj = Path(file_path)
        if not path_obj.is_file():
            logger.warning(f"File không tồn tại: {file_path}")
            return []

        try:
            content = path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path_obj.read_text(encoding="latin-1")
            except Exception as e:
                logger.warning(f"Không thể đọc file {file_path}: {e}")
                return []
        except Exception as e:
            logger.warning(f"Lỗi khi mở file {file_path}: {e}")
            return []

        if not content.strip():
            return []

        lang_normalized = language.lower()
        if lang_normalized not in self._parsers:
            logger.warning(
                f"Ngôn ngữ '{language}' chưa được hỗ trợ bởi Tree-sitter Parser. "
                f"Sử dụng fallback Python nếu khả dụng."
            )
            lang_normalized = "python" if "python" in self._parsers else ""

        if not lang_normalized or lang_normalized not in self._parsers:
            logger.error("Không có Tree-sitter parser nào khả dụng.")
            return []

        parser = self._parsers[lang_normalized]
        try:
            source_bytes = content.encode("utf-8")
            tree = parser.parse(source_bytes)
        except Exception as e:
            logger.warning(f"Tree-sitter parse lỗi trên file {file_path}: {e}")
            return []

        # 1. Trích xuất danh sách imports toàn file
        imports = self._extract_imports(tree.root_node, source_bytes, lang_normalized)

        # 2. Thu thập các symbols (classes, functions, methods)
        nodes: list[CodeNode] = []
        self._traverse_tree(
            node=tree.root_node,
            source_bytes=source_bytes,
            file_path=file_path,
            language=lang_normalized,
            imports=imports,
            parent_class=None,
            collected_nodes=nodes,
        )

        # 3. Nếu file không chứa hàm/class nào (file script thẳng), lưu toàn bộ file như 1 module
        if not nodes:
            lines = content.splitlines()
            start_l = 1
            end_l = max(1, len(lines))
            node_id = compute_code_node_id(file_path, "module", content)
            header = f"# File: {file_path} | Module: root\n{content}"
            module_node = CodeNode(
                node_id=node_id,
                repo_name=self.repo_name,
                file_path=file_path,
                language=lang_normalized,
                symbol_type="module",
                symbol_name=path_obj.stem,
                start_line=start_l,
                end_line=end_l,
                imports=imports,
                raw_code=content,
                context_header=header,
            )
            nodes.append(module_node)

        return nodes

    def inject_context_header(self, node: CodeNode) -> str:
        """
        Tiêm Contextual Header chứa thông tin file, class cha và danh sách imports.
        Giúp embedding model hiểu rõ ngữ cảnh của hàm mà không bị mù thông tin.
        """
        import_str = ", ".join(node.imports[:8]) if node.imports else "none"
        header_parts = [
            f"# File: {node.file_path}",
            f"# Symbol: {node.symbol_name} ({node.symbol_type})",
            f"# Imports: {import_str}",
        ]
        header = "\n".join(header_parts) + "\n" + node.raw_code
        return header

    def _extract_imports(
        self, root_node: Any, source_bytes: bytes, language: str
    ) -> list[str]:
        """Trích xuất tất cả các câu lệnh import ở cấp cao nhất của file."""
        imports: list[str] = []
        import_types = {
            "python": ("import_statement", "import_from_statement"),
            "javascript": ("import_statement",),
            "typescript": ("import_statement",),
            "go": ("import_declaration",),
            "java": ("import_declaration",),
        }
        target_types = import_types.get(language, ("import_statement",))

        for child in root_node.children:
            if child.type in target_types:
                text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                clean_text = " ".join(text.split())
                if clean_text:
                    imports.append(clean_text)

        return imports

    def _traverse_tree(
        self,
        node: Any,
        source_bytes: bytes,
        file_path: str,
        language: str,
        imports: list[str],
        parent_class: str | None,
        collected_nodes: list[CodeNode],
    ) -> None:
        """Đệ quy duyệt cây AST để phát hiện class, function và method."""
        symbol_type: str | None = None
        symbol_name: str | None = None

        if language == "python":
            if node.type == "function_definition":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "method" if parent_class else "function"
            elif node.type == "class_definition":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "class"

        elif language in ("javascript", "typescript"):
            if node.type in ("function_declaration", "arrow_function"):
                symbol_name = self._get_node_name(node, source_bytes) or "anonymous_fn"
                symbol_type = "function"
            elif node.type == "method_definition":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "method"
            elif node.type == "class_declaration":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "class"

        elif language == "go":
            if node.type == "function_declaration":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "function"
            elif node.type == "method_declaration":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "method"

        elif language == "java":
            if node.type == "method_declaration":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "method"
            elif node.type == "class_declaration":
                symbol_name = self._get_node_name(node, source_bytes)
                symbol_type = "class"

        if symbol_type and symbol_name:
            full_symbol = f"{parent_class}.{symbol_name}" if parent_class and symbol_type == "method" else symbol_name
            raw_code = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

            # start_point & end_point trong Tree-sitter là (row, col) với row 0-indexed
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            node_id = compute_code_node_id(file_path, full_symbol, raw_code)

            code_node = CodeNode(
                node_id=node_id,
                repo_name=self.repo_name,
                file_path=file_path,
                language=language,
                symbol_type=symbol_type,
                symbol_name=full_symbol,
                start_line=start_line,
                end_line=end_line,
                imports=imports,
                raw_code=raw_code,
                context_header="",
            )
            code_node.context_header = self.inject_context_header(code_node)
            collected_nodes.append(code_node)

            # Nếu là class, tiếp tục duyệt các phương thức bên trong class
            if symbol_type == "class":
                for child in node.children:
                    self._traverse_tree(
                        node=child,
                        source_bytes=source_bytes,
                        file_path=file_path,
                        language=language,
                        imports=imports,
                        parent_class=symbol_name,
                        collected_nodes=collected_nodes,
                    )
                return

        # Tiếp tục duyệt đệ quy cho các node con
        for child in node.children:
            self._traverse_tree(
                node=child,
                source_bytes=source_bytes,
                file_path=file_path,
                language=language,
                imports=imports,
                parent_class=parent_class,
                collected_nodes=collected_nodes,
            )

    def _get_node_name(self, node: Any, source_bytes: bytes) -> str:
        """Lấy tên định danh (identifier) của node."""
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return "unnamed_symbol"
