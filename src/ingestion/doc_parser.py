"""
Module: src.ingestion.doc_parser
Chức năng: Phân tích và bóc tách tài liệu kỹ thuật, Runbook Markdown theo cấu trúc Heading.
Bảo toàn các khối lệnh (code blocks) và trích xuất danh sách file liên quan.
Tuân thủ Clean Architecture: Chỉ phụ thuộc vào src.domain.
"""
import hashlib
import logging
from pathlib import Path
import re
from src.domain.schemas import IncidentDocNode

logger = logging.getLogger(__name__)

# Regex phát hiện đường dẫn file code thường gặp trong tài liệu
FILE_PATH_PATTERN = re.compile(r"[\w./-]+\.(?:py|ts|js|go|java|yaml|yml|json|toml|sh)")


class RunbookParser:
    """Bộ bóc tách tài liệu Runbook và Markdown kỹ thuật theo cấu trúc phân cấp."""

    def __init__(self, base_url: str = ""):
        self.base_url = base_url

    def parse_markdown(self, file_path: str) -> list[IncidentDocNode]:
        """
        Đọc và phân tách 1 file markdown thành các IncidentDocNode theo các Heading (#, ##, ###).

        Args:
            file_path: Đường dẫn tới file markdown / runbook.

        Returns:
            Danh sách các IncidentDocNode tương ứng với từng phần mục nội dung.
        """
        path_obj = Path(file_path)
        if not path_obj.is_file():
            logger.warning(f"File tài liệu không tồn tại: {file_path}")
            return []

        try:
            raw_text = path_obj.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Không thể đọc file {file_path}: {e}")
            return []

        if not raw_text.strip():
            return []

        sections = self._split_by_headings(raw_text)
        nodes: list[IncidentDocNode] = []

        file_title = path_obj.stem.replace("_", " ").title()

        for idx, (heading, content) in enumerate(sections, start=1):
            full_title = f"{file_title} - {heading}" if heading else file_title
            clean_content = content.strip()
            if not clean_content:
                continue

            # Bóc tách các file liên quan được nhắc tới trong đoạn văn bản
            related_files = sorted(list(set(FILE_PATH_PATTERN.findall(clean_content))))

            # Tạo doc_id deterministic từ file_path và heading
            doc_key = f"{file_path}::{heading}::{idx}"
            doc_id = hashlib.sha256(doc_key.encode("utf-8")).hexdigest()

            doc_node = IncidentDocNode(
                doc_id=doc_id,
                source_type="runbook" if "runbook" in file_path.lower() else "markdown_doc",
                title=full_title,
                content=clean_content,
                related_files=related_files,
                author="system",
                timestamp=None,
                url=f"{self.base_url}/{file_path}" if self.base_url else None,
            )
            nodes.append(doc_node)

        return nodes

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """
        Tách văn bản markdown theo heading nhưng bảo toàn khối code blocks (``` ... ```).
        Trả về danh sách các cặp (heading_title, section_content).
        """
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []

        current_heading = ""
        current_lines: list[str] = []
        in_code_block = False

        heading_regex = re.compile(r"^(#{1,3})\s+(.+)$")

        for line in lines:
            # Kiểm tra trạng thái đang trong code block hay không
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            # Nếu không ở trong code block và gặp dòng heading mới
            if not in_code_block:
                match = heading_regex.match(line.strip())
                if match:
                    # Lưu lại section trước đó nếu có nội dung
                    if current_lines:
                        sections.append((current_heading, "\n".join(current_lines)))
                        current_lines = []
                    current_heading = match.group(2).strip()
                    continue

            current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        return sections
