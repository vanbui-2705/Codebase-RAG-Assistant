"""
Module: harness.metrics
Chức năng: Cung cấp bộ 4 metrics toán học tất định (Deterministic Mathematical Metrics)
để đo lường chính xác hiệu năng của hệ thống RAG mà không phụ thuộc vào LLM Judge:
1. IntentAccuracy: Đo độ chính xác phân loại ý định (Codebase QA vs Incident RCA).
2. FileRetrievalRecall: Đo tỷ lệ tìm đúng file chuẩn trong danh sách golden files.
3. CitationPrecision: Đo tỷ lệ trích dẫn mã nguồn [file:Lstart-Lend] hợp lệ và trỏ đúng file.
4. CulpritCommitAccuracy: Đo độ chính xác phát hiện commit hoặc PR gây ra sự cố.
Tuân thủ Clean Architecture, Python 3.11+ type hints, không có placeholder.
"""

import re
from pathlib import Path
from typing import Sequence

from harness.schema import EvaluationCase, MetricScore


class HarnessMetrics:
    """
    Tập hợp các hàm toán học tất định đánh giá chất lượng truy xuất và sinh nội dung.
    Không dùng LLM-as-a-judge để đảm bảo tính bất biến, tốc độ siêu nhanh và chi phí 0đ.
    """

    # Biểu thức chính quy phát hiện trích dẫn chuẩn [file_path:Lstart-Lend] hoặc [file_path:Lstart]
    CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+):L?(\d+)(?:-L?(\d+))?\]")

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Chuẩn hóa đường dẫn file: chuyển dấu gạch chéo ngược Windows sang gạch chéo xuôi,
        chuyển về chữ thường và loại bỏ khoảng trắng thừa.
        """
        cleaned = path.strip().replace("\\", "/").lower()
        # Loại bỏ tiền tố ./ nếu có
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        return cleaned

    @classmethod
    def _is_file_match(cls, retrieved: str, golden: str) -> bool:
        """
        Kiểm tra xem đường dẫn file truy xuất được có khớp với file golden hay không.
        Hỗ trợ so khớp tương đối, tên file gốc hoặc đuôi đường dẫn.
        """
        norm_r = cls._normalize_path(retrieved)
        norm_g = cls._normalize_path(golden)

        if norm_r == norm_g:
            return True

        # Khớp theo đuôi đường dẫn (VD: 'src/retrieval/vector_store.py' khớp 'vector_store.py')
        if norm_r.endswith("/" + norm_g) or norm_g.endswith("/" + norm_r):
            return True

        # Khớp theo tên file thuần túy (basename) nếu cả hai đều có tên file trùng nhau
        name_r = Path(norm_r).name
        name_g = Path(norm_g).name
        return name_r == name_g and name_r != ""

    @classmethod
    def evaluate_intent(cls, predicted_type: str, expected_type: str) -> float:
        """
        Metric 1: Phân loại ý định tác vụ (Intent Accuracy).

        Args:
            predicted_type: Loại tác vụ dự đoán từ hệ thống ('codebase_qa' | 'incident_rca').
            expected_type: Loại tác vụ chuẩn trong ground truth.

        Returns:
            1.0 nếu khớp hoàn toàn, 0.0 nếu sai lệch.
        """
        if not predicted_type or not expected_type:
            return 0.0

        clean_pred = predicted_type.strip().lower()
        clean_exp = expected_type.strip().lower()

        return 1.0 if clean_pred == clean_exp else 0.0

    @classmethod
    def evaluate_file_retrieval(
        cls,
        retrieved_files: Sequence[str],
        golden_files: Sequence[str],
    ) -> float:
        """
        Metric 2: Tỷ lệ thu hồi file mã nguồn (File Retrieval Recall).
        Recall = (Số lượng file chuẩn có mặt trong danh sách truy xuất) / (Tổng số file chuẩn).

        Args:
            retrieved_files: Danh sách đường dẫn file được hệ thống RAG truy xuất.
            golden_files: Danh sách đường dẫn file chuẩn kỳ vọng.

        Returns:
            Giá trị thực trong dải [0.0, 1.0].
        """
        if not golden_files:
            # Nếu không kỳ vọng file nào (ví dụ câu hỏi ngoài phạm vi bị từ chối)
            return 1.0

        if not retrieved_files:
            return 0.0

        matched_count = 0
        for g_file in golden_files:
            if any(cls._is_file_match(r_file, g_file) for r_file in retrieved_files):
                matched_count += 1

        return float(matched_count / len(golden_files))

    @classmethod
    def evaluate_citation_syntax(
        cls,
        response_text: str,
        valid_context_files: Sequence[str],
    ) -> float:
        """
        Metric 3: Độ chuẩn xác cú pháp và tính xác thực của trích dẫn (Citation Precision).
        Kiểm tra mọi trích dẫn dạng [file:Lstart-Lend] trong văn bản trả về:
        1. Phải đúng cú pháp (dòng bắt đầu > 0, dòng kết thúc >= dòng bắt đầu).
        2. File được trích dẫn phải thuộc danh sách file ngữ cảnh thực tế được cung cấp.

        Precision = (Số trích dẫn hợp lệ) / (Tổng số trích dẫn trong văn bản).

        Args:
            response_text: Câu trả lời hoặc báo cáo RCA do mô hình sinh ra.
            valid_context_files: Danh sách file hợp lệ có trong ngữ cảnh truy xuất được cấp cho LLM.

        Returns:
            Giá trị thực trong dải [0.0, 1.0].
        """
        if not response_text:
            return 0.0

        matches = cls.CITATION_PATTERN.findall(response_text)

        # Trường hợp mô hình không tạo trích dẫn nào:
        if not matches:
            # Nếu là trường hợp từ chối hợp lệ (guardrail) hoặc không có context files
            if not valid_context_files or "Không tìm thấy trong codebase hiện tại" in response_text:
                return 1.0
            # Nếu có ngữ cảnh mã nguồn được cấp mà không thực hiện trích dẫn bắt buộc -> 0.0
            return 0.0

        valid_count = 0
        for match in matches:
            cited_file, start_str, end_str = match
            try:
                start_line = int(start_str)
                end_line = int(end_str) if end_str else start_line
            except ValueError:
                continue

            # Kiểm tra tính hợp lệ của số dòng
            if start_line <= 0 or end_line < start_line:
                continue

            # Kiểm tra xem file được trích dẫn có nằm trong ngữ cảnh thực tế hay không
            file_is_valid = any(
                cls._is_file_match(cited_file, ctx_file) for ctx_file in valid_context_files
            )

            if file_is_valid:
                valid_count += 1

        return float(valid_count / len(matches))

    @classmethod
    def evaluate_culprit_commit(
        cls,
        response_text: str,
        expected_commit: str | None,
    ) -> float:
        """
        Metric 4: Độ chính xác xác định commit/PR gây lỗi (Culprit Commit Accuracy).

        Args:
            response_text: Văn bản báo cáo chẩn đoán sự cố (RCA).
            expected_commit: Mã SHA commit (full hoặc short 7 ký tự) hoặc số PR (VD: '#142').

        Returns:
            1.0 nếu phát hiện đúng mã commit/PR trong câu trả lời (hoặc nếu ca kiểm thử
            không có culprit commit và mô hình không bịa đặt), ngược lại 0.0.
        """
        if expected_commit is None:
            # Không có commit gây lỗi cần tìm -> hoàn thành chỉ tiêu
            return 1.0

        if not response_text:
            return 0.0

        target = expected_commit.strip().lower()

        # Chuẩn bị danh sách các biến thể hợp lệ của commit/PR
        variants = [target]
        # Nếu là commit SHA dài (>= 7 ký tự), hỗ trợ cả short SHA 7 ký tự
        if len(target) >= 7 and not target.startswith("#"):
            variants.append(target[:7])
        # Nếu là PR dạng '#123', hỗ trợ thêm 'pr #123', 'pull request #123', 'pr 123'
        if target.startswith("#"):
            num = target[1:]
            variants.extend(
                [f"pr #{num}", f"pr {num}", f"pull request #{num}", f"pull request {num}"]
            )

        resp_lower = response_text.lower()
        for v in variants:
            if v in resp_lower:
                return 1.0

        return 0.0

    @classmethod
    def evaluate_case(
        cls,
        case: EvaluationCase,
        predicted_type: str,
        retrieved_files: Sequence[str],
        response_text: str,
        valid_context_files: Sequence[str],
        latency_sec: float = 0.0,
    ) -> MetricScore:
        """
        Thực thi tổng thể 4 phép đo trên một ca kiểm thử cụ thể và xác định trạng thái ảo giác.

        Args:
            case: Ca kiểm thử chuẩn (EvaluationCase).
            predicted_type: Loại tác vụ dự đoán.
            retrieved_files: Danh sách các file hệ thống đã tìm được.
            response_text: Câu trả lời hoàn chỉnh sinh ra.
            valid_context_files: Danh sách các file ngữ cảnh thực tế được đưa cho LLM.
            latency_sec: Thời gian phản hồi của ca kiểm thử.

        Returns:
            Đối tượng MetricScore ghi nhận toàn bộ điểm số đo lường.
        """
        intent_acc = cls.evaluate_intent(predicted_type, case.expected_task_type)
        file_rec = cls.evaluate_file_retrieval(retrieved_files, case.golden_files)
        cit_prec = cls.evaluate_citation_syntax(response_text, valid_context_files)
        culprit_acc = cls.evaluate_culprit_commit(response_text, case.culprit_commit)

        # Đánh giá xem có ảo giác hay không (Hallucination check):
        # 1. Nếu là ca rejection: bắt buộc phải có câu từ chối chuẩn hoặc không trích dẫn sai file
        # 2. Nếu là ca bình thường: citation_precision phải đạt 1.0 (mọi trích dẫn đều có căn cứ)
        if case.is_rejection:
            is_hallucination_free = (
                "Không tìm thấy trong codebase hiện tại" in response_text
                or "không tìm thấy" in response_text.lower()
            ) and cit_prec == 1.0
        else:
            is_hallucination_free = cit_prec == 1.0 and (
                culprit_acc == 1.0 if case.culprit_commit else True
            )

        notes_list: list[str] = []
        if intent_acc < 1.0:
            notes_list.append(
                f"Intent sai: dự đoán '{predicted_type}' != kỳ vọng '{case.expected_task_type}'"
            )
        if file_rec < 1.0:
            notes_list.append(f"File Recall chưa đạt 100% ({file_rec * 100:.1f}%)")
        if cit_prec < 1.0:
            notes_list.append(f"Citation Precision vi phạm ({cit_prec * 100:.1f}%)")
        if case.culprit_commit and culprit_acc < 1.0:
            notes_list.append(f"Không nhận diện được commit thủ phạm '{case.culprit_commit}'")

        return MetricScore(
            case_id=case.case_id,
            intent_accuracy=intent_acc,
            file_recall=file_rec,
            citation_precision=cit_prec,
            culprit_accuracy=culprit_acc,
            is_hallucination_free=is_hallucination_free,
            latency_sec=latency_sec,
            predicted_task_type=predicted_type,
            retrieved_files=list(retrieved_files),
            notes="; ".join(notes_list) if notes_list else "Đạt chuẩn tuyệt đối",
        )
