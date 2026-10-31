"""
Module: harness.schema
Chức năng: Định nghĩa các Data Models cho Bộ Khung Đánh Giá & Kiểm Thử Hồi Quy (Evaluation Harness).
Bao gồm thông tin ca kiểm thử (EvaluationCase), điểm số từng tiêu chí (MetricScore)
và bản tổng kết kết quả benchmark (BenchmarkSummary).
Tuân thủ Clean Architecture, Python 3.11+ type hints và Pydantic v2.
"""

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """
    Mô hình biểu diễn một ca kiểm thử chuẩn (Golden Evaluation Case)
    dùng để đánh giá độ chính xác của hệ thống On-call & Codebase RAG.

    Attributes:
        case_id: Mã định danh duy nhất của ca kiểm thử (VD: 'QA-001', 'RCA-005', 'REJ-002').
        input_text: Câu hỏi tự nhiên hoặc log lỗi / stack trace đầu vào từ người dùng.
        expected_task_type: Loại tác vụ kỳ vọng ('codebase_qa' hoặc 'incident_rca').
        golden_files: Danh sách các đường dẫn file mã nguồn / tài liệu chuẩn cần truy xuất.
        culprit_commit: Mã SHA commit hoặc số PR gây ra lỗi (dành cho ca incident_rca).
        expected_symbols: Danh sách các hàm, class, hoặc phương thức quan trọng cần nhắc tới.
        is_rejection: Cờ đánh dấu ca kiểm thử ngoài phạm vi (Out-of-Domain),
                      kỳ vọng hệ thống kích hoạt guardrail chống ảo giác.
        description: Mô tả ngắn gọn ngữ cảnh và mục tiêu kiểm thử của ca này.
        category: Phân loại nhóm kiểm thử (VD: 'architecture', 'auth', 'database', 'stack_trace').
    """

    case_id: str = Field(description="Mã định danh duy nhất của ca kiểm thử")
    input_text: str = Field(description="Câu hỏi hoặc log lỗi đầu vào")
    expected_task_type: str = Field(
        description="Loại tác vụ kỳ vọng: 'codebase_qa' hoặc 'incident_rca'"
    )
    golden_files: list[str] = Field(
        default_factory=list,
        description="Danh sách các file chuẩn cần truy xuất được",
    )
    culprit_commit: str | None = Field(
        default=None,
        description="Mã SHA commit hoặc số PR gây ra lỗi (dành cho ca RCA)",
    )
    expected_symbols: list[str] = Field(
        default_factory=list,
        description="Danh sách symbols (class, function, method) cần nhận diện",
    )
    is_rejection: bool = Field(
        default=False,
        description="True nếu là câu hỏi công nghệ ngoài repo cần từ chối trả lời",
    )
    description: str = Field(
        default="",
        description="Mô tả mục tiêu của ca kiểm thử",
    )
    category: str = Field(
        default="general",
        description="Nhóm kiểm thử chuyên môn (auth, payment, retry, stack_trace, v.v.)",
    )


class MetricScore(BaseModel):
    """
    Kết quả chấm điểm chi tiết cho một ca kiểm thử cụ thể qua 4 metric toán học tất định.

    Attributes:
        case_id: Mã định danh ca kiểm thử tương ứng.
        intent_accuracy: Điểm phân loại ý định (1.0 nếu khớp, 0.0 nếu sai).
        file_recall: Tỷ lệ file chuẩn được truy xuất thành công trong dải [0.0, 1.0].
        citation_precision: Tỷ lệ trích dẫn cú pháp [file:Lstart-Lend] hợp lệ và trỏ đúng file.
        culprit_accuracy: Điểm xác định đúng commit/PR gây lỗi (1.0 nếu đúng, 0.0 nếu sai).
        is_hallucination_free: Cờ xác định phản hồi hoàn toàn không có ảo giác mã nguồn.
        latency_sec: Thời gian thực thi ca kiểm thử tính bằng giây.
        predicted_task_type: Loại tác vụ dự đoán từ hệ thống.
        retrieved_files: Danh sách các file thực tế được hệ thống truy xuất.
        notes: Ghi chú hoặc cảnh báo phát sinh trong quá trình chấm điểm.
    """

    case_id: str = Field(description="Mã ca kiểm thử")
    intent_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Độ chính xác phân loại ý định tác vụ (1.0 hoặc 0.0)",
    )
    file_recall: float = Field(
        ge=0.0,
        le=1.0,
        description="Tỷ lệ thu hồi file chuẩn (Golden File Retrieval Recall)",
    )
    citation_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Độ chuẩn xác trích dẫn mã nguồn [file:Lstart-Lend]",
    )
    culprit_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Độ chính xác xác định commit/PR thủ phạm gây lỗi",
    )
    is_hallucination_free: bool = Field(
        default=True,
        description="True nếu không có bất kỳ trích dẫn ma hoặc file ảo nào",
    )
    latency_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Thời gian thực thi ca kiểm thử (giây)",
    )
    predicted_task_type: str | None = Field(
        default=None,
        description="Task type thực tế do hệ thống dự đoán",
    )
    retrieved_files: list[str] = Field(
        default_factory=list,
        description="Danh sách file được mô hình truy xuất",
    )
    notes: str = Field(
        default="",
        description="Ghi chú chi tiết về kết quả đánh giá",
    )


class BenchmarkSummary(BaseModel):
    """
    Bản tổng kết đo lường toàn diện sau khi chạy xong bộ kiểm thử Golden Dataset.

    Attributes:
        total_cases: Tổng số ca kiểm thử đã thực thi.
        avg_intent_accuracy: Điểm trung bình phân loại ý định (Intent Accuracy).
        avg_file_recall: Điểm trung bình tỷ lệ tìm đúng file (File Retrieval Recall).
        avg_citation_precision: Điểm trung bình độ chuẩn xác trích dẫn (Citation Precision).
        avg_culprit_accuracy: Điểm trung bình tìm đúng commit thủ phạm (Culprit Accuracy).
        execution_time_sec: Tổng thời gian chạy toàn bộ benchmark (giây).
        hallucination_free_rate: Tỷ lệ phần trăm các câu trả lời sạch hoàn toàn ảo giác.
        case_scores: Danh sách kết quả chi tiết từng ca kiểm thử.
    """

    total_cases: int = Field(ge=0, description="Tổng số ca kiểm thử")
    avg_intent_accuracy: float = Field(
        ge=0.0, le=1.0, description="Điểm trung bình Intent Accuracy"
    )
    avg_file_recall: float = Field(
        ge=0.0, le=1.0, description="Điểm trung bình File Retrieval Recall"
    )
    avg_citation_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Điểm trung bình Citation Precision",
    )
    avg_culprit_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Điểm trung bình Culprit Commit Accuracy",
    )
    execution_time_sec: float = Field(
        ge=0.0,
        description="Tổng thời gian chạy benchmark (giây)",
    )
    hallucination_free_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Tỷ lệ ca kiểm thử sạch ảo giác",
    )
    case_scores: list[MetricScore] = Field(
        default_factory=list,
        description="Danh sách điểm số chi tiết từng ca",
    )

    def to_ascii_table(self, summary_only: bool = False) -> str:
        """
        Sinh bảng biểu ASCII hiển thị kết quả trực quan trên giao diện dòng lệnh (CLI).

        Args:
            summary_only: Nếu True chỉ in phần thống kê tổng kết, bỏ qua bảng chi tiết.

        Returns:
            Chuỗi định dạng ASCII table hoàn chỉnh.
        """
        summary_sep = "=" * 80
        summary_block = [
            summary_sep,
            "                   BENCHMARK EVALUATION SUMMARY REPORT                   ",
            summary_sep,
            f" Tổng số ca kiểm thử:           {self.total_cases:>6d}",
            f" Thời gian thực thi:            {self.execution_time_sec:>6.2f} giây",
            f" Tỷ lệ sạch ảo giác:            {self.hallucination_free_rate * 100:>6.1f}%",
            f" 1. Intent Accuracy:            {self.avg_intent_accuracy * 100:>6.2f}%",
            f" 2. File Retrieval Recall:      {self.avg_file_recall * 100:>6.2f}%",
            f" 3. Citation Precision:         {self.avg_citation_precision * 100:>6.2f}%",
            f" 4. Culprit Commit Accuracy:    {self.avg_culprit_accuracy * 100:>6.2f}%",
            summary_sep,
        ]

        if summary_only:
            return "\n".join([""] + summary_block)

        separator = (
            "+"
            + "-" * 14
            + "+"
            + "-" * 16
            + "+"
            + "-" * 14
            + "+"
            + "-" * 20
            + "+"
            + "-" * 18
            + "+"
            + "-" * 10
            + "+"
        )
        col1 = f"| {'Case ID':<12} "
        col2 = f"| {'Task Intent':<14} "
        col3 = f"| {'File Recall':<12} "
        col4 = f"| {'Citation Prec':<18} "
        col5 = f"| {'Culprit Acc':<16} "
        col6 = f"| {'Status':<8} |"
        header = col1 + col2 + col3 + col4 + col5 + col6

        rows = [separator, header, separator]
        for score in self.case_scores:
            status = (
                "PASS" if score.is_hallucination_free and score.intent_accuracy >= 1.0 else "WARN"
            )
            rows.append(
                f"| {score.case_id:<12} "
                f"| {score.intent_accuracy * 100:>12.1f}% "
                f"| {score.file_recall * 100:>10.1f}% "
                f"| {score.citation_precision * 100:>16.1f}% "
                f"| {score.culprit_accuracy * 100:>14.1f}% "
                f"| {status:<8} |"
            )

        rows.append(separator)

        return "\n".join(rows + [""] + summary_block)

    def to_markdown(self) -> str:
        """
        Xuất bản báo cáo đánh giá dưới định dạng Markdown chuẩn cho báo cáo tài liệu.

        Returns:
            Văn bản định dạng Markdown.
        """
        intent_status = "✅ Đạt" if self.avg_intent_accuracy >= 0.95 else "⚠️ Cần cải thiện"
        recall_status = "✅ Đạt" if self.avg_file_recall >= 0.85 else "⚠️ Cần cải thiện"
        prec_status = "✅ Đạt" if self.avg_citation_precision >= 0.90 else "⚠️ Cần cải thiện"
        culprit_status = "✅ Đạt" if self.avg_culprit_accuracy >= 0.80 else "⚠️ Cần cải thiện"
        halluc_status = "✅ Hoàn hảo" if self.hallucination_free_rate >= 0.98 else "⚠️ Có ảo giác"

        lines = [
            "# Báo Cáo Đánh Giá & Đo Lường Hệ Thống On-call RAG Assistant",
            "",
            "## 1. Tổng Quan Chỉ Số",
            "",
            "| Chỉ Số Đo Lường | Giá Trị Trung Bình | Mục Tiêu Kỹ Thuật | Đánh Giá |",
            "| :--- | :---: | :---: | :---: |",
            (
                f"| **Intent Accuracy** | **{self.avg_intent_accuracy * 100:.2f}%** | "
                f"≥ 95.0% | {intent_status} |"
            ),
            (
                f"| **File Retrieval Recall** | **{self.avg_file_recall * 100:.2f}%** | "
                f"≥ 85.0% | {recall_status} |"
            ),
            (
                f"| **Citation Precision** | **{self.avg_citation_precision * 100:.2f}%** | "
                f"≥ 90.0% | {prec_status} |"
            ),
            (
                f"| **Culprit Commit Accuracy** | **{self.avg_culprit_accuracy * 100:.2f}%** | "
                f"≥ 80.0% | {culprit_status} |"
            ),
            (
                f"| **Hallucination-free Rate** | **{self.hallucination_free_rate * 100:.2f}%** | "
                f"100.0% | {halluc_status} |"
            ),
            "",
            f"- **Tổng số ca kiểm thử:** {self.total_cases}",
            f"- **Tổng thời gian thực thi:** {self.execution_time_sec:.2f}s",
            "",
            "## 2. Chi Tiết Từng Ca Kiểm Thử",
            "",
            "| Case ID | Task Intent | File Recall | Citation Precision | "
            "Culprit Accuracy | Hallucination Free |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for s in self.case_scores:
            h_icon = "✅" if s.is_hallucination_free else "❌"
            lines.append(
                f"| `{s.case_id}` | {s.intent_accuracy * 100:.1f}% | {s.file_recall * 100:.1f}% | "
                f"{s.citation_precision * 100:.1f}% | {s.culprit_accuracy * 100:.1f}% | {h_icon} |"
            )

        return "\n".join(lines)
