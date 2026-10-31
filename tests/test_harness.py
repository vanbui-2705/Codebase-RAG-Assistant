"""
Module: tests.test_harness
Chức năng: Unit test và Integration test cho Bộ Khung Đánh Giá & Kiểm Thử Hồi Quy (harness).
Kiểm thử toàn diện:
1. Schemas (EvaluationCase, MetricScore, BenchmarkSummary)
2. Golden Dataset (40 ca kiểm thử, tính toàn vẹn ID, phân bố nhóm)
3. 4 Deterministic Mathematical Metrics trong HarnessMetrics
4. BenchmarkRunner CLI và xuất báo cáo ASCII / JSON / Markdown
"""

import json
from pathlib import Path

import pytest

from harness.cases import get_golden_dataset
from harness.metrics import HarnessMetrics
from harness.runner import BenchmarkRunner, MockRAGSystem
from harness.schema import BenchmarkSummary, EvaluationCase, MetricScore

# =============================================================================
# 1. TEST SCHEMAS
# =============================================================================


def test_evaluation_case_creation() -> None:
    """Kiểm tra khởi tạo đối tượng EvaluationCase với các giá trị mặc định và tùy biến."""
    case = EvaluationCase(
        case_id="TEST-001",
        input_text="Làm thế nào để tìm kiếm Hybrid trong vector store?",
        expected_task_type="codebase_qa",
        golden_files=["src/retrieval/vector_store.py"],
        culprit_commit=None,
        expected_symbols=["QdrantStore", "hybrid_search"],
    )

    assert case.case_id == "TEST-001"
    assert case.expected_task_type == "codebase_qa"
    assert len(case.golden_files) == 1
    assert case.is_rejection is False
    assert case.category == "general"


def test_metric_score_bounds() -> None:
    """Kiểm tra các giá trị đo lường trong MetricScore phải nằm trong dải [0.0, 1.0]."""
    score = MetricScore(
        case_id="TEST-002",
        intent_accuracy=1.0,
        file_recall=0.75,
        citation_precision=1.0,
        culprit_accuracy=0.0,
        is_hallucination_free=True,
        latency_sec=0.15,
    )

    assert score.intent_accuracy == 1.0
    assert score.file_recall == 0.75
    assert score.citation_precision == 1.0
    assert score.culprit_accuracy == 0.0
    assert score.is_hallucination_free is True

    # Kiểm tra validation error khi vượt ngoài [0.0, 1.0]
    with pytest.raises(Exception):
        MetricScore(
            case_id="INVALID",
            intent_accuracy=1.5,  # Sai: vượt quá 1.0
            file_recall=0.5,
            citation_precision=0.5,
            culprit_accuracy=0.5,
        )


def test_benchmark_summary_ascii_and_markdown() -> None:
    """Kiểm tra xuất bảng ASCII và văn bản Markdown từ BenchmarkSummary."""
    score1 = MetricScore(
        case_id="QA-001",
        intent_accuracy=1.0,
        file_recall=1.0,
        citation_precision=1.0,
        culprit_accuracy=1.0,
        is_hallucination_free=True,
    )
    score2 = MetricScore(
        case_id="RCA-001",
        intent_accuracy=1.0,
        file_recall=0.5,
        citation_precision=0.8,
        culprit_accuracy=1.0,
        is_hallucination_free=False,
    )

    summary = BenchmarkSummary(
        total_cases=2,
        avg_intent_accuracy=1.0,
        avg_file_recall=0.75,
        avg_citation_precision=0.9,
        avg_culprit_accuracy=1.0,
        execution_time_sec=1.23,
        hallucination_free_rate=0.5,
        case_scores=[score1, score2],
    )

    # 1. ASCII Table
    ascii_out = summary.to_ascii_table()
    assert "BENCHMARK EVALUATION SUMMARY REPORT" in ascii_out
    assert "QA-001" in ascii_out
    assert "RCA-001" in ascii_out
    assert "100.0%" in ascii_out

    # 2. Markdown output
    md_out = summary.to_markdown()
    assert "# Báo Cáo Đánh Giá & Đo Lường" in md_out
    assert "| `QA-001` |" in md_out
    assert "✅" in md_out


# =============================================================================
# 2. TEST GOLDEN DATASET INTEGRITY
# =============================================================================


def test_golden_dataset_integrity() -> None:
    """Kiểm tra tính toàn vẹn của tập dữ liệu chuẩn: >= 35 ca, không trùng ID, đủ 3 nhóm."""
    dataset = get_golden_dataset()

    # Tổng số lượng ca kiểm thử phải đạt tối thiểu 35-40 ca
    assert len(dataset) >= 35, f"Tập dữ liệu chỉ có {len(dataset)} ca, yêu cầu >= 35 ca"

    # Kiểm tra ID không được trùng nhau
    case_ids = [c.case_id for c in dataset]
    assert len(case_ids) == len(set(case_ids)), "Phát hiện ID bị trùng trong Golden Dataset"

    # Phân loại nhóm
    qa_cases = [c for c in dataset if c.expected_task_type == "codebase_qa" and not c.is_rejection]
    rca_cases = [c for c in dataset if c.expected_task_type == "incident_rca"]
    rej_cases = [c for c in dataset if c.is_rejection]

    # Kiểm tra số lượng từng nhóm theo yêu cầu PRD
    assert len(qa_cases) >= 15, f"Nhóm codebase_qa phải >= 15 ca, hiện có {len(qa_cases)}"
    assert len(rca_cases) >= 15, f"Nhóm incident_rca phải >= 15 ca, hiện có {len(rca_cases)}"
    assert len(rej_cases) >= 5, f"Nhóm rejection phải >= 5 ca, hiện có {len(rej_cases)}"

    # Kiểm tra thuộc tính của nhóm rejection
    for r in rej_cases:
        assert r.is_rejection is True
        assert len(r.golden_files) == 0
        assert r.culprit_commit is None

    # Kiểm tra mọi ca đều có input_text và mô tả
    for c in dataset:
        assert len(c.input_text.strip()) > 0
        assert len(c.description.strip()) > 0


# =============================================================================
# 3. TEST DETERMINISTIC METRICS (HarnessMetrics)
# =============================================================================


def test_metric_intent_accuracy() -> None:
    """Kiểm tra phép đo IntentAccuracy: chuẩn hóa chữ hoa/thường và khoảng trắng."""
    # Khớp tuyệt đối
    assert HarnessMetrics.evaluate_intent("codebase_qa", "codebase_qa") == 1.0
    assert HarnessMetrics.evaluate_intent("incident_rca", "incident_rca") == 1.0

    # Chữ hoa / khoảng trắng
    assert HarnessMetrics.evaluate_intent(" CodeBase_QA ", "codebase_qa") == 1.0
    assert HarnessMetrics.evaluate_intent("INCIDENT_RCA", "incident_rca") == 1.0

    # Không khớp
    assert HarnessMetrics.evaluate_intent("codebase_qa", "incident_rca") == 0.0
    assert HarnessMetrics.evaluate_intent("", "codebase_qa") == 0.0
    assert HarnessMetrics.evaluate_intent("unknown", "incident_rca") == 0.0


def test_metric_file_retrieval_recall() -> None:
    """Kiểm tra phép đo FileRetrievalRecall: hỗ trợ so khớp đường dẫn tương đối và gạch chéo."""
    golden = ["src/retrieval/vector_store.py", "src/retrieval/embedder.py"]

    # Tìm đủ 100%
    retrieved_full = ["src/retrieval/vector_store.py", "src/retrieval/embedder.py", "other/file.py"]
    assert HarnessMetrics.evaluate_file_retrieval(retrieved_full, golden) == 1.0

    # Tìm được 1/2 file (50%)
    retrieved_half = ["src/retrieval/vector_store.py", "some_random_file.py"]
    assert HarnessMetrics.evaluate_file_retrieval(retrieved_half, golden) == 0.5

    # Không tìm được file nào (0%)
    retrieved_none = ["src/agent/workflow.py"]
    assert HarnessMetrics.evaluate_file_retrieval(retrieved_none, golden) == 0.0

    # So khớp qua Windows backslash
    retrieved_win = [r"src\retrieval\vector_store.py", r"src\retrieval\embedder.py"]
    assert HarnessMetrics.evaluate_file_retrieval(retrieved_win, golden) == 1.0

    # So khớp qua basename
    retrieved_base = ["vector_store.py", "embedder.py"]
    assert HarnessMetrics.evaluate_file_retrieval(retrieved_base, golden) == 1.0

    # Trường hợp golden_files rỗng (như ca rejection) -> kỳ vọng 1.0
    assert HarnessMetrics.evaluate_file_retrieval([], []) == 1.0


def test_metric_citation_syntax() -> None:
    """Kiểm tra phép đo CitationPrecision: bắt đúng cú pháp và file trong ngữ cảnh."""
    valid_files = ["src/retrieval/vector_store.py", "src/domain/schemas.py"]

    # 1. Trích dẫn chuẩn hoàn toàn (1.0)
    resp_good = (
        "Xem hàm khởi tạo tại [src/retrieval/vector_store.py:L15-L40] "
        "và [src/domain/schemas.py:L25-L35]."
    )
    assert HarnessMetrics.evaluate_citation_syntax(resp_good, valid_files) == 1.0

    # 2. Trích dẫn một dòng duy nhất [file:L15] (1.0)
    resp_single = "Tham chiếu tại [src/domain/schemas.py:L25]."
    assert HarnessMetrics.evaluate_citation_syntax(resp_single, valid_files) == 1.0

    # 3. Trích dẫn file không nằm trong context (0.0)
    resp_hallucinated = "Code nằm ở [services/payment_service.py:L10-L20]."
    assert HarnessMetrics.evaluate_citation_syntax(resp_hallucinated, valid_files) == 0.0

    # 4. Trích dẫn dải dòng sai (dòng kết thúc < dòng bắt đầu)
    resp_bad_lines = "Xem tại [src/domain/schemas.py:L50-L20]."
    assert HarnessMetrics.evaluate_citation_syntax(resp_bad_lines, valid_files) == 0.0

    # 5. Kết hợp 1 trích dẫn đúng, 1 trích dẫn sai file -> 0.5
    resp_mixed = "Xem [src/domain/schemas.py:L10-L20] và [unrelated/file.py:L5-L10]."
    assert HarnessMetrics.evaluate_citation_syntax(resp_mixed, valid_files) == 0.5

    # 6. Không có trích dẫn nhưng có câu thông báo guardrail hợp lệ -> 1.0
    resp_rejected = "Không tìm thấy trong codebase hiện tại. Công nghệ này không được hỗ trợ."
    assert HarnessMetrics.evaluate_citation_syntax(resp_rejected, []) == 1.0

    # 7. Không có trích dẫn khi có context mã nguồn -> 0.0
    resp_no_cite = "Đây là câu trả lời nhưng tôi lười không trích dẫn dòng code nào."
    assert HarnessMetrics.evaluate_citation_syntax(resp_no_cite, valid_files) == 0.0


def test_metric_culprit_commit() -> None:
    """Kiểm tra phép đo CulpritCommitAccuracy: SHA đầy đủ, SHA rút gọn 7 ký tự, PR number."""
    # 1. Trường hợp không yêu cầu commit (culprit_commit is None)
    assert HarnessMetrics.evaluate_culprit_commit("Báo cáo giải thích kiến trúc", None) == 1.0

    # 2. Khớp mã SHA đầy đủ
    full_sha = "c4b81a2f9011de3b"
    resp_full = f"Sự cố xuất phát từ commit {full_sha} khi cấu hình lại URL."
    assert HarnessMetrics.evaluate_culprit_commit(resp_full, full_sha) == 1.0

    # 3. Khớp mã short SHA (7 ký tự đầu)
    resp_short = "Nghi vấn commit c4b81a2 gây ra lỗi này."
    assert HarnessMetrics.evaluate_culprit_commit(resp_short, full_sha) == 1.0

    # 4. Khớp số PR
    pr_target = "#142"
    assert HarnessMetrics.evaluate_culprit_commit("Được giới thiệu trong PR #142", pr_target) == 1.0
    assert HarnessMetrics.evaluate_culprit_commit("Gây lỗi từ pull request 142", pr_target) == 1.0

    # 5. Không tìm thấy commit trong câu trả lời (0.0)
    assert HarnessMetrics.evaluate_culprit_commit("Lỗi này không rõ nguyên nhân", full_sha) == 0.0
    assert HarnessMetrics.evaluate_culprit_commit("", full_sha) == 0.0


def test_evaluate_case_comprehensive() -> None:
    """Kiểm tra hàm tổng hợp evaluate_case cho cả ca thường và ca rejection."""
    case_qa = EvaluationCase(
        case_id="QA-TEST",
        input_text="Kiến trúc hệ thống thế nào?",
        expected_task_type="codebase_qa",
        golden_files=["src/domain/schemas.py"],
        culprit_commit=None,
    )

    score_qa = HarnessMetrics.evaluate_case(
        case=case_qa,
        predicted_type="codebase_qa",
        retrieved_files=["src/domain/schemas.py"],
        response_text="Xem định nghĩa [src/domain/schemas.py:L10-L30].",
        valid_context_files=["src/domain/schemas.py"],
    )

    assert score_qa.intent_accuracy == 1.0
    assert score_qa.file_recall == 1.0
    assert score_qa.citation_precision == 1.0
    assert score_qa.culprit_accuracy == 1.0
    assert score_qa.is_hallucination_free is True

    # Ca rejection
    case_rej = EvaluationCase(
        case_id="REJ-TEST",
        input_text="Solidity ERC20 transfer?",
        expected_task_type="codebase_qa",
        golden_files=[],
        is_rejection=True,
    )

    score_rej = HarnessMetrics.evaluate_case(
        case=case_rej,
        predicted_type="codebase_qa",
        retrieved_files=[],
        response_text="Không tìm thấy trong codebase hiện tại.",
        valid_context_files=[],
    )

    assert score_rej.intent_accuracy == 1.0
    assert score_rej.file_recall == 1.0
    assert score_rej.citation_precision == 1.0
    assert score_rej.is_hallucination_free is True


# =============================================================================
# 4. TEST BENCHMARK RUNNER
# =============================================================================


def test_mock_rag_system_processing() -> None:
    """Kiểm tra hoạt động của MockRAGSystem trên các loại ca kiểm thử."""
    mock = MockRAGSystem()
    cases = get_golden_dataset()

    for case in cases[:5]:
        out = mock.process(case)
        assert "predicted_task_type" in out
        assert "retrieved_files" in out
        assert "response_text" in out
        if case.is_rejection:
            assert "Không tìm thấy trong codebase hiện tại" in out["response_text"]


def test_benchmark_runner_mock_execution(tmp_path: Path) -> None:
    """Kiểm tra BenchmarkRunner chạy toàn bộ Golden Dataset và xuất file JSON/Markdown."""
    runner = BenchmarkRunner(mode="mock")
    summary = runner.run(limit=10)

    assert summary.total_cases == 10
    assert summary.avg_intent_accuracy >= 0.95
    assert summary.avg_file_recall >= 0.85
    assert summary.avg_citation_precision >= 0.90
    assert summary.avg_culprit_accuracy >= 0.80
    assert summary.hallucination_free_rate >= 0.95

    # Kiểm tra xuất JSON
    json_path = tmp_path / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False)
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    # Kiểm tra xuất Markdown
    md_path = tmp_path / "benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary.to_markdown())
    assert md_path.exists()
    assert md_path.stat().st_size > 0
