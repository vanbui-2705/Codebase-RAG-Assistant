"""
Module: harness
Chức năng: Bộ khung đánh giá và kiểm thử hồi quy độc lập (Evaluation Harness)
cho hệ thống On-call & Codebase RAG Assistant.
Bao gồm:
- Schemas: EvaluationCase, MetricScore, BenchmarkSummary
- Dataset: get_golden_dataset (40 ca kiểm thử thực tế)
- Metrics: HarnessMetrics (4 phép đo toán học tất định)
- Runner: BenchmarkRunner, MockRAGSystem, main
"""

from harness.cases import get_golden_dataset
from harness.metrics import HarnessMetrics
from harness.runner import BenchmarkRunner, MockRAGSystem
from harness.schema import BenchmarkSummary, EvaluationCase, MetricScore

__all__ = [
    "EvaluationCase",
    "MetricScore",
    "BenchmarkSummary",
    "get_golden_dataset",
    "HarnessMetrics",
    "BenchmarkRunner",
    "MockRAGSystem",
]
