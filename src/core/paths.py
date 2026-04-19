from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_FILE = PROJECT_ROOT / "data" / "cleaned" / "benchmarking.json"
BENCHMARK_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

FIRST5_RESULTS_FILE = BENCHMARK_OUTPUT_DIR / "base_benchmark_first5_results.json"
FIRST5_METRICS_FILE = BENCHMARK_OUTPUT_DIR / "base_benchmark_first5_metrics.json"

FULL_RESULTS_FILE = BENCHMARK_OUTPUT_DIR / "base_benchmark_full_results.json"
FULL_METRICS_FILE = BENCHMARK_OUTPUT_DIR / "base_benchmark_full_metrics.json"