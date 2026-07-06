# MLOps Task 0 — Batch Signal Pipeline

A minimal MLOps-style batch job in Python that ingests OHLCV data, computes a rolling mean on `close`, generates a binary trading signal, and produces structured metrics and logs. Designed for reproducibility, observability, and deployment readiness.

## Local Setup & Execution

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker Instructions

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

## Example Output

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 21,
  "seed": 42,
  "status": "success"
}
```

## Configuration Fields

| Field     | Type   | Required | Description                                      | Constraints        |
|-----------|--------|----------|--------------------------------------------------|--------------------|
| `seed`    | `int`  | Yes      | Random seed for deterministic numpy/random ops   | `>= 0`             |
| `window`  | `int`  | Yes      | Rolling window size for the moving average       | `> 0`              |
| `version` | `str`  | Yes      | Pipeline version tag included in metrics output  | Non-empty string   |

## Warm-Up Window (NaN) Policy

The rolling mean computation uses a strict window of size `window`. The first `window - 1` rows have an undefined rolling mean (NaN), and their signal values are also NaN. These warm-up rows are counted in `rows_processed` but excluded from the `signal_rate` calculation via `.dropna()`. No imputation or back-filling is performed.

## Failure Modes

| Scenario                    | Exception              | Exit Code | Behavior                                      |
|-----------------------------|------------------------|-----------|-----------------------------------------------|
| Missing input file          | `DatasetLoadError`     | 1         | Error metrics JSON written, log recorded      |
| Invalid/unparseable CSV     | `DatasetLoadError`     | 1         | Error metrics JSON written, log recorded      |
| Empty CSV (header only)     | `DatasetLoadError`     | 1         | Error metrics JSON written, log recorded      |
| Missing `close` column      | `DatasetValidationError` | 1       | Error metrics JSON written, log recorded      |
| Missing/invalid config file | `ConfigLoadError`      | 1         | Error metrics JSON written (`version: "unknown"`) |
| Invalid config fields       | `ConfigValidationError` | 1        | Error metrics JSON written, log recorded      |
| Unexpected runtime error    | `Exception` (catch-all) | 1        | Full traceback logged, error metrics written  |

In all failure cases, `metrics.json` is always written with a valid error schema, `run.log` captures the error type and message, and the process exits with code 1.

## Attribution

Built for the Primetrade.ai MLOps Engineering Internship — Task 0 Technical Assessment.
