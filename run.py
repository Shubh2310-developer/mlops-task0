"""Batch pipeline for trade-signal preprocessing and metric computation.

This script parses input OHLCV dataset, computes rolling mean and signals,
generates run execution metrics, logs runtime progression, and serializes
results to a JSON file.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Module constants
REQUIRED_CONFIG_KEYS = ("seed", "window", "version")
METRIC_NAME = "signal_rate"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"


# Exception Taxonomy
class PipelineError(Exception):
    """Base class for all handled pipeline failures."""


class ConfigLoadError(PipelineError):
    """Config file missing or not valid YAML. Recoverable."""


class ConfigValidationError(PipelineError):
    """Config parsed but failed field validation. Recoverable."""


class DatasetLoadError(PipelineError):
    """Input file missing, unreadable, or not valid CSV. Recoverable."""


class DatasetValidationError(PipelineError):
    """Dataset parsed but empty or missing required columns. Recoverable."""


# Pure Logic Functions
def validate_config(config_dict: Any) -> Dict[str, Any]:
    """Validates the configuration dictionary types and constraints.

    Args:
        config_dict: Raw configuration dictionary loaded from YAML.

    Returns:
        A dictionary containing the validated config.

    Raises:
        ConfigValidationError: If validation fails.
    """
    if not isinstance(config_dict, dict):
        raise ConfigValidationError(
            f"Config must be a YAML mapping, got {type(config_dict).__name__}"
        )

    for field in REQUIRED_CONFIG_KEYS:
        if field not in config_dict:
            raise ConfigValidationError(
                f"Missing required config field: '{field}'"
            )

    seed = config_dict["seed"]
    window = config_dict["window"]
    version = config_dict["version"]

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigValidationError(
            f"Config field 'seed' must be an integer, got {type(seed).__name__}"
        )
    if seed < 0:
        raise ConfigValidationError(
            f"Config field 'seed' must be a non-negative integer, got {seed}"
        )

    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ConfigValidationError(
            f"Config field 'window' must be a positive integer, got {window}"
        )

    if not isinstance(version, str) or not version.strip():
        raise ConfigValidationError(
            "Config field 'version' must be a non-empty string"
        )

    return {"seed": seed, "window": window, "version": version}


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validates that the dataset contains a numeric 'close' column.

    Args:
        df: The input DataFrame.

    Returns:
        The validated DataFrame with 'close' converted to numeric.

    Raises:
        DatasetValidationError: If the 'close' column is missing or non-numeric.
    """
    if "close" not in df.columns:
        raise DatasetValidationError(
            "Dataset is missing required column: 'close'"
        )

    try:
        # Coerce 'close' to numeric. Preserves NaNs, throws ValueError for strings.
        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="raise")
    except Exception as exc:
        raise DatasetValidationError(
            f"Configured column 'close' is not numeric-coercible: {exc}"
        )

    return df


def compute_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Computes a rolling mean using a specified window size.

    Args:
        series: The pandas Series of close prices.
        window: The size of the rolling window.

    Returns:
        The computed rolling mean series.
    """
    # First window - 1 rows must evaluate to NaN (no warm-up substitution).
    return series.rolling(window=window, min_periods=window).mean()


def generate_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """Generates a binary signal based on close prices relative to their rolling mean.

    Args:
        close: The close price series.
        rolling_mean: The rolling mean series.

    Returns:
        The binary signal series (1, 0, or NaN).
    """
    # Signal is 1 if close > rolling_mean else 0
    # Signal must be NaN where rolling_mean is NaN (warm-up rows or in-series NaNs)
    signal = pd.Series(np.where(close > rolling_mean, 1, 0), index=close.index)
    signal[rolling_mean.isna()] = np.nan
    return signal


def compute_metrics(
    df: pd.DataFrame,
    signal: pd.Series,
    config: Dict[str, Any],
    latency_ms: int,
) -> Dict[str, Any]:
    """Compiles execution metrics into the success schema.

    Args:
        df: The input DataFrame.
        signal: The generated signal Series.
        config: The validated configuration dictionary.
        latency_ms: Real execution wall-clock time in milliseconds.

    Returns:
        A dictionary containing the compiled metrics.
    """
    signal_clean = signal.dropna()
    value = round(float(signal_clean.mean()), 4) if not signal_clean.empty else 0.0

    return {
        "version": config["version"],
        "rows_processed": len(df),
        "metric": METRIC_NAME,
        "value": value,
        "latency_ms": latency_ms,
        "seed": config["seed"],
        "status": "success",
    }


def build_error_metrics(
    error_message: str,
    config: Any,
) -> Dict[str, Any]:
    """Compiles error metrics into the required error schema.

    Args:
        error_message: The error message string.
        config: The configuration dictionary, or None/invalid.

    Returns:
        A dictionary containing the error metrics.
    """
    version = "unknown"
    if isinstance(config, dict) and "version" in config:
        version = config["version"]
    return {
        "version": version,
        "status": "error",
        "error_message": error_message,
    }


# I/O Functions
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The raw configuration dictionary.

    Raises:
        ConfigLoadError: If the file is missing or not valid YAML.
    """
    if not os.path.exists(config_path):
        raise ConfigLoadError(f"Config file not found: {config_path}")
    if not os.path.isfile(config_path):
        raise ConfigLoadError(f"Config path is not a file: {config_path}")
    if not os.access(config_path, os.R_OK):
        raise ConfigLoadError(f"Config file is not readable: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigLoadError(f"Failed to parse YAML config: {exc}")

    return config_dict


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file.

    Args:
        dataset_path: Path to the CSV file.

    Returns:
        The parsed DataFrame.

    Raises:
        DatasetLoadError: If file is missing, unreadable, unparseable, or empty.
    """
    if not os.path.exists(dataset_path):
        raise DatasetLoadError(f"Dataset file not found: {dataset_path}")
    if not os.path.isfile(dataset_path):
        raise DatasetLoadError(f"Dataset path is not a file: {dataset_path}")
    if not os.access(dataset_path, os.R_OK):
        raise DatasetLoadError(f"Dataset file is not readable: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        raise DatasetLoadError(f"Failed to parse CSV dataset: {exc}")

    if df.empty:
        raise DatasetLoadError("Dataset is empty (contains no rows)")

    return df


def write_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """Writes the metrics dictionary to the output path as a JSON file.

    Args:
        metrics: The metrics dictionary to write.
        output_path: Path to write the JSON file.

    Raises:
        IOError: If the output file cannot be written.
    """
    try:
        serialized = json.dumps(metrics, indent=2)
        with open(output_path, "w") as f:
            f.write(serialized)
    except Exception as exc:
        sys.stderr.write(f"Error: Unwritable output path '{output_path}': {exc}\n")
        raise


def setup_logging(log_file: str) -> logging.Logger:
    """Configures the logging system with a file handler and stream handler.

    Args:
        log_file: Path to write the log file.

    Returns:
        The configured Logger instance.
    """
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # File handler (created/truncated in write mode)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler to stderr (logs go to stderr to keep stdout clean for JSON)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# Main Logic Control
def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="MLOps Batch Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument(
        "--log-file", required=True, dest="log_file", help="Path to write log file"
    )
    return parser.parse_args()


def handle_failure(
    exc: Exception, config: Any, output_path: str, logger: logging.Logger
) -> None:
    """Handles pipeline failures by logging, writing error metrics, and exiting.

    Args:
        exc: The caught exception.
        config: The configuration dict, or None if validation failed.
        output_path: Path to write metrics JSON.
        logger: Logger instance to use.
    """
    if isinstance(exc, PipelineError):
        logger.error(f"{type(exc).__name__}: {exc}")
        err_msg = str(exc)
    else:
        logger.exception(f"Unexpected error: {exc}")
        err_msg = f"Unexpected error: {exc}"

    err_metrics = build_error_metrics(err_msg, config)
    try:
        write_metrics(err_metrics, output_path)
    except Exception as write_exc:
        sys.stderr.write(f"Failed to write error metrics: {write_exc}\n")
    print(json.dumps(err_metrics, indent=2))
    logger.info("Job finished — status=error")
    sys.exit(1)


def run_pipeline(
    args: argparse.Namespace, logger: logging.Logger, start_time: float, config_holder: list
) -> None:
    """Executes the full pipeline: load, validate, compute, output.

    Args:
        args: Parsed CLI arguments.
        logger: Logger instance.
        start_time: Time.perf_counter() value at pipeline start.
        config_holder: Mutable list to store config for error handler access.

    Raises:
        PipelineError: On any handled pipeline failure.
    """
    raw_config = load_config(args.config)
    logger.info("Configuration loaded")

    config = validate_config(raw_config)
    config_holder[0] = config
    logger.info(
        f"Configuration validated - seed: {config['seed']}, "
        f"window: {config['window']}, version: {config['version']}"
    )

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    logger.info(f"Random seed set: {config['seed']}")

    raw_df = load_dataset(args.input)
    logger.info(f"Rows loaded: {len(raw_df)}")

    df = validate_dataset(raw_df)
    rolling_mean = compute_rolling_mean(df["close"], config["window"])
    logger.info(f"Rolling mean computed (window={config['window']})")

    signal = generate_signal(df["close"], rolling_mean)
    logger.info("Signal generation complete")

    latency_ms = round((time.perf_counter() - start_time) * 1000)
    metrics = compute_metrics(df, signal, config, latency_ms)
    logger.info(
        f"Metrics: rows_processed={metrics['rows_processed']}, "
        f"signal_rate={metrics['value']:.4f}, latency_ms={metrics['latency_ms']}"
    )

    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))
    logger.info("Job finished — status=success")


def main() -> None:
    """Main execution entrypoint for the MLOps batch pipeline."""
    args = parse_args()
    logger = setup_logging(args.log_file)
    logger.info(
        f"Job started - resolved CLI arguments: input={args.input}, "
        f"config={args.config}, output={args.output}, log_file={args.log_file}"
    )

    config_holder = [None]
    start_time = time.perf_counter()
    try:
        run_pipeline(args, logger, start_time, config_holder)
        sys.exit(0)
    except Exception as exc:
        handle_failure(exc, config_holder[0], args.output, logger)


if __name__ == "__main__":
    main()
