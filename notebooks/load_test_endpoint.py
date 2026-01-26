# Databricks notebook source
# %md
# # Load Testing for Model Serving Endpoint
#
# This notebook performs load testing on the Databricks model serving endpoint to validate autoscaling behavior.
#
# ## Autoscaling Context
# - Databricks autoscaling is based on **concurrency** (concurrent requests), not RPS directly
# - Formula: `max RPS = concurrency / latency_seconds`
# - **Large endpoint (16-64)**: min 16, max 64 concurrent requests
#   - With fast model latency, can handle high RPS
#
# ## Test Scenarios
# - 30 RPS, 60 RPS, 90 RPS - all within large endpoint capacity with fast model inference

# COMMAND ----------
# %pip install marvel_characters-1.0.1-py3-none-any.whl

# COMMAND ----------
# %restart_python

# COMMAND ----------
import time
import os
import requests
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
import threading

from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

from marvel_characters.config import ProjectConfig
from marvel_characters.utils import is_databricks

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# COMMAND ----------
# Initialize Spark and Databricks client
spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=3600).token_value

if not is_databricks():
    load_dotenv()

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Endpoint configuration
ENDPOINT_NAME = "marvel-character-custom-envpack-no-codepath"
SERVING_ENDPOINT = f"{os.environ['DBR_HOST']}/serving-endpoints/{ENDPOINT_NAME}/invocations"

logger.info(f"Target endpoint: {SERVING_ENDPOINT}")

# COMMAND ----------
# Load test data
required_columns = [
    "Height",
    "Weight",
    "Universe",
    "Identity",
    "Gender",
    "Marital_Status",
    "Teams",
    "Origin",
    "Magic",
    "Mutant"
]

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample enough records for load testing (need sufficient data for sustained load)
sampled_records = test_set[required_columns].sample(n=20000, replace=True)
sampled_records = sampled_records.replace({np.nan: None}).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(f"Prepared {len(dataframe_records)} test records")

# COMMAND ----------
@dataclass
class RequestResult:
    """Stores result of a single request."""
    status_code: int
    latency_ms: float
    success: bool
    error: str = None


def call_endpoint(record: list[dict]) -> RequestResult:
    """
    Calls the model serving endpoint with a given input record.
    Returns RequestResult with timing and status information.
    """
    start_time = time.perf_counter()
    try:
        response = requests.post(
            SERVING_ENDPOINT,
            headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
            json={"dataframe_records": record},
            timeout=30
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            status_code=response.status_code,
            latency_ms=latency_ms,
            success=response.status_code == 200,
            error=None if response.status_code == 200 else response.text[:200]
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            status_code=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e)[:200]
        )

# COMMAND ----------
def run_load_test(
    target_rps: int,
    duration_seconds: int,
    records: list[list[dict]],
    max_workers: int = None
) -> dict:
    """
    Runs a load test at the specified RPS for the given duration.

    Args:
        target_rps: Target requests per second
        duration_seconds: How long to run the test
        records: List of request payloads
        max_workers: Max concurrent threads (defaults to target_rps * 2 for headroom)

    Returns:
        Dictionary with test results and statistics
    """
    if max_workers is None:
        max_workers = min(target_rps * 2, 200)  # Cap at 200 threads

    results: list[RequestResult] = []
    results_lock = threading.Lock()

    total_requests = target_rps * duration_seconds
    logger.info("=" * 60)
    logger.info(f"Starting load test: {target_rps} RPS for {duration_seconds} seconds")
    logger.info(f"Total requests planned: {total_requests}")
    logger.info(f"Max concurrent workers: {max_workers}")
    logger.info("=" * 60)

    start_time = time.perf_counter()
    request_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        test_start = time.perf_counter()

        # Submit requests at the target rate
        while time.perf_counter() - test_start < duration_seconds:
            record_idx = request_count % len(records)
            future = executor.submit(call_endpoint, records[record_idx])
            futures.append(future)
            request_count += 1

            # Sleep to maintain target RPS
            elapsed = time.perf_counter() - test_start
            expected_requests = elapsed * target_rps
            if request_count > expected_requests:
                sleep_time = (request_count - expected_requests) / target_rps
                time.sleep(sleep_time)

        # Collect results
        logger.info(f"Submitted {request_count} requests, collecting results...")
        for future in as_completed(futures):
            result = future.result()
            with results_lock:
                results.append(result)

    total_time = time.perf_counter() - start_time

    # Calculate statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in results]

    stats = {
        "target_rps": target_rps,
        "duration_seconds": duration_seconds,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "success_rate": len(successful) / len(results) * 100 if results else 0,
        "actual_rps": len(results) / total_time,
        "avg_latency_ms": np.mean(latencies) if latencies else 0,
        "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
        "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
        "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "min_latency_ms": min(latencies) if latencies else 0,
    }

    # Count error types
    error_counts = defaultdict(int)
    for r in failed:
        error_counts[r.error[:50] if r.error else "Unknown"] += 1
    stats["error_breakdown"] = dict(error_counts)

    return stats


def log_results(stats: dict) -> None:
    """Log load test results."""
    logger.info("=" * 60)
    logger.info(f"LOAD TEST RESULTS - Target: {stats['target_rps']} RPS")
    logger.info("=" * 60)
    logger.info(f"Duration:           {stats['duration_seconds']} seconds")
    logger.info(f"Total Requests:     {stats['total_requests']}")
    logger.info(f"Successful:         {stats['successful_requests']}")
    logger.info(f"Failed:             {stats['failed_requests']}")
    logger.info(f"Success Rate:       {stats['success_rate']:.2f}%")
    logger.info(f"Actual RPS:         {stats['actual_rps']:.2f}")
    logger.info("Latency Statistics (ms):")
    logger.info(f"  Average:          {stats['avg_latency_ms']:.2f}")
    logger.info(f"  P50:              {stats['p50_latency_ms']:.2f}")
    logger.info(f"  P95:              {stats['p95_latency_ms']:.2f}")
    logger.info(f"  P99:              {stats['p99_latency_ms']:.2f}")
    logger.info(f"  Min:              {stats['min_latency_ms']:.2f}")
    logger.info(f"  Max:              {stats['max_latency_ms']:.2f}")

    if stats['error_breakdown']:
        logger.info("Error Breakdown:")
        for error, count in stats['error_breakdown'].items():
            logger.info(f"  {error}: {count}")
    logger.info("=" * 60)

# COMMAND ----------
# Warm-up: Single request to verify endpoint is ready

logger.info("Warming up endpoint with single request...")
warmup_result = call_endpoint(dataframe_records[0])
logger.info(f"Warmup - Status: {warmup_result.status_code}, Latency: {warmup_result.latency_ms:.2f}ms")


# COMMAND ----------
# Load Test Configuration
# Endpoint Size: Large (16-64 concurrent)
# All test scenarios (30, 60, 90 RPS) are within capacity with fast model inference

# COMMAND ----------
# Test 1: 30 RPS

# COMMAND ----------
results_30rps = run_load_test(
    target_rps=30,
    duration_seconds=300,
    records=dataframe_records
)
log_results(results_30rps)

# COMMAND ----------
# Test 2: 60 RPS

# Allow some cool-down between tests
logger.info("Cooling down for 30 seconds before next test...")
time.sleep(30)

results_60rps = run_load_test(
    target_rps=60,
    duration_seconds=300,
    records=dataframe_records
)
log_results(results_60rps)

# COMMAND ----------
# Test 3: 90 RPS

# Allow some cool-down between tests
logger.info("Cooling down for 30 seconds before next test...")
time.sleep(30)

results_90rps = run_load_test(
    target_rps=90,
    duration_seconds=300,
    records=dataframe_records
)
log_results(results_90rps)

# COMMAND ----------
# Summary Comparison

import pandas as pd

summary_df = pd.DataFrame([
    results_30rps,
    results_60rps,
    results_90rps
])

# Select key columns for display
display_columns = [
    "target_rps",
    "actual_rps",
    "total_requests",
    "success_rate",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms"
]

summary_display = summary_df[display_columns].round(2)
summary_display.columns = [
    "Target RPS",
    "Actual RPS",
    "Total Requests",
    "Success Rate %",
    "Avg Latency (ms)",
    "P50 (ms)",
    "P95 (ms)",
    "P99 (ms)"
]

logger.info("=" * 80)
logger.info("LOAD TEST SUMMARY - Large Endpoint (16-64 concurrency)")
logger.info("=" * 80)
logger.info("\n" + summary_display.to_string(index=False))
logger.info("=" * 80)
