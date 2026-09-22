"""
Auto-Approval matches ingestion — S3 event trigger (Lambda).

Fires on ObjectCreated under the auto-approval inbox prefix, resolves the run
parameters from the object key, and starts the ``auto_approval_matches`` Glue job.
The Athena work lives in the Glue job because the enrichment query can outrun a
Lambda's 15-minute ceiling.

Expected key layout (Hive-style, order-independent)
---------------------------------------------------
    auto_approval_inbox/company_code=ctc/product_segment=gm/load_date=2026-07-27/<file>

``company_code`` -> --tenant, ``product_segment`` -> --segment, ``load_date`` ->
--load_date, and the key's parent folder -> --input_prefix. Nothing about the
tenant is hardcoded here.

Trigger on the completion marker, not the data
----------------------------------------------
A producer writing N part files would otherwise start N job runs. TRIGGER_SUFFIX
defaults to ``_SUCCESS`` so exactly one event per drop reaches this handler —
producers must write that marker last. If a producer only ever writes a single
file, set TRIGGER_SUFFIX to ``.csv``; the in-flight guard below still prevents
concurrent runs from colliding on the same output partition.

Environment
-----------
  GLUE_JOB_NAME     Glue job to start, default 'auto_approval_matches'
  TRIGGER_SUFFIX    only keys ending with this start a run, default '_SUCCESS'
  TENANT_JOB_ARGS   optional JSON of per-tenant extra Glue args, e.g.
                    {"ctc": {"--source_store_overrides": "ikea=ikeaca_ikeaca",
                             "--base_catalog_filter": "json_extract_scalar(additional_attributes,'$.cadence') = 'monthly'"}}
                    Keeps tenant quirks in configuration instead of in the query.
"""

import os
import json
import logging
from urllib.parse import unquote_plus

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

glue = boto3.client("glue")

GLUE_JOB_NAME = os.getenv("GLUE_JOB_NAME", "auto_approval_matches")
TRIGGER_SUFFIX = os.getenv("TRIGGER_SUFFIX", "_SUCCESS")

REQUIRED_PARTITIONS = {
    "company_code": "--tenant",
    "product_segment": "--segment",
    "load_date": "--load_date",
}
IN_FLIGHT_STATES = ("STARTING", "RUNNING", "WAITING", "STOPPING")


def lambda_handler(event, context):
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        results.append(_handle_object(bucket, key))

    started = sum(1 for r in results if r["action"] == "started")
    logger.info(f"Processed {len(results)} record(s); started {started} job run(s).")
    return {"statusCode": 200, "body": {"started": started, "records": results}}


def _handle_object(bucket, key):
    if not key.endswith(TRIGGER_SUFFIX):
        logger.info(f"s3://{bucket}/{key} does not end with '{TRIGGER_SUFFIX}' — ignored.")
        return {"key": key, "action": "ignored", "reason": "suffix mismatch"}

    partitions = _parse_partitions(key)
    missing = [p for p in REQUIRED_PARTITIONS if p not in partitions]
    if missing:
        logger.error(f"s3://{bucket}/{key} is missing partition(s) {missing} — ignored.")
        return {"key": key, "action": "ignored", "reason": f"missing {missing}"}

    input_prefix = f"s3://{bucket}/{key.rsplit('/', 1)[0]}/"
    tenant = partitions["company_code"]

    arguments = {"--input_prefix": input_prefix}
    for partition, arg in REQUIRED_PARTITIONS.items():
        arguments[arg] = partitions[partition]
    arguments.update(_tenant_job_args(tenant))

    existing = _in_flight_run(input_prefix)
    if existing:
        logger.info(f"Job run {existing} is already processing {input_prefix} — skipped.")
        return {"key": key, "action": "skipped", "reason": f"in-flight run {existing}"}

    run_id = glue.start_job_run(JobName=GLUE_JOB_NAME, Arguments=arguments)["JobRunId"]
    logger.info(f"Started {GLUE_JOB_NAME} run {run_id} for {input_prefix} (tenant={tenant}).")
    return {"key": key, "action": "started", "job_run_id": run_id, "input_prefix": input_prefix}


def _parse_partitions(key):
    """Collect every ``name=value`` path segment, so the inbox root can move
    without touching this code."""
    partitions = {}
    for segment in key.split("/"):
        if "=" in segment:
            name, value = segment.split("=", 1)
            if name and value:
                partitions[name] = value
    return partitions


def _tenant_job_args(tenant):
    raw = os.getenv("TENANT_JOB_ARGS", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw).get(tenant, {})
    except json.JSONDecodeError as exc:
        logger.warning(f"TENANT_JOB_ARGS is not valid JSON ({exc}) — no per-tenant args applied.")
        return {}


def _in_flight_run(input_prefix):
    """Return the id of a still-running job run for the same drop, if any.

    S3 can deliver an event more than once, and a producer may rewrite the marker.
    Two concurrent runs would both clear and rewrite the same output partition, so
    the second one is dropped rather than raced.
    """
    paginator = glue.get_paginator("get_job_runs")
    for page in paginator.paginate(JobName=GLUE_JOB_NAME):
        for run in page.get("JobRuns", []):
            if run.get("JobRunState") not in IN_FLIGHT_STATES:
                continue
            if run.get("Arguments", {}).get("--input_prefix") == input_prefix:
                return run["Id"]
        # Runs come back newest-first; anything past the in-flight window is
        # finished, so there is no need to page through the full history.
        if all(r.get("JobRunState") not in IN_FLIGHT_STATES for r in page.get("JobRuns", [])):
            break
    return None
