"""
Auto-Approval matches ingestion — Athena execution (parameterized).

Ingests a dropped file of already-decided 3P->1P matches (base = the tenant's own
1P catalog, comp = a 3P marketplace SKU from the PDP warehouse), enriches both
sides with catalog / PDP attributes, applies the auto-approval heuristics, and
UNLOADs the result to the auto_approval_matches report partition that the
downstream promotion process reads.

Trigger
-------
Not scheduled. ``auto_approval_trigger.py`` (S3 ObjectCreated -> Lambda) resolves
tenant / segment / load_date from the dropped object's key and starts this job.
It can also be run by hand with the same arguments.

What the job does
-----------------
1. Sniffs the header of the dropped CSV and registers an external table over the
   drop prefix, so the pipeline never depends on someone hand-creating an
   ``ml_temp_db.<tenant>_reactivate_*`` table first.
2. Resolves the tenant's competitor source_stores from the ``domains`` DynamoDB
   table (same source of truth as adhoc_queue.py) rather than a hardcoded list.
3. Runs the enrichment + heuristics query in Athena and UNLOADs Parquet to
   ``report=<report>/product_segment=<segment>/company_code=<tenant>/load_date=<date>/``.

Heuristics applied (all of these DROP rows)
-------------------------------------------
* base SKU must exist in the tenant's latest catalog snapshot, comp SKU must
  exist in the PDP warehouse for one of the tenant's competitor stores.
* store-level match_library exclusion: a (base_sku, comp_source_store) that
  already has an active match is not given another one.
* pair-level match_library exclusion: a (base_sku, comp_sku, comp_source_store)
  seen in match_library is skipped — including soft-deleted rows, so a pair a
  matcher previously rejected is never re-proposed.
* price sanity: ``abs(base_price - comp_price) / base_price < price_delta_max_ratio``.
  Note this also drops rows with a missing/non-numeric comp price, since the
  comparison evaluates to NULL.
* one match per base SKU — highest score wins, ties broken by comp_source_store
  then comp_sku.

Glue Job arguments
------------------
Required:
  --tenant            e.g. 'ctc'
  --segment           e.g. 'gm'
  --load_date         e.g. '2026-07-27'
  --input_prefix      s3 prefix holding the dropped CSV(s), e.g.
                      's3://ml-stack.prod/auto_approval_inbox/company_code=ctc/product_segment=gm/load_date=2026-07-27/'
Optional:
  --output_location         report root, default 's3://ml-stack.prod/auto_approval_matches/'
  --report                  report partition value, default 'auto_approval_heuristics'
  --competitors             comma list of comp source_stores; default = domains table
  --source_store_overrides  'src=source_store' pairs, e.g. 'ikea=ikeaca_ikeaca', for
                            crawl sources whose store_name does not follow src_src
  --base_catalog_filter     extra predicate on the catalog CTE, e.g.
                            "json_extract_scalar(additional_attributes,'$.cadence') = 'monthly'"
  --price_delta_max_ratio   default '0.8'
  --default_score           score for rows whose input file has no score column, default '100'
  --default_match           match verdict likewise, default 'exact'
  --model_used              default 'ml-V2.0'
  --match_lib_load_date     override match_library snapshot date; default = latest snapshot
  --staging_database        db for the temporary external table, default 'ml_temp_db'
  --athena_output_location  Athena staging, default 's3://ml-stack.prod/athena_staging/'
  --athena_workgroup        Athena workgroup (omit to use the account default)
  --JOB_NAME                label for logs
"""

import sys
import csv
import io
import re
import time
import logging
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION_NAME = "us-east-1"
DOMAINS_TABLE = "domains"

POLL_INTERVAL_SEC = 5
MAX_POLL_ATTEMPTS = 720  # 5s * 720 = up to 60 minutes

athena = boto3.client("athena", region_name=REGION_NAME)
s3 = boto3.client("s3", region_name=REGION_NAME)
dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)

# The columns the downstream promotion process reads, in order. Listed explicitly
# because the final SELECT has to drop the `rn` helper and this Athena engine
# parses EXCEPT only as the set operator, not as column exclusion.
OUTPUT_COLUMNS = [
    "base_source_store", "base_parent_sku", "base_sku", "base_upc",
    "base_custom_attributes", "match", "match_reason",
    "comp_source_store", "comp_sku", "comp_upc", "comp_parent_sku", "comp_custom_sku",
    "comp_url", "comp_img", "comp_brand",
    "base_title", "comp_title", "comp_description",
    "comp_category", "comp_subcategory", "comp_sub_subcategory",
    "comp_size", "comp_uom", "comp_pack_size", "comp_total_size", "comp_custom_attributes",
    "base_price", "comp_price", "comp_alt_size", "comp_alt_uom", "comp_alt_price",
    "comp_dimension", "comp_shipping_weight", "comp_mfr_part_number",
    "model_used", "score",
]

REQUIRED_INPUT_COLUMNS = ["base_sku", "comp_sku", "comp_source_store"]
# Columns the input file may carry itself; when absent we fall back to the
# --default_score / --default_match arguments.
OPTIONAL_INPUT_COLUMNS = ["score", "match"]


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    required = ["tenant", "segment", "load_date", "input_prefix"]
    optional = [
        "JOB_NAME",
        "output_location",
        "report",
        "competitors",
        "source_store_overrides",
        "base_catalog_filter",
        "price_delta_max_ratio",
        "default_score",
        "default_match",
        "model_used",
        "match_lib_load_date",
        "staging_database",
        "athena_output_location",
        "athena_workgroup",
    ]
    present_optional = [o for o in optional if f"--{o}" in sys.argv]
    args = getResolvedOptions(sys.argv, required + present_optional)

    for key in required:
        args[key] = args[key].strip()
        if not args[key]:
            raise ValueError(f"Job argument '{key}' cannot be empty.")

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args["load_date"]):
        raise ValueError(f"Job argument 'load_date' must be YYYY-MM-DD, got '{args['load_date']}'.")

    if not args["input_prefix"].startswith("s3://"):
        raise ValueError(f"--input_prefix must be an s3:// URI, got '{args['input_prefix']}'.")
    if not args["input_prefix"].endswith("/"):
        args["input_prefix"] += "/"

    # Tenant and segment land in SQL identifiers and table names, so keep them to
    # the shape the rest of the platform uses rather than quoting defensively.
    for key in ("tenant", "segment"):
        if not re.fullmatch(r"[a-z0-9_]+", args[key]):
            raise ValueError(f"Job argument '{key}' must be lowercase alphanumeric/underscore.")

    return args


def resolve_competitors(args):
    """Comp source_stores for this tenant: the --competitors override if given,
    else every `primary` registered against the tenant in the domains table."""
    if "competitors" in args:
        stores = [c.strip() for c in args["competitors"].split(",") if c.strip()]
        if not stores:
            raise ValueError("Job argument 'competitors' resolved to an empty list.")
        return stores

    table = dynamodb.Table(DOMAINS_TABLE)
    key_expr = boto3.dynamodb.conditions.Key("client").eq(args["tenant"])
    stores, kwargs = [], {"IndexName": "client-index", "KeyConditionExpression": key_expr}
    while True:
        response = table.query(**kwargs)
        stores.extend(item["primary"] for item in response.get("Items", []))
        if "LastEvaluatedKey" not in response:
            break
        kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    if not stores:
        raise ValueError(
            f"No competitors found in the '{DOMAINS_TABLE}' table for client='{args['tenant']}'."
        )
    return sorted(set(stores))


def parse_source_store_overrides(args):
    """'ikea=ikeaca_ikeaca,foo=bar_bar' -> {'ikea': 'ikeaca_ikeaca', 'foo': 'bar_bar'}.

    Some crawl sources publish a store_name that does not follow the src_src
    convention, so concat(source_name,'_',store_name) would not match the
    competitor list. These map source_name -> the canonical source_store.
    """
    raw = args.get("source_store_overrides", "").strip()
    if not raw:
        return {}
    overrides = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--source_store_overrides entry '{pair}' is not 'source=source_store'.")
        src, store = (p.strip() for p in pair.split("=", 1))
        if not re.fullmatch(r"[a-z0-9_]+", src) or not re.fullmatch(r"[a-z0-9_]+", store):
            raise ValueError(f"--source_store_overrides entry '{pair}' has unexpected characters.")
        overrides[src] = store
    return overrides


# --------------------------------------------------------------------------- #
# Input file inspection + external table registration
# --------------------------------------------------------------------------- #
def read_input_header(input_prefix):
    """Return the dropped CSV's column names, lowercased.

    The header is read from the first data object under the prefix rather than
    assumed, so a producer reordering or renaming columns fails loudly here
    instead of silently loading the wrong values into the report.
    """
    parsed = urlparse(input_prefix)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    paginator = s3.get_paginator("list_objects_v2")
    first_key = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if obj.get("Size", 0) == 0 or key.rsplit("/", 1)[-1].startswith("_"):
                continue
            if not key.lower().endswith(".csv"):
                raise ValueError(
                    f"Only .csv drops are supported; found '{key}' under {input_prefix}."
                )
            first_key = key
            break
        if first_key:
            break

    if not first_key:
        raise ValueError(f"No data objects found under {input_prefix}.")

    head = s3.get_object(Bucket=bucket, Key=first_key, Range="bytes=0-16383")["Body"].read()
    line = head.decode("utf-8", errors="replace").splitlines()[0]
    columns = [c.strip().strip('"').lower() for c in next(csv.reader([line]))]

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"Dropped file {first_key} is missing required column(s) {missing}; header was {columns}."
        )
    if len(set(columns)) != len(columns):
        raise ValueError(f"Dropped file {first_key} has duplicate column names: {columns}.")

    logger.info(f"Input header from {first_key}: {columns}")
    return columns


def build_inbox_ddl(inbox_table, columns, input_prefix):
    """External table over the drop prefix. Every column is string — the query
    casts what it needs, and a permissive schema means a stray value cannot fail
    the whole read."""
    column_sql = ",\n    ".join(f"`{c}` string" for c in columns)
    return f"""
CREATE EXTERNAL TABLE {inbox_table} (
    {column_sql}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"', 'escapeChar' = '\\\\')
LOCATION '{input_prefix}'
TBLPROPERTIES ('skip.header.line.count' = '1')
""".strip()


# --------------------------------------------------------------------------- #
# Query builder
# --------------------------------------------------------------------------- #
def build_query(args, inbox_table, columns, competitors, overrides):
    tenant = args["tenant"]
    segment = args["segment"]
    price_ratio = args.get("price_delta_max_ratio", "0.8").strip()
    model_used = args.get("model_used", "ml-V2.0").strip()

    comp_under = ", ".join(f"'{c}'" for c in competitors)
    base_under = f"{tenant}_{tenant}"

    # Score / match come from the file when the producer supplies them, else from
    # the job defaults — the drop is a decided-match list either way.
    score_expr = (
        "CAST(score AS varchar)" if "score" in columns
        else f"'{args.get('default_score', '100').strip()}'"
    )
    match_expr = (
        "CAST(match AS varchar)" if "match" in columns
        else f"'{args.get('default_match', 'exact').strip()}'"
    )

    # Crawl sources whose store_name is not simply src_src need remapping before
    # they can be joined against the competitor list.
    if overrides:
        cases = "\n            ".join(
            f"WHEN source_name = '{src}' THEN '{store}'" for src, store in sorted(overrides.items())
        )
        comp_source_store_expr = (
            "CASE\n            "
            f"{cases}\n            "
            "ELSE lower(concat(source_name, '_', store_name))\n        END"
        )
    else:
        comp_source_store_expr = "lower(concat(source_name, '_', store_name))"

    base_extra_filter = args.get("base_catalog_filter", "").strip()
    base_extra_sql = f"\n      AND ({base_extra_filter})" if base_extra_filter else ""

    if "match_lib_load_date" in args:
        match_lib_date = f"'{args['match_lib_load_date'].strip()}'"
    else:
        match_lib_date = (
            "(SELECT max(load_date) FROM match_library.match_library_snapshot "
            f"WHERE company_code = '{tenant}')"
        )

    output_projection = ",\n    ".join(OUTPUT_COLUMNS)

    return f"""
WITH raw_input AS (
    SELECT DISTINCT
        lower(trim(base_sku)) AS base_sku,
        lower(trim(comp_sku)) AS comp_sku,
        lower(replace(trim(comp_source_store), '<>', '_')) AS comp_source_store,
        {score_expr} AS score,
        {match_expr} AS match
    FROM {inbox_table}
    WHERE trim(coalesce(base_sku, '')) <> ''
      AND trim(coalesce(comp_sku, '')) <> ''
      AND lower(replace(trim(coalesce(comp_source_store, '')), '<>', '_')) IN ({comp_under})
),
base AS (
    SELECT
        sku AS base_sku, list_price AS base_price, product_title AS base_title,
        category AS base_category, subcategory AS base_subcategory,
        sub_subcategory AS base_sub_subcategory,
        product_description AS base_description, product_url AS base_url,
        brand AS base_brand, uom AS base_uom, size AS base_size, pack_size AS base_pack_size,
        image_url AS base_img, split_part(upc, ',', 1) AS base_upc,
        manufacturer_part_number AS base_mfr_part_number,
        size_alt AS base_alt_size, uom_alt AS base_alt_uom, dimension AS base_dimensions,
        parent_sku AS base_parent_sku, shipping_weight AS base_shipping_weight,
        list_price_alt AS base_alt_price, custom_sku AS base_custom_sku,
        brand_type AS base_brandtype
    FROM bungee_customercatalog.athena_auroradb_catalog
    WHERE source = '{tenant}'
      AND capture_date = (
          SELECT max(capture_date) FROM bungee_customercatalog.athena_auroradb_catalog
          WHERE source = '{tenant}'
      ){base_extra_sql}
),
comp AS (
    SELECT
        lower(sku) AS comp_sku,
        category AS comp_category, subcategory AS comp_subcategory,
        sub_subcategory AS comp_sub_subcategory,
        product_title AS comp_title, product_description AS comp_description,
        product_url AS comp_url, crawled_brand AS comp_brand,
        uom AS comp_uom, size AS comp_size, pack_size AS comp_pack_size,
        image_url AS comp_img, upc AS comp_upc, custom_sku AS comp_custom_sku,
        CAST(list_price AS varchar) AS comp_price,
        size_alt AS comp_alt_size, uom_alt AS comp_alt_uom, dimension AS comp_dimensions,
        parent_sku AS comp_parent_sku, shipping_weight AS comp_shipping_weight,
        manufacturer_part_number AS comp_mfr_part_number,
        {comp_source_store_expr} AS comp_source_store
    FROM pdp_newdev.product_warehouse
    WHERE lower(replace(source_store, '<>', '_')) IN ({comp_under})
      AND product_segment = '{segment}'
),
-- Store-level exclusion: a base SKU that already has an active match at a
-- competitor is not given a second one.
match_lib AS (
    SELECT DISTINCT lower(base_sku) AS base_sku, comp_source_store
    FROM match_library.match_library_snapshot
    WHERE company_code = '{tenant}'
      AND load_date = {match_lib_date}
      AND deleted_date IS NULL
),
-- Pair-level exclusion: deliberately NOT filtered on deleted_date, so a pair a
-- matcher previously rejected is never re-proposed.
match_lib_pairs AS (
    SELECT DISTINCT lower(base_sku) AS base_sku, lower(comp_sku) AS comp_sku, comp_source_store
    FROM match_library.match_library_snapshot
    WHERE company_code = '{tenant}'
      AND load_date = {match_lib_date}
),
raw_matches AS (
    SELECT
        b.base_parent_sku, a.base_sku, a.comp_sku, c.comp_source_store,
        b.base_title, c.comp_title, a.score,
        b.base_brand, c.comp_brand, b.base_url, c.comp_url, b.base_img, c.comp_img,
        b.base_size, b.base_uom, c.comp_size, c.comp_uom,
        b.base_description, c.comp_description, b.base_pack_size, c.comp_pack_size,
        b.base_upc, c.comp_upc, b.base_mfr_part_number, c.comp_mfr_part_number,
        b.base_price, c.comp_price,
        b.base_category, b.base_subcategory, b.base_sub_subcategory,
        b.base_alt_size, b.base_alt_uom, b.base_dimensions, b.base_shipping_weight,
        c.comp_category, c.comp_subcategory, c.comp_sub_subcategory,
        c.comp_alt_size, c.comp_alt_uom, c.comp_dimensions, c.comp_parent_sku,
        c.comp_shipping_weight,
        b.base_alt_price, b.base_custom_sku, c.comp_custom_sku, b.base_brandtype,
        a.match
    FROM raw_input a
    LEFT JOIN base b
        ON ltrim(a.base_sku, '0') = ltrim(lower(b.base_sku), '0')
    LEFT JOIN comp c
        ON ltrim(c.comp_sku, '0') = ltrim(a.comp_sku, '0')
        AND c.comp_source_store = a.comp_source_store
    LEFT JOIN match_lib ml
        ON ltrim(a.base_sku, '0') = ltrim(ml.base_sku, '0')
        AND a.comp_source_store = ml.comp_source_store
    LEFT JOIN match_lib_pairs mlp
        ON ltrim(a.base_sku, '0') = ltrim(mlp.base_sku, '0')
        AND ltrim(a.comp_sku, '0') = ltrim(mlp.comp_sku, '0')
        AND a.comp_source_store = mlp.comp_source_store
    WHERE ml.comp_source_store IS NULL
      AND mlp.comp_source_store IS NULL
      AND b.base_sku IS NOT NULL
      AND c.comp_source_store IS NOT NULL
      AND TRY_CAST(NULLIF(lower(trim(b.base_price)), 'nan') AS DOUBLE) IS NOT NULL
      AND TRY_CAST(NULLIF(lower(trim(b.base_price)), 'nan') AS DOUBLE) != 0
      AND abs(
              TRY_CAST(NULLIF(lower(trim(b.base_price)), 'nan') AS DOUBLE)
              - TRY_CAST(NULLIF(lower(trim(c.comp_price)), 'nan') AS DOUBLE)
          ) / TRY_CAST(NULLIF(lower(trim(b.base_price)), 'nan') AS DOUBLE) < {price_ratio}
),
shaped AS (
    SELECT
        CAST('{base_under}' AS varchar) AS base_source_store,
        base_parent_sku,
        base_sku,
        base_upc,
        CAST('' AS varchar) AS base_custom_attributes,
        match,
        CAST('' AS varchar) AS match_reason,
        comp_source_store,
        comp_sku,
        comp_upc,
        comp_parent_sku,
        comp_custom_sku,
        comp_url,
        comp_img,
        comp_brand,
        base_title,
        comp_title,
        comp_description,
        comp_category,
        comp_subcategory,
        comp_sub_subcategory,
        CAST(comp_size AS varchar) AS comp_size,
        comp_uom,
        comp_pack_size,
        CAST('' AS varchar) AS comp_total_size,
        CAST('' AS varchar) AS comp_custom_attributes,
        CAST(base_price AS varchar) AS base_price,
        CAST(comp_price AS varchar) AS comp_price,
        CAST(comp_alt_size AS varchar) AS comp_alt_size,
        comp_alt_uom,
        CAST('' AS varchar) AS comp_alt_price,
        comp_dimensions AS comp_dimension,
        comp_shipping_weight,
        comp_mfr_part_number,
        CAST('{model_used}' AS varchar) AS model_used,
        CAST(score AS varchar) AS score
    FROM raw_matches
)
SELECT
    {output_projection}
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY base_sku
               ORDER BY TRY_CAST(score AS DOUBLE) DESC, comp_source_store, comp_sku
           ) AS rn
    FROM shaped
) t
WHERE rn = 1
""".strip()


def warn_on_unknown_comp_stores(inbox_table, competitors, database, staging_location, workgroup):
    """Log comp_source_stores in the drop that are not registered for this tenant.

    Those rows are filtered out by the competitor IN-list, so without this the
    only symptom of a competitor missing from the domains table is a quietly
    smaller report.
    """
    sql = (
        "SELECT DISTINCT lower(replace(trim(comp_source_store), '<>', '_')) "
        f"FROM {inbox_table} WHERE trim(coalesce(comp_source_store, '')) <> ''"
    )
    query_id = run_athena(sql, database, staging_location, workgroup)
    results = athena.get_paginator("get_query_results")
    seen = set()
    for page in results.paginate(QueryExecutionId=query_id):
        for row in page["ResultSet"]["Rows"]:
            value = row["Data"][0].get("VarCharValue")
            if value:
                seen.add(value)
    seen.discard("comp_source_store")  # header row of the first page

    unknown = sorted(seen - set(competitors))
    if unknown:
        logger.warning(
            f"{len(unknown)} comp_source_store(s) in the drop are not registered for this "
            f"tenant and their rows will be excluded: {unknown}"
        )
    logger.info(f"Drop covers {len(seen)} comp_source_store(s); {len(seen) - len(unknown)} usable.")


def build_unload_sql(inner_query, partition_path):
    return f"""
UNLOAD (
{inner_query}
)
TO '{partition_path}'
WITH (format = 'PARQUET', compression = 'SNAPPY')
""".strip()


# --------------------------------------------------------------------------- #
# S3 / Athena helpers
# --------------------------------------------------------------------------- #
def clear_s3_prefix(s3_uri):
    """Delete every object under an s3:// prefix (batched). No-op if empty."""
    parsed = urlparse(s3_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    paginator = s3.get_paginator("list_objects_v2")
    batch, deleted = [], 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            batch.append({"Key": obj["Key"]})
            if len(batch) == 1000:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                deleted += len(batch)
                batch = []
    if batch:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)
    logger.info(f"Cleared {deleted} existing object(s) under {s3_uri}")


def copy_s3_prefix(src_uri, dst_uri):
    """Server-side copy every object under src_uri to dst_uri, preserving the key
    suffix. Returns the number of objects copied."""
    src, dst = urlparse(src_uri), urlparse(dst_uri)
    src_bucket, src_prefix = src.netloc, src.path.lstrip("/")
    dst_bucket, dst_prefix = dst.netloc, dst.path.lstrip("/")
    paginator = s3.get_paginator("list_objects_v2")
    copied = 0
    for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            suffix = obj["Key"][len(src_prefix):]
            s3.copy_object(
                Bucket=dst_bucket,
                Key=dst_prefix + suffix,
                CopySource={"Bucket": src_bucket, "Key": obj["Key"]},
            )
            copied += 1
    logger.info(f"Copied {copied} object(s) from {src_uri} to {dst_uri}")
    return copied


def run_athena(sql, database, staging_location, workgroup=None):
    """Start an Athena query and poll to a terminal state. Raises on non-success."""
    kwargs = {
        "QueryString": sql,
        "QueryExecutionContext": {"Database": database},
        "ResultConfiguration": {"OutputLocation": staging_location},
    }
    if workgroup:
        kwargs["WorkGroup"] = workgroup

    query_id = athena.start_query_execution(**kwargs)["QueryExecutionId"]
    logger.info(f"Athena query started: {query_id}")

    for _ in range(MAX_POLL_ATTEMPTS):
        status = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]["Status"]
        state = status["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = status.get("StateChangeReason", "(no reason given)")
                raise RuntimeError(f"Athena query {query_id} {state}: {reason}")
            logger.info(f"Athena query {query_id} SUCCEEDED.")
            return query_id
        time.sleep(POLL_INTERVAL_SEC)

    try:
        athena.stop_query_execution(QueryExecutionId=query_id)
        logger.warning(f"Cancelled Athena query {query_id} after exceeding the poll window.")
    except ClientError as exc:
        logger.warning(f"Could not cancel Athena query {query_id}: {exc}")
    raise TimeoutError(f"Athena query {query_id} did not finish within the poll window.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    logger.info("Step 1: Parsing job arguments...")
    args = parse_args()
    tenant, segment, load_date = args["tenant"], args["segment"], args["load_date"]
    job_name = args.get("JOB_NAME", "auto_approval_matches")

    output_location = args.get("output_location", "s3://ml-stack.prod/auto_approval_matches/")
    if not output_location.endswith("/"):
        output_location += "/"
    report = args.get("report", "auto_approval_heuristics").strip()
    staging_location = args.get("athena_output_location", "s3://ml-stack.prod/athena_staging/")
    workgroup = args.get("athena_workgroup")
    staging_database = args.get("staging_database", "ml_temp_db").strip()

    partition_suffix = (
        f"report={report}/product_segment={segment}/company_code={tenant}/load_date={load_date}/"
    )
    partition_path = f"{output_location}{partition_suffix}"
    # UNLOAD writes here first; the live partition is only touched once the query
    # has succeeded and produced data. '_staging' is underscore-prefixed so Glue
    # crawlers / Spark ignore it.
    stage_path = f"{output_location}_staging/{partition_suffix}"

    inbox_table = f"{staging_database}.auto_approval_inbox_{tenant}_{segment}_{load_date.replace('-', '')}"

    logger.info("Step 2: Resolving competitors and inspecting the dropped file...")
    competitors = resolve_competitors(args)
    overrides = parse_source_store_overrides(args)
    logger.info(f"{len(competitors)} competitor store(s) for '{tenant}'; overrides={overrides or '{}'}")
    columns = read_input_header(args["input_prefix"])

    try:
        logger.info(f"Step 3: Registering external table {inbox_table} over {args['input_prefix']} ...")
        run_athena(f"DROP TABLE IF EXISTS {inbox_table}", staging_database, staging_location, workgroup)
        run_athena(
            build_inbox_ddl(inbox_table, columns, args["input_prefix"]),
            staging_database, staging_location, workgroup,
        )

        warn_on_unknown_comp_stores(
            inbox_table, competitors, staging_database, staging_location, workgroup
        )

        query = build_query(args, inbox_table, columns, competitors, overrides)
        unload_sql = build_unload_sql(query, stage_path)
        logger.info(unload_sql)

        logger.info(f"Step 4: Staging output under {stage_path} ...")
        clear_s3_prefix(stage_path)

        logger.info("Step 5: Running Athena UNLOAD (this may take a while)...")
        query_id = run_athena(unload_sql, staging_database, staging_location, workgroup)

        # Step 6: promote. Only now do we clear the live partition and copy the
        # freshly staged files in — a failed query never touches existing data.
        # If the query produced nothing, leave the old partition intact rather
        # than replacing good data with an empty result.
        probe = s3.list_objects_v2(
            Bucket=urlparse(stage_path).netloc,
            Prefix=urlparse(stage_path).path.lstrip("/"),
            MaxKeys=1,
        )
        if probe.get("KeyCount", 0) == 0:
            logger.warning(
                f"{job_name} produced 0 rows for tenant '{tenant}' (segment '{segment}', "
                f"load_date '{load_date}'). The existing partition was left unchanged — "
                f"every input row was excluded by the heuristics, or the drop was empty."
            )
        else:
            logger.info(f"Step 6: Promoting staged data to {partition_path} ...")
            clear_s3_prefix(partition_path)
            copied = copy_s3_prefix(stage_path, partition_path)
            logger.info(f"Step 7: Done — {copied} file(s) at {partition_path} (query {query_id}).")
    finally:
        try:
            clear_s3_prefix(stage_path)
        except ClientError as cleanup_exc:
            logger.warning(f"Staging cleanup failed for {stage_path}: {cleanup_exc}")
        try:
            run_athena(f"DROP TABLE IF EXISTS {inbox_table}", staging_database, staging_location, workgroup)
        except (ClientError, RuntimeError, TimeoutError) as cleanup_exc:
            logger.warning(f"Could not drop {inbox_table}: {cleanup_exc}")


if __name__ == "__main__":
    main()
