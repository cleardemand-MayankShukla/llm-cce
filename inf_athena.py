"""
INF Recall Analysis — Athena execution (parameterized).

Replaces the Spark/Glue "resilient view" job (inf.py). Instead of loading every
source table into Spark DataFrames through the fragile ``load_resilient_view``
machinery, this job runs the INF recall-analysis query (see inf.sql) directly in
Athena and UNLOADs the result to S3, partitioned by tenant / run_date.

Why Athena instead of Spark
---------------------------
1. The query is written in Trino/Athena dialect (1-indexed arrays such as
   ``split(uuid_a,'<>',3)[3]``, ``array_agg``, ``a||b`` string concat). Running
   that verbatim through ``spark.sql`` silently breaks — Spark arrays are
   0-indexed — which is the exact bug class the old job suffered from.
2. Athena reads columns lazily. It only touches the columns a query references,
   so the ``bungeedatalake.bungee_competitiveintelligence_datalake`` table's
   "bad" columns (the ones that made the Spark catalog load blow up) are never
   read — we simply never select them. That removes the need for any
   resilient-loading workaround and lets us bring the deep_crawl CTE back.

Glue Job arguments
------------------
Required:
  --segment         e.g. 'gm'
  --input_table     qualified db.table, e.g. 'ml_temp_db.ctc_inf_1507'
  --tenant          e.g. 'ctc'
  --competitors     comma list, e.g. 'amazonca,homedepotca,homehardwareca,ronaca,walmartca'
  --run_date        e.g. '2026-07-16'
Optional:
  --output_location         default 's3://ml-inf-recall/inf_report_final/'
  --athena_output_location  Athena staging, default 's3://ml-inf-recall/athena_staging/'
  --athena_workgroup        Athena workgroup (omit to use the account default)
  --athena_database         QueryExecutionContext db (default: input_table's db)
  --match_lib_load_date     override match_library snapshot date; default = latest snapshot
  --deep_crawl_years        comma list, e.g. '2026,2025,2024'; default = run_date year + 2 prior
  --JOB_NAME                label for logs/Slack (Spark jobs auto-supply it; Python Shell does not)
"""

import sys
import re
import json
import time
import logging
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

import boto3
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION_NAME = "us-east-1"
SLACK_SECRET_NAME = "ml-inf-recall-mail-slack"

POLL_INTERVAL_SEC = 5
MAX_POLL_ATTEMPTS = 720  # 5s * 720 = up to 60 minutes for the heavy query + UNLOAD

athena = boto3.client("athena", region_name=REGION_NAME)
s3 = boto3.client("s3", region_name=REGION_NAME)


# --------------------------------------------------------------------------- #
# Slack (optional, best-effort, never fatal). Uses urllib so the job has no
# dependency on `requests` being present in the Glue Python environment.
# --------------------------------------------------------------------------- #
def _get_slack_webhook():
    try:
        client = boto3.session.Session().client("secretsmanager", region_name=REGION_NAME)
        secret = json.loads(client.get_secret_value(SecretId=SLACK_SECRET_NAME)["SecretString"])
        return secret.get("slack_webhook_url")
    except (ClientError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning(f"Slack webhook lookup skipped: {exc}")
        return None


def send_slack(message):
    webhook = _get_slack_webhook()
    if not webhook:
        logger.info(f"[slack-disabled] {message}")
        return
    payload = {"channel": "#ml-inf-recall", "username": "INF_ANALYSIS", "text": message}
    try:
        req = Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=10) as resp:
            logger.info(f"Slack notification sent ({resp.status}).")
    except (URLError, TimeoutError) as exc:
        logger.warning(f"Slack notification failed: {exc}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    required = ["segment", "input_table", "tenant", "competitors", "run_date"]
    optional = [
        # JOB_NAME is auto-supplied by Spark (glueetl) jobs but NOT by Python Shell
        # jobs, so it's optional here (used only for log/Slack labels).
        "JOB_NAME",
        "output_location",
        "athena_output_location",
        "athena_workgroup",
        "athena_database",
        "match_lib_load_date",
        "deep_crawl_years",
    ]
    present_optional = [o for o in optional if f"--{o}" in sys.argv]
    args = getResolvedOptions(sys.argv, required + present_optional)

    for key in ("segment", "input_table", "tenant", "run_date"):
        args[key] = args[key].strip()
        if not args[key]:
            raise ValueError(f"Job argument '{key}' cannot be empty.")

    # run_date is used both as a partition value and to derive deep_crawl years,
    # so fail fast with a clear message rather than deep inside fragment-building.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args["run_date"]):
        raise ValueError(f"Job argument 'run_date' must be YYYY-MM-DD, got '{args['run_date']}'.")

    competitors = [c.strip() for c in args["competitors"].split(",") if c.strip()]
    if not competitors:
        raise ValueError("Job argument 'competitors' resolved to an empty list.")
    args["competitors_list"] = competitors
    return args


# --------------------------------------------------------------------------- #
# SQL fragment builders — reproduce every literal shape inf.sql relies on.
# --------------------------------------------------------------------------- #
def build_fragments(args):
    tenant = args["tenant"]
    competitors = args["competitors_list"]

    frags = {
        # 'amazonca<>amazonca', 'homedepotca<>homedepotca', ...
        "comp_bracket": ", ".join(f"'{c}<>{c}'" for c in competitors),
        # 'amazonca_amazonca', 'homedepotca_homedepotca', ...
        "comp_under": ", ".join(f"'{c}_{c}'" for c in competitors),
        # 'amazonca', 'homedepotca', ...  (deep_crawl source_name)
        "comp_names": ", ".join(f"'{c}'" for c in competitors),
        # 'ctc<>ctc' / 'ctc_ctc'
        "base_bracket": f"'{tenant}<>{tenant}'",
        "base_under": f"'{tenant}_{tenant}'",
        # comp underscore list + tenant underscore  (csf_ml_input IN clause)
        "all_under": ", ".join([f"'{c}_{c}'" for c in competitors] + [f"'{tenant}_{tenant}'"]),
    }

    # deep_crawl year filter: explicit override, else run_date's year + 2 prior years.
    if "deep_crawl_years" in args:
        years = [y.strip() for y in args["deep_crawl_years"].split(",") if y.strip()]
    else:
        ry = int(args["run_date"][:4])
        years = [str(ry), str(ry - 1), str(ry - 2)]
    frags["years_sql"] = ", ".join(f"'{y}'" for y in years)

    # match_library snapshot date: explicit override, else the latest snapshot for
    # this tenant. Using MAX(load_date) instead of "today" (the old bug) means a
    # backfill or an early run never silently loses every match-library flag.
    if "match_lib_load_date" in args:
        frags["load_date_pred"] = f"AND load_date = '{args['match_lib_load_date'].strip()}'"
    else:
        frags["load_date_pred"] = (
            "AND load_date = (\n"
            "            SELECT max(load_date) FROM match_library.match_library_snapshot\n"
            f"            WHERE base_source_store = {frags['base_under']}\n"
            "              AND active = true AND deleted_date IS NULL\n"
            "        )"
        )
    return frags


# --------------------------------------------------------------------------- #
# Query builder — the inf.sql query, parameterized, with the known defects fixed:
#   * csf_semantic output now reads from `cs` (was wrongly duplicating `cg`).
#   * match_library filtered by latest snapshot, not wall-clock "today".
#   * junk '_' entries dropped from the IN lists.
#   * final `rn` helper column excluded from the output.
# The CTE/join/split logic is otherwise a faithful copy of the validated inf.sql.
# --------------------------------------------------------------------------- #
def build_inner_query(args, frags):
    input_table = args["input_table"]
    tenant = args["tenant"]
    segment = args["segment"]

    comp_bracket = frags["comp_bracket"]
    comp_under = frags["comp_under"]
    comp_names = frags["comp_names"]
    base_bracket = frags["base_bracket"]
    base_under = frags["base_under"]
    all_under = frags["all_under"]
    years_sql = frags["years_sql"]
    load_date_pred = frags["load_date_pred"]

    return f"""
WITH my_cte AS (
    SELECT base_sku, base_source_store, comp_source_store, lower(comp_sku) as comp_sku
    FROM {input_table}
),
crawl_attempt AS (
    SELECT
        LOWER(CAST(a.base_sku AS varchar)) AS base_sku,
        CASE WHEN b.product_title is null THEN FALSE ELSE TRUE end as base_sku_in_catalog,
        b.upc as upc,
        b.product_title as base_title,
        comp_sku,
        comp_source_store,
        base_source_store,
        b.segment,
        b.manufacturer_part_number as base_mpn
    FROM my_cte as a
    LEFT JOIN (
        Select sku, product_title, upc, segment, manufacturer_part_number
        FROM bungee_customercatalog.athena_auroradb_catalog
        WHERE source = '{tenant}'
        and capture_date = (
            select max(capture_date)
            from bungee_customercatalog.athena_auroradb_catalog
            where source = '{tenant}'
        )
        and (is_active = 'True' AND is_discontinued = 'False')
    ) as b
    ON(lower(cast(a.base_sku as varchar)) = lower(b.sku))
),
-- deep_crawl: the datalake table has columns that break a full-schema load, so
-- we project ONLY the scalar columns we need. Athena reads columns lazily, so
-- the problematic columns are never touched.
deep_crawl AS (
    SELECT DISTINCT
        LOWER(sku) AS dc_sku,
        LOWER(upc) AS dc_upc,
        LOWER(CONCAT_WS('_', source_name, store_name)) AS dc_source_store,
        MIN(capture_date) AS comp_sku_first_date
    FROM "bungeedatalake"."bungee_competitiveintelligence_datalake"
    WHERE year IN ({years_sql})
      AND source_name IN ({comp_names})
    GROUP BY 1, 2, 3
),
pdp AS (
    SELECT
        DISTINCT LOWER(sku) AS pdp_sku,
        product_title AS comp_title,
        product_url AS comp_url,
        LOWER(upc) AS pdp_upc,
        manufacturer_part_number AS pdp_mpn,
        REPLACE(source_store, '<>', '_') AS pdp_source_store,
        year||month||day as pdp_date
    FROM pdp_newdev.product_warehouse
    WHERE source_store IN ({comp_bracket})
      AND product_segment = '{segment}'
),
fl_dump AS (
    Select DISTINCT
        lower(base_sku) as base_sku,
        replace(base_source_store,'<>','_') as base_source_store,
        lower(comp_sku) as comp_sku,
        replace(comp_source_store,'<>','_') as comp_source_store,
        score, match_date, inserted_date, answer,
        array_distinct(array_agg(queue_name)) as fastlane_queue_name
    FROM ml_internal_uat.fastlane_dump
    WHERE base_source_store = {base_under}
      AND comp_source_store IN ({comp_under})
      AND segment = '{segment}'
    GROUP BY 1,2,3,4,5,6,7,8
),
csf_upc as(
    Select DISTINCT
        split_part(uuid_a,'<>',1) as base_upc,
        lower(split_part(uuid_a,'<>',2)) as base_sku,
        split_part(uuid_b,'<>',1) as comp_upc,
        lower(split_part(uuid_b,'<>',2)) as comp_sku,
        score,
        replace(split(uuid_a,'<>',3)[3],'<>','_') as base_source_store,
        replace(split(uuid_b,'<>',3)[3],'<>','_') as comp_source_store,
        MIN(year||month||day) as upc_sugg_date
    FROM product_seam_prod.type_upc_matches
    WHERE split(uuid_a,'<>',3)[3] = {base_bracket}
      AND split(uuid_b,'<>',3)[3] IN ({comp_bracket})
      AND segment = '{segment}'
    GROUP BY 1,2,3,4,5,6,7
),
csf_mpn as(
    Select DISTINCT
        lower(split_part(uuid_a,'<>',1)) as base_sku,
        lower(split_part(uuid_b,'<>',2)) as comp_sku,
        score,
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store,
        MIN(year||month||day) as mpn_sugg_date
    FROM product_seam_prod.type_mpn_matches
    WHERE base_source_store = {base_bracket}
      AND comp_source_store IN ({comp_bracket})
      AND segment = '{segment}'
    GROUP BY 1,2,3,4,5
),
match_lib AS (
    SELECT
        LOWER(base_sku) AS matchlib_sku,
        base_upc AS matchlib_upc,
        comp_sku AS matchlib_comp,
        comp_upc AS matchlib_comp_upc,
        base_source_store, comp_source_store, match_date, matcher_comments,
        MAX(CASE WHEN deleted_date IS NULL THEN true ELSE false END) AS is_present_in_match_library,
        MAX(CASE WHEN deleted_date IS NOT NULL THEN true ELSE false END) AS is_deleted_in_match_library,
        MAX(deleted_date) AS last_deleted_date,
        MIN(load_date) AS match_library_inserted_date
    FROM match_library.match_library_snapshot
    WHERE active = true and deleted_date is null
      AND base_source_store = {base_under}
      and comp_source_store IN ({comp_under})
      {load_date_pred}
    GROUP BY 1,2,3,4,5,6,7,8
),
csf_ml_input as(
    SELECT LOWER(sku) as inp_sku, product_title as inp_title, upc, source_store as inp_source_store, capture_date as inp_capture_date
    FROM product_seam_prod.checkpoint_precision_model_type_ml_input
    WHERE year||month||day IN (select max(year||month||day) from product_seam_prod.checkpoint_precision_model_type_ml_input WHERE source_store IN ({all_under}))
      AND source_store IN ({all_under}) AND segment = '{segment}'
    UNION
    SELECT LOWER(sku) as inp_sku, product_title as inp_title, upc, source_store as inp_source_store, capture_date as inp_capture_date
    FROM product_seam_prod.checkpoint_generic_model_type_ml_input
    WHERE year||month||day IN (select max(year||month||day) from product_seam_prod.checkpoint_generic_model_type_ml_input WHERE source_store IN ({all_under}))
      AND source_store IN ({all_under}) AND segment = '{segment}'
),
csf_precision as(
    Select DISTINCT
        lower(split_part(base_sku_uuid,'<>',1)) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku,
        score,
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store,
        MIN(year||month||day) as first_sugg_date,
        max(score) as model_score
    from product_seam_prod.checkpoint_precision_model_type_directed_pairs
    where base_source_store = {base_bracket}
      AND comp_source_store IN ({comp_bracket})
      AND segment = '{segment}'
    GROUP BY 1,2,3,4,5
),
csf_generic as (
    Select DISTINCT
        split_part(base_sku_uuid,'<>',1) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku,
        score,
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store,
        MIN(year||month||day) as first_sugg_date,
        max(score) as model_score
    from product_seam_prod.checkpoint_generic_model_type_directed_pairs
    where base_source_store = {base_bracket}
      AND comp_source_store IN ({comp_bracket})
      AND segment = '{segment}'
    GROUP BY 1,2,3,4,5
),
csf_semantic as (
    Select DISTINCT
        split_part(base_sku_uuid,'<>',1) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku,
        score,
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store,
        MIN(year||month||day) as first_sugg_date,
        max(score) as model_score
    from product_seam_prod.checkpoint_semantic_minilm_l6_v2_type_directed_pairs
    where base_source_store = {base_bracket}
      AND comp_source_store IN ({comp_bracket})
      AND segment = '{segment}'
    GROUP BY 1,2,3,4,5
),
csf_queue as (
    Select DISTINCT
        lower(base_sku) as base_sku,
        base_source_store,
        lower(comp_sku) as comp_sku,
        comp_source_store,
        MAX(aggregated_score) as queue_score,
        MIN(year||month||day) as queue_date,
        array_distinct(array_agg(queue_name)) as queue_name
    FROM product_seam_prod.type_queue
    WHERE tenant = '{tenant}'
      AND segment = '{segment}'
    GROUP BY 1,2,3,4
),
afm AS (
    select
        product_segment AS segment,
        company_code AS tenant,
        base_sku, search_key, search_key_type, search_type, comp_source_store,
        concat(base_sku,'<>',replace(comp_source_store,'_','<>')) as afm_request_key
    from ml_internal.afm_request_data
    where company_code = '{tenant}'
      and comp_source_store IN ({comp_under})
      AND product_segment = '{segment}'
),
result AS (
    SELECT
        a.*,
        CASE WHEN dc_sku IS NOT NULL THEN TRUE ELSE FALSE END AS sku_in_dc,
        comp_sku_first_date,
        dc_upc,
        CASE WHEN dc_upc IS NOT NULL THEN TRUE ELSE FALSE END AS upc_in_dc,
        CASE WHEN pdp_sku IS NOT NULL THEN TRUE ELSE FALSE END AS sku_in_pdp,
        pdp_date,
        pdp_upc,
        pdp_mpn,
        CASE WHEN pdp_upc IS NOT NULL THEN TRUE ELSE FALSE END AS upc_in_pdp,
        CASE WHEN f.inserted_date IS NOT NULL THEN TRUE ELSE FALSE END AS is_fl_attempted,
        f.inserted_date,
        f.answer,
        f.fastlane_queue_name,
        comp_title,
        CASE WHEN h.is_present_in_match_library THEN TRUE ELSE FALSE END AS is_present_in_match_library,
        h.is_deleted_in_match_library,
        h.last_deleted_date,
        h.match_library_inserted_date,
        h.matcher_comments,
        REGEXP_REPLACE(a.upc, '^0+', '') AS normalized_base_upc,
        REGEXP_REPLACE(b.dc_upc, '^0+', '') AS normalized_dc_upc,
        REGEXP_REPLACE(c.pdp_upc, '^0+', '') AS normalized_pdp_upc,
        CASE
            WHEN REGEXP_REPLACE(a.upc, '^0+', '') = REGEXP_REPLACE(b.dc_upc, '^0+', '')
              OR REGEXP_REPLACE(a.upc, '^0+', '') = REGEXP_REPLACE(c.pdp_upc, '^0+', '')
            THEN TRUE ELSE FALSE
        END AS has_upc_overlap,

        -- CSF UPC
        CASE WHEN cu.score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_upc,
        cu.score AS csf_upc_score,
        cu.upc_sugg_date AS csf_upc_first_sugg_date,
        cu.base_upc AS csf_base_upc,
        cu.comp_upc AS csf_comp_upc,

        -- CSF MPN
        CASE WHEN cm.score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_mpn,
        cm.score AS csf_mpn_score,
        cm.mpn_sugg_date AS csf_mpn_first_sugg_date,

        -- CSF Precision Model
        CASE WHEN cp.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_precision,
        cp.model_score AS csf_precision_score,
        cp.first_sugg_date AS csf_precision_first_sugg_date,

        -- CSF Generic Model
        CASE WHEN cg.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_generic,
        cg.model_score AS csf_generic_score,
        cg.first_sugg_date AS csf_generic_first_sugg_date,

        -- CSF Semantic Model  (reads from cs — the semantic CTE — not cg)
        CASE WHEN cs.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_semantic,
        cs.model_score AS csf_semantic_score,
        cs.first_sugg_date AS csf_semantic_first_sugg_date,

        -- CSF Queue
        CASE WHEN cq.queue_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_pair_in_csf_queue,
        cq.queue_score AS csf_queue_score,
        cq.queue_date AS csf_queue_date,
        cq.queue_name AS csf_queue_name,

        -- CSF ML Input: Base SKU
        CASE WHEN cmi_base.inp_sku IS NOT NULL THEN TRUE ELSE FALSE END AS base_sku_is_present_in_csf_ml_input,
        cmi_base.inp_title AS base_title_csf_ml_input,
        cmi_base.inp_capture_date AS base_capture_date_csf,

        -- CSF ML Input: Comp SKU
        CASE WHEN cmi_comp.inp_sku IS NOT NULL THEN TRUE ELSE FALSE END AS comp_sku_is_present_in_csf_ml_input,
        cmi_comp.inp_title AS comp_title_csf_ml_input,
        cmi_comp.inp_capture_date AS comp_capture_date_csf,

        -- AFM Suggestions
        CASE WHEN afm.afm_request_key IS NOT NULL THEN TRUE ELSE FALSE END AS has_afm_request,
        afm.search_type AS afm_search_type,
        afm.search_key_type AS afm_search_key_type
    FROM crawl_attempt AS a
    LEFT JOIN deep_crawl AS b
        ON a.comp_source_store = b.dc_source_store
        AND a.comp_sku = b.dc_sku
    LEFT JOIN pdp AS c
        ON a.comp_source_store = c.pdp_source_store
        AND a.comp_sku = c.pdp_sku
    LEFT JOIN fl_dump AS f
        ON a.comp_source_store = f.comp_source_store
        AND a.base_sku = f.base_sku
        AND a.comp_sku = f.comp_sku
    LEFT JOIN match_lib AS h
        ON a.comp_source_store = h.comp_source_store
        AND a.base_sku = h.matchlib_sku
        AND a.comp_sku = h.matchlib_comp
    LEFT JOIN csf_generic as cg
        ON a.comp_source_store = cg.comp_source_store
        AND a.base_sku = cg.base_sku
        AND a.comp_sku = cg.comp_sku
    LEFT JOIN csf_semantic as cs
        ON a.comp_source_store = cs.comp_source_store
        AND a.base_sku = cs.base_sku
        AND a.comp_sku = cs.comp_sku
    LEFT JOIN csf_precision AS cp
        ON a.comp_source_store = cp.comp_source_store
        AND a.base_sku = cp.base_sku
        AND a.comp_sku = cp.comp_sku
    LEFT JOIN csf_queue AS cq
        ON a.comp_source_store = cq.comp_source_store
        AND a.base_sku = cq.base_sku
        AND a.comp_sku = cq.comp_sku
    LEFT JOIN csf_ml_input as cmi_base
       ON a.base_sku = cmi_base.inp_sku
       AND a.base_source_store = cmi_base.inp_source_store
    LEFT JOIN csf_ml_input AS cmi_comp
        ON a.comp_sku = cmi_comp.inp_sku
        AND a.comp_source_store = cmi_comp.inp_source_store
    LEFT JOIN afm as afm
        ON a.base_sku = afm.base_sku
       AND a.comp_source_store = afm.comp_source_store
    LEFT JOIN csf_upc as cu
        ON a.base_sku = cu.base_sku
        AND a.comp_source_store = cu.comp_source_store
        AND a.comp_sku = cu.comp_sku
    LEFT JOIN csf_mpn as cm
        ON a.base_sku = cm.base_sku
        AND a.comp_source_store = cm.comp_source_store
        AND a.comp_sku = cm.comp_sku
)
-- Keep one row per (base_sku, comp_sku, comp_source_store). We SELECT * (which
-- carries a constant rn column, exactly as inf.sql does) rather than
-- `SELECT * EXCEPT (rn)` — this Athena engine parses EXCEPT only as the set
-- operator, not column exclusion.
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY base_sku, comp_sku, comp_source_store
               ORDER BY base_sku DESC
           ) AS rn
    FROM result
) t
WHERE rn = 1
""".strip()


def build_unload_sql(inner_query, partition_path):
    """Wrap the analysis query in an Athena UNLOAD that writes Parquet to the
    tenant/run_date partition path. The path is cleared beforehand, so the
    location is empty and only this partition is (re)written."""
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
    """Server-side copy every object under src_uri to dst_uri, preserving the
    key suffix. Returns the number of objects copied."""
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
        # NOTE: if `workgroup` has "enforce workgroup configuration" enabled, Athena
        # ignores this OutputLocation in favour of the workgroup's own — so
        # --athena_output_location has no effect in that case (not an error).
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

    # Timed out waiting — cancel the still-running query so it can't keep writing
    # output to staging after we've already reported failure.
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
    tenant, run_date, segment = args["tenant"], args["run_date"], args["segment"]
    job_name = args.get("JOB_NAME", "inf_recall_analysis")

    output_location = args.get("output_location", "s3://ml-inf-recall/inf_report_final/")
    if not output_location.endswith("/"):
        output_location += "/"
    staging_location = args.get("athena_output_location", "s3://ml-inf-recall/athena_staging/")
    workgroup = args.get("athena_workgroup")
    # Fully-qualified table names make the context db mostly cosmetic; default to
    # the input table's database.
    default_db = args["input_table"].split(".")[0] if "." in args["input_table"] else "default"
    database = args.get("athena_database", default_db)

    partition_path = f"{output_location}tenant={tenant}/run_date={run_date}/"
    # UNLOAD writes here first; the live partition is only touched once the query
    # has succeeded and produced data. '_staging' is underscore-prefixed so Glue
    # crawlers / Spark ignore it. (Assumes the same tenant+run_date is not
    # processed by two concurrent runs.)
    stage_path = f"{output_location}_staging/tenant={tenant}/run_date={run_date}/"

    logger.info("Step 2: Building parameterized SQL...")
    frags = build_fragments(args)
    inner_query = build_inner_query(args, frags)
    unload_sql = build_unload_sql(inner_query, stage_path)
    print(unload_sql)

    try:
        # Step 3: UNLOAD into the scratch prefix. Clear it first to drop any
        # orphans from a previously killed run (UNLOAD requires an empty target).
        logger.info(f"Step 3: Staging output under {stage_path} ...")
        clear_s3_prefix(stage_path)

        logger.info("Step 4: Running Athena UNLOAD (this may take a while)...")
        query_id = run_athena(unload_sql, database, staging_location, workgroup)

        # Step 5: promote. Only now do we clear the live partition and copy the
        # freshly staged files in — a failed query never touches existing data.
        # If the query produced nothing (0 rows), leave the old partition intact
        # rather than replacing good data with an empty result.
        probe = s3.list_objects_v2(
            Bucket=urlparse(stage_path).netloc,
            Prefix=urlparse(stage_path).path.lstrip("/"),
            MaxKeys=1,
        )
        if probe.get("KeyCount", 0) == 0:
            msg = (
                f"⚠️ INF recall analysis `{job_name}` produced 0 rows for tenant "
                f"`{tenant}` (segment `{segment}`, run_date `{run_date}`). The existing "
                f"partition was left unchanged — check competitors / segment / dates."
            )
            logger.warning(msg)
            # send_slack(msg)  # alerts disabled
        else:
            logger.info(f"Step 5: Promoting staged data to {partition_path} ...")
            clear_s3_prefix(partition_path)
            copied = copy_s3_prefix(stage_path, partition_path)
            logger.info(f"Step 6: Done — {copied} file(s) at {partition_path} (query {query_id}).")
            # send_slack(  # alerts disabled
            #     f"✅ INF recall analysis `{job_name}` completed for tenant `{tenant}` "
            #     f"(segment `{segment}`, run_date `{run_date}`, {copied} file(s))."
            # )
    except Exception as exc:
        logger.error(f"INF recall analysis failed: {exc}")
        # send_slack(  # alerts disabled
        #     f"❌ INF recall analysis `{job_name}` FAILED for tenant `{tenant}` "
        #     f"(run_date `{run_date}`): {exc}"
        # )
        raise
    finally:
        # Always remove the scratch prefix, whether we promoted, skipped, or failed.
        try:
            clear_s3_prefix(stage_path)
        except ClientError as cleanup_exc:
            logger.warning(f"Staging cleanup failed for {stage_path}: {cleanup_exc}")


if __name__ == "__main__":
    main()
