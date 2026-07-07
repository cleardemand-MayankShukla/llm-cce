#!/usr/bin/env python3
"""
AWS Glue Job: Build hierarchical retriever .pkl bundles for a given tenant.

This job combines two data sources:
  1. Own-tenant matches: from match_library + fastlane_dump filtered by the tenant's base_source_store.
  2. Other-tenant matches (optional): matches where the competitor (comp_source_store) belongs to
     a competitor list defined in a JSON file, filtered by a category mapping parquet.

Job Parameters (set in the Glue job's --job-parameters):
  --tenant                  Primary tenant code (e.g. 'wfca')
  --segment                 The tenant's own segment (e.g. 'grocery'). Used to restrict synthetic
                            data to the same segment when include_other_segments=false.
  --include_other_tenants   'true'/'false' — master gate: if false, skip synthetic data entirely.
                            If true, pull matches from other tenants' comp pairs via category mapping.
  --include_other_segments  'true'/'false' — only applies when include_other_tenants=true.
                            If false, restrict synthetic rows to base SKUs whose catalog segment
                            matches --segment. If true, all segments are included.
  --category_map_file_path  Optional S3 folder path containing:
                              - parquet file(s): category mapping table
                              - competitor_list.json: list of competitor source-store prefixes
  --output_file_path        Base S3 folder for the output. The job scopes this
                            itself by segment/tenant — category_retriever.pkl,
                            segment_retriever.pkl, and metrics.json all land at:
                              {output_file_path}/{segment}/{tenant}/
                            (e.g. --output_file_path s3://my-bucket/retrievers
                             -> s3://my-bucket/retrievers/grocery/shamrock/category_retriever.pkl)

Dependencies (set via Glue job's --additional-python-modules):
  pandas, pyarrow, scikit-learn, boto3
"""

import sys
import json
import datetime
import pickle
import io

import boto3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext

# ---------------------------------------------------------------------------
# Glue / Spark bootstrap
# ---------------------------------------------------------------------------
sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

# Required args — Glue raises a clear error if any are missing
_args = getResolvedOptions(
    sys.argv,
    [
        "tenant",
        "segment",
        "include_other_tenants",
        "include_other_segments",
        "output_file_path",
        "category_map_file_path",
    ],
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_LABELS = {"exact_match", "equivalent_match", "not_a_match"}
MIN_EXAMPLES_PER_LABEL = 3

MATCH_LIBRARY_TABLE = 'match_library.match_library_snapshot'
FASTLANE_TABLE      = 'ml_internal_uat.fastlane_dump'
CATALOG_TABLE       = 'bungee_customercatalog.athena_auroradb_catalog'
PDP_TABLE           = 'pdp_newdev.product_warehouse'


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client():
    return boto3.client("s3")


def _parse_s3_uri(s3_uri: str):
    """Return (bucket, key) from an s3://bucket/key URI."""
    assert s3_uri.startswith("s3://"), f"Not an S3 URI: {s3_uri}"
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def s3_upload_bytes(data: bytes, s3_uri: str) -> None:
    """Upload raw bytes to an S3 URI."""
    bucket, key = _parse_s3_uri(s3_uri)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    print(f"  Uploaded {len(data):,} bytes → s3://{bucket}/{key}")


def s3_download_bytes(s3_uri: str) -> bytes:
    """Download raw bytes from an S3 URI."""
    bucket, key = _parse_s3_uri(s3_uri)
    resp = _s3_client().get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def s3_list_keys(s3_folder_uri: str, suffix: str = "") -> list:
    """List all object keys under an S3 folder URI that match a suffix."""
    bucket, prefix = _parse_s3_uri(s3_folder_uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# Category mapping loader
# ---------------------------------------------------------------------------

def load_category_mapping(s3_folder_uri: str) -> tuple:
    """
    Read category mapping from an S3 folder that contains:
      - One or more .parquet files  → pandas DataFrame (the mapping table)
      - competitor_list.json        → list of competitor source-store prefixes

    Returns:
        (mapping_df, competitor_list)
        mapping_df       : pd.DataFrame with columns [retailer, comp_category,
                           comp_subcategory, comp_sub_subcategory]
        competitor_list  : list[str]  e.g. ['londondrug', 'shoppers']
    """
    bucket, _ = _parse_s3_uri(s3_folder_uri)

    # --- Load competitor JSON ---
    json_keys = s3_list_keys(s3_folder_uri, suffix="competitor_list.json")
    if not json_keys:
        raise FileNotFoundError(
            f"No 'competitor_list.json' found under {s3_folder_uri}"
        )
    json_bytes = _s3_client().get_object(Bucket=bucket, Key=json_keys[0])["Body"].read()
    json_obj = json.loads(json_bytes)

    if isinstance(json_obj, list):
        competitor_list = json_obj
    elif isinstance(json_obj, dict):
        competitor_list = json_obj.get("competitors", [])
    else:
        raise ValueError("Invalid competitor_list.json format")
    
    if not all(isinstance(x, str) for x in competitor_list):
        raise ValueError("'competitors' must be a list of strings")

    # --- Load parquet mapping ---
    parquet_keys = s3_list_keys(s3_folder_uri, suffix=".parquet")
    if not parquet_keys:
        raise FileNotFoundError(
            f"No .parquet files found under {s3_folder_uri}"
        )
    frames = []
    for key in parquet_keys:
        raw = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        frames.append(pd.read_parquet(io.BytesIO(raw)))
    mapping_df = pd.concat(frames, ignore_index=True)
    print(
        f"  Loaded category mapping: {len(mapping_df):,} rows "
        f"from {len(parquet_keys)} parquet file(s)"
    )

    return mapping_df, competitor_list


# ---------------------------------------------------------------------------
# SQL query builders
# ---------------------------------------------------------------------------

def build_own_tenant_query(tenant: str) -> str:
    """
    SQL for own-tenant matches.
    Reads match_library + fastlane, deduplicates, joins catalog (base)
    and product_warehouse (comp), returns enriched rows.
    """

    return f"""
WITH match_library_data AS (
    SELECT
        base_sku,
        base_source_store,
        comp_sku,
        comp_source_store,
        match AS match_norm
    FROM {MATCH_LIBRARY_TABLE}
    WHERE active
      AND match IS NOT NULL
      AND match IN ('exact_match', 'equivalent_match', 'not_a_match')
      AND SPLIT_PART(base_source_store, '_', 1) IN ('{tenant}')
      AND company_code IN ('{tenant}')
      AND load_date = (
            SELECT MAX(load_date)
            FROM {MATCH_LIBRARY_TABLE}
            where company_code IN ('{tenant}')
          )
      AND deleted_date IS NULL
      AND comp_sku IS NOT NULL
),
fastlane_data AS (
    SELECT
        base_sku,
        base_source_store,
        comp_sku,
        comp_source_store,
        answer AS match_norm
    FROM {FASTLANE_TABLE}
    WHERE is_active
      AND answer IS NOT NULL
      AND answer IN ('exact_match', 'equivalent_match', 'not_a_match')
      AND SPLIT_PART(base_source_store, '_', 1) IN ('{tenant}')
),
match_union AS (
    SELECT * FROM match_library_data
    UNION ALL
    SELECT * FROM fastlane_data
),
deduped AS (
    SELECT DISTINCT
        base_sku,
        base_source_store,
        comp_sku,
        comp_source_store,
        match_norm
    FROM match_union
),
latest_catalog AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY source_store, sku
                ORDER BY capture_date DESC
            ) AS rn
        FROM {CATALOG_TABLE}
        WHERE source IN ('{tenant}')
    ) t
    WHERE rn = 1
),
base_with_categories AS (
    SELECT
        d.base_sku,
        d.base_source_store,
        d.comp_sku,
        d.comp_source_store,
        d.match_norm,
        ac.product_title   AS base_title,
        ac.brand           AS base_brand,
        ac.category        AS base_category,
        ac.subcategory     AS base_subcategory,
        ac.sub_subcategory AS base_sub_subcategory,
        ac.segment         AS base_segment
    FROM deduped d
    INNER JOIN latest_catalog ac
        ON  LOWER(d.base_sku)   = LOWER(ac.sku)
        AND d.base_source_store = ac.source_store
),
pdp_deduped AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY sku, REPLACE(source_store, '<>', '_')
                ORDER BY product_segment
            ) AS rn
        FROM {PDP_TABLE}
    ) t
    WHERE rn = 1
)
SELECT
    b.base_sku,
    b.base_source_store,
    b.comp_sku,
    b.comp_source_store,
    b.match_norm,
    b.base_title,
    b.base_brand,
    b.base_category,
    b.base_subcategory,
    b.base_sub_subcategory,
    pw.product_title      AS comp_title,
    pw.standardized_brand AS comp_brand,
    pw.category           AS comp_category,
    pw.subcategory        AS comp_subcategory,
    pw.sub_subcategory    AS comp_sub_subcategory
FROM base_with_categories b
INNER JOIN pdp_deduped pw
    ON  LOWER(b.comp_sku)   = LOWER(pw.sku)
    AND b.comp_source_store = REPLACE(pw.source_store, '<>', '_')
"""


def build_category_mapping_query(
    tenant: str,
    tenant_segment: str,
    include_other_segments: bool,
    competitor_list: list,
) -> str:
    """
    SQL for synthetic (other-tenant) matches filtered by category mapping.

    The temp view 'category_mapping_view' must already be registered by the
    caller before this SQL is executed.

    Args:
        tenant               : Primary tenant code — used to scope company_code filter.
        tenant_segment       : The tenant's own segment value from the catalog.
        include_other_segments: If False, a WHERE clause restricts base SKUs to only
                               those whose catalog segment matches tenant_segment.
                               If True, all segments are included.
        competitor_list      : List of competitor source-store prefixes from the
                               competitor_list.json in the category mapping folder.
    """
    comp_list_sql = ", ".join(f"'{c}'" for c in competitor_list)
    segment_filter = "" if include_other_segments else f"WHERE ac.segment = '{tenant_segment}'"

    # Drop the f-string entirely — use .format() for all substitutions so
    # the segment_filter (which may be empty) is never mixed with f-string braces.
    return """
WITH match_library_data AS (
    SELECT
        base_sku,
        base_source_store,
        comp_sku,
        comp_source_store,
        match AS match_norm
    FROM {match_library_table}
    WHERE active
      AND match IS NOT NULL
      AND match IN ('exact_match', 'equivalent_match', 'not_a_match')
      AND SPLIT_PART(comp_source_store, '_', 1) IN ({comp_list_sql})
      AND company_code IN ('{tenant}')
      AND load_date = (
            SELECT MAX(load_date)
            FROM {match_library_table}
          )
      AND deleted_date IS NULL
      AND comp_sku IS NOT NULL
),
fastlane_data AS (
    SELECT
        base_sku,
        base_source_store,
        comp_sku,
        comp_source_store,
        answer AS match_norm
    FROM {fastlane_table}
    WHERE is_active
      AND answer IS NOT NULL
      AND answer IN ('exact_match', 'equivalent_match', 'not_a_match')
      AND SPLIT_PART(comp_source_store, '_', 1) IN ({comp_list_sql})
),
match_union AS (
    SELECT * FROM match_library_data
    UNION ALL
    SELECT * FROM fastlane_data
),
deduped AS (
    SELECT DISTINCT
        base_sku,
        base_source_store,
        comp_sku,
        comp_source_store,
        match_norm
    FROM match_union
),
latest_catalog AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY source_store, sku
                ORDER BY capture_date DESC
            ) AS rn
        FROM {catalog_table}
    ) t
    WHERE rn = 1
),
base_with_categories AS (
    SELECT
        d.base_sku,
        d.base_source_store,
        d.comp_sku,
        d.comp_source_store,
        d.match_norm,
        ac.product_title   AS base_title,
        ac.brand           AS base_brand,
        ac.category        AS base_category,
        ac.subcategory     AS base_subcategory,
        ac.sub_subcategory AS base_sub_subcategory,
        ac.segment         AS base_segment
    FROM deduped d
    INNER JOIN latest_catalog ac
        ON  LOWER(d.base_sku)   = LOWER(ac.sku)
        AND d.base_source_store = ac.source_store
    {segment_filter}
),
pdp_deduped AS (
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY sku, REPLACE(source_store, '<>', '_')
                ORDER BY product_segment
            ) AS rn
        FROM {pdp_table}
    ) t
    WHERE rn = 1
),
enriched AS (
    SELECT
        b.base_sku,
        b.base_source_store,
        b.comp_sku,
        b.comp_source_store,
        b.match_norm,
        b.base_title,
        b.base_brand,
        b.base_category,
        b.base_subcategory,
        b.base_sub_subcategory,
        b.base_segment,
        pw.product_title      AS comp_title,
        pw.standardized_brand AS comp_brand,
        pw.category           AS comp_category,
        pw.subcategory        AS comp_subcategory,
        pw.sub_subcategory    AS comp_sub_subcategory
    FROM base_with_categories b
    INNER JOIN pdp_deduped pw
        ON  LOWER(b.comp_sku)   = LOWER(pw.sku)
        AND b.comp_source_store = REPLACE(pw.source_store, '<>', '_')
)
SELECT
    e.base_sku,
    e.base_source_store,
    e.comp_sku,
    e.comp_source_store,
    e.match_norm,
    e.base_title,
    e.base_brand,
    e.base_category,
    e.base_subcategory,
    e.base_sub_subcategory,
    e.base_segment,
    e.comp_title,
    e.comp_brand,
    e.comp_category,
    e.comp_subcategory,
    e.comp_sub_subcategory
FROM enriched e
JOIN category_mapping_view m
    ON  SPLIT_PART(e.comp_source_store, '_', 1) = m.retailer
    AND e.comp_category                         = m.comp_category
    AND e.comp_subcategory                      = m.comp_subcategory
    AND e.comp_sub_subcategory                  = m.comp_sub_subcategory
""".format(
        match_library_table=MATCH_LIBRARY_TABLE,
        fastlane_table=FASTLANE_TABLE,
        catalog_table=CATALOG_TABLE,
        pdp_table=PDP_TABLE,
        comp_list_sql=comp_list_sql,
        tenant=tenant,
        segment_filter=segment_filter,
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_spark_query(sql: str, spark_session) -> pd.DataFrame:
    """Run a SQL query via Spark and return a pandas DataFrame."""
    print(f"  Executing query ({len(sql):,} chars)...")
    print("=" * 80)
    print(sql)
    print("=" * 80)
    df_spark = spark_session.sql(sql)
    df = df_spark.toPandas()
    print(f"  Query returned {len(df):,} rows")
    return df


def fetch_data(
    tenant: str,
    tenant_segment: str,
    include_other_tenants: bool,
    include_other_segments: bool,
    category_map_file_path: str | None,
    spark_session,
) -> pd.DataFrame:
    """
    Orchestrate data collection from both SQL sources.

    Logic:
    - Always fetch own-tenant matches.
    - include_other_tenants is the master gate for synthetic data. If False,
      the category-mapping query is skipped entirely regardless of other flags.
    - If include_other_tenants=True and category_map_file_path is provided,
      run the category-mapping query. include_other_segments then controls
      whether that query is restricted to base SKUs in tenant_segment only.

    Returns:
        Combined, deduplicated pandas DataFrame.
    """
    frames = []

    # ---- 1. Own-tenant data ------------------------------------------------
    print("\n[1/2] Fetching own-tenant matches...")
    own_sql = build_own_tenant_query(tenant)
    own_df = fetch_spark_query(own_sql, spark_session)
    frames.append(own_df)
    print(f"  Own-tenant rows: {len(own_df):,}")

    # ---- 2. Synthetic (other-tenant) data — gated by include_other_tenants ---
    # include_other_tenants is the master on/off switch.
    # include_other_segments only controls the segment filter inside the query.
    if not include_other_tenants:
        print("\n[2/2] Skipping synthetic data (include_other_tenants=False)")
    elif category_map_file_path is None:
        print("\n[2/2] Skipping synthetic data (no category_map_file_path provided)")
    else:
        seg_note = "all segments" if include_other_segments else f"segment='{tenant_segment}' only"
        print(f"\n[2/2] Fetching synthetic competitor matches ({seg_note})...")

        mapping_df, competitor_list = load_category_mapping(category_map_file_path)

        # Register parquet mapping as an in-memory Spark temp view.
        # No Glue catalog table or Athena view needs to be pre-created.
        mapping_spark = spark_session.createDataFrame(mapping_df)
        mapping_spark.createOrReplaceTempView("category_mapping_view")
        print("  Registered 'category_mapping_view' as Spark temp view")

        cat_sql = build_category_mapping_query(
            tenant=tenant,
            tenant_segment=tenant_segment,
            include_other_segments=include_other_segments,
            competitor_list=competitor_list,
        )
        cat_df = fetch_spark_query(cat_sql, spark_session)
        frames.append(cat_df)
        print(f"  Synthetic rows: {len(cat_df):,}")

    # ---- Combine & deduplicate --------------------------------------------
    synth_df = frames[1] if len(frames) > 1 else pd.DataFrame(columns=own_df.columns)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["base_sku", "base_source_store", "comp_sku", "comp_source_store"]
    )
    print(
        f"\nCombined: {before:,} rows → {len(combined):,} after deduplication "
        f"(removed {before - len(combined):,} duplicates)"
    )
    return combined, own_df, synth_df


# ---------------------------------------------------------------------------
# Retriever build helpers  (ported from build_retriever.py)
# ---------------------------------------------------------------------------

def compose_weighted_query(
    base_title: str,
    comp_title: str,
    base_brand: str = "",
    comp_brand: str = "",
    category: str = "",
    subcategory: str = "",
    brand_weight: int = 3,
    category_weight: int = 2,
) -> str:
    """
    Build a single weighted text query from product fields.
    Tokens are repeated to simulate TF-IDF weight boosting.
    """
    parts = [base_title, comp_title]
    if base_brand:
        parts.extend([base_brand] * brand_weight)
    if comp_brand:
        parts.extend([comp_brand] * brand_weight)
    if category:
        parts.extend([category] * category_weight)
    if subcategory:
        parts.extend([subcategory] * category_weight)
    return " ".join(parts)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the combined DataFrame before building retrievers."""
    required = [
        "base_title", "comp_title", "match_norm",
        "base_brand", "comp_brand",
        "base_category", "base_subcategory",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Combined DataFrame is empty — cannot build retriever")

    invalid = set(df["match_norm"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Invalid match_norm labels found: {invalid}. "
            f"Expected one of: {VALID_LABELS}"
        )

    print("\n--- Label Distribution ---")
    dist = df["match_norm"].value_counts()
    for label in VALID_LABELS:
        count = int(dist.get(label, 0))
        pct = (count / len(df)) * 100
        flag = " ⚠️  LOW" if count < MIN_EXAMPLES_PER_LABEL else ""
        print(f"  {label}: {count:,} ({pct:.1f}%){flag}")
    print(f"  Total: {len(df):,} rows")


def build_bundle(
    df: pd.DataFrame,
    level: str,
    brand_weight: int = 3,
    category_weight: int = 2,
) -> dict:
    """
    Build a TF-IDF + KNN retriever bundle.

    Args:
        df            : Combined match DataFrame
        level         : 'category' or 'segment'
        brand_weight  : Token repetition weight for brand
        category_weight: Token repetition weight for category/subcategory

    Returns:
        Bundle dict ready for pickling.
    """
    print(f"\nBuilding {level}-level retriever bundle...")

    queries = []
    for _, row in df.iterrows():
        cat    = str(row.get("base_category", ""))    if level == "category" else None
        subcat = str(row.get("base_subcategory", "")) if level == "segment"  else None

        q = compose_weighted_query(
            base_title=str(row.get("base_title", "")),
            comp_title=str(row.get("comp_title", "")),
            base_brand=str(row.get("base_brand", "")),
            comp_brand=str(row.get("comp_brand", "")),
            category=cat or "",
            subcategory=subcat or "",
            brand_weight=brand_weight,
            category_weight=category_weight,
        )
        queries.append(q)

    rows_copy = df.copy()
    rows_copy["q"] = queries

    # TF-IDF
    print(f"  Fitting TF-IDF on {len(queries):,} documents...")
    vec = TfidfVectorizer(sublinear_tf=True, max_features=50_000)
    X = vec.fit_transform(queries)
    print(f"  Vocabulary size: {len(vec.vocabulary_):,}")

    # KNN
    k = min(20, len(queries))
    print(f"  Fitting NearestNeighbors (k={k})...")
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(X)

    bundle = {
        "vec": vec,
        "nn": nn,
        "rows": rows_copy,
        "level": level,
        "built_at": datetime.datetime.utcnow().isoformat(),
        "allowed_labels": list(VALID_LABELS),
    }
    print(f"  ✓ {level}-level bundle ready")
    return bundle


def sanity_check(bundle: dict) -> None:
    """Quick smoke test — run a dummy query through the bundle."""
    vec  = bundle["vec"]
    nn   = bundle["nn"]
    rows = bundle["rows"]

    dummy_q = "test product dummy brand"
    qv = vec.transform([dummy_q])
    distances, indices = nn.kneighbors(qv, n_neighbors=min(3, len(rows)))

    assert len(indices[0]) > 0, "KNN returned no neighbours"
    print(
        f"  ✓ Sanity check passed — retrieved {len(indices[0])} neighbours "
        f"for dummy query (level={bundle['level']})"
    )



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _brand_counts(df: pd.DataFrame) -> dict:
    """Return per-brand record counts from base_brand, sorted descending."""
    if df.empty or "base_brand" not in df.columns:
        return {}
    return (
        df["base_brand"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )


def compute_metrics(own_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    Build a metrics dict with row counts and per-brand breakdowns
    for own-tenant and synthetic data separately.
    """
    return {
        "own_data": {
            "count": len(own_df),
            "brand": _brand_counts(own_df),
        },
        "synth_data": {
            "count": len(synth_df),
            "brand": _brand_counts(synth_df),
        },
    }


def save_metrics_to_s3(metrics: dict, s3_uri: str) -> None:
    """Serialise metrics as indented JSON and upload to S3."""
    payload = json.dumps(metrics, indent=2).encode("utf-8")
    s3_upload_bytes(payload, s3_uri)

# ---------------------------------------------------------------------------
# Pickle save / load
# ---------------------------------------------------------------------------

def save_bundle_to_s3(bundle: dict, s3_uri: str) -> None:
    """Serialize a bundle with pickle and upload to S3."""
    buf = io.BytesIO()
    pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    s3_upload_bytes(buf.read(), s3_uri)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

tenant                 = _args["tenant"]
tenant_segment         = _args["segment"]
include_other_tenants  = _args["include_other_tenants"].lower() == "true"
include_other_segments = _args["include_other_segments"].lower() == "true"
category_map_file_path_raw   = _args.get("category_map_file_path", "-").strip()
category_map_file_path        = None if category_map_file_path_raw .lower() == "-" else category_map_file_path_raw
output_file_path       = _args["output_file_path"].rstrip("/")

# Scoped so every tenant/segment writes to its own folder:
#   {output_file_path}/{segment}/{tenant}/category_retriever.pkl
#   {output_file_path}/{segment}/{tenant}/segment_retriever.pkl
#   {output_file_path}/{segment}/{tenant}/metrics.json
scoped_output_path = f"{output_file_path}/{tenant_segment}/{tenant}"

print("=" * 60)
print("  Retriever Build Job")
print(f"  tenant                 : {tenant}")
print(f"  segment                : {tenant_segment}")
print(f"  include_other_tenants  : {include_other_tenants}")
print(f"  include_other_segments : {include_other_segments}")
print(f"  category_map_file_path : {category_map_file_path or '(not provided)'}")
print(f"  scoped_output_path     : {scoped_output_path}")
print("=" * 60)

# --- Step 1: Fetch data -----------------------------------------------------
df, own_df, synth_df = fetch_data(
    tenant=tenant,
    tenant_segment=tenant_segment,
    include_other_tenants=include_other_tenants,
    include_other_segments=include_other_segments,
    category_map_file_path=category_map_file_path,
    spark_session=spark,
)

# --- Step 2: Validate -------------------------------------------------------
print("\nValidating combined dataset...")
validate_dataframe(df)

# --- Step 3: Build bundles --------------------------------------------------
category_bundle = build_bundle(df, level="category")
segment_bundle  = build_bundle(df, level="segment")

# --- Step 4: Sanity checks --------------------------------------------------
print("\nRunning sanity checks...")
sanity_check(category_bundle)
sanity_check(segment_bundle)

# --- Step 5: Compute and save metrics ---------------------------------------
metrics = compute_metrics(own_df, synth_df)

metrics_path = f"{scoped_output_path}/metrics.json"
print(f"\nSaving metrics to: {metrics_path}")
save_metrics_to_s3(metrics, metrics_path)
print(json.dumps(metrics, indent=2))

# --- Step 6: Package and save -----------------------------------------------
# category_bundle and segment_bundle are written as separate pkls — a
# consumer that only needs one level doesn't have to load the other.
category_bundle["tenant"] = tenant
segment_bundle["tenant"]  = tenant

category_pkl_path = f"{scoped_output_path}/category_retriever.pkl"
segment_pkl_path  = f"{scoped_output_path}/segment_retriever.pkl"

print(f"\nSaving category-level bundle to: {category_pkl_path}")
save_bundle_to_s3(category_bundle, category_pkl_path)

print(f"\nSaving segment-level bundle to: {segment_pkl_path}")
save_bundle_to_s3(segment_bundle, segment_pkl_path)

print("\n✓ Retriever build job complete.")