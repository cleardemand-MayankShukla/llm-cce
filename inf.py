import sys
import json
import logging
import requests
import boto3
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from urllib.parse import urlparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_secret(secret_name, region_name="us-east-1"):
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {e}")
        raise e
    
    secret_string = get_secret_value_response.get('SecretString')
    if secret_string:
        return json.loads(secret_string)
    raise ValueError("SecretString not found in the response.")

def send_message_to_slack(message):
    """Send a message to Slack using webhook from Secrets Manager."""
    try:
        secret = get_secret("ml-inf-recall-mail-slack")
        slack_url = secret.get("slack_webhook_url")

        if not slack_url:
            raise ValueError("Slack webhook URL not found in secret.")

        payload = {
            "channel": "#ml-inf-recall",
            "username": "INF_ANALYSIS",
            "text": message
        }

        response = requests.post(
            slack_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        logger.info("✅ Slack notification sent.")
    except Exception as e:
        logger.error(f"❌ Slack notification failed: {e}")

# 1. Parse Glue Job arguments
logger.info("Step 1: Parsing Glue Job arguments...")
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'segment',     # e.g., 'grocery'
    'input_table',  # 1. e.g., 'ml_temp_table.chewy_non_core_inf_analysis_20251203'
    'tenant',       # 2. e.g., 'chewy'
    'competitors',  # 3. e.g., 'amazon,walmart,target'
    'run_date'      # e.g., '2025-12-03'
])

input_table = args['input_table'].strip()
tenant = args['tenant'].strip()
run_date = args['run_date'].strip()
segment = args['segment'].strip()

if not input_table or not tenant or not run_date or not segment:
    raise ValueError(f"Job arguments cannot be empty strings! Received: input_table='{input_table}', tenant='{tenant}', run_date='{run_date}', segment='{segment}'")
# 'amazon,walmart,target' -> ['amazon', 'walmart', 'target']
competitors_list = [c.strip() for c in args['competitors'].split(',')]

# 2. Prepare dynamically formatted strings to substitute in the SQL query
logger.info("Step 2: Preparing dynamic SQL substituting strings...")
# Request 3: competitors list as plain text IN clause
comp_list_sql = ", ".join([f"'{c}'" for c in competitors_list])

# Request 4 & 5: competitor<>competitor and tenant<>tenant
comp_bracket_list_sql = ", ".join([f"'{c}<>{c}'" for c in competitors_list])
base_bracket = f"'{tenant}<>{tenant}'"

# For tables that use underscore instead of angled brackets (e.g., 'amazon_amazon')
comp_under_list_sql = ", ".join([f"'{c}_{c}'" for c in competitors_list])
base_under = f"'{tenant}_{tenant}'"

# For ml_input where both tenant_tenant and competitor_competitor are in the IN clause
all_under_list = [f"'{c}_{c}'" for c in competitors_list] + [base_under]
all_under_list_sql = ", ".join(all_under_list)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Enable native Spark Parquet reader (crucial for ignoreMissingFiles to work)
spark.conf.set("spark.sql.hive.convertMetastoreParquet", "true")
spark.conf.set("spark.sql.legacy.allowNegativeScaleOfDecimal", "true")
spark.conf.set("spark.sql.parquet.mergeSchema", "false")
spark.conf.set("spark.sql.parquet.filterPushdown", "true")

# ---- FIX: Ignore missing S3 files/partitions that exist in Glue Catalog but not S3 ----
spark.conf.set("spark.sql.files.ignoreMissingFiles", "true")
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
spark.conf.set("spark.sql.parquet.ignoreMissingFiles", "true")
spark.conf.set("spark.sql.hive.verifyPartitionPath", "false")
spark.conf.set("spark.hadoop.mapred.input.dir.existing.only", "true")
spark.conf.set("spark.hadoop.mapreduce.input.fileinputformat.input.dir.existing.only", "true")
spark.conf.set("spark.hadoop.fs.s3.ignore.missing.files", "true")

# Advanced Spark configs to prevent partition listing errors
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")
spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")

# Apply Hadoop configurations directly to Spark Context (most robust for Glue)
hadoop_conf = sc._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3.ignore.missing.files", "true")
hadoop_conf.set("mapred.input.dir.existing.only", "true")
hadoop_conf.set("mapreduce.input.fileinputformat.input.dir.existing.only", "true")
hadoop_conf.set("spark.sql.hive.verifyPartitionPath", "false")
# ---- ULTIMATE FIX: Schema & Location-aware S3 loading with Boto3 verification ----
from urllib.parse import urlparse

def get_table_location_from_glue_catalog(catalog_table):
    try:
        glue = boto3.client('glue')
        if '.' not in catalog_table:
            logger.warning(f"Glue catalog lookup requires qualified table name: {catalog_table}")
            return None
        database, table = catalog_table.split('.', 1)
        response = glue.get_table(DatabaseName=database, Name=table)
        location = response['Table']['StorageDescriptor'].get('Location')
        if location:
            return location.rstrip('/') + '/'
    except Exception as e:
        logger.warning(f"Glue API location lookup failed for {catalog_table}: {e}")
    return None


def get_table_location(catalog_table):
    """Helper to find the S3 location of a catalog table."""
    try:
        desc = spark.sql(f"DESCRIBE FORMATTED {catalog_table}").collect()
        for row in desc:
            if row['col_name'].strip() == 'Location':
                return row['data_type'].strip().rstrip('/') + '/'
    except Exception as e:
        logger.warning(f"Could not find location for {catalog_table} via DESCRIBE FORMATTED: {e}")
    return get_table_location_from_glue_catalog(catalog_table)


def load_resilient_view(catalog_table, view_name, partitions=None):
    """
    1. Gets schema from Glue Catalog.
    2. Verifies physical S3 paths exist using Boto3.
    3. Loads verified paths into a View.
    """
    logger.info(f"Preparing resilient view {view_name} using catalog {catalog_table}...")
    
    # Step A: Get metadata from the catalog
    try:
        table_df = spark.table(catalog_table)
        schema = table_df.schema
        base_s3_path = get_table_location(catalog_table)
        
        # FIX: Manual override for the PDP bucket if the catalog is wrong
        if "pdp_newdev" in catalog_table and ("pdp-newdev.east1" in str(base_s3_path) or not base_s3_path):
             base_s3_path = "s3://pdp-newdev.east1/product_warehouse/"
             logger.info(f"Using manual S3 override for PDP: {base_s3_path}")
             
        if not base_s3_path:
            raise ValueError(f"Could not determine S3 location for {catalog_table}")
    except Exception as e:
        logger.warning(f"Metadata fetch failed for {catalog_table}: {e}. Attempting schema-less load from S3 paths.")
        schema = None  # Will load without schema
        base_s3_path = get_table_location(catalog_table)
        if not base_s3_path:
            logger.warning(f"Could not determine S3 location for {catalog_table} after metadata failure. Creating empty view instead.")
            spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW {view_name} AS SELECT CAST(NULL AS STRING) as dummy LIMIT 0")
            return

    # Step B: Verify physical S3 paths
    s3_client = boto3.client('s3')
    parsed = urlparse(base_s3_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    existing_paths = []
    if not partitions:
        partitions = [""]

    for part in partitions:
        full_prefix = f"{prefix}{part}"
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=full_prefix, MaxKeys=1)
        if 'Contents' in response:
            candidate = f"s3://{bucket}/{full_prefix}"
            if candidate not in existing_paths:
                existing_paths.append(candidate)
            logger.info(f"Verified: {view_name} -> {part}")

    if not existing_paths:
        # No verified physical path for this table / partition set.
        logger.warning(f"No S3 data found for {view_name} under {base_s3_path}. Creating empty view.")
        if schema:
            spark.createDataFrame(sc.emptyRDD(), schema).createOrReplaceTempView(view_name)
        else:
            fallback_cols = ["dummy"]
            select_clause = ", ".join([f"CAST(NULL AS STRING) as {c}" for c in fallback_cols])
            spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW {view_name} AS SELECT {select_clause} LIMIT 0")
        return

    def build_reader(use_schema, recursive=False):
        reader = spark.read \
            .option("mergeSchema", "true") \
            .option("ignoreCorruptFiles", "true")
        if recursive:
            reader = reader.option("recursiveFileLookup", "true")
        if use_schema and schema:
            reader = reader.schema(schema).option("basePath", base_s3_path)
        return reader

    try:
        reader = build_reader(use_schema=bool(schema), recursive=False)
        df = reader.parquet(*existing_paths)
        df.createOrReplaceTempView(view_name)
        logger.info(f"✅ View {view_name} ready ({len(existing_paths)} partitions).")
        return
    except Exception as e:
        logger.warning(f"Physical load failed for {view_name}: {e}. Retrying with recursive file lookup.")

    try:
        reader = build_reader(use_schema=bool(schema), recursive=True)
        load_paths = existing_paths if existing_paths else [base_s3_path]
        df = reader.parquet(*load_paths)
        df.createOrReplaceTempView(view_name)
        logger.info(f"✅ View {view_name} ready with recursiveFileLookup ({len(load_paths)} paths).")
        return
    except Exception as e:
        logger.error(f"Recursive fallback failed for {view_name}: {e}. Falling back to empty schema.")
        if schema:
            spark.createDataFrame(sc.emptyRDD(), schema).createOrReplaceTempView(view_name)
        else:
            fallback_cols = ["dummy"]
            select_clause = ", ".join([f"CAST(NULL AS STRING) as {c}" for c in fallback_cols])
            spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW {view_name} AS SELECT {select_clause} LIMIT 0")

# These are the sub-folders we expect to see on S3.
years_months = [
    "year=2025/month=12/",
    "year=2026/month=01/",
    "year=2026/month=02/",
    "year=2026/month=03/",
    "year=2026/month=04/"
]

pet_partitions = [f"segment=pets/{ym}" for ym in years_months]
tenant_partitions = [f"tenant={tenant}/{ym}" for ym in years_months]
standard_partitions = years_months

logger.info("Step 3: Loading all tables with Schema/Location-aware S3 verification...")

# Load models and matches
load_resilient_view("product_seam_prod.type_upc_matches", "type_upc_matches_resilient", pet_partitions)
load_resilient_view("product_seam_prod.type_mpn_matches", "type_mpn_matches_resilient", pet_partitions)
load_resilient_view("product_seam_prod.checkpoint_precision_model_type_directed_pairs", "checkpoint_precision_resilient", pet_partitions)
load_resilient_view("product_seam_prod.checkpoint_generic_model_type_directed_pairs", "checkpoint_generic_resilient", pet_partitions)
load_resilient_view("product_seam_prod.checkpoint_semantic_minilm_l6_v2_type_directed_pairs", "checkpoint_semantic_resilient", pet_partitions)

# Load ML inputs (trying both pet and standard partitions)
load_resilient_view("product_seam_prod.checkpoint_precision_model_type_ml_input", "precision_ml_input_resilient", pet_partitions + standard_partitions)
load_resilient_view("product_seam_prod.checkpoint_generic_model_type_ml_input", "generic_ml_input_resilient", pet_partitions + standard_partitions)


# Load PDP
load_resilient_view("pdp_newdev.product_warehouse", "pdp_resilient", standard_partitions)

# Queue table (often uses tenant in the path)
load_resilient_view("product_seam_prod.type_queue", "type_queue_resilient", tenant_partitions + standard_partitions)

# Load deep crawl data
load_resilient_view("bungeedatalake.bungee_competitiveintelligence_datalake", "deep_crawl_resilient", standard_partitions)

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ---- Datalake load removed due to persistent metadata errors ----
# We skip querying bungeedatalake.bungee_competitiveintelligence_datalake entirely
# and will rely on other catalog sources.

# 3. Parameterized SQL query
logger.info("Step 4: Compiling main parameterised SQL query...")
current_load_date = datetime.now().strftime('%Y-%m-%d')
sql_query = f"""
WITH my_cte AS (
    SELECT base_sku, base_source_store, comp_source_store, lower(comp_sku) as comp_sku 
    FROM {input_table}
),
crawl_attempt AS (
    SELECT
        LOWER(CAST(a.base_sku AS STRING)) AS base_sku,
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
    ON(lower(cast(a.base_sku as string)) = lower(b.sku))
),
deep_crawl AS (
    SELECT CAST(NULL AS STRING) AS dc_sku, CAST(NULL AS STRING) AS dc_upc, CAST(NULL AS STRING) AS dc_source_store, CAST(NULL AS STRING) AS comp_sku_first_date
    WHERE 1=0
),
pdp AS (
    SELECT 
        DISTINCT LOWER(sku) AS pdp_sku, 
        product_title AS comp_title, 
        product_url AS comp_url, 
        LOWER(upc) AS pdp_upc, 
        manufacturer_part_number AS pdp_mpn,
        REPLACE(source_store, '<>', '_') AS pdp_source_store,
        concat(year, month, day) as pdp_date
    FROM pdp_resilient
    WHERE source_store IN ({comp_bracket_list_sql}) 
    AND product_segment = '{segment}'
),
fl_dump AS (
    Select DISTINCT 
        lower(base_sku) as base_sku, 
        replace(base_source_store,'<>','_') as base_source_store,
        lower(comp_sku) as comp_sku, 
        replace(comp_source_store,'<>','_') as comp_source_store, 
        score, match_date, inserted_date, answer, queue_name
    FROM ml_internal_uat.fastlane_dump
    WHERE base_source_store = {base_under}
    AND comp_source_store IN ({comp_under_list_sql})
    AND segment = '{segment}'
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
        MIN(concat(year,month,day)) as upc_sugg_date
    FROM type_upc_matches_resilient
    WHERE split(uuid_a,'<>',3)[3] = {base_bracket}
    AND split(uuid_b,'<>',3)[3] IN ({comp_bracket_list_sql}) 
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
        MIN(concat(year,month,day)) as mpn_sugg_date
    FROM type_mpn_matches_resilient
    WHERE base_source_store = {base_bracket}
    AND comp_source_store IN ({comp_bracket_list_sql}) 
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
    and comp_source_store IN ({comp_under_list_sql})
    and load_date = '{current_load_date}'
    GROUP BY 1,2,3,4,5,6,7,8
),
csf_ml_input as(
    SELECT LOWER(sku) as inp_sku, product_title as inp_title, upc, source_store as inp_source_store, capture_date as inp_capture_date
    FROM precision_ml_input_resilient 
    WHERE concat(year,month,day) IN (select max(concat(year,month,day)) from precision_ml_input_resilient WHERE source_store IN ({all_under_list_sql})) 
    AND source_store IN ({all_under_list_sql}) AND segment = '{segment}'
    UNION 
    SELECT LOWER(sku) as inp_sku, product_title as inp_title, upc, source_store as inp_source_store, capture_date as inp_capture_date
    FROM generic_ml_input_resilient 
    WHERE concat(year,month,day) IN (select max(concat(year,month,day)) from generic_ml_input_resilient WHERE source_store IN ({all_under_list_sql})) 
    AND source_store IN ({all_under_list_sql}) AND segment = '{segment}'
),
csf_precision as(
    Select DISTINCT 
        lower(split_part(base_sku_uuid,'<>',1)) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku, 
        score, 
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store, 
        MIN(concat(year,month,day)) as first_sugg_date, 
        max(score) as model_score 
    from checkpoint_precision_resilient 
    where base_source_store = {base_bracket}
    AND comp_source_store IN ({comp_bracket_list_sql}) 
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
        MIN(concat(year,month,day)) as first_sugg_date, 
        max(score) as model_score
    from checkpoint_generic_resilient 
    where base_source_store = {base_bracket}
    AND comp_source_store IN ({comp_bracket_list_sql}) 
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
        MIN(concat(year,month,day)) as first_sugg_date, 
        max(score) as model_score
    from checkpoint_semantic_resilient 
    where base_source_store = {base_bracket}
    AND comp_source_store IN ({comp_bracket_list_sql}) 
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
        MIN(concat(year,month,day)) as queue_date, 
        array_distinct(collect_list(queue_name)) as queue_name
    FROM type_queue_resilient
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
    and comp_source_store IN ({comp_under_list_sql}) 
    AND segment = '{segment}'
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
        f.queue_name,
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
        cm.base_mpn AS csf_base_mpn,
        cm.comp_mpn AS csf_comp_mpn,
        
        -- CSF Precision Model
        CASE WHEN cp.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_precision,
        cp.model_score AS csf_precision_score,
        cp.first_sugg_date AS csf_precision_first_sugg_date,
        
        -- CSF Generic Model
        CASE WHEN cg.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_generic,
        cg.model_score AS csf_generic_score,
        cg.first_sugg_date AS csf_generic_first_sugg_date,
        
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
"""

print(sql_query)
try:
    logger.info("Step 5: Executing main SQL Query (this may take a while)...")
    # Execute the query
    df = spark.sql(sql_query)
    logger.info("Main SQL query executed successfully. Adding partition columns...")
    
    # Add partition columns to the dataframe
    from pyspark.sql.functions import lit
    df = df.withColumn("tenant", lit(tenant)) \
           .withColumn("run_date", lit(run_date))
    
    # Write the output to S3 partitioned by tenant and run_date
    output_path = "s3://ml-inf-recall/inf_report_analysis/"
    logger.info(f"Step 6: Writing output DataFrame to S3 partitions: {output_path}...")
    
    df.write \
      .mode("overwrite") \
      .partitionBy("tenant", "run_date") \
      .format("parquet") \
      .save(output_path)
    
    logger.info("Output successfully written to S3.")
    # Send success notification
    # send_message_to_slack(f"✅ Glue Job `{args['JOB_NAME']}` completed successfully for tenant `{tenant}`.")
    
except Exception as e:
    logger.error(f"Glue job failed: {e}")  # keep full error in logs
    # send_message_to_slack(f"❌ Glue Job `{args['JOB_NAME']}` FAILED for tenant `{tenant}`.")
    raise e

job.commit()
