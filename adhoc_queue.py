"""
Lambda 2: Run the parameterized INSERT INTO query for adhoc queue population.

Design:
  - Base product data: from ML Input (latest partition)
  - Comp product data: COALESCE(ML Input, PDP) — ML Input first, PDP as fallback
  - Mandatory fields are configurable; rows missing any mandatory field are excluded

Input event:
{
    "output_database": "ml_internal",
    "output_table": "type_adhoc_queue",

    "temp_input_database": "ml_temp_table",
    "temp_input_table": "feedersupply_0if_llm_verdict_queue",

    "match_library_database": "match_library",
    "match_library_table": "match_library_snapshot",
    "product_database": "product_seam_prod",
    "product_table": "checkpoint_precision_model_type_ml_input",
    "pdp_database": "pdp_newdev",                     (optional, default: "pdp_newdev")
    "pdp_table": "product_warehouse",                  (optional, default: "product_warehouse")

    "base_store_name_display": "Feederssup",
    "base_source_store_filter": "feederssup_feederssup",

    "domain": "pets",
    "segment": "pets",
    "queue_name": "feedersupply_0if_llm_verdict_adhoc_queue",

    "temp_table_where_filter": "verdict='exact_match'",
    "score_column": "score",

    "mandatory_fields": ["base_title", "base_url", "comp_title", "comp_url"],  (optional)

    "athena_output_location": "s3://product-seam.prod/athena_output/"
}
"""

import boto3
import time
import json
import os
from urllib.request import urlopen, Request
from urllib.error import URLError

athena_client = boto3.client("athena")
dynamodb = boto3.resource("dynamodb")

DOMAINS_TABLE = "domains"

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

POLL_INTERVAL_SEC = 2
MAX_POLL_ATTEMPTS = 150  # ~5 min max wait

DEFAULT_MANDATORY_FIELDS = ["base_title", "comp_title"]

MANDATORY_FIELD_SQL = {
    "base_title": "i_base.product_title",
    "base_url": "i_base.product_url",
    "base_img": "i_base.image_url",
    "base_description": "i_base.product_description",
    "base_brand": "i_base.brand",
    "comp_title": "COALESCE(NULLIF(i_comp.product_title,''), pdp.product_title)",
    "comp_url": "COALESCE(NULLIF(i_comp.product_url,''), pdp.product_url)",
    "comp_img": "COALESCE(NULLIF(i_comp.image_url,''), pdp.image_url)",
    "comp_description": "COALESCE(NULLIF(i_comp.product_description,''), pdp.product_description)",
    "comp_brand": "COALESCE(NULLIF(i_comp.brand,''), pdp.standardized_brand)",
}


def lambda_handler(event, context):
    output_database = event["output_database"]
    output_table = event["output_table"]
    output_full = f"{output_database}.{output_table}"

    temp_input_database = event["temp_input_database"]
    temp_input_table = event["temp_input_table"]
    temp_input_full = f"{temp_input_database}.{temp_input_table}"

    match_library_database = event["match_library_database"]
    match_library_table = event["match_library_table"]
    match_library_full = f"{match_library_database}.{match_library_table}"

    product_database = event["product_database"]
    product_table = event["product_table"]
    product_full = f"{product_database}.{product_table}"

    pdp_database = event.get("pdp_database", "pdp_newdev")
    pdp_table = event.get("pdp_table", "product_warehouse")
    pdp_full = f"{pdp_database}.{pdp_table}"

    base_store_display = event["base_store_name_display"]
    base_source_store_filter = event["base_source_store_filter"]

    company_code = base_source_store_filter.split("_")[0]

    output_location = event.get("athena_output_location", "s3://product-seam.prod/athena_output/")

    comp_source_stores = _get_comp_source_stores(company_code)
    if not comp_source_stores:
        return {
            "statusCode": 400,
            "body": {
                "message": f"No competitors found in domains table for client='{company_code}'"
            },
        }

    domain = event["domain"]
    segment = event["segment"]
    queue_name = event["queue_name"]

    match_lib_load_date = _get_latest_match_lib_load_date(
        match_library_full, company_code, output_location
    )
    if not match_lib_load_date:
        return {
            "statusCode": 400,
            "body": {
                "message": (
                    f"No load_date found in {match_library_full} for "
                    f"company_code='{company_code}'"
                )
            },
        }

    product_partition_date = event.get("product_partition_date", "")
    if product_partition_date:
        parts = product_partition_date.split("-")
        product_partition_year = parts[0]
        product_partition_month = parts[1]
        product_partition_day = parts[2]
    else:
        partition = _get_latest_product_partition(
            product_full, segment, output_location
        )
        if not partition:
            return {
                "statusCode": 400,
                "body": {
                    "message": (
                        f"No partitions found in {product_full} for "
                        f"segment='{segment}'"
                    )
                },
            }
        product_partition_year = partition["year"]
        product_partition_month = partition["month"]
        product_partition_day = partition["day"]

    temp_table_where_filter = event.get("temp_table_where_filter", "")

    bungee_review_state = event.get("bungee_review_state", "verified")
    customer_review_state = event.get("customer_review_state", "unverified")
    match_type = event.get("match_type", "exact")
    matcher = event.get("matcher", "MODEL")
    match_id = event.get("match_id", "c65p951d-09iv-4l12-8l09-vphl8p9184r5")
    match_status = event.get("match_status", "product_found")
    model_used = event.get("model_used", "afm-model")
    item_status = event.get("item_status", "new")
    status = event.get("status", "completed")
    expiration_days = event.get("expiration_days", 21)
    btfastlane_company_code = event.get("btfastlane_company_code", "btfastlane")
    score_column = event.get("score_column", "aggregated_score")

    mandatory_fields = event.get("mandatory_fields", DEFAULT_MANDATORY_FIELDS)

    # Narrow comp stores to only those present in the temp input table
    actual_comp_stores = _get_distinct_comp_stores_from_temp_table(
        temp_input_full, output_location
    )
    if actual_comp_stores:
        valid = set(comp_source_stores)
        actual_comp_stores = [s for s in actual_comp_stores if s in valid]

    comp_source_stores_to_use = actual_comp_stores if actual_comp_stores else comp_source_stores
    comp_stores_in = ", ".join(f"'{s}'" for s in comp_source_stores_to_use)
    comp_stores_pdp_in = ", ".join(
        f"'{s.replace('_', '<>')}'" for s in comp_source_stores_to_use
    )

    mandatory_where = _build_mandatory_where(mandatory_fields)

    sql = f"""
INSERT INTO {output_full}
SELECT * FROM (
SELECT
    CAST(CONCAT(filtered.base_sku, '_', filtered.base_source_store, '<>', filtered.comp_sku, '_', filtered.comp_source_store) AS VARCHAR) AS pair_id,
    CAST('True' AS VARCHAR) AS active,

    CAST(i_base.size_alt AS VARCHAR) AS base_alt_size,
    CAST(i_base.uom_alt AS VARCHAR) AS base_alt_uom,
    CAST(i_base.brand AS VARCHAR) AS base_brand,
    CAST(i_base.category AS VARCHAR) AS base_category,
    CAST(
        '{{base_strength_concentration=null, base_strength_concentration_uom=null,base_pharmacy_package_quantity=null, base_pharmacy_package_quantity_uom=null,base_total_quantity=null, base_total_quantity_uom=null,base_total_quantity_uom_normalized=null, base_product_total_size=null,base_product_total_uom=null}}' AS VARCHAR
    ) AS base_custom_attributes,

    CAST(i_base.product_description AS VARCHAR) AS base_description,
    CAST(i_base.dimension AS VARCHAR) AS base_dimensions,
    CAST(i_base.manufacturer_part_number AS VARCHAR) AS base_mfr_part_number,
    CAST(i_base.image_url AS VARCHAR) AS base_img,
    CAST(i_base.parent_sku AS VARCHAR) AS base_parent_sku,

    CAST(i_base.effective_price AS VARCHAR) AS base_price,
    CAST(i_base.shipping_weight AS VARCHAR) AS base_shipping_weight,
    CAST(i_base.size AS VARCHAR) AS base_size,
    CAST(filtered.base_sku AS VARCHAR) AS base_sku,

    CAST(
        CONCAT(
            COALESCE(NULLIF(i_base.match_sku, ''), filtered.base_sku),
            '_', filtered.base_source_store, '_', filtered.comp_source_store
        ) AS VARCHAR
    ) AS match_sku_base_source_store_comp_source_store,

    CAST(filtered.base_source_store AS VARCHAR) AS base_source_store,
    CAST(CONCAT(filtered.base_source_store, '_', filtered.comp_sku, '_', filtered.comp_source_store) AS VARCHAR)
        AS base_source_store_comp_sku_comp_source_store,

    CAST('{base_store_display}' AS VARCHAR) AS base_store_name_display,

    CAST(i_base.subcategory AS VARCHAR) AS base_subcategory,
    CAST(i_base.sub_subcategory AS VARCHAR) AS base_sub_subcategory,
    CAST(i_base.product_title AS VARCHAR) AS base_title,
    CAST(i_base.uom AS VARCHAR) AS base_uom,
    CAST(i_base.upc AS VARCHAR) AS base_upc,
    CAST(i_base.product_url AS VARCHAR) AS base_url,

    CAST('{bungee_review_state}' AS VARCHAR) AS bungee_review_state,
    CAST('{btfastlane_company_code}' AS VARCHAR) AS company_code,
    CAST(CONCAT('{btfastlane_company_code}', '_', filtered.base_source_store) AS VARCHAR)
        AS company_code_base_source_store,

    CAST(COALESCE(NULLIF(CAST(i_comp.size_alt AS VARCHAR),''), CAST(pdp.size_alt AS VARCHAR)) AS VARCHAR) AS comp_alt_size,
    CAST(COALESCE(NULLIF(i_comp.uom_alt,''), pdp.uom_alt) AS VARCHAR) AS comp_alt_uom,
    CAST(COALESCE(NULLIF(i_comp.brand,''), pdp.standardized_brand) AS VARCHAR) AS comp_brand,
    CAST(COALESCE(NULLIF(i_comp.category,''), pdp.category) AS VARCHAR) AS comp_category,
    CAST(
        '{{comp_strength_concentration=null, comp_strength_concentration_uom=null,comp_pharmacy_package_quantity=null, comp_pharmacy_package_quantity_uom=null,comp_total_quantity=null,comp_total_quantity_uom=null,comp_total_quantity_uom_normalized=null, comp_converted_quantity=null,comp_converted_quantity_uom_normalized=null, comp_product_total_size=null,comp_product_total_uom=null}}' AS VARCHAR
    ) AS comp_custom_attributes,

    CAST(COALESCE(NULLIF(i_comp.product_description,''), pdp.product_description) AS VARCHAR) AS comp_description,
    CAST(COALESCE(NULLIF(i_comp.dimension,''), pdp.dimension) AS VARCHAR) AS comp_dimensions,
    CAST(COALESCE(NULLIF(i_comp.image_url,''), pdp.image_url) AS VARCHAR) AS comp_img,
    CAST(COALESCE(NULLIF(i_comp.parent_sku,''), pdp.parent_sku) AS VARCHAR) AS comp_parent_sku,

    CAST(COALESCE(NULLIF(i_comp.effective_price,''), CAST(pdp.effective_price AS VARCHAR), '0.0') AS VARCHAR) AS comp_price,
    CAST(COALESCE(NULLIF(i_comp.shipping_weight,''), pdp.shipping_weight) AS VARCHAR) AS comp_shipping_weight,
    CAST(COALESCE(NULLIF(CAST(i_comp.size AS VARCHAR),''), CAST(pdp.size AS VARCHAR)) AS VARCHAR) AS comp_size,
    CAST(filtered.comp_sku AS VARCHAR) AS comp_sku,

    CAST(CONCAT(filtered.comp_sku, '_', filtered.comp_source_store, '_', filtered.base_source_store) AS VARCHAR)
        AS comp_sku_comp_source_store_base_source_store,

    CAST(filtered.comp_source_store AS VARCHAR) AS comp_source_store,

    CAST(
        CONCAT(
            filtered.comp_source_store, '_',
            COALESCE(NULLIF(i_comp.match_sku, ''), filtered.base_sku),
            '_', filtered.base_source_store
        ) AS VARCHAR
    ) AS comp_source_store_match_sku_base_source_store,

    CAST(split_part(filtered.comp_source_store, '_', 2) AS VARCHAR) AS comp_store_name_display,
    CAST(COALESCE(NULLIF(i_comp.subcategory,''), pdp.subcategory) AS VARCHAR) AS comp_subcategory,
    CAST(COALESCE(NULLIF(i_comp.sub_subcategory,''), pdp.sub_subcategory) AS VARCHAR) AS comp_sub_subcategory,
    CAST(COALESCE(NULLIF(i_comp.product_title,''), pdp.product_title) AS VARCHAR) AS comp_title,
    CAST(COALESCE(NULLIF(i_comp.manufacturer_part_number,''), pdp.manufacturer_part_number) AS VARCHAR) AS comp_mfr_part_number,
    CAST(COALESCE(NULLIF(i_comp.uom,''), pdp.uom) AS VARCHAR) AS comp_uom,
    CAST(COALESCE(NULLIF(i_comp.upc,''), pdp.upc) AS VARCHAR) AS comp_upc,
    CAST(COALESCE(NULLIF(i_comp.product_url,''), pdp.product_url) AS VARCHAR) AS comp_url,

    CAST('{customer_review_state}' AS VARCHAR) AS customer_review_state,
    CAST('{domain}' AS VARCHAR) AS domain,
    CAST(FLOOR(RAND() * 90000) + 10000 AS INTEGER) AS enqueue_index,
    CAST(TO_UNIXTIME(CURRENT_DATE + INTERVAL '{expiration_days}' DAY) AS INTEGER) AS expiration_date,

    CAST('{item_status}' AS VARCHAR) AS item_status,
    CAST('{match_type}' AS VARCHAR) AS match,
    CAST('{matcher}' AS VARCHAR) AS matcher,
    CAST(TO_UNIXTIME(CURRENT_DATE) AS INTEGER) AS match_date,
    CAST('{match_id}' AS VARCHAR) AS match_id,

    CAST(COALESCE(NULLIF(i_base.match_sku, ''), filtered.base_sku) AS VARCHAR) AS match_sku,
    CAST('{match_status}' AS VARCHAR) AS match_status,
    CAST('{model_used}' AS VARCHAR) AS model_used,

    CAST('{queue_name}_{item_status}' AS VARCHAR) AS queue_name_item_status,
    CAST({score_column} AS DOUBLE) AS score,
    CAST('{status}' AS VARCHAR) AS status,
    CAST('{queue_name}' AS VARCHAR) AS queue_name,

    CAST('{queue_name}' AS VARCHAR) AS adhoc_queue_name,
    CAST(current_date AS VARCHAR) AS load_date

FROM (
    SELECT deduped.*
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY base_sku, comp_sku, base_source_store, comp_source_store
            ) AS rnk
        FROM {temp_input_full}
        -- WHERE {temp_table_where_filter}
    ) deduped
    LEFT JOIN (
        SELECT LOWER(base_sku) AS base_sku,
               comp_source_store,
               base_source_store
        FROM {match_library_full}
        WHERE load_date = '{match_lib_load_date}'
          AND company_code = '{company_code}'
          AND active
          AND deleted_date IS NULL
    ) b
      ON LOWER(deduped.base_sku) = b.base_sku
     AND deduped.comp_source_store = b.comp_source_store
     AND deduped.base_source_store = b.base_source_store
    WHERE deduped.rnk = 1
      AND b.base_sku IS NULL
) filtered

-- Base: ML Input (latest partition)
LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY sku, source_store ORDER BY sku
    ) AS rn_base
    FROM {product_full}
    WHERE year = '{product_partition_year}'
      AND month = '{product_partition_month}'
      AND day = '{product_partition_day}'
      AND segment = '{segment}'
      AND source_store = '{base_source_store_filter}'
) i_base
  ON LOWER(filtered.base_sku) = LOWER(i_base.sku)
 AND LOWER(filtered.base_source_store) = LOWER(i_base.source_store)
 AND i_base.rn_base = 1

-- Comp: ML Input (latest partition, only stores present in temp input)
LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY sku, source_store ORDER BY sku
    ) AS rn_comp
    FROM {product_full}
    WHERE year = '{product_partition_year}'
      AND month = '{product_partition_month}'
      AND day = '{product_partition_day}'
      AND segment = '{segment}'
      AND source_store IN ({comp_stores_in})
) i_comp
  ON LOWER(filtered.comp_sku) = LOWER(i_comp.sku)
 AND LOWER(filtered.comp_source_store) = LOWER(i_comp.source_store)
 AND i_comp.rn_comp = 1

-- Comp fallback: PDP (only stores present in temp input)
LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY sku, source_store ORDER BY capture_date DESC
    ) AS rn_pdp
    FROM {pdp_full}
    WHERE source_store IN ({comp_stores_pdp_in})
) pdp
  ON LOWER(filtered.comp_sku) = LOWER(pdp.sku)
 AND LOWER(replace(filtered.comp_source_store, '_', '<>')) = LOWER(pdp.source_store)
 AND pdp.rn_pdp = 1
) enriched
{mandatory_where}
"""

    print(sql)
    result = _execute_query(sql, output_database, output_location)

    if result["status"] == "SUCCEEDED":
        row_count = _verify_insert_success(
            result["query_execution_id"], output_full, output_location
        )

        print(f"INSERT query completed. Rows inserted: {row_count}")

        if row_count is None:
            error_msg = "INSERT reported success but could not verify row count"
            _send_slack_notification(
                "failure",
                error_msg,
                {
                    "Table": output_full,
                    "Query ID": result["query_execution_id"],
                    "Issue": "Verification query failed",
                },
            )
            return {
                "statusCode": 500,
                "body": {
                    "message": error_msg,
                    "query_execution_id": result["query_execution_id"],
                },
            }
        elif row_count == 0:
            warning_msg = "INSERT completed but inserted 0 rows. Check if records match filtering criteria."
            _send_slack_notification(
                "warning",
                warning_msg,
                {
                    "Table": output_full,
                    "Query ID": result["query_execution_id"],
                    "Rows Inserted": row_count,
                    "Possible Cause": "All records filtered by mandatory field validation",
                },
            )
            return {
                "statusCode": 200,
                "body": {
                    "message": warning_msg,
                    "query_execution_id": result["query_execution_id"],
                    "rows_inserted": row_count,
                },
            }
        else:
            success_msg = f"INSERT into {output_full} completed successfully"
            _send_slack_notification(
                "success",
                success_msg,
                {
                    "Table": output_full,
                    "Query ID": result["query_execution_id"],
                    "Rows Inserted": row_count,
                    "Queue Name": queue_name,
                },
            )
            return {
                "statusCode": 200,
                "body": {
                    "message": success_msg,
                    "query_execution_id": result["query_execution_id"],
                    "rows_inserted": row_count,
                },
            }
    else:
        error_msg = f"Query failed: {result['reason']}"
        _send_slack_notification(
            "failure",
            error_msg,
            {
                "Table": output_full,
                "Query ID": result["query_execution_id"],
                "Status": result["status"],
                "Reason": result["reason"],
            },
        )
        return {
            "statusCode": 500,
            "body": {
                "message": error_msg,
                "query_execution_id": result["query_execution_id"],
                "status": result["status"],
            },
        }


def _get_distinct_comp_stores_from_temp_table(temp_input_full, output_location):
    """Get distinct comp_source_store values actually present in the temp input."""
    sql = f"""
SELECT DISTINCT comp_source_store
FROM {temp_input_full}
WHERE comp_source_store IS NOT NULL
"""
    result = _execute_query(sql, temp_input_full.split(".")[0], output_location)
    if result["status"] != "SUCCEEDED":
        return []

    response = athena_client.get_query_results(
        QueryExecutionId=result["query_execution_id"]
    )
    rows = response["ResultSet"]["Rows"]
    return [
        row["Data"][0].get("VarCharValue", "")
        for row in rows[1:]
        if row["Data"][0].get("VarCharValue")
    ]


def _verify_insert_success(query_execution_id, output_table, output_location):
    time.sleep(1)
    verify_sql = f"""
SELECT COUNT(*) as row_count
FROM {output_table}
WHERE load_date = CAST(CURRENT_DATE AS VARCHAR)
LIMIT 1
"""
    result = _execute_query(
        verify_sql,
        output_table.split(".")[0],
        output_location,
    )

    if result["status"] != "SUCCEEDED":
        print(f"Verification query failed: {result['reason']}")
        return None

    try:
        response = athena_client.get_query_results(
            QueryExecutionId=result["query_execution_id"]
        )
        rows = response["ResultSet"]["Rows"]
        if len(rows) < 2:
            print("No results from verification query")
            return None
        return int(rows[1]["Data"][0].get("VarCharValue", "0"))
    except Exception as e:
        print(f"Error parsing verification query results: {str(e)}")
        return None


def _send_slack_notification(status, message, details=None):
    if not SLACK_WEBHOOK_URL or "YOUR/WEBHOOK" in SLACK_WEBHOOK_URL:
        print(f"Slack webhook not configured. Message: {message}")
        return False

    color_map = {
        "success": "#36a64f",
        "failure": "#ff0000",
        "warning": "#ffcc00",
    }

    payload = {
        "attachments": [
            {
                "color": color_map.get(status, "#808080"),
                "title": f"Lambda Insert Queue - {status.upper()}",
                "text": message,
                "fields": [],
                "ts": int(time.time()),
            }
        ]
    }

    if details:
        for key, value in details.items():
            payload["attachments"][0]["fields"].append(
                {"title": key, "value": str(value), "short": True}
            )

    try:
        req = Request(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=10) as response:
            print(f"Slack notification sent: {response.status}")
            return True
    except URLError as e:
        print(f"Error sending Slack notification: {str(e)}")
        return False


def _build_mandatory_where(mandatory_fields):
    if not mandatory_fields:
        return ""
    conditions = [
        f"{field} IS NOT NULL AND {field} != ''" for field in mandatory_fields
    ]
    return "WHERE " + "\n  AND ".join(conditions)


def _get_latest_match_lib_load_date(match_library_full, company_code, output_location):
    sql = f"""
SELECT load_date
FROM {match_library_full}
WHERE company_code = '{company_code}'
ORDER BY load_date DESC
LIMIT 1
"""
    result = _execute_query(sql, match_library_full.split(".")[0], output_location)
    if result["status"] != "SUCCEEDED":
        return None

    response = athena_client.get_query_results(
        QueryExecutionId=result["query_execution_id"]
    )
    rows = response["ResultSet"]["Rows"]
    if len(rows) < 2:
        return None
    return rows[1]["Data"][0].get("VarCharValue", "")


def _get_latest_product_partition(product_full, segment, output_location):
    sql = f"""
SELECT year, month, day
FROM {product_full}
WHERE segment = '{segment}'
ORDER BY (year || month || day) DESC
LIMIT 1
"""
    result = _execute_query(sql, product_full.split(".")[0], output_location)
    if result["status"] != "SUCCEEDED":
        return None

    response = athena_client.get_query_results(
        QueryExecutionId=result["query_execution_id"]
    )
    rows = response["ResultSet"]["Rows"]
    if len(rows) < 2:
        return None

    data = rows[1]["Data"]
    return {
        "year": data[0].get("VarCharValue", ""),
        "month": data[1].get("VarCharValue", ""),
        "day": data[2].get("VarCharValue", ""),
    }


def _get_comp_source_stores(tenant):
    table = dynamodb.Table(DOMAINS_TABLE)
    competitors = []

    response = table.query(
        IndexName="client-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("client").eq(tenant),
    )
    competitors.extend(item["primary"] for item in response.get("Items", []))

    while "LastEvaluatedKey" in response:
        response = table.query(
            IndexName="client-index",
            KeyConditionExpression=boto3.dynamodb.conditions.Key("client").eq(tenant),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        competitors.extend(item["primary"] for item in response.get("Items", []))

    return competitors


def _execute_query(sql, database, output_location):
    response = athena_client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    query_execution_id = response["QueryExecutionId"]

    for _ in range(MAX_POLL_ATTEMPTS):
        result = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        state = result["QueryExecution"]["Status"]["State"]

        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            reason = (
                result["QueryExecution"]["Status"].get("StateChangeReason", "")
            )
            return {
                "query_execution_id": query_execution_id,
                "status": state,
                "reason": reason,
            }
        time.sleep(POLL_INTERVAL_SEC)

    return {
        "query_execution_id": query_execution_id,
        "status": "TIMEOUT",
        "reason": "Exceeded max poll attempts",
    }
