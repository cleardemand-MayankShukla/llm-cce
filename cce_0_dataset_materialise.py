import os
import re
import abc
import sys
import json
import tarfile
import boto3
import numpy as np

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, ArrayType, DoubleType
)
from pyspark.sql.functions import pandas_udf
from pyspark import SparkFiles
from pyspark import StorageLevel
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

# ============================================================
# Glue Initialization
# ============================================================

sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session

# ============================================================
# Arguments
# ============================================================
#
# Glue job parameters:
# --tenant                          : staples
# --load_date                       : 2026-05-25
# --dynamodb_table                  : verdict_audit_env_config
# --dynamodb_config_key             : cce_dataset_automation_config
# --additional-python-modules       : s3://ml-etl-test-dataset/cce_automated_dataset/glue_deps/dependencies.gluewheels.zip
# --python-modules-installer-option : --no-index
# --extra-files                     : s3://ml-etl-test-dataset/cce_automated_dataset/models/model.tar.gz
# --enable-glue-datacatalog         : true
# --job-language                    : python

args = getResolvedOptions(
    sys.argv,
    ["tenant", "load_date", "dynamodb_table", "dynamodb_config_key"]
)

TENANT              = args["tenant"]
LOAD_DATE           = args["load_date"]
DYNAMODB_TABLE      = args["dynamodb_table"]
DYNAMODB_CONFIG_KEY = args["dynamodb_config_key"]

# ============================================================
# Config Loader — DynamoDB
# ============================================================

def load_config(table_name: str, config_key: str) -> dict:
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table    = dynamodb.Table(table_name)
    response = table.get_item(Key={"pk": config_key})
    item     = response.get("Item")
    if not item:
        raise ValueError(
            f"No config found in DynamoDB table '{table_name}' "
            f"for config_key='{config_key}'"
        )
    item.pop("pk", None)
    return item

# ============================================================
# Query Builder
# ============================================================

def build_query(tenant, load_date, ml_db, fastlane_table):
    query = """WITH match_lib AS (
            SELECT
                base_sku,
                base_source_store,
                comp_sku,
                comp_source_store,

                concat_ws('<>', base_sku, replace(base_source_store,'_','<>'), comp_sku, replace(comp_source_store,'_','<>')) AS pair_id,
                concat_ws('<>', base_sku, replace(base_source_store,'_','<>'))                                               AS sku_uuid_a,
                concat_ws('<>', comp_sku, replace(comp_source_store,'_','<>'))                                               AS sku_uuid_b,

                match         AS answer,
                segment,
                NULLIF(TRIM(base_title),       '') AS base_title,
                NULLIF(TRIM(base_description), '') AS base_description,
                NULLIF(TRIM(base_brand),       '') AS base_brand,
                base_upc, base_price, base_pack_size,

                NULLIF(TRIM(comp_title),       '') AS comp_title,
                NULLIF(TRIM(comp_description), '') AS comp_description,
                NULLIF(TRIM(comp_brand),       '') AS comp_brand,
                comp_upc, comp_price, comp_pack_size,

                1 AS source_priority

            FROM match_library.match_library_snapshot
            WHERE active       = true
              AND deleted_date IS NULL
              AND company_code = '${TENANT}'
              AND load_date    = '${LOAD_DATE}'
              AND match        IN ('exact', 'equivalent')
        ),

        match_lib_pair_ids AS (
            SELECT pair_id FROM match_lib
        ),

        fastlane AS (
            SELECT
                f.base_sku, f.base_source_store, f.comp_sku, f.comp_source_store,

                concat_ws('<>', f.base_sku, replace(f.base_source_store,'_','<>'), f.comp_sku, replace(f.comp_source_store,'_','<>')) AS pair_id,
                concat_ws('<>', f.base_sku, replace(f.base_source_store,'_','<>'))                                                    AS sku_uuid_a,
                concat_ws('<>', f.comp_sku, replace(f.comp_source_store,'_','<>'))                                                    AS sku_uuid_b,

                f.answer, f.segment,
                NULL AS base_title, NULL AS base_description, NULL AS base_brand,
                f.base_upc,
                NULL AS base_price, NULL AS base_pack_size,
                NULL AS comp_title, NULL AS comp_description, NULL AS comp_brand,
                f.comp_upc,
                NULL AS comp_price, NULL AS comp_pack_size,
                2 AS source_priority

            FROM ${ML_DB}.${FASTLANE_TABLE} f
            LEFT JOIN match_lib_pair_ids m
                   ON m.pair_id = concat_ws('<>', f.base_sku, replace(f.base_source_store,'_','<>'),
                                                 f.comp_sku,  replace(f.comp_source_store,'_','<>'))
            WHERE f.year || f.month  >= '202401'
              AND f.answer            IN ('exact', 'equivalent', 'not_a_match')
              AND f.base_source_store = '${TENANT}_${TENANT}'
              AND m.pair_id           IS NULL
        ),

        combined AS (
            SELECT * FROM match_lib
            UNION ALL
            SELECT * FROM fastlane
        ),

        combined_deduped AS (
            SELECT *
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY pair_id ORDER BY source_priority) AS rn
                FROM combined
            )
            WHERE rn = 1
        ),

        latest_catalog AS (
            SELECT
                sku,
                concat(source, '_', source) AS source_store,
                NULLIF(TRIM(product_title),       '') AS product_title,
                NULLIF(TRIM(product_description), '') AS product_description,
                NULLIF(TRIM(brand),               '') AS brand,
                manufacturer_part_number
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY source_store, sku
                           ORDER BY capture_date DESC
                       ) AS rn
                FROM bungee_customercatalog.athena_auroradb_catalog
                WHERE source = '${TENANT}'
                  AND sku IN (SELECT DISTINCT base_sku FROM combined_deduped)
            )
            WHERE rn = 1
        ),

        relevant_comp_skus AS (
            SELECT DISTINCT comp_sku, comp_source_store
            FROM combined_deduped
        ),

        pdp_filtered AS (
            SELECT p.*
            FROM pdp_newdev.product_warehouse p
            INNER JOIN relevant_comp_skus r
                   ON p.sku                              = r.comp_sku
                  AND replace(p.source_store, '<>', '_') = r.comp_source_store
        ),

        pdp_deduped AS (
            SELECT
                sku,
                replace(source_store, '<>', '_')      AS source_store,
                NULLIF(TRIM(product_title),       '') AS product_title,
                NULLIF(TRIM(product_description), '') AS product_description,
                NULLIF(TRIM(standardized_brand),  '') AS standardized_brand,
                manufacturer_part_number,
                upc_list
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY sku, source_store
                           ORDER BY product_segment
                       ) AS rn
                FROM pdp_filtered
            )
            WHERE rn = 1
        ),

        enriched AS (
            SELECT
                c.sku_uuid_a, c.sku_uuid_b, c.pair_id, c.answer, c.segment,
                c.base_sku, c.base_source_store,
                COALESCE(c.base_title,       cat.product_title)       AS base_title,
                COALESCE(c.base_description, cat.product_description) AS base_description,
                COALESCE(c.base_brand,       cat.brand)               AS base_brand,
                c.base_upc, c.base_price, c.base_pack_size,
                c.comp_sku, c.comp_source_store,
                COALESCE(c.comp_title,       p.product_title)         AS comp_title,
                COALESCE(c.comp_description, p.product_description)   AS comp_description,
                COALESCE(c.comp_brand,       p.standardized_brand)    AS comp_brand,
                c.comp_upc, c.comp_price, c.comp_pack_size,
                cat.manufacturer_part_number AS base_mpn,
                p.manufacturer_part_number   AS comp_mpn,
                p.upc_list                   AS comp_upc_list,
                c.source_priority

            FROM combined_deduped c
            LEFT JOIN latest_catalog cat ON c.base_sku = cat.sku AND c.base_source_store = cat.source_store
            LEFT JOIN pdp_deduped p      ON c.comp_sku = p.sku   AND c.comp_source_store = p.source_store
        )

        SELECT * FROM enriched
        WHERE base_title IS NOT NULL
          AND comp_title  IS NOT NULL
    """

    query = query.replace("${TENANT}",         tenant)
    query = query.replace("${LOAD_DATE}",      load_date)
    query = query.replace("${ML_DB}",          ml_db)
    query = query.replace("${FASTLANE_TABLE}", fastlane_table)

    return query

# ============================================================
# Dataset Extraction
# ============================================================

def extract_dataset(query):
    print("Executing extraction query")
    df = spark.sql(query)
    print(f"Rows extracted: {df.count()}")
    return df

# ============================================================
# NLTK Bootstrap
# ============================================================

def bootstrap_nltk(
    nltk_data_path: str = "/tmp/nltk_data",
    s3_bucket:      str = None,
    s3_prefix:      str = None
) -> None:
    """
    Ensure NLTK resources are available on the worker.
    Thread-safe via a lock instantiated inside the function — never
    captured by cloudpickle so no serialisation errors.
    Idempotent: no-ops if directory already populated.
    """
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    if os.path.exists(nltk_data_path) and os.listdir(nltk_data_path):
        return

    import threading as _threading
    _nltk_bootstrap_lock = _threading.Lock()

    with _nltk_bootstrap_lock:
        if os.path.exists(nltk_data_path) and os.listdir(nltk_data_path):
            return
        os.makedirs(nltk_data_path, exist_ok=True)
        if s3_bucket and s3_prefix:
            _download_s3_dir(s3_bucket, s3_prefix, nltk_data_path)
        else:
            for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
                nltk.download(resource, download_dir=nltk_data_path, quiet=True)

# ============================================================
# S3 Download Helper  (NLTK only — model no longer downloaded per worker)
# ============================================================

def _download_s3_dir(s3_bucket: str, s3_prefix: str, local_dir: str) -> None:
    """Recursively download an S3 prefix to a local directory."""
    s3        = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            s3_key        = obj["Key"]
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_path    = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(s3_bucket, s3_key, local_path)

# ============================================================
# Model extraction — SparkFiles approach (no S3 race condition)
# ============================================================
#
# Glue distributes model.tar.gz to every worker node via --extra-files.
# SparkFiles.get("model.tar.gz") returns the local path to that file.
# We extract it once per worker process into /tmp/model_dir/ and cache
# the loaded SentenceTransformer in a module-level dict — plain dict,
# not a lock, so cloudpickle has nothing unpicklable to capture.
#
# This completely replaces _download_model() and eliminates the S3
# race condition: the file is already on disk before the job starts.

_worker_model_cache = {}

def _load_model_from_spark_files(
    tar_filename: str = "model.tar.gz",
    extract_path: str = "/tmp/model_dir"   # this is the FINAL model directory
) -> "SentenceTransformer":
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    print(f"[MODEL LOAD] sentence_transformers version: {sentence_transformers.__version__}")

    if extract_path in _worker_model_cache:
        print(f"[MODEL LOAD] Cache hit — returning existing model instance")
        return _worker_model_cache[extract_path]

    import threading as _threading
    _extract_lock = _threading.Lock()

    with _extract_lock:
        if extract_path in _worker_model_cache:
            print(f"[MODEL LOAD] Cache hit inside lock — returning existing model instance")
            return _worker_model_cache[extract_path]

        tar_path = SparkFiles.get(tar_filename)
        print(f"[MODEL LOAD] SparkFiles resolved tar path: {tar_path}")
        print(f"[MODEL LOAD] Tar file exists: {os.path.exists(tar_path)}")
        print(f"[MODEL LOAD] Tar file size (bytes): {os.path.getsize(tar_path) if os.path.exists(tar_path) else 'N/A'}")

        if not os.path.exists(extract_path):
            # Extract to /tmp/ so that model_dir/ inside the tar
            # lands directly at /tmp/model_dir/ — no double nesting
            print(f"[MODEL LOAD] Extracting tar to /tmp/")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path="/tmp/")
            print(f"[MODEL LOAD] Extraction complete. Contents of /tmp/model_dir/: {os.listdir(extract_path)}")
        else:
            print(f"[MODEL LOAD] Already extracted at: {extract_path}")

        modules_json_path = os.path.join(extract_path, "modules.json")
        if os.path.exists(modules_json_path):
            with open(modules_json_path) as f:
                print(f"[MODEL LOAD] modules.json: {f.read()}")
        else:
            print(f"[MODEL LOAD] WARNING: modules.json not found at {modules_json_path}")

        print(f"[MODEL LOAD] Loading SentenceTransformer from: {extract_path}")
        model = SentenceTransformer(extract_path)
        print(f"[MODEL LOAD] Model loaded successfully")
        _worker_model_cache[extract_path] = model

    return model

# ============================================================
# Text Preprocessor (stateless, worker-safe)
# ============================================================

class TextPreprocessor:
    """Stateless preprocessor — safe to instantiate on driver or workers."""

    def __init__(self):
        nltk.data.path.append("/tmp/nltk_data")
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        text   = str(text).lower()
        text   = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [
            self._lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self._stop_words
        ]
        return " ".join(tokens)

    def preprocess_batch(self, texts: list) -> list:
        return [self.preprocess(t) for t in texts]

# ============================================================
# Abstract Base
# ============================================================

class SimilarityGenerator(abc.ABC):

    def __init__(self, spark: SparkSession, num_partitions: int = 200):
        self.spark          = spark
        self.num_partitions = num_partitions

    @property
    @abc.abstractmethod
    def score_column_name(self) -> str: ...

    @abc.abstractmethod
    def _build_scores(self, title_pair_df: DataFrame) -> DataFrame: ...

    def generate(self, training_df: DataFrame) -> DataFrame:
        title_pair_df = self._build_title_pairs(training_df)
        score_df      = self._build_scores(title_pair_df)
        return training_df.join(score_df, on=["base_title", "comp_title"], how="left")

    def _build_title_pairs(self, df: DataFrame) -> DataFrame:
        return (
            df
            .select("base_title", "comp_title")
            .distinct()
            .repartition(self.num_partitions)
        )

# ============================================================
# BERT Similarity Generator
# ============================================================

class BertSimilarityGenerator(SimilarityGenerator):

    def __init__(
        self,
        spark:          SparkSession,
        num_partitions: int = 200
    ):
        super().__init__(spark, num_partitions)

    @property
    def score_column_name(self) -> str:
        return "bert_similarity"

    def _build_scores(self, title_pair_df: DataFrame) -> DataFrame:

        title_df = (
            title_pair_df
            .select(F.col("base_title").alias("title"))
            .union(title_pair_df.select(F.col("comp_title").alias("title")))
            .distinct()
            .repartition(self.num_partitions)
        )

        embedding_schema = StructType([
            StructField("title",     StringType(),           True),
            StructField("embedding", ArrayType(FloatType()), True)
        ])

        def encode_partition(rows):
            titles = [row["title"] for row in rows]
            if not titles:
                return

            # Load model from the --extra-files tarball.
            # No S3 call, no race condition — file is already on the worker disk.
            model = _load_model_from_spark_files()

            embeddings = model.encode(
                titles,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            for title, emb in zip(titles, embeddings):
                yield (title, emb.tolist())

        embedding_df = (
            title_df.rdd
            .mapPartitions(encode_partition)
            .toDF(embedding_schema)
        )

        base_emb_df = (embedding_df
                       .withColumnRenamed("title",     "base_title")
                       .withColumnRenamed("embedding", "base_embedding"))
        comp_emb_df = (embedding_df
                       .withColumnRenamed("title",     "comp_title")
                       .withColumnRenamed("embedding", "comp_embedding"))

        pair_with_emb = (
            title_pair_df
            .join(base_emb_df, on="base_title", how="left")
            .join(comp_emb_df, on="comp_title", how="left")
        )

        @pandas_udf(DoubleType())
        def dot_product_udf(base_emb: pd.Series, comp_emb: pd.Series) -> pd.Series:
            def _dot(v1, v2):
                if v1 is None or v2 is None:
                    return None
                return float(np.dot(np.array(v1), np.array(v2)))
            return pd.Series([_dot(a, b) for a, b in zip(base_emb, comp_emb)])

        return (
            pair_with_emb
            .withColumn(self.score_column_name, dot_product_udf("base_embedding", "comp_embedding"))
            .select("base_title", "comp_title", self.score_column_name)
        )

# ============================================================
# TF-IDF Similarity Generator
# ============================================================

class TFIDFSimilarityGenerator(SimilarityGenerator):

    def __init__(
        self,
        spark:          SparkSession,
        nltk_bucket:    str,
        nltk_prefix:    str,
        num_partitions: int   = 200,
        max_features:   int   = 5000,
        ngram_range:    tuple = (1, 2)
    ):
        super().__init__(spark, num_partitions)
        self.nltk_bucket  = nltk_bucket
        self.nltk_prefix  = nltk_prefix
        self.max_features = max_features
        self.ngram_range  = ngram_range

    @property
    def score_column_name(self) -> str:
        return "tfidf_similarity"

    def _build_scores(self, title_pair_df: DataFrame) -> DataFrame:

        # Step 1: collect distinct titles to driver for fitting
        all_titles = (
            title_pair_df
            .select(F.col("base_title").alias("title"))
            .union(title_pair_df.select(F.col("comp_title").alias("title")))
            .distinct()
            .toPandas()["title"]
            .tolist()
        )

        # Step 2: preprocess + fit on driver
        preprocessor       = TextPreprocessor()
        processed_titles   = preprocessor.preprocess_batch(all_titles)
        title_to_processed = dict(zip(all_titles, processed_titles))

        vectorizer = TfidfVectorizer(
            analyzer     = "word",
            ngram_range  = self.ngram_range,
            max_features = self.max_features,
            stop_words   = "english"
        )
        vectorizer.fit(processed_titles)

        # Step 3: broadcast fitted vectorizer and preprocessed title map to workers
        bc_vectorizer         = self.spark.sparkContext.broadcast(vectorizer)
        bc_title_to_processed = self.spark.sparkContext.broadcast(title_to_processed)

        nltk_bucket = self.nltk_bucket
        nltk_prefix = self.nltk_prefix

        # Step 4: score on workers via pandas UDF
        # _nltk_bootstrapped flag ensures bootstrap runs once per worker process,
        # not once per Arrow batch — avoids repeated filesystem checks per batch.
        _nltk_bootstrapped = False

        @pandas_udf(DoubleType())
        def tfidf_similarity_udf(
            base_titles: pd.Series,
            comp_titles: pd.Series
        ) -> pd.Series:
            nonlocal _nltk_bootstrapped
            if not _nltk_bootstrapped:
                bootstrap_nltk(s3_bucket=nltk_bucket, s3_prefix=nltk_prefix)
                _nltk_bootstrapped = True

            vec       = bc_vectorizer.value
            title_map = bc_title_to_processed.value

            base_processed = base_titles.map(lambda t: title_map.get(t, str(t).lower()))
            comp_processed = comp_titles.map(lambda t: title_map.get(t, str(t).lower()))

            base_vecs = normalize(vec.transform(base_processed), norm="l2")
            comp_vecs = normalize(vec.transform(comp_processed), norm="l2")

            scores = np.asarray(base_vecs.multiply(comp_vecs).sum(axis=1)).flatten()
            return pd.Series(np.round(scores, 3))

        return (
            title_pair_df
            .withColumn(self.score_column_name, tfidf_similarity_udf("base_title", "comp_title"))
            .select("base_title", "comp_title", self.score_column_name)
        )

# ============================================================
# Pipeline Runner
# ============================================================

def run_similarity_pipeline(training_df, generators):
    df = training_df
    for generator in generators:
        print(f"Running {generator.__class__.__name__}...")
        df = generator.generate(df)
        print(f"  → column '{generator.score_column_name}' added")
    return df

# ============================================================
# Output Writer
# ============================================================

def write_output(
    df:            DataFrame,
    output_bucket: str,
    output_prefix: str,
    tenant:        str,
    load_date:     str
) -> None:
    
    output_path = f"s3://{output_bucket}/{output_prefix}/tenant={tenant}/load_date={load_date}/"
    print(f"Writing output to: {output_path}")

    # Drop embedding columns
    cols_to_drop = [c for c in df.columns if c.endswith("_embedding")]
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        df = df.drop(*cols_to_drop)

    # Persist once — this is the single materialisation point.
    # Everything upstream (query, BERT, TF-IDF) executes exactly once here.
    print("Persisting DataFrame...")
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    count = df.count()
    print(f"Persisted. Row count: {count}")

    spark.conf.set("spark.sql.shuffle.partitions", "50")
    spark.conf.set("spark.sql.files.maxRecordsPerFile", "15000")
    df.write.mode("overwrite").parquet(output_path)
    print("Write complete.")

    df.unpersist()
# ============================================================
# Main
# ============================================================

def main():
    config = load_config(DYNAMODB_TABLE, DYNAMODB_CONFIG_KEY)
    print(f"Config loaded from DynamoDB table '{DYNAMODB_TABLE}', key '{DYNAMODB_CONFIG_KEY}'")

    bootstrap_nltk(
        s3_bucket = config["nltk_bucket"],
        s3_prefix = config["nltk_prefix"]
    )

    query       = build_query(TENANT, LOAD_DATE, config["ml_db"], config["fastlane_table"])
    training_df = extract_dataset(query)

    generators = [
        BertSimilarityGenerator(
            spark          = spark,
            num_partitions = 200
            # model_bucket and model_prefix removed — model comes from --extra-files
        ),
        TFIDFSimilarityGenerator(
            spark          = spark,
            nltk_bucket    = config["nltk_bucket"],
            nltk_prefix    = config["nltk_prefix"],
            num_partitions = 200,
            max_features   = 5000,
            ngram_range    = (1, 2)
        ),
    ]

    training_df = run_similarity_pipeline(training_df, generators)

    write_output(
        df            = training_df,
        output_bucket = config["output_bucket"],
        output_prefix = config.get("output_prefix", ""),
        tenant        = TENANT,
        load_date     = LOAD_DATE
    )

if __name__ == "__main__":
    main()