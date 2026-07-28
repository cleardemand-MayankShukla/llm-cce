import os
import re
import sys
import json
import tarfile
import numpy as np

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, ArrayType, DoubleType
)
from pyspark.sql.functions import pandas_udf
from pyspark.sql.window import Window
from pyspark import StorageLevel, SparkFiles
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import boto3

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
# Job parameters:
# --tenant                          : staples
# --competitor_list                 : amazon,officedepot,target
# --threshold                       : 0.5              (applied on combined_score)
# --output_file_path                : s3://my-bucket/category-similarity/output/
# --nltk_location                   : s3://my-bucket/nltk_data/
# --prev_date                       : 2026-04-01       (optional; pass literal
#                                       string "null" or omit to run full flow)
# --run_date                        : 2026-07-07        (optional; overrides
#                                       today's date for the output path —
#                                       {output_file_path}/{tenant}/{run_date}.
#                                       Omit to use date.today() as before.
#                                       Lets a caller like a Step Functions
#                                       state machine pin the same date this
#                                       job used, so a downstream job can
#                                       reliably locate its output.)
#
# Glue system parameters:
# --extra-files                     : s3://ml-etl-test-dataset/cce_automated_dataset/models/model.tar.gz
# --additional-python-modules       : s3://ml-etl-test-dataset/cce_automated_dataset/glue_deps/dependencies.zip
# --python-modules-installer-option : --no-index
# --enable-glue-datacatalog         : true
# --job-language                    : python
#
# Incremental behaviour (--prev_date provided and not "null"):
#   1. Read previous scored parquet from output_file_path.
#   2. Read previous competitor_list.json from the same path.
#   3. Compute new pairs = full cross-join MINUS pairs already present in the
#      previous output (anti-join on the 7-column taxonomy key).
#      "New pairs" arise from:
#        • New base taxonomy nodes added since prev_date.
#        • New comp taxonomy nodes or a newly added competitor.
#        • A competitor dropped from the list is excluded at query time so its
#          rows naturally disappear from comp_nodes; they are NOT carried forward
#          from the previous output — this gives a clean refresh.
#   4. Score only the new pairs.
#   5. Union new scored rows with previous output rows whose retailer is still
#      in the current competitor_list, then re-rank the combined set.
#   6. Overwrite output parquet + write a fresh competitor_list.json.

args = getResolvedOptions(
    sys.argv,
    ["tenant", "competitor_list", "threshold", "output_file_path",
     "nltk_location", "prev_date"]
)

TENANT           = args["tenant"]
COMPETITOR_LIST  = args["competitor_list"]
THRESHOLD        = float(args["threshold"])
OUTPUT_FILE_PATH = args["output_file_path"].rstrip("/")
NLTK_LOCATION    = args["nltk_location"].rstrip("/")
_prev_date_raw   = args.get("prev_date", "-").strip()
PREV_DATE        = None if _prev_date_raw.lower() == "-" else _prev_date_raw

# --run_date is intentionally NOT in the getResolvedOptions list above --
# it's an optional override for callers (e.g. a Step Functions state machine)
# that need this job and a downstream job to agree on the same date rather
# than each independently resolving date.today(). Existing callers that never
# pass --run_date are unaffected; the job falls back to today's date exactly
# as before.
def _get_optional_arg(argv: list, name: str) -> str | None:
    flag = f"--{name}"
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None

_run_date_raw  = _get_optional_arg(sys.argv, "run_date")
RUN_DATE_OVERRIDE = None if _run_date_raw is None or _run_date_raw.strip().lower() == "-" else _run_date_raw.strip()

# Score weightings
BERT_WEIGHT  = 0.7
TFIDF_WEIGHT = 0.3

# Pair identity columns — used for anti-join and union deduplication
PAIR_KEY_COLS = [
    "base_category", "base_subcategory", "base_sub_subcategory",
    "retailer",
    "comp_category", "comp_subcategory", "comp_sub_subcategory"
]

# ============================================================
# S3 URI helpers
# ============================================================

def _parse_s3_uri(s3_uri: str) -> tuple:
    """Split 's3://bucket/prefix/path' → ('bucket', 'prefix/path')."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri}")
    without_scheme = s3_uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def _s3_path_exists(s3_uri: str) -> bool:
    """Return True if the S3 prefix contains at least one object."""
    bucket, prefix = _parse_s3_uri(s3_uri)
    s3     = boto3.client("s3")
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
    return result.get("KeyCount", 0) > 0


def _read_json_from_s3(s3_uri: str) -> dict:
    bucket, key = _parse_s3_uri(s3_uri)
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _write_json_to_s3(data: dict, s3_uri: str) -> None:
    bucket, key = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket      = bucket,
        Key         = key,
        Body        = json.dumps(data, indent=2).encode("utf-8"),
        ContentType = "application/json"
    )
    print(f"Competitor list written to: {s3_uri}")

# ============================================================
# NLTK Bootstrap
# ============================================================

def bootstrap_nltk(
    nltk_s3_uri:    str,
    nltk_data_path: str = "/tmp/nltk_data"
) -> None:
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    if os.path.exists(nltk_data_path) and os.listdir(nltk_data_path):
        return

    import threading as _threading
    _lock = _threading.Lock()

    with _lock:
        if os.path.exists(nltk_data_path) and os.listdir(nltk_data_path):
            return
        os.makedirs(nltk_data_path, exist_ok=True)
        s3_bucket, s3_prefix = _parse_s3_uri(nltk_s3_uri)
        _download_s3_dir(s3_bucket, s3_prefix, nltk_data_path)


def _download_s3_dir(s3_bucket: str, s3_prefix: str, local_dir: str) -> None:
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
# Model loader  —  SparkFiles approach (no S3 race condition)
# ============================================================

_worker_model_cache = {}

def _load_model_from_spark_files(
    tar_filename: str = "model.tar.gz",
    extract_path: str = "/tmp/model_dir"
) -> "SentenceTransformer":
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    print(f"[MODEL LOAD] sentence_transformers version: {sentence_transformers.__version__}")

    if extract_path in _worker_model_cache:
        print("[MODEL LOAD] Cache hit — returning existing model instance")
        return _worker_model_cache[extract_path]

    import threading as _threading
    _extract_lock = _threading.Lock()

    with _extract_lock:
        if extract_path in _worker_model_cache:
            print("[MODEL LOAD] Cache hit inside lock — returning existing model instance")
            return _worker_model_cache[extract_path]

        tar_path = SparkFiles.get(tar_filename)
        print(f"[MODEL LOAD] SparkFiles resolved tar path  : {tar_path}")
        print(f"[MODEL LOAD] Tar file exists               : {os.path.exists(tar_path)}")
        print(f"[MODEL LOAD] Tar file size (bytes)         : {os.path.getsize(tar_path) if os.path.exists(tar_path) else 'N/A'}")

        if not os.path.exists(extract_path):
            print("[MODEL LOAD] Extracting tar to /tmp/")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path="/tmp/")
            print(f"[MODEL LOAD] Extraction complete. Contents: {os.listdir(extract_path)}")
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
        print("[MODEL LOAD] Model loaded successfully")
        _worker_model_cache[extract_path] = model

    return model

# ============================================================
# Competitor source_store builder
# ============================================================

def build_source_store_list(competitor_list_csv: str) -> list:
    """
    "amazon,officedepot" → ["'amazon<>amazon'", "'officedepot<>officedepot'"]
    """
    competitors = [c.strip() for c in competitor_list_csv.split(",") if c.strip()]
    if not competitors:
        raise ValueError("--competitor_list must contain at least one competitor name.")
    return [f"'{c}<>{c}'" for c in competitors]


def parse_competitor_names(competitor_list_csv: str) -> list:
    """Return plain name list: "amazon,officedepot" → ["amazon", "officedepot"]"""
    return [c.strip() for c in competitor_list_csv.split(",") if c.strip()]

# ============================================================
# Query Builders
# ============================================================

def build_tenant_query(tenant: str) -> str:
    return f"""
        SELECT DISTINCT
            category,
            subcategory,
            sub_subcategory
        FROM bungee_customercatalog.athena_auroradb_catalog
        WHERE source = '{tenant}'
          AND capture_date = (
              SELECT MAX(capture_date)
              FROM bungee_customercatalog.athena_auroradb_catalog
              WHERE source = '{tenant}'
          )
    """


def build_competitor_query(source_store_list: list) -> str:
    in_clause = ", ".join(source_store_list)
    return f"""
        SELECT DISTINCT
            SPLIT_PART(source_store, '<>', 1) AS retailer,
            category,
            subcategory,
            sub_subcategory
        FROM pdp_newdev.product_warehouse
        WHERE source_store IN ({in_clause})
    """

# ============================================================
# Dataset Extraction
# ============================================================

def extract_dataset(query: str, label: str = "") -> DataFrame:
    tag = f": {label}" if label else ""
    print(f"Executing extraction query{tag}")
    df = spark.sql(query)
    print(f"  Rows extracted: {df.count()}")
    return df

# ============================================================
# Text Preprocessor
# ============================================================

class TextPreprocessor:

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
# Taxonomy text builder
# ============================================================

def build_taxonomy_text(category, subcategory, sub_subcategory) -> str:
    parts = [
        str(p).strip()
        for p in [category, subcategory, sub_subcategory]
        if p is not None and str(p).strip()
    ]
    return " ".join(parts)

# ============================================================
# BERT Similarity Generator
# ============================================================

class BertSimilarityGenerator:
    """
    Adds bert_similarity column to pair_df using the SentenceTransformer
    loaded from model.tar.gz distributed via --extra-files.
    """

    def __init__(self, spark: SparkSession, num_partitions: int = 200):
        self.spark          = spark
        self.num_partitions = num_partitions

    def generate(self, pair_df: DataFrame) -> DataFrame:

        title_df = (
            pair_df
            .select(
                F.concat_ws(" ",
                    F.col("base_category"),
                    F.col("base_subcategory"),
                    F.col("base_sub_subcategory")
                ).alias("title")
            )
            .union(
                pair_df.select(
                    F.concat_ws(" ",
                        F.col("comp_category"),
                        F.col("comp_subcategory"),
                        F.col("comp_sub_subcategory")
                    ).alias("title")
                )
            )
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
            model      = _load_model_from_spark_files()
            embeddings = model.encode(
                titles,
                batch_size           = 64,
                show_progress_bar    = False,
                normalize_embeddings = True
            )
            for title, emb in zip(titles, embeddings):
                yield (title, emb.tolist())

        embedding_df = (
            title_df.rdd
            .mapPartitions(encode_partition)
            .toDF(embedding_schema)
        )

        base_text_col = F.concat_ws(" ",
            F.col("base_category"), F.col("base_subcategory"), F.col("base_sub_subcategory"))
        comp_text_col = F.concat_ws(" ",
            F.col("comp_category"), F.col("comp_subcategory"), F.col("comp_sub_subcategory"))

        base_emb_df = (embedding_df
                       .withColumnRenamed("title",     "base_title")
                       .withColumnRenamed("embedding", "base_embedding"))
        comp_emb_df = (embedding_df
                       .withColumnRenamed("title",     "comp_title")
                       .withColumnRenamed("embedding", "comp_embedding"))

        pair_with_emb = (
            pair_df
            .withColumn("base_title", base_text_col)
            .withColumn("comp_title", comp_text_col)
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
            .withColumn("bert_similarity", dot_product_udf("base_embedding", "comp_embedding"))
            .drop("base_title", "comp_title", "base_embedding", "comp_embedding")
        )

# ============================================================
# TF-IDF Similarity Generator
# ============================================================

class TFIDFSimilarityGenerator:
    """
    Adds tfidf_similarity column to pair_df using TF-IDF cosine similarity
    on preprocessed taxonomy text.
    """

    def __init__(
        self,
        spark:          SparkSession,
        nltk_s3_uri:    str,
        num_partitions: int   = 200,
        max_features:   int   = 5000,
        ngram_range:    tuple = (1, 2)
    ):
        self.spark          = spark
        self.nltk_s3_uri    = nltk_s3_uri
        self.num_partitions = num_partitions
        self.max_features   = max_features
        self.ngram_range    = ngram_range

    def generate(self, pair_df: DataFrame) -> DataFrame:
        bc_vectorizer, bc_base_map, bc_comp_map = self._fit_vectorizer(pair_df)
        nltk_s3_uri = self.nltk_s3_uri
        _nltk_bootstrapped = False

        @pandas_udf(DoubleType())
        def tfidf_similarity_udf(
            base_cats:    pd.Series,
            base_subs:    pd.Series,
            base_subsubs: pd.Series,
            comp_cats:    pd.Series,
            comp_subs:    pd.Series,
            comp_subsubs: pd.Series
        ) -> pd.Series:
            nonlocal _nltk_bootstrapped
            if not _nltk_bootstrapped:
                bootstrap_nltk(nltk_s3_uri=nltk_s3_uri)
                _nltk_bootstrapped = True

            vec      = bc_vectorizer.value
            base_map = bc_base_map.value
            comp_map = bc_comp_map.value

            base_keys = list(zip(base_cats, base_subs, base_subsubs))
            comp_keys = list(zip(comp_cats, comp_subs, comp_subsubs))

            base_processed = [
                base_map.get(k, " ".join(str(p) for p in k if p)) for k in base_keys
            ]
            comp_processed = [
                comp_map.get(k, " ".join(str(p) for p in k if p)) for k in comp_keys
            ]

            base_vecs = normalize(vec.transform(base_processed), norm="l2")
            comp_vecs = normalize(vec.transform(comp_processed), norm="l2")

            scores = np.asarray(base_vecs.multiply(comp_vecs).sum(axis=1)).flatten()
            return pd.Series(np.round(scores, 4))

        return (
            pair_df
            .withColumn(
                "tfidf_similarity",
                tfidf_similarity_udf(
                    "base_category",      "base_subcategory",      "base_sub_subcategory",
                    "comp_category",      "comp_subcategory",      "comp_sub_subcategory"
                )
            )
        )

    def _fit_vectorizer(self, pair_df: DataFrame):
        bootstrap_nltk(nltk_s3_uri=self.nltk_s3_uri)
        preprocessor = TextPreprocessor()

        base_rows = (
            pair_df.select("base_category", "base_subcategory", "base_sub_subcategory")
            .distinct().collect()
        )
        base_raw = {
            (r["base_category"], r["base_subcategory"], r["base_sub_subcategory"]):
            build_taxonomy_text(r["base_category"], r["base_subcategory"], r["base_sub_subcategory"])
            for r in base_rows
        }

        comp_rows = (
            pair_df.select("comp_category", "comp_subcategory", "comp_sub_subcategory")
            .distinct().collect()
        )
        comp_raw = {
            (r["comp_category"], r["comp_subcategory"], r["comp_sub_subcategory"]):
            build_taxonomy_text(r["comp_category"], r["comp_subcategory"], r["comp_sub_subcategory"])
            for r in comp_rows
        }

        all_raw_unique = list(set(base_raw.values()) | set(comp_raw.values()))
        all_proc       = preprocessor.preprocess_batch(all_raw_unique)
        raw_to_proc    = dict(zip(all_raw_unique, all_proc))

        base_map = {k: raw_to_proc[v] for k, v in base_raw.items()}
        comp_map = {k: raw_to_proc[v] for k, v in comp_raw.items()}

        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=self.ngram_range,
            max_features=self.max_features, stop_words="english"
        )
        vectorizer.fit(all_proc)
        print(f"  TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

        bc_vectorizer = self.spark.sparkContext.broadcast(vectorizer)
        bc_base_map   = self.spark.sparkContext.broadcast(base_map)
        bc_comp_map   = self.spark.sparkContext.broadcast(comp_map)
        return bc_vectorizer, bc_base_map, bc_comp_map

# ============================================================
# Pipeline Runner
# ============================================================

def run_similarity_pipeline(pair_df: DataFrame, generators: list) -> DataFrame:
    df = pair_df
    for generator in generators:
        print(f"Running {generator.__class__.__name__}...")
        df = generator.generate(df)
        print(f"  → done")
    return df

# ============================================================
# Combined score + threshold + rank
# ============================================================

def apply_combined_score_and_rank(df: DataFrame, threshold: float) -> DataFrame:
    """
    1. combined_score = 0.7 * bert_similarity + 0.3 * tfidf_similarity
    2. Filter rows where combined_score >= threshold.
    3. Rank within (base_category, base_subcategory, base_sub_subcategory, retailer)
       ordered by combined_score DESC.

    Returns DataFrame with final output schema.
    """
    df = df.withColumn(
        "combined_score",
        F.round(
            F.lit(BERT_WEIGHT)  * F.col("bert_similarity") +
            F.lit(TFIDF_WEIGHT) * F.col("tfidf_similarity"),
            4
        )
    )

    df = df.filter(F.col("combined_score") >= threshold)
    print(f"  Threshold {threshold} applied on combined_score.")

    window_spec = Window.partitionBy(
        "base_category", "base_subcategory", "base_sub_subcategory", "retailer"
    ).orderBy(F.col("combined_score").desc())

    return (
        df
        .withColumn("rank", F.row_number().over(window_spec))
        .select(
            "base_category",
            "base_subcategory",
            "base_sub_subcategory",
            "retailer",
            "comp_category",
            "comp_subcategory",
            "comp_sub_subcategory",
            "bert_similarity",
            "tfidf_similarity",
            "combined_score",
            "rank"
        )
    )

# ============================================================
# Competitor list writer
# ============================================================

def write_competitor_list(
    tenant:          str,
    competitors:     list,
    run_date:        str,
    prev_date:       str,
    output_file_path: str
) -> None:
    """
    Write a JSON sidecar next to the output parquet so future runs can
    compare which competitors were included/excluded.

    Schema:
    {
        "tenant"      : "staples",
        "run_date"    : "2026-06-01",
        "prev_date"   : "2026-04-01",        // null for full runs
        "competitors" : ["amazon", "target"],
        "competitor_count": 2
    }
    """
    competitor_list_uri = f"{output_file_path}/competitor_list.json"
    # output_file_path is already scoped to tenant/run_date by the caller
    payload = {
        "tenant"           : tenant,
        "run_date"         : run_date,
        "prev_date"        : prev_date,
        "competitors"      : sorted(competitors),
        "competitor_count" : len(competitors)
    }
    _write_json_to_s3(payload, competitor_list_uri)

# ============================================================
# Output Writer
# ============================================================

def write_output(df: DataFrame, output_file_path: str) -> None:
    print(f"Writing output parquet to: {output_file_path}")

    cols_to_drop = [c for c in df.columns if c.endswith("_embedding")]
    if cols_to_drop:
        print(f"Dropping embedding columns: {cols_to_drop}")
        df = df.drop(*cols_to_drop)

    print("Persisting DataFrame before write...")
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    count = df.count()
    print(f"Persisted. Row count: {count}")

    spark.conf.set("spark.sql.shuffle.partitions", "50")
    spark.conf.set("spark.sql.files.maxRecordsPerFile", "15000")
    df.write.mode("overwrite").parquet(output_file_path)
    print("Write complete.")
    df.unpersist()

# ============================================================
# Full run
# ============================================================

def run_full(
    tenant_df:     DataFrame,
    competitor_df: DataFrame,
    generators:    list,
    threshold:     float
) -> DataFrame:
    """
    Score every (base_node × comp_node) pair from scratch.
    """
    base_nodes = (
        tenant_df
        .select("category", "subcategory", "sub_subcategory")
        .distinct()
        .withColumnRenamed("category",       "base_category")
        .withColumnRenamed("subcategory",     "base_subcategory")
        .withColumnRenamed("sub_subcategory", "base_sub_subcategory")
    )

    comp_nodes = (
        competitor_df
        .select("retailer", "category", "subcategory", "sub_subcategory")
        .distinct()
        .withColumnRenamed("category",       "comp_category")
        .withColumnRenamed("subcategory",     "comp_subcategory")
        .withColumnRenamed("sub_subcategory", "comp_sub_subcategory")
    )

    print(f"  Base taxonomy nodes : {base_nodes.count()}")
    print(f"  Comp taxonomy nodes : {comp_nodes.count()}")

    pair_df = base_nodes.crossJoin(comp_nodes).repartition(200)
    print(f"  Total pairs         : {pair_df.count()}")

    scored_df = run_similarity_pipeline(pair_df, generators)
    return apply_combined_score_and_rank(scored_df, threshold)

# ============================================================
# Incremental run
# ============================================================

def run_incremental(
    tenant_df:       DataFrame,
    competitor_df:   DataFrame,
    generators:      list,
    threshold:       float,
    output_file_path: str,
    current_competitors: list
) -> DataFrame:
    """
    1. Load previous output parquet.
    2. Retain only rows for competitors still in current_competitors
       (dropped competitors are removed cleanly).
    3. Build full current cross-join of base × comp nodes.
    4. Anti-join against the retained previous output to find new pairs only.
    5. Score the new pairs.
    6. Union new scored rows with retained previous rows.
    7. Re-rank the full combined set (combined_score already present on
       previous rows; re-ranking ensures ranks are consistent post-union).
    """
    # --- Load previous output --------------------------------
    print(f"Loading previous output from: {output_file_path}")
    prev_df = spark.read.parquet(output_file_path)
    print(f"  Previous rows: {prev_df.count()}")

    # --- Drop rows for competitors no longer in scope --------
    current_retailers = [c.strip() for c in current_competitors]
    prev_df_filtered  = prev_df.filter(F.col("retailer").isin(current_retailers))
    dropped_count     = prev_df.count() - prev_df_filtered.count()
    print(f"  Rows dropped (competitors removed from list): {dropped_count}")
    print(f"  Retained previous rows: {prev_df_filtered.count()}")

    # --- Build full current pair space -----------------------
    base_nodes = (
        tenant_df
        .select("category", "subcategory", "sub_subcategory")
        .distinct()
        .withColumnRenamed("category",       "base_category")
        .withColumnRenamed("subcategory",     "base_subcategory")
        .withColumnRenamed("sub_subcategory", "base_sub_subcategory")
    )

    comp_nodes = (
        competitor_df
        .select("retailer", "category", "subcategory", "sub_subcategory")
        .distinct()
        .withColumnRenamed("category",       "comp_category")
        .withColumnRenamed("subcategory",     "comp_subcategory")
        .withColumnRenamed("sub_subcategory", "comp_sub_subcategory")
    )

    full_pair_df = base_nodes.crossJoin(comp_nodes)
    print(f"  Full current pair space: {full_pair_df.count()}")

    # --- Anti-join: keep only pairs not already scored -------
    # Use a broadcast hint on the previous keys since they fit in memory
    prev_keys_df = prev_df_filtered.select(PAIR_KEY_COLS).distinct()

    new_pair_df = (
        full_pair_df
        .join(F.broadcast(prev_keys_df), on=PAIR_KEY_COLS, how="left_anti")
        .repartition(200)
    )
    print(f"  New pairs to score   : {new_pair_df.count()}")

    if new_pair_df.rdd.isEmpty():
        print("  No new pairs found — re-ranking existing data only.")
        # Still re-rank in case the competitor set changed
        return apply_combined_score_and_rank(
            prev_df_filtered.drop("combined_score", "rank"),
            threshold
        )

    # --- Score new pairs only --------------------------------
    new_scored_df = run_similarity_pipeline(new_pair_df, generators)
    new_scored_df = new_scored_df.withColumn(
        "combined_score",
        F.round(
            F.lit(BERT_WEIGHT)  * F.col("bert_similarity") +
            F.lit(TFIDF_WEIGHT) * F.col("tfidf_similarity"),
            4
        )
    )
    # Apply threshold before union to avoid storing low-quality new rows
    new_scored_df = new_scored_df.filter(F.col("combined_score") >= threshold)
    print(f"  New rows after threshold filter: {new_scored_df.count()}")

    # --- Union with retained previous rows -------------------
    # Drop rank from previous rows — we will re-rank the full combined set
    # so that rank numbers are globally consistent after additions/removals.
    prev_for_union = prev_df_filtered.drop("rank")

    combined_df = prev_for_union.unionByName(
        new_scored_df.select(prev_for_union.columns)
    )
    print(f"  Combined rows before re-rank: {combined_df.count()}")

    # --- Re-rank the full combined set -----------------------
    window_spec = Window.partitionBy(
        "base_category", "base_subcategory", "base_sub_subcategory", "retailer"
    ).orderBy(F.col("combined_score").desc())

    return (
        combined_df
        .withColumn("rank", F.row_number().over(window_spec))
        .select(
            "base_category",
            "base_subcategory",
            "base_sub_subcategory",
            "retailer",
            "comp_category",
            "comp_subcategory",
            "comp_sub_subcategory",
            "bert_similarity",
            "tfidf_similarity",
            "combined_score",
            "rank"
        )
    )

# ============================================================
# Main
# ============================================================

def main():
    from datetime import date

    run_date = RUN_DATE_OVERRIDE or str(date.today())

    print("=" * 60)
    print(f"  tenant           : {TENANT}")
    print(f"  competitor_list  : {COMPETITOR_LIST}")
    print(f"  threshold        : {THRESHOLD}")
    print(f"  output_file_path : {OUTPUT_FILE_PATH}")
    print(f"  nltk_location    : {NLTK_LOCATION}")
    print(f"  prev_date        : {PREV_DATE}")
    print(f"  run_date         : {run_date}")
    print(f"  score weights    : BERT={BERT_WEIGHT}, TF-IDF={TFIDF_WEIGHT}")
    print("=" * 60)

    # Bootstrap NLTK on the driver
    bootstrap_nltk(nltk_s3_uri=NLTK_LOCATION)

    # Parse competitors
    competitors       = parse_competitor_names(COMPETITOR_LIST)
    source_store_list = build_source_store_list(COMPETITOR_LIST)
    print(f"Competitors       : {competitors}")
    print(f"Source store list : {source_store_list}")

    # Extract taxonomy data
    tenant_df     = extract_dataset(build_tenant_query(TENANT),                label="tenant taxonomy")
    competitor_df = extract_dataset(build_competitor_query(source_store_list), label="competitor taxonomy")

    # Shared generators for both full and incremental runs
    generators = [
        BertSimilarityGenerator(
            spark          = spark,
            num_partitions = 200
        ),
        TFIDFSimilarityGenerator(
            spark          = spark,
            nltk_s3_uri    = NLTK_LOCATION,
            num_partitions = 200,
            max_features   = 5000,
            ngram_range    = (1, 2)
        ),
    ]

    # ---- Scoped output path: base/tenant/run_date -----------
    # Both the parquet and competitor_list.json land here so each
    # run is self-contained and trivially comparable across dates.
    scoped_output_path = f"{OUTPUT_FILE_PATH}/{TENANT}/{run_date}"
    print(f"  scoped_output_path : {scoped_output_path}")

    # ---- Full run or Incremental run ------------------------
    if PREV_DATE is None:
        print("\n--- FULL RUN (no prev_date provided) ---")
        result_df = run_full(tenant_df, competitor_df, generators, THRESHOLD)
    else:
        print(f"\n--- INCREMENTAL RUN (prev_date={PREV_DATE}) ---")
        prev_scoped_path = f"{OUTPUT_FILE_PATH}/{TENANT}/{PREV_DATE}"
        if not _s3_path_exists(prev_scoped_path):
            print(
                f"WARNING: prev_date supplied but no previous output found at "
                f"'{prev_scoped_path}'. Falling back to full run."
            )
            result_df = run_full(tenant_df, competitor_df, generators, THRESHOLD)
        else:
            result_df = run_incremental(
                tenant_df           = tenant_df,
                competitor_df       = competitor_df,
                generators          = generators,
                threshold           = THRESHOLD,
                output_file_path    = prev_scoped_path,
                current_competitors = competitors
            )

    # ---- Write output parquet -------------------------------
    write_output(result_df, scoped_output_path)

    # ---- Write competitor list sidecar ----------------------
    write_competitor_list(
        tenant           = TENANT,
        competitors      = competitors,
        run_date         = run_date,
        prev_date        = PREV_DATE,
        output_file_path = scoped_output_path
    )


if __name__ == "__main__":
    main()