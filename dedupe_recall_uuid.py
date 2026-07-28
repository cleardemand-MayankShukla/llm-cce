"""
Recall-suggestion UUID de-duplication — Glue Python Shell job.

Fixes a cardinality defect in the semantic-recall suggestion CSVs
(``base_sku_uuid,comp_sku_uuid,score``) written under one or more
``comp_source_store=<comp>/`` partitions. Point ``--input_prefix`` at the parent
(e.g. ``s3://kaushik-temp/``) and every partition below it is processed; point it
at a single partition to do just that one.

The defect
----------
Every ``base_sku_uuid`` / ``comp_sku_uuid`` is a ``<>``-delimited tuple whose
last three parts are ``sku<>source<>store``. Some rows also carry a leading UPC
part (``upc<>sku<>source<>store``, 4 parts) while others do not
(``sku<>source<>store``, 3 parts). The SAME product therefore appears under two
different uuid strings — once with its UPC, once without — so a single logical
base->comp suggestion is emitted as two rows. Downstream steps (CCE / Verdict)
then double-count that pair. In one observed file 38,046 rows carried only
30,904 distinct base->comp pairs.

The fix
-------
1. Normalise every uuid to its canonical ``sku<>source<>store`` form by dropping
   the leading UPC part (keep the last three ``<>`` segments). This is exactly
   the ``sku_uuid_a`` / ``sku_uuid_b`` shape ``cce_0_dataset_materialise.py``
   builds, so the output lines up with the rest of the pipeline.
2. Drop rows that become byte-identical ``(base, comp, score)`` triples after
   normalisation — these are the pure UPC-with/without duplicates and carry no
   new information.
3. KEEP everything else. Where the with-UPC and without-UPC twins were scored
   slightly differently by the model, both scores are genuinely distinct data,
   so both rows are retained (the pair simply stays on more than one row). The
   score is treated as opaque text and never parsed, so its exact value is
   preserved bit-for-bit.

Scope / assumptions
-------------------
* The prefix is walked recursively. Only keys whose path contains the partition
  marker (default ``comp_source_store=``) are processed, so sibling prefixes such
  as ``athena_output/`` are left untouched. Set ``--partition_marker ''`` to
  process every matching file regardless of partition.
* De-duplication is per file. The suggestion CSVs are split by base-SKU range
  (e.g. ``1_3837.csv``, ``3838_7674.csv``), so a base SKU and all of its comp
  twins live in the same file — per-file exact-triple removal fully collapses
  the UPC duplication. If a base SKU could ever span two files, this would miss
  those cross-file duplicates (none were present in the data checked).
* Default is OVERWRITE IN PLACE. Pass ``--output_prefix`` to write elsewhere
  (the ``comp_source_store=.../file.csv`` sub-path is mirrored under it), or
  ``--dry_run true`` to report the row deltas without writing anything.

Glue Job arguments
------------------
Required:
  --input_prefix    parent (or single-partition) s3 prefix, e.g.
                    's3://kaushik-temp/'  or
                    's3://kaushik-temp/comp_source_store=bestbuyca<>bestbuyca/'
Optional:
  --output_prefix   where to write cleaned files (default: --input_prefix, i.e.
                    overwrite in place). The sub-path below --input_prefix is preserved.
  --partition_marker only process keys whose path contains this (default
                    'comp_source_store='; pass '' to disable the filter).
  --file_suffix     only process keys ending with this (default '.csv').
  --dry_run         'true' to compute and log deltas without writing (default 'false').
  --JOB_NAME        label for logs (Spark jobs auto-supply it; Python Shell does not).
"""

import sys
import csv
import io
import logging
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION_NAME = "us-east-1"

s3 = boto3.client("s3", region_name=REGION_NAME)


# --------------------------------------------------------------------------- #
# Argument parsing (mirrors inf_athena.py: required + only-present optionals)
# --------------------------------------------------------------------------- #
def parse_args():
    required = ["input_prefix"]
    optional = ["JOB_NAME", "output_prefix", "partition_marker", "file_suffix", "dry_run"]
    present_optional = [o for o in optional if f"--{o}" in sys.argv]
    args = getResolvedOptions(sys.argv, required + present_optional)

    args["input_prefix"] = args["input_prefix"].strip()
    if not args["input_prefix"].startswith("s3://"):
        raise ValueError(f"--input_prefix must be an s3:// URI, got '{args['input_prefix']}'.")
    if not args["input_prefix"].endswith("/"):
        args["input_prefix"] += "/"

    out = args.get("output_prefix", args["input_prefix"]).strip()
    if not out.startswith("s3://"):
        raise ValueError(f"--output_prefix must be an s3:// URI, got '{out}'.")
    if not out.endswith("/"):
        out += "/"
    args["output_prefix"] = out

    # partition_marker may legitimately be '' (disable filtering), so only default
    # it when the arg was not supplied at all.
    args["partition_marker"] = args["partition_marker"] if "partition_marker" in args else "comp_source_store="
    args["file_suffix"] = args.get("file_suffix", ".csv").strip()
    args["dry_run"] = str(args.get("dry_run", "false")).strip().lower() == "true"
    return args


# --------------------------------------------------------------------------- #
# Core transform
# --------------------------------------------------------------------------- #
def normalize_uuid(u):
    """Collapse a ``<>``-delimited sku uuid to its canonical ``sku<>source<>store``
    form by keeping only the last three parts (drops a leading UPC when present).
    Values with fewer than three parts are returned unchanged."""
    parts = u.split("<>")
    if len(parts) < 3:
        return u
    return "<>".join(parts[-3:])


def dedupe_csv(text):
    """Normalise uuids and drop exact-duplicate (base, comp, score) triples.

    Returns (out_text, stats). ``score`` is kept as the original string — never
    parsed — so distinct-but-numerically-close scores are preserved as separate
    rows and exact values never drift.
    """
    reader = csv.reader(io.StringIO(text))
    out_buf = io.StringIO()
    # The source files use LF line endings; csv.writer defaults to CRLF, so pin LF.
    writer = csv.writer(out_buf, lineterminator="\n")

    header = None
    rows_in = rows_out = malformed = 0
    seen = set()

    for i, row in enumerate(reader):
        if i == 0:
            header = row
            writer.writerow(row)          # preserve header verbatim
            continue
        if not row:
            continue                      # skip blank trailing lines
        rows_in += 1
        if len(row) != 3:
            # Unexpected shape — keep it rather than silently drop data.
            malformed += 1
            writer.writerow(row)
            rows_out += 1
            continue
        base, comp, score = row
        base_n, comp_n = normalize_uuid(base), normalize_uuid(comp)
        key = (base_n, comp_n, score)
        if key in seen:
            continue                      # pure UPC-with/without duplicate
        seen.add(key)
        writer.writerow([base_n, comp_n, score])
        rows_out += 1

    stats = {
        "header": header,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "removed": rows_in - rows_out,
        "malformed": malformed,
    }
    return out_buf.getvalue(), stats


# --------------------------------------------------------------------------- #
# S3 helpers
# --------------------------------------------------------------------------- #
def list_csv_keys(bucket, prefix, suffix, marker):
    """Recursively list object keys under prefix ending with suffix. Keeps only
    keys whose path contains ``marker`` (skips siblings like ``athena_output/``);
    an empty marker disables that filter. 0-byte folder markers are skipped.
    Returns (kept_keys, skipped_count)."""
    paginator = s3.get_paginator("list_objects_v2")
    keys, skipped = [], 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(suffix) or obj.get("Size", 0) == 0:
                continue
            if marker and marker not in key:
                skipped += 1
                continue
            keys.append(key)
    return keys, skipped


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    logger.info("Step 1: Parsing job arguments...")
    args = parse_args()
    job_name = args.get("JOB_NAME", "dedupe_recall_uuid")

    in_parsed = urlparse(args["input_prefix"])
    out_parsed = urlparse(args["output_prefix"])
    in_bucket, in_prefix = in_parsed.netloc, in_parsed.path.lstrip("/")
    out_bucket, out_prefix = out_parsed.netloc, out_parsed.path.lstrip("/")
    in_place = args["output_prefix"] == args["input_prefix"]

    mode = "DRY RUN (no writes)" if args["dry_run"] else (
        "OVERWRITE IN PLACE" if in_place else f"write to {args['output_prefix']}"
    )
    logger.info(
        f"[{job_name}] input={args['input_prefix']} marker='{args['partition_marker']}' "
        f"suffix={args['file_suffix']} mode={mode}"
    )

    logger.info("Step 2: Listing input files...")
    keys, skipped = list_csv_keys(in_bucket, in_prefix, args["file_suffix"], args["partition_marker"])
    if skipped:
        logger.info(f"Skipped {skipped} file(s) not under a '{args['partition_marker']}' partition.")
    if not keys:
        logger.warning(f"No '{args['file_suffix']}' objects found under {args['input_prefix']} — nothing to do.")
        return
    logger.info(f"Found {len(keys)} file(s) to process across all partitions.")

    def partition_of(name):
        # Group log lines by the comp_source_store=... segment when present.
        for seg in name.split("/"):
            if args["partition_marker"] and seg.startswith(args["partition_marker"]):
                return seg
        return "(root)"

    tot_in = tot_out = tot_removed = tot_malformed = 0
    per_part = {}  # partition -> [files, rows_in, rows_out, removed]
    for key in keys:
        body = s3.get_object(Bucket=in_bucket, Key=key)["Body"].read().decode("utf-8")
        out_text, st = dedupe_csv(body)
        tot_in += st["rows_in"]
        tot_out += st["rows_out"]
        tot_removed += st["removed"]
        tot_malformed += st["malformed"]

        name = key[len(in_prefix):] if key.startswith(in_prefix) else key.rsplit("/", 1)[-1]
        part = partition_of(name)
        agg = per_part.setdefault(part, [0, 0, 0, 0])
        agg[0] += 1; agg[1] += st["rows_in"]; agg[2] += st["rows_out"]; agg[3] += st["removed"]

        pct = (st["removed"] / st["rows_in"] * 100) if st["rows_in"] else 0.0
        extra = f", malformed(kept)={st['malformed']}" if st["malformed"] else ""
        logger.info(
            f"  {name}: {st['rows_in']} -> {st['rows_out']} rows "
            f"(removed {st['removed']} dup, {pct:.1f}%){extra}"
        )

        if args["dry_run"]:
            continue
        dst_key = out_prefix + name
        s3.put_object(
            Bucket=out_bucket,
            Key=dst_key,
            Body=out_text.encode("utf-8"),
            ContentType="text/csv",
        )

    logger.info("Step 3: Per-partition summary:")
    for part in sorted(per_part):
        f, ri, ro, rm = per_part[part]
        logger.info(f"  {part}: {f} file(s), {ri} -> {ro} rows, removed {rm} dup")

    action = "would remove" if args["dry_run"] else "removed"
    logger.info(
        f"Step 4: Done. {len(keys)} file(s) across {len(per_part)} partition(s): "
        f"{tot_in} -> {tot_out} rows, {action} {tot_removed} duplicate row(s)."
        + (f" {tot_malformed} malformed row(s) kept unchanged." if tot_malformed else "")
    )
    if args["dry_run"]:
        logger.info("Dry run — no objects were written.")


if __name__ == "__main__":
    main()
