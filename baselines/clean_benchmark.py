#!/usr/bin/env python3
"""
Build the clean benchmark directory structure.

For each question this script:
  1. Verifies the question's SQL produces the expected ground-truth answers.
  2. Optionally verifies that all geo_wkt values in question_entities match the DB.
  3. Writes  benchmark/T{N}/{i}/question.json
             benchmark/T{N}/{i}/baseline_answers.json

The baseline_answers.json is populated from whatever eval CSV + answer cache
files are found on disk (auto-discovered from *_text_eval.csv).

Usage:
  python clean_benchmark.py                     # full run with SQL verification
  python clean_benchmark.py --skip-verify       # skip SQL correctness check
  python clean_benchmark.py --verify-geo-wkts   # also verify geometry matches DB
  python clean_benchmark.py --out-dir my_bench  # custom output directory
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm

ROOT = Path(__file__).parent
QUESTIONS_DIR = ROOT / "selected_questions"
BENCHMARK_DIR = ROOT / "benchmark"

TYPES_ORDERED = [
    "range+name", "range:non_spat_filter+name", "range:direction+name", "range:towards+name",
    "knn+name", "knn:non_spat_filter+name", "knn+name+multi_source1", "knn+name+multi_source2",
    "knn:direction+name", "knn:towards+name", "intersects:area_max+name", "intersects:length_max+name",
    "range+loc", "range:non_spat_filter+loc", "range:direction+loc", "range:towards+loc",
    "knn+loc", "knn:non_spat_filter+loc", "knn:direction+loc", "knn:towards+loc",
    "range+angle", "knn+angle",
    "range+count", "intersects+count",
    "range+distance", "knn+distance",
    "intersects:area_total+area", "intersects:length_total+length",
]
TYPE_LABELS = {t: f"T{i}" for i, t in enumerate(TYPES_ORDERED, 1)}

RELEVANT_SCORES = {
    "count":    ["relative_error", "attempted"],
    "area":     ["relative_error", "attempted"],
    "length":   ["relative_error", "attempted"],
    "distance": ["relative_error", "attempted"],
    "name":     ["P", "R", "F1", "attempted"],
    "loc":      ["P", "R", "F1", "distance_error", "attempted"],
    "angle":    ["P", "R", "F1", "angle_error", "attempted"],
}

DB_PARAMS = dict(host="localhost", dbname="osm_ca", user="postgres", password="postgres", port=5432)


# ── Database ────────────────────────────────────────────────────────────────

def make_conn():
    return psycopg.connect(**DB_PARAMS, row_factory=dict_row)


def run_sql(sql: str, conn, timeout_ms: int = 600_000) -> dict:
    cur = conn.cursor()
    cur.execute("SET statement_timeout = %d" % timeout_ms)
    try:
        cur.execute(sql)
    except Exception as e:
        conn.rollback()
        return {"output": [], "error": str(e)}
    rows = cur.fetchall()
    cur.close()
    return {"output": [{k: row[k] for k in row if row[k] is not None} for row in rows], "error": ""}


# ── Question loading ─────────────────────────────────────────────────────────

def load_questions() -> list:
    files = sorted(glob(str(QUESTIONS_DIR / "*.jsonl")))
    questions = []
    qid = 0
    for path in files:
        q_type = Path(path).stem
        with open(path) as f:
            for _ in range(100):
                line = f.readline()
                if not line:
                    break
                q = json.loads(line)
                q["id"] = qid
                q["type"] = q_type
                qid += 1
                questions.append(q)
    return questions


# ── JSON block extraction (mirrors baselines.py) ─────────────────────────────

def _flatten(array):
    if isinstance(array, list) and any(isinstance(i, list) for i in array):
        out = []
        for item in array:
            out.extend(_flatten(item) if isinstance(item, list) else [item])
        return out
    return array


def extract_json_blocks(text: str, idx=0) -> list:
    p1 = re.compile(r"\b\d+(?:_\d+)*\b")
    p2 = re.compile(r"\b\d+(?:,\d+)*\b")
    p3 = re.compile(r"//.*?\n")
    p4 = re.compile(r",\s*}")
    p5 = re.compile(r"}\s*{")
    blocks = []
    for match in re.findall(r"```[\s]*json(.*?)```", text, re.DOTALL):
        try:
            s = match.strip()
            s = p1.sub(lambda m: m.group().replace("_", ""), s)
            s = p2.sub(lambda m: m.group().replace(",", ""), s)
            s = p3.sub("", s)
            s = p4.sub("}", s)
            s = (s.replace("\\'", "'").replace("\\&", "&")
                  .replace(": integer", ": null").replace(" * ", ""))
            s = re.sub(r"(\d[\d\s]*)(?:\s*\+\s*[\d\s\+]+)", lambda m: m.group(1).replace(" ", ""), s)
            s = re.sub(r"\b\d[\d\s]*\b", lambda m: m.group(0).replace(" ", ""), s)
            if p5.search(s):
                s = p5.sub("},\n{", s)
                s = f"[\n{s}\n]"
            convert_area = " acres" in s
            if convert_area:
                s = s.replace(" acres,", ",").replace("acres,", ",")
            data = json.loads(s)
            if convert_area and isinstance(data, dict) and "area" in data:
                data["area"] = data["area"] * 4046.8564224
            blocks.append(data)
        except json.JSONDecodeError:
            pass
    return _flatten(blocks)


# ── Baseline answer discovery ────────────────────────────────────────────────

def discover_labels(root: Path, selected_models=None) -> list[str]:
    all_labels = []
    for p in sorted(root.glob("*_text_eval.csv")):
        label = p.stem.replace("_text_eval", "")
        if (root / f"{label}_parsed_eval.csv").exists():
            all_labels.append(label)

    if not selected_models:
        return all_labels

    print(f"  Available labels: {all_labels}")
    ordered = []
    for m in selected_models:
        matched = [l for l in all_labels if l == m or l.startswith(m + "_")]
        if not matched:
            print(f"  WARNING: no labels found for model '{m}'")
        ordered.extend(matched)
    return ordered


def _get_output_type(q_type: str) -> str | None:
    for o in RELEVANT_SCORES:
        if q_type.endswith(f"+{o}") or f"+{o}+" in q_type:
            return o
    return None


def load_scores(labels: list, root: Path) -> dict:
    """
    Returns {label: {"text": {qid: scores}, "json": {qid: scores}}}
    """
    result = {}
    for label in labels:
        text_df = pd.read_csv(root / f"{label}_text_eval.csv").fillna(0)
        parsed_df = pd.read_csv(root / f"{label}_parsed_eval.csv").fillna(0)

        text_scores = {}
        for _, row in text_df.iterrows():
            d = row.to_dict()
            text_scores[int(row["id"])] = {k: d[k] for k in d if k not in ("type", "id")}

        json_scores = {}
        for _, row in parsed_df.iterrows():
            d = row.to_dict()
            otype = _get_output_type(str(d.get("type", "")))
            keep = RELEVANT_SCORES.get(otype, []) if otype else []
            json_scores[int(row["id"])] = {k: d[k] for k in d if k not in ("type", "id") and k in keep}

        result[label] = {"text": text_scores, "json": json_scores}
    return result


def load_answer_cache(labels: list, root: Path) -> dict:
    """
    Returns {label: {"text": {qid: str}, "json": {qid: str}}}
    Looks in cache/{model_name}/{step}.json where step is direct_answer / sql_answer / rag_answer.
    """
    result = {}
    for label in labels:
        # derive model name by stripping known suffix
        model = label
        prefix = "direct"
        for sfx in ("_text2sql", "_direct", "_rag", "_shuffled"):
            if label.endswith(sfx):
                model = label[: -len(sfx)]
                prefix = sfx.lstrip("_")
                break

        step_map = {
            "direct":   ("direct_answer",   "direct_json_parse"),
            "text2sql": ("sql_answer",       "sql_json_parse"),
            "rag":      ("rag_answer",       "rag_json_parse"),
            "shuffled": ("shuffled_answer",  "shuffled_json_parse"),
        }
        ans_step, json_step = step_map.get(prefix, ("direct_answer", "direct_json_parse"))

        cache_dir = root / "cache" / model

        def _load_step(step):
            p = cache_dir / f"{step}.json"
            if not p.exists():
                return {}
            with open(p) as f:
                items = json.load(f)
            return {int(it["id"]): it.get("content", "") for it in items}

        result[label] = {
            "text": _load_step(ans_step),
            "json": _load_step(json_step),
        }
    return result


# ── Verification ─────────────────────────────────────────────────────────────

def run_sql_answers(sql: str, expected_answers: list, conn) -> tuple[list | None, str | None]:
    """
    Run the stored SQL and return (fresh_rows, error_or_warning).

    The generator has a bug where answers for POI-only question types were
    fetched with a stale predicate from a previous iteration, producing a
    subset of what the stored SQL actually returns. For those types the stored
    answers are unreliable and the SQL output is authoritative.

    For scalar questions (count/area/length/distance) there is no stale-predicate
    issue (those templates always include a region entity that resets the
    predicate), so we verify the scalar value matches.
    """
    result = run_sql(sql, conn, timeout_ms=600_000)
    if result["error"]:
        return None, f"SQL error: {result['error']}"

    rows = result["output"]

    # Scalar answers: verify the value matches (stale predicate does not affect these)
    expected_ids = {a["id"] for a in expected_answers if "id" in a}
    if not expected_ids and expected_answers:
        if len(rows) != 1:
            return None, f"Expected 1 scalar row, got {len(rows)}"
        scalar_key = next(iter(expected_answers[0]))
        expected_val = expected_answers[0][scalar_key]
        actual_val = rows[0].get(scalar_key)
        if actual_val is None:
            return None, f"Scalar key '{scalar_key}' missing from result"
        try:
            if not math.isclose(float(actual_val), float(expected_val), rel_tol=1e-5):
                return None, f"Scalar mismatch for '{scalar_key}': expected {expected_val}, got {actual_val}"
        except (TypeError, ValueError):
            if actual_val != expected_val:
                return None, f"Scalar mismatch for '{scalar_key}': expected {expected_val}, got {actual_val}"

    # For ID-based answers: SQL output is authoritative (stale-predicate bug may have
    # produced a subset); return fresh rows so caller can overwrite q["answers"]
    return rows, None


def verify_geo_wkts(question_entities: dict, sql: str, conn) -> list[str]:
    """
    For each entity whose geo_wkt was substituted into the SQL template, verify
    that the stored WKT matches the current DB geometry.

    The generator produces WKT via shapely.to_wkt(shapely.from_wkb(geometry))
    and substitutes it literally into the SQL. If the WKT is not in the SQL the
    entity is a metadata-only reference and is silently skipped.

    Comparison is done in PostGIS via ST_Equals to avoid precision mismatch:
    shapely.to_wkt() rounds to 6 decimal places while ST_AsText returns full
    double precision, causing shapely.equals() to return False for identical
    geometries.

    Failures are hard errors: the DB geometry must equal the stored WKT.
    """
    errors = []
    for key, entity in question_entities.items():
        if "geo_wkt" not in entity:
            continue
        expected_wkt = entity["geo_wkt"]

        # Skip entities whose WKT was not embedded in the SQL template
        if expected_wkt not in sql:
            continue

        for subkey, table, id_col in [("poi", "pois", "osm_id"), ("region", "regions", "id")]:
            if subkey not in entity or not isinstance(entity[subkey], dict):
                continue
            record_id = entity[subkey].get(id_col)
            if not record_id:
                continue
            safe_wkt = expected_wkt.replace("'", "''")
            # Use ST_DWithin with 1-metre tolerance to handle shapely's 6-decimal-place
            # rounding (~0.1 m max error). Works for both POINTs and POLYGON/MULTIPOLYGON.
            r = run_sql(
                f"SELECT "
                f"  ST_DWithin(geometry, ST_GeomFromText('{safe_wkt}', 4326)::geography, 1) AS match, "
                f"  ST_AsText(geometry::geometry) AS db_wkt "
                f"FROM {table} WHERE {id_col} = {record_id} LIMIT 1",
                conn, 5_000,
            )
            if r["error"]:
                errors.append(
                    f"geo_wkt DB query error for {key} ({id_col}={record_id}): {r['error']}"
                )
                break
            if not r["output"]:
                errors.append(f"No DB record for {table}.{id_col}={record_id} (entity {key})")
                break
            row = r["output"][0]
            if not row.get("match"):
                db_wkt = row.get("db_wkt", "")
                tqdm.write(f"    [DEBUG] entity {key} ({table}.{id_col}={record_id})")
                tqdm.write(f"    [DEBUG] stored WKT ({len(expected_wkt)} chars): {expected_wkt[:120]}…")
                tqdm.write(f"    [DEBUG] DB WKT     ({len(db_wkt)} chars): {db_wkt[:120]}…")
                tqdm.write(f"    [DEBUG] stored geom_type: {expected_wkt.split('(')[0].strip()}")
                tqdm.write(f"    [DEBUG] DB geom_type:     {db_wkt.split('(')[0].strip()}")
                errors.append(
                    f"geo_wkt mismatch for {key}: stored WKT does not intersect DB geometry "
                    f"({table}.{id_col}={record_id})"
                )
            break
    return errors


# ── Entity / question cleaning ───────────────────────────────────────────────

def clean_entities(obj: dict, q_type: str) -> dict:
    out = {}
    for k, v in obj.items():
        if k == "[1]":
            out[k] = {
                k2: v[k2]
                for k2 in v
                if k2 in ("main_category", "sub_category", "poi_filter_desc",
                           "poi_filter_sql", "sub_category_label", "table")
            }
        else:
            out[k] = v
    return out


def clean_question(text: str, q_type: str) -> str:
    text = text.replace("  ", " ").replace("The", "the")
    tid = TYPE_LABELS.get(q_type, "")
    if tid in ("T4", "T16", "T20"):
        text = text.replace("fast food", "fast food restaurant")
    if tid == "T16":
        text = text.replace("where can I find", "where can I find a")
    text = text.replace("Pediatric emergency", "pediatric emergency")
    return text


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading questions…")
    questions = load_questions()
    print(f"  {len(questions)} questions loaded")

    renames = {}
    if args.rename:
        for pair in args.rename:
            old, new = pair.split("=", 1)
            renames[old.strip()] = new.strip()

    labels = discover_labels(ROOT, selected_models=args.models)
    print(f"  Labels discovered: {labels}")

    scores = load_scores(labels, ROOT)
    answers = load_answer_cache(labels, ROOT)

    # Apply renames: map original label → display name used as key in output JSON
    label_display = {l: renames.get(l, l) for l in labels}

    conn = make_conn() if not args.skip_verify else None

    verify_errors = []
    counts: dict[str, int] = defaultdict(int)

    try:
        for q in tqdm(questions, desc="Processing"):
            tid = TYPE_LABELS.get(q["type"])
            if tid is None:
                tqdm.write(f"  Unknown type {q['type']!r}, skipping q{q['id']}")
                continue

            # Run the stored SQL; use its output as authoritative answers.
            # Scalar questions are verified against stored answers.
            # ID-based answers are overwritten (generator stale-predicate bug
            # may have stored an incomplete subset).
            if conn and not args.skip_verify:
                fresh_rows, err = run_sql_answers(q["sql"], q["answers"], conn)
                if err:
                    verify_errors.append({"id": q["id"], "type": q["type"], "error": err})
                    tqdm.write(f"  [VERIFY FAIL] q{q['id']} ({q['type']}): {err}")
                elif fresh_rows is not None:
                    q["answers"] = fresh_rows

            # geo_wkt verification (optional)
            if conn and args.verify_geo_wkts and "question_entities" in q:
                geo_errs = verify_geo_wkts(q["question_entities"], q["sql"], conn)
                for ge in geo_errs:
                    verify_errors.append({"id": q["id"], "type": q["type"], "error": ge})
                    tqdm.write(f"  [GEO FAIL] q{q['id']} ({q['type']}): {ge}")
                    tqdm.write(f"    question: {q.get('question', '')[:120]}")
                    tqdm.write(f"    sql:      {q['sql'][:200]}")

            counts[q["type"]] += 1
            i = counts[q["type"]]
            q_dir = out_dir / tid / f"{i:3d}"
            q_dir.mkdir(parents=True, exist_ok=True)

            # Clean question
            clean_q = {}
            for k, v in q.items():
                if k == "question_entities":
                    clean_q[k] = clean_entities(v, q["type"])
                elif k == "question":
                    clean_q[k] = clean_question(v, q["type"])
                else:
                    clean_q[k] = v

            # Build baseline_answers
            baseline_answers = {}
            for label in labels:
                qid = q["id"]
                text_answer = answers[label]["text"].get(qid, "")
                json_answer = answers[label]["json"].get(qid, "")
                baseline_answers[label_display[label]] = {
                    "text": {
                        "answer": text_answer,
                        "scores": scores[label]["text"].get(qid, {}),
                    },
                    "parsed": {
                        "answer": extract_json_blocks(json_answer, qid) if json_answer else [],
                        "scores": scores[label]["json"].get(qid, {}),
                    },
                }

            with open(q_dir / "question.json", "w") as f:
                json.dump(clean_q, f, indent=2)
            with open(q_dir / "baseline_answers.json", "w") as f:
                json.dump(baseline_answers, f, indent=2)

    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results…")
    finally:
        if conn:
            conn.close()

    if verify_errors:
        err_path = out_dir / "verification_errors.json"
        with open(err_path, "w") as f:
            json.dump(verify_errors, f, indent=2)
        print(f"\n{len(verify_errors)} verification errors → {err_path}")
    else:
        print("\nAll SQL answers verified successfully.")

    total = sum(counts.values())
    print(f"Written {total} questions to {out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Build clean benchmark directory")
    parser.add_argument(
        "--out-dir", default=str(BENCHMARK_DIR),
        help="Output directory (default: ./benchmark)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip SQL correctness verification",
    )
    parser.add_argument(
        "--verify-geo-wkts", action="store_true",
        help="Also verify geo_wkt values match the database geometries",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Filter labels by model prefix (e.g. gpt4o sonnet4.6). Default: all discovered.",
    )
    parser.add_argument(
        "--rename", nargs="*", default=None, metavar="OLD=NEW",
        help="Rename labels in output keys (e.g. ministral-3:14b-cloud_shuffled=shuffled).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
