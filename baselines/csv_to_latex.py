#!/usr/bin/env python3
"""
Convert CSV files in exp_tables/ to LaTeX format.

Edit MODEL_MAP below to customize short labels for each model variant.
Short-name convention used here:
  base model letter  +  suffix for retrieval method
  M = ministral-3:14b-cloud,  S = sonnet4.6,  G = gpt4o
  (no suffix) = direct,  R = rag,  T = text2sql
  standalone "R" = shuffled baseline
"""

import glob
import json
import os
import pandas as pd

# ── Customize model short labels here ────────────────────────────────────────
MODEL_MAP = {
    # ministral
    "ministral-3:14b-cloud_direct":   "M",
    "ministral-3:14b-cloud_rag":      "MR",
    "ministral-3:14b-cloud_text2sql": "MT",
    # sonnet
    "sonnet4.6_direct":               "S",
    "sonnet4.6_rag":                  "SR",
    "sonnet4.6_text2sql":             "ST",
    # gpt-4o
    "gpt4o_direct":                   "G",
    "gpt4o_rag":                      "GR",
    "gpt4o_text2sql":                 "GT",
    # shuffled baseline
    "shuffled":                       "R",
}
# ── Output-type row ordering ──────────────────────────────────────────────────
OUTPUT_TYPE_ORDER = ["name", "loc", "angle", "area", "count", "distance", "length"]
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "exp_tables")
OUTPUT_DIR = os.path.join(INPUT_DIR, "latex")

# Sort longest-first so that e.g. "ministral-3:14b-cloud_direct" is replaced
# before a hypothetical shorter substring could match first.
_SORTED_MAP = sorted(MODEL_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
# Short name for the shuffled baseline (used to find and move its columns last).
_SHUFFLED_SHORT = MODEL_MAP["shuffled"]


def rename_str(s: str) -> str:
    """Replace every full model name in *s* with its short label."""
    for full, short in _SORTED_MAP:
        s = s.replace(full, short)
    return s


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply model-name renaming to column headers, index, and string cells."""
    df = df.copy()
    df.columns = [rename_str(str(c)) for c in df.columns]
    df.index   = [rename_str(str(i)) for i in df.index]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: rename_str(str(x)) if pd.notna(x) else x
            )
    return df


def _hline_to_booktabs(latex: str) -> str:
    """Replace booktabs rules with \\hline (booktabs package not available)."""
    latex = latex.replace(r"\toprule", r"\hline")
    latex = latex.replace(r"\midrule", r"\hline")
    latex = latex.replace(r"\bottomrule", r"\hline")
    return latex


def reorder_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. If an output_type column is present, sort rows by OUTPUT_TYPE_ORDER.
    2. Move shuffled-baseline columns (exactly '_SHUFFLED_SHORT' or
       ending with '_<shuffled_short>') to the rightmost position.
    Called *after* process_df so model names are already short.
    """
    # ── Row ordering ─────────────────────────────────────────────────────────
    type_col = next(
        (c for c in df.columns if c in ("output_type", "type")), None
    )
    if type_col is not None:
        present  = [t for t in OUTPUT_TYPE_ORDER if t in df[type_col].values]
        leftover = [v for v in df[type_col] if v not in OUTPUT_TYPE_ORDER]
        new_order = present + leftover
        df = (
            df.set_index(type_col)
              .reindex(new_order)
              .reset_index()
              .rename(columns={"index": type_col})
        )

    # ── Column ordering: M, MT, MR, S, ST, SR, G, GT, GR, R ─────────────────
    # Order all metric columns by the desired model sequence, preserving any
    # non-metric (string) columns in their original position.
    MODEL_COL_ORDER = ["M", "MT", "MR", "S", "ST", "SR", "G", "GT", "GR", "R"]

    def model_of(col):
        """Return the model short-name embedded in a column name, or None."""
        for m in sorted(MODEL_COL_ORDER, key=len, reverse=True):
            if col == m or col.endswith(f"_{m}"):
                # Make sure we don't match 'MR' when looking for 'R', etc.
                # The endswith check is unambiguous because we check longest first.
                return m
        return None

    # Separate non-metric columns (string dtype or no model suffix) from metric ones
    non_metric = [c for c in df.columns if model_of(c) is None]
    metric = [c for c in df.columns if model_of(c) is not None]
    ordered_metric = sorted(metric, key=lambda c: MODEL_COL_ORDER.index(model_of(c)))
    return df[non_metric + ordered_metric]


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in a plain string."""
    for ch, repl in [("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                     ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(ch, repl)
    return s


def _fmt(val) -> str:
    """Format a cell value for LaTeX: float → 4 dp, NaN → --."""
    try:
        if pd.isna(val):
            return "--"
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return str(val)


def make_name_text_recall_table(df: pd.DataFrame) -> str:
    """
    Custom two-row grouped header for name_text_recall_parsed_f1.

    Layout:
      TID || Recall on full text output (R_* cols) || F1 on parsed output (F1_* cols)
    TID uses \\multirow{2}{*} to span the two header rows.
    A vertical rule separates the two metric groups.
    """
    df = process_df(df)

    # Model order after renaming (direct → text2sql → rag → shuffled)
    models = ["M", "S", "G", "MT", "ST", "GT", "MR", "SR", "GR", "R"]
    recall_cols = [f"R_{m}" for m in models]
    f1_cols     = [f"F1_{m}" for m in models]
    n = len(models)   # 10

    col_spec = f"l|{'c' * n}|{'c' * n}"

    lines = [
        r"\begin{table}",
        r"\caption{Name Text Recall Parsed F1}",
        r"\label{tab:name_text_recall_parsed_f1}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        # Header row 1 – group labels; TID spans both rows.
        # Use |c| on the first multicolumn to keep the left border explicit
        # (multicolumn would otherwise suppress the | from the column spec).
        (
            r"\multirow{2}{*}{TID}"
            f" & \\multicolumn{{{n}}}{{|c|}}{{Recall on full text output}}"
            f" & \\multicolumn{{{n}}}{{c}}{{F1 on parsed output}} \\\\"
        ),
        # Partial hline between the group-label row and the model-name row.
        # Starts at col 2 so it doesn't cut through the TID \multirow cell.
        f"\\cline{{2-{1 + 2*n}}}",
        # Header row 2 – individual model names.
        # \multicolumn{1}{l|}{} on col 1 ensures the right border of the TID
        # column is drawn in this row too (otherwise the empty \multirow cell
        # may suppress it in some renderers).
        r"\multicolumn{1}{l|}{} & " + " & ".join(models) + " & " + " & ".join(models) + r" \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        tid_cell  = str(row["TID"])
        r_vals  = " & ".join(_fmt(row.get(c)) for c in recall_cols)
        f1_vals = " & ".join(_fmt(row.get(c)) for c in f1_cols)
        lines.append(f"{tid_cell} & {r_vals} & {f1_vals} \\\\")

    # Average row
    lines.append(r"\hline")
    avg_r  = " & ".join(f"{df[c].mean():.2f}" if c in df.columns else "--" for c in recall_cols)
    avg_f1 = " & ".join(f"{df[c].mean():.2f}" if c in df.columns else "--" for c in f1_cols)
    lines.append(f"\\textbf{{Avg}} & {avg_r} & {avg_f1} \\\\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _insert_avg_row(latex: str, df: pd.DataFrame) -> str:
    """
    Append a midrule + Average row before \\bottomrule.
    Numeric columns are averaged; string columns get 'Avg'.
    The first string column (e.g. output_type) gets the label 'Avg'.
    """
    avg_cells = []
    first_str = True
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            avg_cells.append(f"{df[col].mean():.2f}")
        else:
            avg_cells.append("\\textbf{Avg}" if first_str else "")
            first_str = False

    avg_line = " & ".join(avg_cells) + r" \\"
    insertion = r"\hline" + "\n" + avg_line + "\n"
    # Replace the *last* \hline (closing rule) so the avg row goes at the bottom
    idx = latex.rfind(r"\hline")
    return latex[:idx] + insertion + latex[idx:]


def make_loc_parsed_scores_table(df: pd.DataFrame) -> str:
    """
    Custom two-row grouped header for loc_parsed_scores.

    Layout:
      TID | P (10 models) | R (10 models) | F1 (10 models) | distance_error (10 models)
    TID uses \\multirow{2}{*}.  Vertical rules separate the four metric groups.
    """
    df = process_df(df)

    models = ["M", "S", "G", "MT", "ST", "GT", "MR", "SR", "GR", "R"]
    metric_keys    = ["F1",                  "distance_error"]
    metric_labels  = ["F1 for address text", "Distance Error"]
    n = len(models)   # 10
    g = len(metric_keys)  # 2

    col_spec = "l|" + "|".join("c" * n for _ in range(g))

    group_headers = []
    for i, label in enumerate(metric_labels):
        fmt = "|c|" if i < g - 1 else "c"
        group_headers.append(f"\\multicolumn{{{n}}}{{{fmt}}}{{{label}}}")

    lines = [
        r"\begin{table}",
        r"\caption{Loc Parsed Scores}",
        r"\label{tab:loc_parsed_scores}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        r"\multirow{2}{*}{TID} & " + " & ".join(group_headers) + r" \\",
        f"\\cline{{2-{1 + g*n}}}",
        r"\multicolumn{1}{l|}{} & " + " & ".join(
            " & ".join(models) for _ in range(g)
        ) + r" \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        tid_cell = str(row["TID"])
        cells = []
        for mk in metric_keys:
            for m in models:
                cells.append(_fmt(row.get(f"{mk}_{m}")))
        lines.append(f"{tid_cell} & " + " & ".join(cells) + r" \\")

    # Average row
    lines.append(r"\hline")
    avg_cells = []
    for mk in metric_keys:
        for m in models:
            col = f"{mk}_{m}"
            avg_cells.append(f"{df[col].mean():.2f}" if col in df.columns else "--")
    lines.append(r"\textbf{Avg} & " + " & ".join(avg_cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_angle_parsed_scores_table(df: pd.DataFrame) -> str:
    """Two-group header: F1 for parsed direction | Angle Error. No avg row."""
    df = process_df(df)

    models = ["M", "S", "G", "MT", "ST", "GT", "MR", "SR", "GR", "R"]
    metric_keys   = ["F1",                       "angle_error"]
    metric_labels = ["F1 for parsed direction",  "Angle Error"]
    n = len(models)
    g = len(metric_keys)

    col_spec = "l|" + "|".join("c" * n for _ in range(g))

    group_headers = []
    for i, label in enumerate(metric_labels):
        fmt = "|c|" if i < g - 1 else "c"
        group_headers.append(f"\\multicolumn{{{n}}}{{{fmt}}}{{{label}}}")

    lines = [
        r"\begin{table}",
        r"\caption{Angle Parsed Scores}",
        r"\label{tab:angle_parsed_scores}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        r"\multirow{2}{*}{TID} & " + " & ".join(group_headers) + r" \\",
        f"\\cline{{2-{1 + g*n}}}",
        r"\multicolumn{1}{l|}{} & " + " & ".join(
            " & ".join(models) for _ in range(g)
        ) + r" \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        tid_cell = str(row["TID"])
        cells = []
        for mk in metric_keys:
            for m in models:
                cells.append(_fmt(row.get(f"{mk}_{m}")))
        lines.append(f"{tid_cell} & " + " & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def df_to_latex(df: pd.DataFrame, caption: str, label: str,
                add_avg: bool = False) -> str:
    df = process_df(df)
    df = reorder_df(df)
    latex = df.to_latex(
        caption=caption,
        label=label,
        escape=True,
        float_format=lambda x: f"{x:.2f}",
        na_rep="--",
        index=False,
    )
    latex = _hline_to_booktabs(latex)
    if add_avg:
        latex = _insert_avg_row(latex, df)
    return latex


def make_relative_error_table() -> str:
    """
    Combined relative-error table across Count, Distance, Area, Length.

    Row structure (mirrors the user's example):
      - Output type (multirow when multiple TIDs) | TID | one col per model
      - \cline{2-N} between TIDs inside the same type group
      - \midrule between different type groups
      - Minimum value per row is bolded
    """
    models = ["M", "S", "G", "MT", "ST", "GT", "MR", "SR", "GR", "R"]
    col_key = "relative_error"
    n = len(models)

    # ── Load and tag each CSV ─────────────────────────────────────────────────
    groups = [
        ("Count",    "count_parsed_scores.csv"),
        ("Distance", "distance_parsed_scores.csv"),
        ("Area",     "area_parsed_scores.csv"),
        ("Length",   "length_parsed_scores.csv"),
    ]

    # Build list of (group_label, tid, {model: value})
    rows = []
    for group_label, fname in groups:
        df = pd.read_csv(os.path.join(INPUT_DIR, fname))
        df = process_df(df)
        for _, row in df.iterrows():
            vals = {m: row.get(f"{col_key}_{m}") for m in models}
            rows.append((group_label, str(row["TID"]), vals))

    # ── Column spec: type | TID | models ─────────────────────────────────────
    col_spec = "l|l|" + "c" * n

    lines = [
        r"\begin{table}",
        r"\caption{Relative Error}",
        r"\label{tab:relative_error}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        "\\textbf{Type} & \\textbf{TID} & " + " & ".join(f"\\textbf{{{m}}}" for m in models) + r" \\",
        r"\hline",
    ]

    # Group rows by type label to know how many TIDs each has
    from itertools import groupby
    grouped = [(lbl, list(grp)) for lbl, grp in
               groupby(rows, key=lambda r: r[0])]

    last_group_idx = len(grouped) - 1
    for g_idx, (group_label, group_rows) in enumerate(grouped):
        n_rows = len(group_rows)
        for r_idx, (_, tid, vals) in enumerate(group_rows):
            # Bold the minimum value in this row
            numeric = {m: v for m, v in vals.items()
                       if v is not None and not pd.isna(v)}
            min_val = min(numeric.values()) if numeric else None

            cells = []
            for m in models:
                v = vals.get(m)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    cells.append("--")
                else:
                    s = f"{float(v):.2f}"
                    cells.append(f"\\textbf{{{s}}}" if float(v) == min_val else s)

            if r_idx == 0:
                if n_rows > 1:
                    # Wrap \multirow inside \multicolumn so the | border is
                    # explicitly declared in every row of the group.
                    inner = f"\\multirow{{{n_rows}}}{{*}}{{\\textbf{{{group_label}}}}}"
                    type_cell = f"\\multicolumn{{1}}{{l|}}{{{inner}}}"
                else:
                    type_cell = f"\\multicolumn{{1}}{{l|}}{{\\textbf{{{group_label}}}}}"
            else:
                type_cell = r"\multicolumn{1}{l|}{}"
            tid_cell = f"\\textbf{{{tid}}}"
            lines.append(f"{type_cell} & {tid_cell} & " + " & ".join(cells) + r" \\")


        # \midrule between groups (not after the last one)
        if g_idx < last_group_idx:
            lines.append(r"\hline")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _is_correct(scores: dict, output_type: str) -> bool:
    """
    Return True if parsed scores meet the correctness threshold for output_type.
      name / loc / angle : parsed F1 >= 0.5
      count / area / length / distance : relative_error <= 0.1
    """
    if not scores.get("attempted", False):
        return False
    numeric_types = {"count", "area", "length", "distance"}
    if output_type in numeric_types:
        err = scores.get("relative_error")
        return err is not None and err <= 0.1
    else:
        f1 = scores.get("F1", 0) or 0
        return f1 >= 0.5


def _sql_correct_counts() -> dict:
    """
    For each text2sql base model (M, S, G), count how many questions had:
      - attempted=True in parsed scores (SQL returned usable data)
      - AND met the per-output-type correctness threshold in parsed scores

    Thresholds: F1 >= 0.5 for name/loc/angle; relative_error <= 0.1 for numerics.

    Returns dict: model_short → {"valid": int, "correct": int, "total": int}
    """
    base_models = {
        "M": "ministral-3:14b-cloud",
        "S": "sonnet4.6",
        "G": "gpt4o",
    }
    answer_keys = {
        "M": "ministral-3:14b-cloud_text2sql",
        "S": "sonnet4.6_text2sql",
        "G": "gpt4o_text2sql",
    }

    # Build lookup: question_id → (output_type, baseline_answers path)
    q_files = glob.glob(os.path.join(BASE_DIR, "benchmark", "T*", "*", "question.json"))
    id_to_meta = {}
    for qf in q_files:
        try:
            q = json.load(open(qf))
            # Derive output type from the last '+'-separated part of the type string
            raw_type = q.get("type", "")
            output_type = raw_type.split("+")[-1]
            id_to_meta[q["id"]] = {
                "output_type": output_type,
                "ans_path": qf.replace("question.json", "baseline_answers.json"),
            }
        except Exception:
            continue

    results = {}
    for short, model_folder in base_models.items():
        cache_path = os.path.join(BASE_DIR, "cache", model_folder, "sql_exec.json")
        if not os.path.exists(cache_path):
            results[short] = None
            continue

        sql_exec = {r["id"]: r for r in json.load(open(cache_path))}
        model_key = answer_keys[short]

        total = len(sql_exec)
        valid = 0
        correct = 0
        for qid in sql_exec:
            meta = id_to_meta.get(qid)
            if meta is None:
                continue
            ans_path = meta["ans_path"]
            if not os.path.exists(ans_path):
                continue
            try:
                answers = json.load(open(ans_path))
            except Exception:
                continue

            parsed_scores = answers.get(model_key, {}).get("parsed", {}).get("scores", {})
            if not parsed_scores.get("attempted", False):
                continue

            valid += 1
            if _is_correct(parsed_scores, meta["output_type"]):
                correct += 1

        results[short] = {"total": total, "valid": valid, "correct": correct}

    return results


def make_sql_error_table(df: pd.DataFrame) -> str:
    """
    Transposed SQL error summary.
    Columns = model short labels.
    Rows = Valid, Not Valid, then individual error types.
    """
    df = process_df(df)  # renames the 'label' column values

    # SQL errors are per-model (not per-method variant), so only base models
    models = ["M", "S", "G"]
    present = [m for m in models if m in df["label"].values]

    # Index by model label for easy lookup
    data = df.set_index("label")

    error_cols = [c for c in df.columns if c != "label" and c != "valid"]

    # Pretty-print error type names (keys are raw column names, no escaping)
    label_map = {
        "timeout":                   "Timeout",
        "column_does_not_exist":     "Column not found",
        "sub_query_error":           "Sub-query error",
        "function_does_not_exist":   "Function not found",
        "other":                     "Other",
        "missing_from":              "Missing FROM",
        "syntax_error":              "Syntax error",
        "relation_does_not_exist":   "Relation not found",
        "operator_does_not_exist":   "Operator not found",
    }

    col_spec = "l|" + "c" * len(present)

    lines = [
        r"\begin{table}",
        r"\caption{SQL Error Summary}",
        r"\label{tab:sql_error_summary}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        "\\textbf{Error Type} & " + " & ".join(f"\\textbf{{{m}}}" for m in present) + r" \\",
        r"\hline",
    ]

    # The CSV counts individual SQL execution records (one question can produce
    # multiple SQL blocks), so we derive the true per-model totals from the cache.
    model_folder = {"M": "ministral-3:14b-cloud", "S": "sonnet4.6", "G": "gpt4o"}
    totals = {}
    for m in present:
        folder = model_folder.get(m)
        cache_path = os.path.join(BASE_DIR, "cache", folder, "sql_exec.json") if folder else None
        if cache_path and os.path.exists(cache_path):
            cache = json.load(open(cache_path))
            totals[m] = sum(len(item.get("records", [])) or 1 for item in cache)
        else:
            totals[m] = 2800  # fallback

    def pct(n, m):
        den = totals.get(m, 2800)
        return f"{n / den * 100:.1f}\\%" if den else "--"

    def get_count(m, col):
        if m in data.index and col in data.columns and not pd.isna(data.loc[m, col]):
            return int(data.loc[m, col])
        return 0

    error_type_cols = [c for c in error_cols if c != "timeout"]
    correct_stats = _sql_correct_counts()

    # ── Valid SQL group ───────────────────────────────────────────────────────
    # Valid SQL = executed without error (valid) + timed out (timeout)
    valid_counts   = [get_count(m, "valid")   for m in present]
    timeout_counts = [get_count(m, "timeout") for m in present]
    valid_sql      = [v + t for v, t in zip(valid_counts, timeout_counts)]
    correct_counts = [
        correct_stats[m]["correct"] if correct_stats.get(m) else 0
        for m in present
    ]
    incorrect_counts = [
        vs - t - c
        for vs, t, c in zip(valid_sql, timeout_counts, correct_counts)
    ]

    lines.append(
        "\\textbf{Valid SQL} & "
        + " & ".join(pct(v, m) for v, m in zip(valid_sql, present)) + r" \\"
    )
    lines.append(
        r"\quad Timeout & "
        + " & ".join(pct(t, m) for t, m in zip(timeout_counts, present)) + r" \\"
    )
    lines.append(
        r"\quad Correct Output & "
        + " & ".join(pct(c, m) for c, m in zip(correct_counts, present)) + r" \\"
    )
    lines.append(
        r"\quad Incorrect Output & "
        + " & ".join(pct(i, m) for i, m in zip(incorrect_counts, present)) + r" \\"
    )
    lines.append(r"\hline")

    # ── Invalid SQL group ─────────────────────────────────────────────────────
    invalid_counts = [
        sum(get_count(m, c) for c in error_type_cols)
        for m in present
    ]
    lines.append(
        "\\textbf{Invalid SQL} & "
        + " & ".join(pct(v, m) for v, m in zip(invalid_counts, present)) + r" \\"
    )
    for col in error_type_cols:
        pretty = label_map.get(col, col.replace("_", " ").title())
        vals = [get_count(m, col) for m in present]
        lines.append(
            f"\\quad {pretty} & "
            + " & ".join(pct(v, m) for v, m in zip(vals, present)) + r" \\"
        )

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_prf_text_table() -> str:
    """
    Average Precision, Recall and F1 on free text output, grouped by output type.

    Source: text_eval_by_type.csv
    Rows: output types (name, loc, angle, area, count, distance, length) + Avg
    Columns: three groups (P | R | F1), each with 10 model columns.
    """
    df = pd.read_csv(os.path.join(INPUT_DIR, "text_eval_by_type.csv"))
    df.columns = [rename_str(c) for c in df.columns]

    # Derive output type from the template name (everything after the last '+')
    def extract_output_type(t: str) -> str:
        parts = t.split("+")
        # Walk parts right-to-left to find the first known output type
        for part in reversed(parts):
            if part in OUTPUT_TYPE_ORDER:
                return part
        return parts[-1]  # fallback

    df["output_type"] = df["type"].apply(extract_output_type)

    models  = ["M", "MT", "MR", "S", "ST", "SR", "G", "GT", "GR", "R"]
    metrics = ["F1"]
    n = len(models)
    g = len(metrics)

    # Aggregate: mean per (output_type, metric, model)
    agg_rows = []
    for ot in OUTPUT_TYPE_ORDER:
        subset = df[df["output_type"] == ot]
        if subset.empty:
            continue
        row: dict = {"output_type": ot.capitalize()}
        for metric in metrics:
            for m in models:
                col = f"{metric}_{m}"
                row[col] = subset[col].mean() if col in subset.columns else float("nan")
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    col_spec = "l|" + "|".join("c" * n for _ in range(g))

    group_headers = []
    for i, label in enumerate(metrics):
        border = "|c|" if i < g - 1 else "c"
        group_headers.append(f"\\multicolumn{{{n}}}{{{border}}}{{{label}}}")

    lines = [
        r"\begin{table*}",
        r"\caption{Average Precision, Recall and F1 on Free Text Output}",
        r"\label{tab:prf_text_average}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        r"\multirow{2}{*}{Type} & " + " & ".join(group_headers) + r" \\",
        f"\\cline{{2-{1 + g * n}}}",
        r"\multicolumn{1}{l|}{} & "
        + " & ".join(" & ".join(models) for _ in range(g))
        + r" \\",
        r"\hline",
    ]

    for _, row in agg_df.iterrows():
        cells = []
        for metric in metrics:
            for m in models:
                cells.append(_fmt(row.get(f"{metric}_{m}")))
        lines.append(str(row["output_type"]) + " & " + " & ".join(cells) + r" \\")

    # Average row across all output types
    lines.append(r"\hline")
    avg_cells = []
    for metric in metrics:
        for m in models:
            col = f"{metric}_{m}"
            avg_cells.append(f"{agg_df[col].mean():.2f}" if col in agg_df.columns else "--")
    lines.append(r"\textbf{Avg} & " + " & ".join(avg_cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".csv"))
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    for fname in csv_files:
        base    = fname[:-4]                          # strip .csv
        inpath  = os.path.join(INPUT_DIR, fname)
        outpath = os.path.join(OUTPUT_DIR, base + ".tex")

        df      = pd.read_csv(inpath)
        caption = base.replace("_", " ").title()
        label   = f"tab:{base}"

        if base == "name_text_recall_parsed_f1":
            latex = make_name_text_recall_table(df)
        elif base == "loc_parsed_scores":
            latex = make_loc_parsed_scores_table(df)
        elif base == "angle_parsed_scores":
            latex = make_angle_parsed_scores_table(df)
        elif base == "sql_error_summary":
            latex = make_sql_error_table(df)
        elif base == "attempted_rates":
            latex = df_to_latex(df, caption=caption, label=label, add_avg=True)
        else:
            latex = df_to_latex(df, caption=caption, label=label)

        with open(outpath, "w") as fh:
            fh.write(latex)
        print(f"  {fname:<45}  →  latex/{base}.tex")

    # ── Extra: combined relative-error table ─────────────────────────────────
    rel_path = os.path.join(OUTPUT_DIR, "relative_error_scores.tex")
    with open(rel_path, "w") as fh:
        fh.write(make_relative_error_table())
    print(f"  {'(combined)relative_error_scores':<45}  →  latex/relative_error_scores.tex")

    # ── Extra: average P/R/F1 on free text output ─────────────────────────────
    prf_path = os.path.join(OUTPUT_DIR, "prf_text_average.tex")
    with open(prf_path, "w") as fh:
        fh.write(make_prf_text_table())
    print(f"  {'(combined)prf_text_average':<45}  →  latex/prf_text_average.tex")

    print(f"\nDone. {len(csv_files) + 2} table(s) written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
