from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VALIDATION_DIR = ROOT / "reports" / "validation"

VERDICT_MD = VALIDATION_DIR / "intraday_strategy_verdict_mar2026.md"
APEX_WF_JSON = VALIDATION_DIR / "apex_walkforward_8yr.json"
QUANT_LAB_JSON = VALIDATION_DIR / "quant_lab_intraday_8y.json"
TIMEEDGE_JSON = VALIDATION_DIR / "timeedge_lab_8y.json"
OUT_JSON = VALIDATION_DIR / "intraday_reality_bundle_mar2026.json"

TRADE_ACTIVITY_REQUIREMENT = {
    "preferred_cadence": "~1 trade per trading day",
    "minimum_acceptable_cadence": "2-3 trades per week",
    "reject_if_below": "<2 trades per week",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def key_to_snake(label: str) -> str:
    label = label.strip().lower()
    label = label.replace("/", "_")
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return label.strip("_")


def parse_number(value: str) -> Any:
    raw = value.strip()

    ratio_match = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", raw)
    if ratio_match:
        left = int(ratio_match.group(1))
        right = int(ratio_match.group(2))
        return {
            "raw": raw,
            "numerator": left,
            "denominator": right,
            "ratio": left / right if right else None,
        }

    clean = (
        raw.replace("Rs", "")
        .replace(",", "")
        .replace("%", "")
        .replace("+", "")
        .strip()
    )

    if clean.lower() in {"inf", "infinity"}:
        return float("inf")

    try:
        if re.fullmatch(r"[+-]?\d+", clean):
            return int(clean)
        if re.fullmatch(r"[+-]?\d*\.\d+", clean) or re.fullmatch(r"[+-]?\d+\.\d*", clean):
            return float(clean)
    except ValueError:
        return raw

    return raw


def parse_bullets(block: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in block.splitlines():
        if not line.startswith("- "):
            continue
        payload = line[2:].strip()
        if ":" not in payload:
            continue
        key, val = payload.split(":", 1)
        out[key_to_snake(key)] = parse_number(val)
    return out


def extract_section(text: str, header: str) -> str:
    idx = text.find(header)
    if idx == -1:
        return ""
    start = idx + len(header)
    tail = text[start:]
    next_section = re.search(r"\n## ", tail)
    if next_section:
        return tail[: next_section.start()].strip()
    return tail.strip()


def parse_verdict_markdown(text: str) -> dict[str, Any]:
    data_constraints_text = extract_section(text, "## Data and constraints")
    baseline_text = extract_section(text, "## Existing Apex baseline (current engine)")

    constraints = [
        line[2:].strip()
        for line in data_constraints_text.splitlines()
        if line.strip().startswith("- ")
    ]

    baseline_horizons: list[dict[str, Any]] = []
    for match in re.finditer(r"### (?P<label>.+?)\n(?P<body>(?:- .+\n)+)", baseline_text):
        label = match.group("label").strip()
        body = match.group("body")
        metrics = parse_bullets(body)
        baseline_horizons.append({"label": label, "metrics": metrics})

    deploy_stance_block = extract_section(text, "Most honest deploy stance right now:")
    deploy_stance = [
        re.sub(r"^\d+\.\s*", "", line.strip())
        for line in deploy_stance_block.splitlines()
        if re.match(r"^\d+\.\s+", line.strip())
    ]

    interpretation = ""
    for line in baseline_text.splitlines():
        if line.startswith("Interpretation:"):
            interpretation = line.replace("Interpretation:", "").strip()
            break

    return {
        "constraints": constraints,
        "baseline_horizons": baseline_horizons,
        "baseline_interpretation": interpretation,
        "deploy_stance": deploy_stance,
    }


def assess_strategy(
    *,
    name: str,
    net_return: float | None,
    profit_factor: float | None,
    win_rate: float | None,
    max_dd: float | None,
    avg_pnl: float | None,
    positive_folds: int | None,
    folds: int | None,
) -> dict[str, Any]:
    reasons: list[str] = []

    if profit_factor is not None and profit_factor < 1.0:
        reasons.append("negative_expectancy_after_costs")

    if net_return is not None and net_return < 0:
        reasons.append("negative_out_of_sample_or_full_period_return")

    if folds and positive_folds is not None:
        ratio = positive_folds / folds if folds else 0.0
        if ratio < 0.5:
            reasons.append("regime_instability_low_positive_fold_ratio")

    if win_rate is not None and avg_pnl is not None and win_rate >= 50 and avg_pnl < 0:
        reasons.append("win_rate_illusion_payoff_or_cost_drag")

    if max_dd is not None and max_dd < -20:
        reasons.append("high_drawdown_profile")

    status = "failed" if reasons else "unclear"

    return {
        "name": name,
        "status": status,
        "reasons": reasons,
        "metrics": {
            "net_return": net_return,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "avg_pnl": avg_pnl,
            "positive_folds": positive_folds,
            "folds": folds,
        },
    }


def build_bundle() -> dict[str, Any]:
    verdict_text = read_text(VERDICT_MD)
    apex_wf = read_json(APEX_WF_JSON)
    quant = read_json(QUANT_LAB_JSON)
    timeedge = read_json(TIMEEDGE_JSON)

    verdict = parse_verdict_markdown(verdict_text)

    folds = apex_wf.get("folds", [])
    wf_pos = sum(1 for f in folds if isinstance(f.get("net_return"), (int, float)) and f["net_return"] > 0)
    wf_n = len(folds)
    wf_summary = apex_wf.get("summary", {})

    quant_top = (quant.get("top10") or [{}])[0]
    quant_agg = quant_top.get("agg", {})
    quant_best_full = quant.get("best_full_8y", {})

    time_top = (timeedge.get("top15") or [{}])[0]
    time_agg = time_top.get("agg", {})
    robust_count = timeedge.get("robust_count")

    baseline_long = None
    baseline_120 = None
    baseline_59 = None
    for item in verdict["baseline_horizons"]:
        label = item.get("label", "")
        if "2014 trading days" in label:
            baseline_long = item
        elif "120 trading days" in label:
            baseline_120 = item
        elif "59 trading days" in label:
            baseline_59 = item

    assessments = [
        assess_strategy(
            name="apex_walkforward_crosscheck",
            net_return=wf_summary.get("net_return_mean"),
            profit_factor=wf_summary.get("profit_factor_mean"),
            win_rate=wf_summary.get("win_rate_mean"),
            max_dd=wf_summary.get("max_dd_mean"),
            avg_pnl=None,
            positive_folds=wf_pos,
            folds=wf_n,
        ),
        assess_strategy(
            name="quant_lab_top_candidate_walkforward",
            net_return=quant_agg.get("net_return_mean"),
            profit_factor=quant_agg.get("profit_factor_mean"),
            win_rate=quant_agg.get("win_rate_mean"),
            max_dd=quant_agg.get("max_dd_mean"),
            avg_pnl=quant_agg.get("avg_pnl_mean"),
            positive_folds=quant_agg.get("positive_folds"),
            folds=quant_agg.get("folds"),
        ),
        assess_strategy(
            name="quant_lab_top_candidate_full_8y",
            net_return=quant_best_full.get("net_return"),
            profit_factor=quant_best_full.get("profit_factor"),
            win_rate=quant_best_full.get("win_rate"),
            max_dd=quant_best_full.get("max_dd"),
            avg_pnl=quant_best_full.get("avg_pnl"),
            positive_folds=None,
            folds=None,
        ),
        assess_strategy(
            name="timeedge_top_candidate_walkforward",
            net_return=time_agg.get("net_return_mean"),
            profit_factor=time_agg.get("profit_factor_mean"),
            win_rate=time_agg.get("win_rate_mean"),
            max_dd=time_agg.get("max_dd_mean"),
            avg_pnl=time_agg.get("avg_pnl_mean"),
            positive_folds=time_agg.get("positive_folds"),
            folds=time_agg.get("folds"),
        ),
    ]

    data_limitations = []
    for c in verdict["constraints"]:
        low = c.lower()
        if "volume is 100% zero" in low:
            data_limitations.append("volume_field_unusable_zero_values")
        if "cost model" in low:
            data_limitations.append("costs_included_tax_and_slippage")

    if robust_count == 0:
        data_limitations.append("timeedge_search_found_zero_robust_candidates")

    cross_observations = [
        "Short-window profitability exists (59-day patch) but fails long-horizon robustness checks.",
        "Across independent strategy families, top candidates still show profit_factor < 1 and negative mean returns.",
        "Fold-by-fold outcomes are unstable, indicating strong regime sensitivity.",
        "Observed win-rate pockets do not convert to positive expectancy after costs.",
    ]

    next_requirements = [
        "Require majority positive OOS folds before any deployment decisions.",
        "Require profit_factor > 1.0 and mean net_return > 0 after explicit cost model.",
        "Add richer market-state features before another strategy sweep (options chain/OI/IV where available).",
        "Do not use volume-dependent alpha while current dataset volume field remains unusable.",
        "Use minimum trade-count guardrails to avoid short-window overinterpretation.",
        "Trade frequency gate: target ~1 trade/day; minimum acceptable is 2-3 trades/week; reject lower cadence.",
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": "2026-03-29",
        "source_files": {
            "intraday_strategy_verdict": str(VERDICT_MD.relative_to(ROOT)).replace("\\", "/"),
            "apex_walkforward": str(APEX_WF_JSON.relative_to(ROOT)).replace("\\", "/"),
            "quant_lab": str(QUANT_LAB_JSON.relative_to(ROOT)).replace("\\", "/"),
            "timeedge_lab": str(TIMEEDGE_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "dataset_context": {
            "constraints": verdict["constraints"],
            "detected_limitations": sorted(set(data_limitations)),
        },
        "user_preferences": {
            "trade_activity_requirement": TRADE_ACTIVITY_REQUIREMENT,
        },
        "baseline_apex": {
            "horizons": verdict["baseline_horizons"],
            "interpretation": verdict["baseline_interpretation"],
        },
        "walkforward_crosscheck": {
            "summary": wf_summary,
            "positive_folds": wf_pos,
            "folds": wf_n,
            "fold_details": folds,
        },
        "independent_searches": {
            "quant_lab_top_candidate": quant_top,
            "quant_lab_best_full_8y": quant_best_full,
            "timeedge_top_candidate": time_top,
            "timeedge_robust_candidate_count": robust_count,
        },
        "assessment_matrix": assessments,
        "cross_strategy_observations": cross_observations,
        "why_strategies_failed": {
            "primary": [
                "negative_expectancy_after_costs",
                "regime_instability",
                "win_rate_without_positive_payoff",
                "data_feature_limitations",
            ],
            "notes": [
                "Most tested families fail profitability and robustness gates simultaneously.",
                "One positive fold/year does not generalize across OOS folds.",
                "Short-window gains are insufficient evidence for scalable edge.",
            ],
        },
        "next_strategy_requirements": next_requirements,
        "deploy_stance": verdict["deploy_stance"],
        "ai_ingestion_summary": {
            "overall_verdict": "No robust intraday strategy edge validated on current substrate.",
            "trade_activity_gate": TRADE_ACTIVITY_REQUIREMENT,
            "minimum_next_validation_gates": {
                "profit_factor": "> 1.0",
                "mean_oos_net_return": "> 0",
                "positive_fold_ratio": ">= 0.6",
                "drawdown_control": "bounded and stable across folds",
            },
        },
    }


def main() -> None:
    missing = [
        path
        for path in [VERDICT_MD, APEX_WF_JSON, QUANT_LAB_JSON, TIMEEDGE_JSON]
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in missing)
        raise FileNotFoundError(f"Missing required input artifacts: {names}")

    bundle = build_bundle()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"Bundle generated: {str(OUT_JSON.relative_to(ROOT)).replace('\\', '/')}")


if __name__ == "__main__":
    main()
