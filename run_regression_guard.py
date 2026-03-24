import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_SYMBOLS = ["NIFTY 50", "NIFTY BANK", "SENSEX"]
DEFAULT_DAYS = [1825, 3650, 5475]


@dataclass
class SnapshotResult:
    name: str
    root: Path
    backtests: Dict[Tuple[str, int], Dict[str, float]]
    walkforward_oos: Dict[str, Dict[str, float]]


def run_cmd(args: List[str], cwd: Path, timeout: int = 0) -> str:
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=None if timeout <= 0 else timeout,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n{output[:4000]}"
        )
    return output


def parse_float(raw: str) -> float:
    cleaned = raw.strip()
    cleaned = cleaned.replace("Rs", "").replace(",", "")
    cleaned = cleaned.replace("+", "")
    cleaned = cleaned.replace("%", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"Cannot parse numeric value from: {raw}")
    return float(match.group(0))


def parse_metric_from_text(text: str, label: str) -> float:
    m = re.search(rf"{re.escape(label)}:\s*([^\r\n]+)", text)
    if not m:
        raise ValueError(f"Metric not found: {label}")
    return parse_float(m.group(1))


def extract_block(text: str, header: str, window: int = 5000) -> str:
    idx = text.find(header)
    if idx < 0:
        raise ValueError(f"Block header not found: {header}")
    return text[idx : idx + window]


def parse_backtest_metrics(text: str) -> Dict[str, float]:
    return {
        "Return": parse_metric_from_text(text, "Total Return"),
        "CAGR": parse_metric_from_text(text, "CAGR"),
        "PF": parse_metric_from_text(text, "Profit Factor"),
        "Sharpe": parse_metric_from_text(text, "Sharpe Ratio"),
        "Calmar": parse_metric_from_text(text, "Calmar Ratio"),
        "MaxDD": parse_metric_from_text(text, "Max Drawdown"),
    }


def parse_walkforward_oos_metrics(text: str, symbol: str) -> Dict[str, float]:
    symbol_tag = symbol.replace(" ", "_")
    block = extract_block(text, f"BACKTEST RESULTS: WF_test_{symbol_tag}")
    pbo_match = re.search(r"PBO score:\s*([0-9.]+)", text)
    if not pbo_match:
        raise ValueError(f"PBO score not found for {symbol}")
    metrics = {
        "Return": parse_metric_from_text(block, "Total Return"),
        "PF": parse_metric_from_text(block, "Profit Factor"),
        "Sharpe": parse_metric_from_text(block, "Sharpe Ratio"),
        "Calmar": parse_metric_from_text(block, "Calmar Ratio"),
        "MaxDD": parse_metric_from_text(block, "Max Drawdown"),
        "PBO": float(pbo_match.group(1)),
    }
    return metrics


def run_snapshot(
    snapshot_name: str,
    root: Path,
    symbols: List[str],
    days_list: List[int],
    data_source: str,
    fetch_retries: int,
) -> SnapshotResult:
    backtests: Dict[Tuple[str, int], Dict[str, float]] = {}
    walkforward_oos: Dict[str, Dict[str, float]] = {}

    def run_with_parse_retry(cmd: List[str], parse_fn, parse_label: str, attempts: int = 3):
        last_exc: Optional[Exception] = None
        last_output = ""
        for attempt in range(1, attempts + 1):
            try:
                output = run_cmd(cmd, cwd=root)
                parsed = parse_fn(output)
                return parsed, output
            except Exception as exc:
                last_exc = exc
                last_output = output if "output" in locals() else ""
                if attempt < attempts:
                    time.sleep(2)
                    continue
        preview = (last_output or "")[:2000]
        raise RuntimeError(
            f"Failed to parse {parse_label} after {attempts} attempts. "
            f"Last error: {last_exc}\nOutput preview:\n{preview}"
        )

    for symbol in symbols:
        print(f"[{snapshot_name}] walkforward: {symbol}")
        walk_cmd = [
            "python",
            "prometheus/main.py",
            "walkforward",
            "--symbol",
            symbol,
            "--parrondo",
            "--data-source",
            data_source,
            "--fetch-retries",
            str(fetch_retries),
        ]
        parsed_wf, _ = run_with_parse_retry(
            walk_cmd,
            lambda txt: parse_walkforward_oos_metrics(txt, symbol),
            f"walkforward {symbol}",
        )
        walkforward_oos[symbol] = parsed_wf

        for days in days_list:
            print(f"[{snapshot_name}] backtest: {symbol} {days}d")
            bt_cmd = [
                "python",
                "prometheus/main.py",
                "backtest",
                "--days",
                str(days),
                "--symbol",
                symbol,
                "--parrondo",
                "--data-source",
                data_source,
                "--fetch-retries",
                str(fetch_retries),
            ]
            parsed_bt, _ = run_with_parse_retry(
                bt_cmd,
                parse_backtest_metrics,
                f"backtest {symbol} {days}",
            )
            backtests[(symbol, days)] = parsed_bt

    return SnapshotResult(
        name=snapshot_name,
        root=root,
        backtests=backtests,
        walkforward_oos=walkforward_oos,
    )


def add_worktree(ref: str, prefix: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    run_cmd(["git", "worktree", "add", "--detach", str(temp_dir), ref], cwd=Path.cwd())
    return temp_dir


def hydrate_runtime_files(source_root: Path, target_root: Path) -> None:
    """
    Copy local runtime-only files required for command execution into temporary worktrees.
    This keeps strict code snapshot comparison while allowing local credentials-dependent CLI runs.
    """
    rel_paths = [
        Path("prometheus/config/credentials.yaml"),
    ]
    for rel in rel_paths:
        src = source_root / rel
        dst = target_root / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def remove_worktree(path: Path) -> None:
    try:
        run_cmd(["git", "worktree", "remove", "--force", str(path)], cwd=Path.cwd())
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def compare_no_degrade(base: float, target: float, higher_is_better: bool) -> Tuple[float, str]:
    delta = target - base
    if abs(delta) < 1e-9:
        return delta, "PASS"
    if higher_is_better:
        return delta, "PASS" if delta >= 0 else "FAIL"
    return delta, "PASS" if delta <= 0 else "FAIL"


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_deployment_filters(
    out_path: Path,
    walkforward_rows: List[Dict[str, object]],
) -> None:
    whitelist: List[str] = []
    symbol_rules: Dict[str, Dict[str, float]] = {}

    for row in walkforward_rows:
        symbol = str(row["Symbol"])
        status_pf = row["PF_Status"] == "PASS"
        status_sh = row["Sharpe_Status"] == "PASS"
        status_ca = row["Calmar_Status"] == "PASS"
        status_dd = row["MaxDD_Status"] == "PASS"
        status_pbo = row["PBO_Status"] == "PASS"
        tgt_pbo = float(row["Target_PBO"])
        tgt_pf = float(row["Target_PF"])
        tgt_sh = float(row["Target_Sharpe"])
        tgt_dd = float(row["Target_MaxDD"])

        robust_enough = tgt_pbo < 0.50 and tgt_pf >= 1.50 and tgt_sh >= 1.50
        no_degrade = status_pf and status_sh and status_ca and status_dd and status_pbo

        if robust_enough and no_degrade:
            whitelist.append(symbol)
            if tgt_dd <= 25:
                symbol_rules[symbol] = {
                    "max_position_risk_pct": 1.00,
                    "max_daily_loss_pct": 3.00,
                    "max_concurrent_positions": 2,
                }
            elif tgt_dd <= 35:
                symbol_rules[symbol] = {
                    "max_position_risk_pct": 0.80,
                    "max_daily_loss_pct": 2.50,
                    "max_concurrent_positions": 1,
                }
            else:
                symbol_rules[symbol] = {
                    "max_position_risk_pct": 0.60,
                    "max_daily_loss_pct": 2.00,
                    "max_concurrent_positions": 1,
                }

    lines = [
        "# Auto-generated by run_regression_guard.py",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "policy: strict_no_degrade",
        "global:",
        "  enabled: true",
        "  require_pbo_below: 0.50",
        "  require_no_metric_degradation: true",
        "symbols:",
    ]

    if whitelist:
        for symbol in whitelist:
            rule = symbol_rules[symbol]
            lines.extend(
                [
                    f"  \"{symbol}\":",
                    "    trade_enabled: true",
                    f"    max_position_risk_pct: {rule['max_position_risk_pct']:.2f}",
                    f"    max_daily_loss_pct: {rule['max_daily_loss_pct']:.2f}",
                    f"    max_concurrent_positions: {int(rule['max_concurrent_positions'])}",
                ]
            )
    else:
        lines.append("  {}")

    lines.extend(
        [
            "symbol_whitelist:",
            *(f"  - \"{s}\"" for s in whitelist),
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strict A/B regression guard: compare previous commit vs current snapshot "
            "using PF, Sharpe, Calmar, MaxDD, and PBO with PASS/FAIL drift thresholds."
        )
    )
    parser.add_argument("--base-ref", default="HEAD~1", help="Baseline git ref (default: HEAD~1)")
    parser.add_argument(
        "--target-ref",
        default="WORKTREE",
        help="Target git ref or WORKTREE for current uncommitted snapshot",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to evaluate",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=DEFAULT_DAYS,
        help="Backtest day windows",
    )
    parser.add_argument(
        "--out-dir",
        default="regression/ab_compare",
        help="Output directory for delta tables",
    )
    parser.add_argument(
        "--data-source",
        default="yfinance",
        choices=["auto", "kite", "angelone", "yfinance"],
        help="Force identical historical source for both snapshots (default: yfinance).",
    )
    parser.add_argument(
        "--fetch-retries",
        type=int,
        default=2,
        help="Retries per fetch call (passed through to main.py).",
    )
    parser.add_argument(
        "--allow-degrade",
        action="store_true",
        help="Do not fail process on threshold breaches (report-only mode).",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    base_wt: Optional[Path] = None
    target_wt: Optional[Path] = None

    try:
        print(f"Preparing baseline snapshot: {args.base_ref}")
        base_wt = add_worktree(args.base_ref, "ab_base_")
        hydrate_runtime_files(repo_root, base_wt)

        if args.target_ref.upper() == "WORKTREE":
            target_root = repo_root
            target_name = "WORKTREE"
        else:
            print(f"Preparing target snapshot: {args.target_ref}")
            target_wt = add_worktree(args.target_ref, "ab_target_")
            hydrate_runtime_files(repo_root, target_wt)
            target_root = target_wt
            target_name = args.target_ref

        baseline = run_snapshot(
            "BASE",
            base_wt,
            args.symbols,
            args.days,
            args.data_source,
            args.fetch_retries,
        )
        target = run_snapshot(
            target_name,
            target_root,
            args.symbols,
            args.days,
            args.data_source,
            args.fetch_retries,
        )

        backtest_rows: List[List[object]] = []
        for symbol in args.symbols:
            for days in args.days:
                b = baseline.backtests[(symbol, days)]
                t = target.backtests[(symbol, days)]

                d_pf, s_pf = compare_no_degrade(b["PF"], t["PF"], True)
                d_sh, s_sh = compare_no_degrade(b["Sharpe"], t["Sharpe"], True)
                d_ca, s_ca = compare_no_degrade(b["Calmar"], t["Calmar"], True)
                d_dd, s_dd = compare_no_degrade(b["MaxDD"], t["MaxDD"], False)

                backtest_rows.append(
                    [
                        symbol,
                        days,
                        b["PF"],
                        t["PF"],
                        d_pf,
                        s_pf,
                        b["Sharpe"],
                        t["Sharpe"],
                        d_sh,
                        s_sh,
                        b["Calmar"],
                        t["Calmar"],
                        d_ca,
                        s_ca,
                        b["MaxDD"],
                        t["MaxDD"],
                        d_dd,
                        s_dd,
                    ]
                )

        wf_rows_for_csv: List[List[object]] = []
        wf_rows_for_rules: List[Dict[str, object]] = []
        for symbol in args.symbols:
            b = baseline.walkforward_oos[symbol]
            t = target.walkforward_oos[symbol]

            d_pf, s_pf = compare_no_degrade(b["PF"], t["PF"], True)
            d_sh, s_sh = compare_no_degrade(b["Sharpe"], t["Sharpe"], True)
            d_ca, s_ca = compare_no_degrade(b["Calmar"], t["Calmar"], True)
            d_dd, s_dd = compare_no_degrade(b["MaxDD"], t["MaxDD"], False)
            d_pbo, s_pbo = compare_no_degrade(b["PBO"], t["PBO"], False)

            wf_row_dict = {
                "Symbol": symbol,
                "Base_PF": b["PF"],
                "Target_PF": t["PF"],
                "Delta_PF": d_pf,
                "PF_Status": s_pf,
                "Base_Sharpe": b["Sharpe"],
                "Target_Sharpe": t["Sharpe"],
                "Delta_Sharpe": d_sh,
                "Sharpe_Status": s_sh,
                "Base_Calmar": b["Calmar"],
                "Target_Calmar": t["Calmar"],
                "Delta_Calmar": d_ca,
                "Calmar_Status": s_ca,
                "Base_MaxDD": b["MaxDD"],
                "Target_MaxDD": t["MaxDD"],
                "Delta_MaxDD": d_dd,
                "MaxDD_Status": s_dd,
                "Base_PBO": b["PBO"],
                "Target_PBO": t["PBO"],
                "Delta_PBO": d_pbo,
                "PBO_Status": s_pbo,
            }
            wf_rows_for_rules.append(wf_row_dict)
            wf_rows_for_csv.append(list(wf_row_dict.values()))

        write_csv(
            out_dir / "backtest_delta_table.csv",
            [
                "Symbol",
                "Days",
                "Base_PF",
                "Target_PF",
                "Delta_PF",
                "PF_Status",
                "Base_Sharpe",
                "Target_Sharpe",
                "Delta_Sharpe",
                "Sharpe_Status",
                "Base_Calmar",
                "Target_Calmar",
                "Delta_Calmar",
                "Calmar_Status",
                "Base_MaxDD",
                "Target_MaxDD",
                "Delta_MaxDD",
                "MaxDD_Status",
            ],
            backtest_rows,
        )

        write_csv(
            out_dir / "walkforward_oos_delta_table.csv",
            list(wf_rows_for_rules[0].keys()) if wf_rows_for_rules else [],
            wf_rows_for_csv,
        )

        deploy_path = repo_root / "prometheus" / "config" / "deployment_filters.yaml"
        generate_deployment_filters(deploy_path, wf_rows_for_rules)

        print("\n=== STRICT NO-DEGRADE SUMMARY (WALKFORWARD OOS) ===")
        overall_pass = True
        for row in wf_rows_for_rules:
            statuses = [
                row["PF_Status"],
                row["Sharpe_Status"],
                row["Calmar_Status"],
                row["MaxDD_Status"],
                row["PBO_Status"],
            ]
            overall = "PASS" if all(x == "PASS" for x in statuses) else "FAIL"
            if overall == "FAIL":
                overall_pass = False
            print(
                f"{row['Symbol']}: {overall} | "
                f"PF {row['Delta_PF']:+.3f} ({row['PF_Status']}), "
                f"Sharpe {row['Delta_Sharpe']:+.3f} ({row['Sharpe_Status']}), "
                f"Calmar {row['Delta_Calmar']:+.3f} ({row['Calmar_Status']}), "
                f"MaxDD {row['Delta_MaxDD']:+.3f} ({row['MaxDD_Status']}), "
                f"PBO {row['Delta_PBO']:+.3f} ({row['PBO_Status']})"
            )

        print("\nOutputs:")
        print(f"- {out_dir / 'backtest_delta_table.csv'}")
        print(f"- {out_dir / 'walkforward_oos_delta_table.csv'}")
        print(f"- {deploy_path}")

        if not overall_pass:
            if not args.allow_degrade:
                print("\nSTRICT NO-DEGRADE RESULT: FAIL (threshold breach detected)")
                return 2
            print("\nSTRICT NO-DEGRADE RESULT: FAIL (report-only mode; --allow-degrade set)")
            return 0

        print("\nSTRICT NO-DEGRADE RESULT: PASS")

        return 0
    finally:
        if base_wt is not None:
            remove_worktree(base_wt)
        if target_wt is not None:
            remove_worktree(target_wt)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
