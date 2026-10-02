"""Run a registered, hash-bound offline optimizer comparison without fitting forecasts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence-dir", type=Path, required=True)
args = parser.parse_args()
e = args.evidence_dir.resolve()
plan = json.loads((e / "plan.json").read_text())
for name, digest in plan["hashes"].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"registered source/input changed: {name}")
if any((e/f"{kind}_{engine}.csv").exists() for kind in ("standard","origin") for engine in ("legacy","solio")):
    raise FileExistsError("comparison outputs already exist; register a fresh run")
env = {**os.environ, "PYTHONPATH": str(ROOT)}
summary = {}
for kind in ("standard", "origin"):
    for engine in ("legacy", "solio"):
        output = e / f"{kind}_{engine}.csv"
        command = [sys.executable, "-m", "fpl.milp.optimize" if engine == "legacy" else "fpl.milp.solio",
                   "--predictions-csv", str(e/f"{kind}_matched_predictions.csv"),
                   "--start-gw", "153", "--max-gw", "183", "--horizon", "3", "--time-limit", "1200",
                   "--output", str(output)]
        command += ["--scoring-mode", "corrected"] if engine == "legacy" else ["--season", "2024-25"]
        print(f"Starting {kind}/{engine}", flush=True)
        started = time.perf_counter()
        with (e/f"{kind}_{engine}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter()-started
        frame = pd.read_csv(output)
        if len(frame) != 31:
            raise RuntimeError("incomplete backtest")
        if engine == "legacy":
            if (e/f"{kind}_{engine}.log").read_text().count("Status: Optimal") != 31:
                raise RuntimeError("legacy backtest did not prove every solve optimal")
        elif not frame.solver_status.eq("Optimal").all() or not frame.mip_gap.eq(0).all():
            raise RuntimeError("Solio backtest did not prove every solve optimal")
        net = frame.actual_total_points - 4*frame.alpha
        summary[f"{kind}_{engine}"] = {"elapsed_seconds":elapsed,"gross_points":float(frame.actual_total_points.sum()),
            "net_points":float(net.sum()),"hits":int(frame.alpha.sum()),"optimal_solves":31}
        net_frame = frame.copy()
        net_frame["actual_total_points"] = net
        net_frame.to_csv(e/f"{kind}_{engine}_net.csv",index=False)
        (e/"summary.json").write_text(json.dumps(summary,indent=2)+'\n')
        print(f"Completed {kind}/{engine}: {summary[f'{kind}_{engine}']}",flush=True)
    for metric in ("gross","net"):
        suffix = "" if metric == "gross" else "_net"
        command = [sys.executable,"-m","fpl.milp.compare_backtests",str(e/f"{kind}_solio{suffix}.csv"),
                   str(e/f"{kind}_legacy{suffix}.csv"),"--n-boot","10000","--block-len","3","--seed","0"]
        with (e/f"{kind}_{metric}_comparison.log").open("x") as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
print("All optimizer comparisons complete",flush=True)
