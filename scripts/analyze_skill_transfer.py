#!/usr/bin/env python3
"""
Predict full-task scores from diagnostic skill profiles.

Uses a linear model: predicted_score = sum(skill_weight[skill] * diagnostic_score[skill])
per task. Fits weights from task_categories.json and diagnostic JSON. Useful to extrapolate
from quick diagnostic runs without running full benchmarks.

Inputs:
  - --diagnostic: logs/{agent}_diagnostic.json (skill_profile or diagnostic_results)
  - --full-benchmark: logs dir or JSON with actual full-task scores
  - --categories: data/task_categories.json (skill_weights per env)

Output: data/{agent}_transfer.json — predictions, R², per-task actual vs predicted
"""

import argparse
import json
from pathlib import Path

SKILLS = ("spatial", "deductive", "inductive", "grounded")


def _diagnostic_scores(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    p = d.get("skill_profile", {})
    if p:
        return {s: float(p.get(s, 0)) for s in SKILLS}
    r = d.get("diagnostic_results", {})
    return {s: float(r.get(s, {}).get("average_score", 0)) for s in SKILLS}


def _full_benchmark(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        out = {}
        for f in p.rglob("*.json"):
            if "diagnostic" in f.name.lower():
                continue
            try:
                s = json.load(open(f))
                if task := s.get("env_name"):
                    if score := s.get("norm_score") or s.get("final/Normalized Score"):
                        out[task] = float(score)
            except (json.JSONDecodeError, KeyError):
                pass
        return out
    d = json.load(open(p))
    if isinstance(d, dict) and "results" in d:
        return {r["task"]: float(r.get("norm_score", r.get("score", 0))) for r in d["results"]}
    if isinstance(d, dict) and "predictions" in d:
        return {x["task"]: float(x["actual_score"]) for x in d["predictions"]}
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, dict):
            if s := v.get("norm_score") or v.get("score") or v.get("actual_score"):
                out[k] = float(s)
    return out


def _r2(actual: list, pred: list) -> float:
    if len(actual) < 2:
        return 0.0
    m = sum(actual) / len(actual)
    ss_tot = sum((a - m) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, pred))
    return 1 - (ss_res / ss_tot) if ss_tot else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--diagnostic", required=True)
    p.add_argument("-f", "--full-benchmark", required=True)
    p.add_argument("-c", "--categories", default="data/task_categories.json")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    diag = _diagnostic_scores(args.diagnostic)
    cats = json.load(open(args.categories))
    actual = _full_benchmark(args.full_benchmark)

    preds = []
    for task, act in actual.items():
        c = cats.get(task)
        if not c or "skill_weights" not in c:
            continue
        w = c["skill_weights"]
        contrib = {s: round(w.get(s, 0) * diag.get(s, 0), 4) for s in SKILLS}
        pred = sum(contrib.values())
        preds.append({
            "task": task, "skill_weights": w, "predicted_score": round(pred, 4),
            "actual_score": round(act, 4), "error": round(act - pred, 4),
            "skill_contributions": contrib,
        })

    act_l, pred_l = [x["actual_score"] for x in preds], [x["predicted_score"] for x in preds]
    ta = {
        "r_squared": round(_r2(act_l, pred_l), 4),
        "mse": round(sum((a - p) ** 2 for a, p in zip(act_l, pred_l)) / len(preds), 4) if preds else 0,
        "mae": round(sum(abs(a - p) for a, p in zip(act_l, pred_l)) / len(preds), 4) if preds else 0,
        "predictions": preds,
    }
    agent = json.load(open(args.diagnostic)).get("agent", "unknown")
    out = {"agent": agent, "diagnostic_scores": diag, "transfer_analysis": ta}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote to {args.output}  R²={ta['r_squared']}  n={len(preds)}")


if __name__ == "__main__":
    main()
