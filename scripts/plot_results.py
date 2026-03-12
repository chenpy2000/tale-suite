# Plot diagnostic comparison and skill profiles from diagnostic JSON files.

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SKILLS = ("spatial", "deductive", "inductive", "grounded")
LABELS = {s: s.title() for s in SKILLS}
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
HYBRID_COLORS = ["#8e44ad", "#9b59b6", "#a569bd", "#bb8fce", "#d7bde2"]


def _load(files):
    out = []
    for path in files:
        d = json.load(open(path))
        scores, stds = {}, {}
        profile = d.get("skill_profile", {})
        diag = d.get("diagnostic_results", {})
        for s in SKILLS:
            if s in profile:
                scores[s], stds[s] = float(profile[s]), 0.0
            elif s in diag:
                tasks = diag[s].get("tasks", [])
                vals = [np.mean(t["episode_scores"]) if "episode_scores" in t else t.get("score", 0) for t in tasks]
                scores[s] = float(np.mean(vals)) if vals else 0.0
                stds[s] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                scores[s], stds[s] = 0.0, 0.0
        out.append((d.get("agent", Path(path).stem), scores, stds))
    return out


def diagnostic_comparison(files, out_dir):
    data = _load(files)
    if not data:
        raise ValueError("No data")
    x = np.arange(len(SKILLS))
    w = 0.8 / len(data)
    off = (len(data) - 1) * w / 2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, scores, stds) in enumerate(data):
        vals = [scores[s] * 100 for s in SKILLS]
        errs = [stds[s] * 100 for s in SKILLS]
        ax.bar(x - off + i * w, vals, w, label=name, color=COLORS[i % len(COLORS)],
               yerr=errs if any(e > 0 for e in errs) else None, capsize=3)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Skill")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SKILLS])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = Path(out_dir) / "diagnostic_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def skill_profiles(files, out_dir):
    data = _load(files)
    if not data:
        raise ValueError("No data")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SKILLS))
    for i, (name, scores, _) in enumerate(data):
        ax.plot(x, [scores[s] * 100 for s in SKILLS], marker="o", lw=2, color=COLORS[i % len(COLORS)], label=name)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Skill")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SKILLS])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(out_dir) / "skill_profiles.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _parse_hybrid_weights(path):
    """Parse graph/vqvae weights from filename like graph-vqvae-80-20.json."""
    stem = Path(path).stem
    m = re.match(r"graph-vqvae-(\d+)-(\d+)", stem, re.I)
    if m:
        g, v = int(m.group(1)), int(m.group(2))
        return g / 100.0, v / 100.0
    return None, None


def _overall_score(d):
    """Compute overall score from diagnostic JSON."""
    profile = d.get("skill_profile", {})
    diag = d.get("diagnostic_results", {})
    vals = []
    for s in SKILLS:
        if s in profile:
            vals.append(float(profile[s]))
        elif s in diag:
            tasks = diag[s].get("tasks", [])
            v = [np.mean(t["episode_scores"]) if "episode_scores" in t else t.get("score", 0) for t in tasks]
            vals.append(float(np.mean(v)) if v else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _expand_metrics(files):
    """Expand glob patterns in metrics file list."""
    out = []
    for f in files:
        expanded = glob.glob(f)
        out.extend(expanded if expanded else [f])
    return out


def plot_hybrid_weights(files, out_dir):
    """Line plot: performance vs graph weight for Graph+VQ-VAE hybrids."""
    files = _expand_metrics(files)
    points = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        g_w, v_w = _parse_hybrid_weights(path)
        if g_w is not None:
            score = _overall_score(d)
            points.append((g_w, score, Path(path).stem))
    if not points:
        raise ValueError("No Graph+VQ-VAE hybrid metrics found (expect graph-vqvae-X-Y.json)")
    points.sort(key=lambda x: x[0])
    x_vals = [p[0] for p in points]
    y_vals = [p[1] * 100 for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals, "o-", color="#9b59b6", lw=2, markersize=8)
    ax.set_xlabel("Graph weight")
    ax.set_ylabel("Overall score (%)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    for i, (g, s, name) in enumerate(points):
        ax.annotate(f"{g:.0%}", (g, s * 100), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    path = Path(out_dir) / "hybrid_weight_sensitivity.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _is_hybrid(name_or_path):
    """Check if agent/metric is a hybrid (by name or filename)."""
    s = str(name_or_path).lower()
    return "hybrid" in s or "graph-vqvae" in s or "memory-react" in s


def plot_hybrid_comparison(files, out_dir):
    """Bar chart: single agents and hybrids side-by-side, hybrids in purple shades."""
    files = _expand_metrics(files)
    data = _load(files)
    if not data:
        raise ValueError("No data")
    hybrids = [(n, s, std) for n, s, std in data if _is_hybrid(n)]
    singles = [(n, s, std) for n, s, std in data if not _is_hybrid(n)]
    # Order: singles first, then hybrids
    ordered = singles + hybrids
    x = np.arange(len(SKILLS))
    w = 0.8 / len(ordered)
    off = (len(ordered) - 1) * w / 2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, scores, stds) in enumerate(ordered):
        vals = [scores[s] * 100 for s in SKILLS]
        errs = [stds[s] * 100 for s in SKILLS]
        color = HYBRID_COLORS[i % len(HYBRID_COLORS)] if _is_hybrid(name) else COLORS[i % len(COLORS)]
        ax.bar(
            x - off + i * w,
            vals,
            w,
            label=name,
            color=color,
            yerr=errs if any(e > 0 for e in errs) else None,
            capsize=3,
        )
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Skill")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SKILLS])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = Path(out_dir) / "hybrid_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnostic-comparison")
    d.add_argument("-m", "--metrics", nargs="+", required=True)
    d.add_argument("-o", "--output", default="plots")
    s = sub.add_parser("skill-profiles")
    s.add_argument("-m", "--metrics", nargs="+", required=True)
    s.add_argument("-o", "--output", default="plots")
    hw = sub.add_parser("hybrid-weights")
    hw.add_argument("-m", "--metrics", nargs="+", required=True)
    hw.add_argument("-o", "--output", default="plots")
    hc = sub.add_parser("hybrid-comparison")
    hc.add_argument("-m", "--metrics", nargs="+", required=True)
    hc.add_argument("-o", "--output", default="plots")
    args = p.parse_args()
    if args.cmd == "diagnostic-comparison":
        diagnostic_comparison(args.metrics, args.output)
    elif args.cmd == "skill-profiles":
        skill_profiles(args.metrics, args.output)
    elif args.cmd == "hybrid-weights":
        plot_hybrid_weights(args.metrics, args.output)
    else:
        plot_hybrid_comparison(args.metrics, args.output)


if __name__ == "__main__":
    main()
