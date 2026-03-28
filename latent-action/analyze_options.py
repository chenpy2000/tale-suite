#!/usr/bin/env python3
# Analyze VQ-VAE checkpoint: option codes, usage, rewards, visualizations.
# Run after training to review codebook utilization and option semantics.

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from vqvae import VQVAE, ActionVocab

PAD = "<PAD>"


def load_model(checkpoint_path, trajectories_root, device="cpu"):
    """Load VQ-VAE from checkpoint with layer remap for older checkpoints."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    args = ckpt.get("args", {})
    traj_dir = str(Path(trajectories_root).resolve())
    model = VQVAE(
        text_model_name=args.get("text_model", "all-MiniLM-L6-v2"),
        action_vocab=ActionVocab(ckpt["action_vocab"]),
        trajectories_root=traj_dir,
        latent_dim=args.get("latent_dim", 256),
        num_codes=args.get("num_codes", 128),
        commitment_beta=args.get("commitment_beta", 0.25),
    )
    sd = ckpt["model_state_dict"]
    remap = {
        "encoder.input_proj": "encoder.proj",
        "encoder.positional_encoding.pe": "encoder.pe",
        "encoder.output_proj": "encoder.out",
        "decoder.option_proj": "decoder.opt_proj",
        "decoder.decoder": "decoder.lstm",
        "decoder.output_head": "decoder.head",
    }
    for old, new in remap.items():
        for k in list(sd.keys()):
            if k.startswith(old + "."):
                sd[new + k[len(old) :]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, args


def build_windows_with_rewards(root_dir, window_size, min_non_pad=3, max_windows=None):
    """Yield (obs_seq, action_seq, rewards, file_path) for each sliding window."""
    root = Path(root_dir)
    pad = PAD
    yielded = 0
    for p in sorted(root.rglob("episode_*.json")):
        if max_windows and yielded >= max_windows:
            break
        try:
            data = json.load(p.open(encoding="utf-8"))
            steps = data.get("steps", [])
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(steps, list):
            continue
        obs = [str(s.get("obs", "") or "") for s in steps]
        act = [str(s.get("action", "") or "") for s in steps]
        rewards = [float(s.get("reward", 0)) for s in steps]
        n = len(obs)
        if n < window_size:
            continue
        for i in range(n - window_size + 1):
            if max_windows and yielded >= max_windows:
                break
            wo, wa = obs[i : i + window_size], act[i : i + window_size]
            wr = rewards[i : i + window_size]
            if sum(1 for a in wa if a and a != pad) < min_non_pad:
                continue
            yielded += 1
            yield wo, wa, wr, str(p)


def get_option_ids(model, obs_batch, action_batch, device):
    """Run encoder + quantizer to get option_ids for a batch. No gradient."""
    obs_batch, action_batch = model._normalize_batch(obs_batch, action_batch)
    obs_emb, action_emb = model._encode_text_batch(obs_batch, action_batch)
    with torch.no_grad():
        z = model.encoder(obs_emb, action_emb)
        _, option_ids, _ = model.quantizer(z)
    return option_ids.cpu().numpy().flatten()


def run_analysis(model, trajectories_dir, window_size, device, batch_size=32, max_windows=None):
    """Collect all windows, map to option codes, aggregate by code."""
    windows = list(build_windows_with_rewards(trajectories_dir, window_size, max_windows=max_windows))
    if not windows:
        return {}, []

    code_data = defaultdict(lambda: {"actions": [], "obs_snippets": [], "rewards": []})
    all_option_ids = []

    for b in range(0, len(windows), batch_size):
        batch = windows[b : b + batch_size]
        obs_batch = [x[0] for x in batch]
        action_batch = [x[1] for x in batch]
        rewards_batch = [x[2] for x in batch]

        ids = get_option_ids(model, obs_batch, action_batch, device)
        for i, code_id in enumerate(ids):
            code_id = int(code_id)
            all_option_ids.append(code_id)
            wo, wa, wr = obs_batch[i], action_batch[i], rewards_batch[i]
            code_data[code_id]["actions"].append(wa)
            code_data[code_id]["obs_snippets"].append(wo[-1][:200] if wo else "")  # last obs, truncated
            code_data[code_id]["rewards"].extend(wr)

    return dict(code_data), all_option_ids


def truncate(s, max_len=80):
    s = (s or "").strip().replace("\n", " ")
    return (s[:max_len] + "...") if len(s) > max_len else s


def write_report(code_data, all_option_ids, num_codes, report_path):
    """Write option_analysis.txt report."""
    usage = Counter(all_option_ids)
    active = len([c for c in range(num_codes) if usage[c] > 0])
    pct = 100 * active / num_codes if num_codes else 0
    target_met = "YES" if pct >= 90 else "NO (want >90%)"

    lines = [
        "=" * 60,
        "VQ-VAE Option Analysis Report",
        "=" * 60,
        f"Total windows analyzed: {len(all_option_ids)}",
        f"Codebook: {active}/{num_codes} codes active ({pct:.1f}%)",
        f"Target >90% utilization: {target_met}",
        "",
    ]

    for code_id in sorted(code_data.keys()):
        d = code_data[code_id]
        n = len(d["actions"])
        avg_r = sum(d["rewards"]) / len(d["rewards"]) if d["rewards"] else 0
        action_seqs = d["actions"]
        action_counter = Counter()
        for seq in action_seqs:
            for a in seq:
                if a and a != PAD:
                    action_counter[a] += 1
        top_actions = action_counter.most_common(5)
        top_action_seqs = [tuple(s) for s in action_seqs[:3]]

        lines.extend([
            f"\n--- Option {code_id} (n={n}, avg_reward={avg_r:.3f}) ---",
            "Top actions: " + ", ".join(f"{a}({c})" for a, c in top_actions),
            "Sample action sequences:",
        ])
        for seq in top_action_seqs:
            clean = [a for a in seq if a and a != PAD]
            lines.append("  " + " -> ".join(clean[:5]) + (" ..." if len(clean) > 5 else ""))
        if d["obs_snippets"]:
            sample_obs = truncate(d["obs_snippets"][0], 120)
            lines.append(f"Sample game state: {sample_obs}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {report_path}")


def plot_tsne(embeddings, num_codes, out_path):
    """t-SNE plot of codebook embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn/matplotlib not installed; skipping t-SNE plot")
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_codes - 1))
    xy = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(xy[:, 0], xy[:, 1], alpha=0.7)
    for i in range(num_codes):
        plt.annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=8, alpha=0.8)
    plt.title("t-SNE of Option Embeddings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"t-SNE plot saved to {out_path}")


def plot_histogram(usage_counts, num_codes, out_path):
    """Histogram of option usage frequency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping histogram")
        return
    counts = [usage_counts.get(i, 0) for i in range(num_codes)]
    plt.figure(figsize=(12, 4))
    plt.bar(range(num_codes), counts, color="steelblue", edgecolor="navy", alpha=0.8)
    plt.xlabel("Option code")
    plt.ylabel("Usage count")
    plt.title("Option Usage Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Histogram saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Analyze VQ-VAE checkpoint options")
    p.add_argument("--checkpoint", default="checkpoints/vqvae_checkpoint.pt")
    p.add_argument("--trajectories-dir", default=None)
    p.add_argument("--output-dir", default=".", help="Directory for report and plots (default: . -> option_analysis.txt)")
    p.add_argument("--report-name", default="option_analysis.txt")
    p.add_argument("--max-windows", type=int, default=None, help="Limit windows for faster analysis")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    traj_dir = args.trajectories_dir or str(root / "data" / "trajectories_cleaned")
    if not Path(traj_dir).exists():
        traj_dir = str(root / "data" / "trajectories")
    if not Path(traj_dir).exists():
        raise SystemExit(f"Trajectories not found: {traj_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_args = load_model(args.checkpoint, traj_dir, device)
    model.to(device)
    window_size = ckpt_args.get("window_size", 5)
    num_codes = model.quantizer.num_codes

    print("Running analysis (encoding windows; may take a few minutes)...")
    code_data, all_option_ids = run_analysis(
        model, traj_dir, window_size, device, max_windows=args.max_windows
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / args.report_name

    write_report(code_data, all_option_ids, num_codes, report_path)

    usage = Counter(all_option_ids)
    embeddings = model.quantizer.codebook.weight.detach().cpu().numpy()
    plot_tsne(embeddings, num_codes, str(out_dir / "tsne_options.png"))
    plot_histogram(usage, num_codes, str(out_dir / "option_usage_histogram.png"))

    active = sum(1 for c in range(num_codes) if usage[c] > 0)
    print(f"Done. {active}/{num_codes} codes active ({100*active/num_codes:.1f}%)")


if __name__ == "__main__":
    main()
