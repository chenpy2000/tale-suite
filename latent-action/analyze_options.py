#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from vqvae import VQVAE, ActionVocab


def build_windows(traj_dir, w, max_n=None):
    out = []
    for path in sorted(Path(traj_dir).rglob("episode_*.json")):
        try:
            d = json.load(path.open(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        steps = d.get("steps", [])
        if not steps:
            continue
        pad = ActionVocab.PAD_TOKEN
        if len(steps) < w:
            obs = [str(s.get("obs", "")) for s in steps] + [""] * (w - len(steps))
            act = [str(s.get("action", pad)) for s in steps] + [pad] * (w - len(steps))
            out.append({"obs_seq": obs, "action_seq": act, "reward_sum": sum(float(s.get("reward", 0)) for s in steps),
                       "state_summary": (obs[0] or "")[:160], "game": d.get("game", path.parent.name)})
        else:
            for i in range(len(steps) - w + 1):
                chunk = steps[i:i + w]
                obs = [str(s.get("obs", "")) for s in chunk]
                act = [str(s.get("action", pad)) for s in chunk]
                out.append({"obs_seq": obs, "action_seq": act, "reward_sum": sum(float(s.get("reward", 0)) for s in chunk),
                           "state_summary": (obs[0] or "")[:160], "game": d.get("game", path.parent.name)})
                if max_n and len(out) >= max_n:
                    return out
        if max_n and len(out) >= max_n:
            return out
    return out


def main():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    p.add_argument("--checkpoint-path", default=str(root / "checkpoints" / "vqvae_checkpoint.pt"))
    p.add_argument("--trajectories-dir", default=str(root / "data" / "trajectories_cleaned"))
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--analysis-dir", default=str(root / "analysis"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    a = ckpt.get("args", {})
    model = VQVAE(text_model_name=a.get("text_model", "all-MiniLM-L6-v2"), action_vocab=ActionVocab(ckpt["action_vocab"]),
                  trajectories_root=args.trajectories_dir, latent_dim=a.get("latent_dim", 256),
                  num_codes=a.get("num_codes", 128), commitment_beta=a.get("commitment_beta", 0.25))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    w = args.window_size or int(a.get("window_size", 5))
    windows = build_windows(args.trajectories_dir, w, args.max_windows)
    if not windows:
        raise SystemExit("No windows")

    # encode all windows
    opt_actions = defaultdict(Counter)
    opt_counts = defaultdict(int)
    opt_reward = defaultdict(float)
    model.quantizer.reset_usage_stats()

    for i in range(0, len(windows), args.batch_size):
        batch = windows[i:i + args.batch_size]
        obs_b = [x["obs_seq"] for x in batch]
        act_b = [x["action_seq"] for x in batch]
        obs_n, act_n = model._normalize_batch(obs_b, act_b)
        obs_emb, act_emb = model._encode_text_batch(obs_n, act_n)
        _, ids, _ = model.quantizer(model.encoder(obs_emb, act_emb))
        for x, c in zip(batch, ids.cpu().tolist()):
            opt_actions[c][tuple(x["action_seq"])] += 1
            opt_counts[c] += 1
            opt_reward[c] += x["reward_sum"]

    u = model.quantizer.usage_stats()
    n_active = (u["counts"] > 0).sum().item()
    out_dir = Path(args.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # report
    lines = [f"ckpt={args.checkpoint_path}", f"windows={len(windows)} w={w}",
             f"codes={n_active}/{len(u['counts'])} ppl={u['perplexity']:.2f}", ""]
    for c in sorted(opt_counts.keys(), key=lambda x: -opt_counts[x]):
        avg_r = opt_reward[c] / max(1, opt_counts[c])
        lines.append(f"opt{c}: n={opt_counts[c]} avg_r={avg_r:.3f}")
        for seq, n in opt_actions[c].most_common(3):
            lines.append(f"  {n}x: {'|'.join(seq)}")
        lines.append("")

    report = out_dir / "option_analysis.txt"
    report.write_text("\n".join(lines))
    print(f"report: {report}")

    # plots
    try:
        import matplotlib.pyplot as plt
        counts = u["counts"].cpu().numpy()
        plt.figure(figsize=(10, 3))
        plt.bar(range(len(counts)), counts)
        plt.savefig(out_dir / "option_usage_histogram.png", dpi=120)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
