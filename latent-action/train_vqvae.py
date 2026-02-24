#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from vqvae import VQVAE


class TrajectoryWindowDataset(Dataset):
    def __init__(self, root_dir, window_size=5, pad_short=True, min_non_pad=3, augment=False, aug_p=0.3):
        self.root = Path(root_dir)
        self.w = window_size
        self.pad_short = pad_short
        self.min_np = min_non_pad
        self.augment = augment
        self.aug_p = aug_p
        self.pad = "<PAD>"
        self.examples = []
        for p in sorted(self.root.rglob("episode_*.json")):
            try:
                steps = json.load(p.open(encoding="utf-8")).get("steps", [])
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(steps, list):
                continue
            obs = [str(s.get("obs", "") or "") for s in steps]
            act = [str(s.get("action", "") or "") for s in steps]
            n = len(obs)
            if n == 0:
                continue
            if n < self.w:
                if not self.pad_short or sum(1 for a in act if a and a != self.pad) < self.min_np:
                    continue
                self.examples.append((obs + [self.pad] * (self.w - n), act + [self.pad] * (self.w - n)))
            else:
                for i in range(n - self.w + 1):
                    wo, wa = obs[i:i + self.w], act[i:i + self.w]
                    if sum(1 for a in wa if a and a != self.pad) >= self.min_np:
                        self.examples.append((wo, wa))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        obs, act = list(self.examples[i][0]), list(self.examples[i][1])
        if self.augment and random.random() < self.aug_p and len(obs) > 3:
            j = random.randint(0, len(obs) - 1)
            obs.pop(j)
            act.pop(j)
            obs += [self.pad] * (self.w - len(obs))
            act += [self.pad] * (self.w - len(act))
        return obs, act


class BalancedTrajectoryDataset(Dataset):
    def __init__(self, root_dir, window_size=10, min_non_pad=7):
        self.root = Path(root_dir)
        self.w = window_size
        self.min_np = min_non_pad
        self.pad = "<PAD>"
        self.examples = []
        for p in sorted(self.root.rglob("episode_*.json")):
            try:
                steps = json.load(p.open(encoding="utf-8")).get("steps", [])
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(steps, list) or len(steps) < self.w:
                continue
            for i in range(len(steps) - self.w + 1):
                w = steps[i:i + self.w]
                obs = [str(s.get("obs", "") or "") for s in w]
                act = [str(s.get("action", "") or "") for s in w]
                if sum(1 for a in act if not a.strip() or a == self.pad) <= self.w - self.min_np:
                    self.examples.append((obs, act))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return list(self.examples[i][0]), list(self.examples[i][1])


def collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch]


class CodebookReset:
    def __init__(self, model, thresh=10, freq=500):
        self.model, self.thresh, self.freq = model, thresh, freq
        self.usage, self.steps = defaultdict(int), 0

    def update(self, ids):
        for c in ids.cpu().tolist():
            self.usage[int(c)] += 1
        self.steps += 1
        if self.steps % self.freq == 0:
            self._reset()

    def _reset(self):
        q = self.model.quantizer
        dead = set(range(q.num_codes)) - set(self.usage.keys())
        rare = {c for c, n in self.usage.items() if n < self.thresh}
        to_reset = dead | rare
        active = [c for c, n in self.usage.items() if n >= self.thresh]
        if not to_reset or not active:
            self.usage.clear()
            return
        with torch.no_grad():
            for c in to_reset:
                src = random.choice(active)
                q.codebook.weight[c] = q.codebook.weight[src].clone() + torch.randn_like(q.codebook.weight[src]) * 0.01
        self.usage.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories-dir", default=None)
    p.add_argument("--window-size", type=int, default=5, choices=[5, 10, 15])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-codes", type=int, default=64, choices=[32, 64, 128, 256])
    p.add_argument("--commitment-beta", type=float, default=1.0)
    p.add_argument("--text-model", default="all-MiniLM-L6-v2")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--checkpoint-path", default="checkpoints/vqvae_checkpoint.pt")
    p.add_argument("--seed", type=int, default=20241001)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--commitment-schedule", action="store_true")
    p.add_argument("--balanced", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    traj_dir = args.trajectories_dir or str(root / "data" / "trajectories_cleaned")
    if not Path(traj_dir).exists():
        traj_dir = str(root / "data" / "trajectories")

    if args.balanced:
        ds = BalancedTrajectoryDataset(traj_dir, args.window_size, min(args.window_size - 3, args.window_size))
    else:
        ds = TrajectoryWindowDataset(traj_dir, args.window_size, pad_short=True, min_non_pad=3, augment=args.augment)
    if len(ds) == 0:
        raise SystemExit("No trajectories. Run collect_data first.")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    model = VQVAE(text_model_name=args.text_model, trajectories_root=traj_dir,
                  latent_dim=args.latent_dim, num_codes=args.num_codes, commitment_beta=args.commitment_beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / Path(args.checkpoint_path).name
    resetter = CodebookReset(model, 10, 500)
    sched = (lambda e: 0.25 + 1.75 * min(1, e / 10)) if args.commitment_schedule else None

    for epoch in range(1, args.epochs + 1):
        if sched:
            model.commitment_beta = sched(epoch - 1)
        model.train()
        model.quantizer.reset_usage_stats()
        total_loss = 0.0
        n = 0
        for obs_batch, action_batch in loader:
            opt.zero_grad(set_to_none=True)
            _, option_ids, loss, m = model(obs_batch, action_batch)
            resetter.update(option_ids)
            loss.backward()
            opt.step()
            total_loss += m["loss/total"]
            n += 1
        u = model.quantizer.usage_stats()
        print(f"ep{epoch} loss={total_loss/max(1,n):.4f} codes={u['used_codes']}/{model.quantizer.num_codes} ppl={u['perplexity']:.2f}")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "args": vars(args),
                "action_vocab": model.action_vocab.stoi}, ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
