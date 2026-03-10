#!/usr/bin/env python3
# Train VQ-VAE on (obs, action) windows from trajectories_cleaned.
# Sliding windows over episodes; reconstruct next action; codebook reset for dead codes.

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

from vqvae import VQVAE, normalize_text, PAD


class TrajectoryWindowDataset(Dataset):
    """Sliding windows of (obs, action) from episode_*.json. Can pad short episodes, optionally augment by dropping a random step."""
    def __init__(self, root_dir, window_size=5, pad_short=True, min_non_pad=3, augment=False, aug_p=0.3):
        self.root = Path(root_dir)
        self.w = window_size
        self.pad_short = pad_short
        self.min_np = min_non_pad
        self.augment = augment
        self.aug_p = aug_p
        self.examples = []
        for p in sorted(self.root.rglob("episode_*.json")):
            try:
                steps = json.load(p.open(encoding="utf-8")).get("steps", [])
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(steps, list):
                continue
            obs = [normalize_text(s.get("obs", "") or "") for s in steps]
            act = []
            for s in steps:
                a = str(s.get("action", "") or "")
                act.append(PAD if a.strip() == PAD else normalize_text(a))
            n = len(obs)
            if n == 0:
                continue
            if n < self.w:
                if not self.pad_short or sum(1 for a in act if a and a != PAD) < self.min_np:
                    continue
                # Left-pad to match inference (short history pads on left)
                self.examples.append(([""] * (self.w - n) + obs, [PAD] * (self.w - n) + act))
            else:
                for i in range(n - self.w + 1):
                    wo, wa = obs[i:i + self.w], act[i:i + self.w]
                    if sum(1 for a in wa if a and a != PAD) >= self.min_np:
                        self.examples.append((wo, wa))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        obs, act = list(self.examples[i][0]), list(self.examples[i][1])
        if self.augment and random.random() < self.aug_p and len(obs) > 3:
            j = random.randint(0, len(obs) - 1)
            obs.pop(j)
            act.pop(j)
            # Left-pad to match inference
            obs = [""] * (self.w - len(obs)) + obs
            act = [PAD] * (self.w - len(act)) + act
        return obs, act


_PASSIVE_PREFIXES = ("examine ", "look", "inventory")


def _is_passive(action):
    a = action.strip().lower()
    return any(a.startswith(p) or a == p for p in _PASSIVE_PREFIXES)


class BalancedTrajectoryDataset(Dataset):
    """Sliding windows, only from episodes >= window_size. Skips windows with too many pads.
    With reward_weight=True, assigns higher sampling weight to windows near positive rewards."""
    def __init__(self, root_dir, window_size=10, min_non_pad=7, reward_weight=False):
        self.root = Path(root_dir)
        self.w = window_size
        self.min_np = min_non_pad
        self.examples = []
        self.weights = []
        for p in sorted(self.root.rglob("episode_*.json")):
            try:
                data = json.load(p.open(encoding="utf-8"))
                steps = data.get("steps", [])
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(steps, list) or len(steps) < self.w:
                continue

            # Per-step rewards for weighting
            step_rewards = [float(s.get("reward", 0)) for s in steps]
            ep_total_reward = data.get("total_reward", sum(step_rewards))

            for i in range(len(steps) - self.w + 1):
                w = steps[i:i + self.w]
                obs = [normalize_text(s.get("obs", "") or "") for s in w]
                act = []
                for s in w:
                    a = str(s.get("action", "") or "")
                    act.append(PAD if a.strip() == PAD else normalize_text(a))
                if sum(1 for a in act if not a.strip() or a == PAD) > self.w - self.min_np:
                    continue

                self.examples.append((obs, act))

                if reward_weight:
                    # Weight: base 1.0, boosted if window contains positive reward or is from high-reward episode
                    window_reward = sum(step_rewards[i:i + self.w])
                    # Passive-heavy windows get downweighted
                    passive_frac = sum(1 for a in act if _is_passive(a)) / len(act)
                    passive_penalty = max(0.2, 1.0 - passive_frac)
                    w_val = (1.0 + max(0, window_reward) + max(0, ep_total_reward) * 0.01) * passive_penalty
                    self.weights.append(w_val)
                else:
                    self.weights.append(1.0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return list(self.examples[i][0]), list(self.examples[i][1])


def collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch]


def collate_tensors(batch):
    """Collate for precomputed embeddings: stack tensors."""
    obs_emb = torch.stack([x[0] for x in batch])
    act_emb = torch.stack([x[1] for x in batch])
    target_ids = torch.stack([x[2] for x in batch])
    return obs_emb, act_emb, target_ids


class PrecomputedEmbeddingDataset(Dataset):
    """Dataset of precomputed obs/action embeddings. Avoids encoding at training time."""
    def __init__(self, base_dataset, text_encoder, action_vocab, device, encode_batch_size=512):
        self.base = base_dataset
        self.device = device
        obs_list = [base_dataset[i][0] for i in range(len(base_dataset))]
        act_list = [base_dataset[i][1] for i in range(len(base_dataset))]
        all_obs = [" ".join((t or "").strip().split()) for seq in obs_list for t in seq]
        all_act = [" ".join((t or "").strip().split()) for seq in act_list for t in seq]
        n, w = len(obs_list), len(obs_list[0])
        with torch.no_grad():
            obs_emb, act_emb = [], []
            for b in tqdm(range(0, n * w, encode_batch_size), desc="Precomputing obs", leave=False):
                batch_obs = all_obs[b : b + encode_batch_size]
                if batch_obs:
                    e = text_encoder.model.encode(batch_obs, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=False)
                    obs_emb.append(e)
            obs_emb = torch.cat(obs_emb, dim=0).view(n, w, -1)
            for b in tqdm(range(0, n * w, encode_batch_size), desc="Precomputing act", leave=False):
                batch_act = all_act[b : b + encode_batch_size]
                if batch_act:
                    e = text_encoder.model.encode(batch_act, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=False)
                    act_emb.append(e)
            act_emb = torch.cat(act_emb, dim=0).view(n, w, -1)
        target_ids = torch.tensor([[action_vocab.encode(a) for a in seq] for seq in act_list], dtype=torch.long)
        self.obs_emb = obs_emb.cpu()
        self.act_emb = act_emb.cpu()
        self.target_ids = target_ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.obs_emb[i], self.act_emb[i], self.target_ids[i]


class CodebookReset:
    """Every freq steps, reset dead/rare codes by copying from active codes + small noise."""
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
    """Load trajectories, train VQ-VAE, save checkpoint with action_vocab and args."""
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories-dir", default=None)
    p.add_argument("--window-size", type=int, default=5, choices=[5, 10, 15])
    p.add_argument("--batch-size", type=int, default=128, help="Larger batches = faster on GPU")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", "--lr", type=float, default=0.0001, dest="learning_rate")
    p.add_argument("--num-codes", type=int, default=64, choices=[32, 64, 128, 256])
    p.add_argument("--commitment-beta", type=float, default=1.0)
    p.add_argument("--commitment-start", type=float, default=0.5, help="Start value when using --commitment-schedule")
    p.add_argument("--commitment-end", type=float, default=2.5, help="End value when using --commitment-schedule")
    p.add_argument("--text-model", default="all-MiniLM-L6-v2")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--checkpoint-path", default="checkpoints/vqvae_checkpoint.pt")
    p.add_argument("--resume", default=None, help="Resume from checkpoint (e.g. checkpoints/vqvae_checkpoint.pt)")
    p.add_argument("--seed", type=int, default=20241001)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--commitment-schedule", action="store_true")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--reward-weight", action="store_true",
                   help="Weight training windows by nearby reward (upweight goal actions, downweight passive)")
    p.add_argument("--precompute-embeddings", action="store_true",
                   help="Encode all text once before training; much faster for multi-epoch runs")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0=main process only)")
    p.add_argument("--device", default=None, help="Device (cuda/cpu). Default: cuda if available")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cudnn.benchmark = True  # faster conv/RNN kernels
    else:
        print("Using CPU", flush=True)

    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    traj_dir = args.trajectories_dir or str(root / "data" / "trajectories_cleaned")
    if not Path(traj_dir).exists():
        traj_dir = str(root / "data" / "trajectories")

    if args.balanced:
        ds = BalancedTrajectoryDataset(traj_dir, args.window_size, min(args.window_size - 3, args.window_size),
                                       reward_weight=args.reward_weight)
    else:
        ds = TrajectoryWindowDataset(traj_dir, args.window_size, pad_short=True, min_non_pad=3, augment=args.augment)
    if len(ds) == 0:
        raise SystemExit(
            f"No trajectories (window_size={args.window_size}). "
            "Run collect_data first. With --balanced, episodes need >= window_size steps."
        )

    print(f"Dataset: {len(ds)} windows from {traj_dir}")
    print("Loading VQ-VAE model (SentenceTransformer may download on first run)...", flush=True)
    model = VQVAE(text_model_name=args.text_model, trajectories_root=traj_dir,
                  latent_dim=args.latent_dim, num_codes=args.num_codes, commitment_beta=args.commitment_beta,
                  device=device)

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        if resume_path.exists():
            resume_ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
            start_epoch = resume_ckpt.get("epoch", 0) + 1
            print(f"Resumed from {resume_path} (starting epoch {start_epoch})", flush=True)
        else:
            print(f"Resume path not found: {resume_path}, starting from scratch", flush=True)

    if args.precompute_embeddings:
        print("Precomputing embeddings (one-time cost)...", flush=True)
        ds = PrecomputedEmbeddingDataset(ds, model.text_encoder, model.action_vocab, device)
        collate_fn = collate_tensors
    else:
        collate_fn = collate

    pin_memory = device == "cuda"
    nw = 0 if args.precompute_embeddings else args.num_workers  # precompute: data in RAM, workers duplicate memory

    sampler = None
    do_shuffle = True
    if args.reward_weight and args.balanced and hasattr(ds, 'weights') and ds.weights:
        weights_tensor = torch.tensor(ds.weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights_tensor, num_samples=len(ds), replacement=True)
        do_shuffle = False
        print(f"Using reward-weighted sampling (weight range: {min(ds.weights):.2f} - {max(ds.weights):.2f})", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=do_shuffle if sampler is None else False,
        sampler=sampler, num_workers=nw, collate_fn=collate_fn,
        pin_memory=pin_memory, persistent_workers=nw > 0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if resume_ckpt and "optimizer_state_dict" in resume_ckpt:
        opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("Restored optimizer state", flush=True)
    n_batches = len(loader)
    print(f"Training: {n_batches} batches/epoch, {args.epochs} epochs (batch_size={args.batch_size}, num_workers={nw})", flush=True)

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / Path(args.checkpoint_path).name
    resetter = CodebookReset(model, 10, 500)
    # Ramp commitment beta from commitment_start to commitment_end over first 10 epochs
    if args.commitment_schedule:
        def sched(e):
            t = min(1, e / 10)
            return args.commitment_start + (args.commitment_end - args.commitment_start) * t
    else:
        sched = None

    for epoch in range(start_epoch, args.epochs + 1):
        if sched:
            model.commitment_beta = sched(epoch - 1)
        model.train()
        model.quantizer.reset_usage_stats()
        total_loss = 0.0
        n = 0
        for bi, batch in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            if args.precompute_embeddings:
                obs_emb, act_emb, target_ids = batch
                obs_emb = obs_emb.to(device, non_blocking=True)
                act_emb = act_emb.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)
                _, option_ids, loss, m = model.forward_from_embeddings(obs_emb, act_emb, target_ids)
            else:
                obs_batch, action_batch = batch
                _, option_ids, loss, m = model(obs_batch, action_batch)
            resetter.update(option_ids)
            loss.backward()
            opt.step()
            total_loss += float(m["loss/total"])
            n += 1
            if (bi + 1) % 10 == 0 or bi == 0:
                print(f"  ep{epoch} batch {bi+1}/{n_batches} loss={m['loss/total']:.4f}", flush=True)
        u = model.quantizer.usage_stats()
        print(f"ep{epoch} done loss={total_loss/max(1,n):.4f} codes={u['used_codes']}/{model.quantizer.num_codes} ppl={u['perplexity']:.2f}", flush=True)

        # Save after every epoch so we can resume if interrupted
        torch.save({
                    "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "args": vars(args),
            "action_vocab": model.action_vocab.stoi,
        }, ckpt_path)
        print(f"  saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
