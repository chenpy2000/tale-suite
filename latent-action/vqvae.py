# VQ-VAE for option learning from (obs, action) sequences.
# Requires: torch, sentence-transformers

import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


PAD = "<PAD>"
UNK = "<UNK>"


class ActionVocab:
    PAD_TOKEN = PAD
    UNK_TOKEN = UNK

    def __init__(self, stoi):
        self.stoi = dict(stoi)
        self.itos = {i: t for t, i in self.stoi.items()}
        assert PAD in self.stoi and UNK in self.stoi

    @classmethod
    def from_trajectories(cls, root_dir, min_freq=1):
        root = Path(root_dir) if root_dir else Path(__file__).resolve().parent / "data" / "trajectories"
        cnt = Counter()
        for p in root.rglob("episode_*.json"):
            try:
                for s in json.load(p.open(encoding="utf-8")).get("steps", []):
                    a = str(s.get("action", "")).strip()
                    if a:
                        cnt[a] += 1
            except (json.JSONDecodeError, OSError):
                pass
        tokens = [PAD, UNK] + [a for a, n in cnt.items() if n >= min_freq]
        return cls({t: i for i, t in enumerate(tokens)})

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def size(self):
        return len(self.stoi)

    def encode(self, a):
        return self.stoi.get(str(a), self.stoi[UNK])

    def decode(self, i):
        return self.itos.get(int(i), UNK)


class TextEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _enc(self, text):
        t = " ".join((text or "").strip().split())
        e = self.model.encode(t, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=False)
        return e.squeeze(0) if e.ndim > 1 else e

    def encode_observation(self, obs):
        return self._enc(obs)

    def encode_action(self, action):
        return self._enc(action)


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=128, embedding_dim=256, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.register_buffer("usage_counts", torch.zeros(num_codes, dtype=torch.long))
        self.register_buffer("total_assignments", torch.zeros(1, dtype=torch.long))

    def reset_usage_stats(self):
        self.usage_counts.zero_()
        self.total_assignments.zero_()

    def usage_stats(self):
        total = int(self.total_assignments.item())
        used = int((self.usage_counts > 0).sum().item())
        ppl = 0.0
        if total > 0:
            p = self.usage_counts.float() / total
            nz = p[p > 0]
            ppl = float(torch.exp(-(nz * torch.log(nz)).sum()).item())
        return {"used_codes": used, "perplexity": ppl, "counts": self.usage_counts.detach().cpu().clone()}

    def forward(self, z):
        shape = z.shape
        z_flat = z.reshape(-1, self.embedding_dim)
        codebook = self.codebook.weight
        d = (z_flat ** 2).sum(1, keepdim=True) + (codebook ** 2).sum(1) - 2 * z_flat @ codebook.t()
        codes = d.argmin(1)
        quantized = self.codebook(codes)
        cb_loss = F.mse_loss(quantized, z_flat.detach())
        cmt_loss = F.mse_loss(z_flat, quantized.detach())
        vq_loss = cb_loss + self.beta * cmt_loss
        out = z_flat + (quantized - z_flat).detach()
        with torch.no_grad():
            self.usage_counts.index_add_(0, codes, torch.ones_like(codes, dtype=self.usage_counts.dtype))
            self.total_assignments += codes.numel()
        return out.reshape(*shape), codes.reshape(*shape[:-1]), vq_loss


class Encoder(nn.Module):
    def __init__(self, obs_dim, action_dim, model_dim=256, output_dim=256, num_layers=4, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(obs_dim + action_dim, model_dim)
        pe = torch.zeros(512, model_dim)
        pos = torch.arange(512).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos * torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000) / model_dim)))
        pe[:, 1::2] = torch.cos(pos * torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000) / model_dim)))
        self.register_buffer("pe", pe.unsqueeze(0))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout, activation="gelu", batch_first=True, norm_first=True),
            num_layers
        )
        self.out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, output_dim))

    def forward(self, obs_emb, action_emb):
        x = torch.cat([obs_emb, action_emb], dim=-1)
        x = self.proj(x) + self.pe[:, :x.size(1)]
        x = self.transformer(x)
        return self.out(x[:, -1])


class Decoder(nn.Module):
    def __init__(self, obs_dim, quant_dim, vocab_size, hidden=256, layers=2, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.opt_proj = nn.Linear(quant_dim, hidden)
        self.lstm = nn.LSTM(hidden * 2, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, vocab_size))

    def forward(self, quantized, obs_seq):
        B, T, _ = obs_seq.shape
        oh = self.obs_proj(obs_seq)
        qh = self.opt_proj(quantized).unsqueeze(1).expand(B, T, -1)
        h, _ = self.lstm(torch.cat([oh, qh], -1))
        return self.head(h)

    def action_loss(self, logits, targets):
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.pad_id)


class VQVAE(nn.Module):
    def __init__(self, text_model_name="all-MiniLM-L6-v2", action_vocab=None, trajectories_root="data/trajectories",
                 latent_dim=256, num_codes=128, commitment_beta=1.0, device=None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.commitment_beta = float(commitment_beta)
        self.text_encoder = TextEncoder(text_model_name, str(self.device))
        self.text_dim = self.text_encoder.dim
        self.action_vocab = action_vocab or ActionVocab.from_trajectories(trajectories_root)
        self.encoder = Encoder(self.text_dim, self.text_dim, latent_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_codes, latent_dim)
        self.decoder = Decoder(self.text_dim, latent_dim, self.action_vocab.size, pad_id=self.action_vocab.pad_id)
        self.to(self.device)

    def _normalize_batch(self, obs_seq, action_seq):
        if isinstance(obs_seq[0], str):
            obs_seq, action_seq = [list(obs_seq)], [list(action_seq)]
        else:
            obs_seq = [list(s) for s in obs_seq]
            action_seq = [list(s) for s in action_seq]
        max_len = max(max(len(s) for s in obs_seq), max(len(s) for s in action_seq))
        for i in range(len(obs_seq)):
            obs_seq[i] = obs_seq[i] + [""] * (max_len - len(obs_seq[i]))
            action_seq[i] = action_seq[i] + [PAD] * (max_len - len(action_seq[i]))
        return obs_seq, action_seq

    def _encode_text_batch(self, obs_batch, action_batch):
        obs_embs = [torch.stack([self.text_encoder.encode_observation(t) for t in seq]) for seq in obs_batch]
        act_embs = [torch.stack([self.text_encoder.encode_action(t) for t in seq]) for seq in action_batch]
        return torch.stack(obs_embs).to(self.device), torch.stack(act_embs).to(self.device)

    def forward(self, obs_sequence, action_sequence):
        obs_batch, action_batch = self._normalize_batch(obs_sequence, action_sequence)
        obs_emb, action_emb = self._encode_text_batch(obs_batch, action_batch)
        target_ids = torch.tensor([[self.action_vocab.encode(a) for a in seq] for seq in action_batch], dtype=torch.long, device=self.device)

        z = self.encoder(obs_emb, action_emb)
        quantized, option_ids, vq_loss = self.quantizer(z)
        logits = self.decoder(quantized, obs_emb)
        recon_loss = self.decoder.action_loss(logits, target_ids)
        commitment_loss = self.commitment_beta * F.mse_loss(z, quantized.detach())
        total_loss = recon_loss + vq_loss + commitment_loss

        u = self.quantizer.usage_stats()
        metrics = {"loss/total": float(total_loss.item()), "loss/reconstruction": float(recon_loss.item()),
                   "loss/vq": float(vq_loss.item()), "loss/commitment": float(commitment_loss.item())}
        return logits, option_ids, total_loss, metrics
