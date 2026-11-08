"""Group 10's NRMS-inspired hybrid model, reimplemented layer for layer.

Source: 6_NRMS_Models/NRMS_(2_Heads,_Dropout=0.1)_(Test_2).ipynb at commit 3007ee3 of
github.com/bt422210/BT4222-Group-10. Module names, sizes, defaults, pooling and masking
follow that notebook. Two mechanical differences, neither of which changes the arithmetic:
news inputs are gathered from on-device tensors by index instead of being stacked per
sample in Python, and the user encoder receives already-encoded history vectors.

Training follows their loop (Adam, lr 5e-4, batch 32, 10 epochs, cross-entropy over one
click and four sampled non-clicks) with two procedural differences, so the baseline is treated
like our own models: seeds are fixed, and the epoch with the best validation score is restored
before anything is scored for test.
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nrms_inputs import USER_CAT_COLS


# ---------------------------------------------------------------------------
# Encoders (their sections 2 and 3)
# ---------------------------------------------------------------------------
class TextNewsEncoder(nn.Module):
    def __init__(self, input_dim, num_subclusters, model_dim=128, num_heads=2, dropout=0.1):
        super().__init__()
        self.subcluster_embeddings = nn.Embedding(num_subclusters, 32)
        self.subcluster_proj = nn.Linear(32, model_dim)
        self.proj = nn.Linear(input_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, title_abs_emb, subcluster_ids):
        x = torch.stack([self.proj(title_abs_emb), self.subcluster_proj(self.subcluster_embeddings(subcluster_ids))], dim=1)
        attn, _ = self.attention(x, x, x)
        return self.ffn(attn.mean(dim=1))


class EntityEncoder(nn.Module):
    def __init__(self, kg_input_dim=100, ner_input_dim=384, model_dim=128, num_heads=2, dropout=0.1):
        super().__init__()
        self.kg_proj = nn.Linear(kg_input_dim, model_dim)
        self.ner_proj = nn.Linear(ner_input_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, kg_title, kg_abstract, ner_title, ner_abstract, flags):
        x = torch.stack([self.kg_proj(kg_title), self.kg_proj(kg_abstract),
                         self.ner_proj(ner_title), self.ner_proj(ner_abstract)], dim=1)   # [B, 4, D]
        pad = flags == 0
        all_masked = pad.all(dim=1)
        safe = pad.clone()
        safe[all_masked] = False
        attn, _ = self.attention(x, x, x, key_padding_mask=safe)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (attn * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        pooled = pooled.clone()
        pooled[all_masked] = 0.0
        return self.ffn(pooled)


class NewsEncoder(nn.Module):
    def __init__(self, text_input_dim, kg_input_dim, ner_input_dim, num_subclusters, model_dim=128, num_heads=2, dropout=0.1):
        super().__init__()
        self.text_encoder = TextNewsEncoder(text_input_dim, num_subclusters, model_dim, num_heads, dropout)
        self.entity_encoder = EntityEncoder(kg_input_dim, ner_input_dim, model_dim, num_heads, dropout)
        self.fusion_layer = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, title_abs, subcluster_ids, kg_title, kg_abstract, ner_title, ner_abstract, flags):
        text_vec = self.text_encoder(title_abs, subcluster_ids)
        entity_vec = self.entity_encoder(kg_title, kg_abstract, ner_title, ner_abstract, flags)
        return self.fusion_layer(torch.cat([text_vec, entity_vec], dim=-1))


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=128, num_head=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask):
        all_padded = key_padding_mask.all(dim=1)
        safe = key_padding_mask.clone()
        safe[all_padded] = False
        attn, _ = self.attn(x, x, x, key_padding_mask=safe)
        x = self.norm1(x + self.dropout1(attn))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        if all_padded.any():
            x = x.clone()
            x[all_padded] = 0.0
        return x


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, mask):
        scores = torch.tanh(self.proj(x)) @ self.query
        all_padded = mask.all(dim=1)
        safe = mask.clone()
        safe[all_padded] = False
        weights = torch.softmax(scores.masked_fill(safe, -1e9), dim=1).masked_fill(mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        out = (weights.unsqueeze(-1) * x).sum(dim=1)
        if all_padded.any():
            out = out.clone()
            out[all_padded] = 0.0
        return out


class UserEncoder(nn.Module):
    """Self-attention then additive attention over the last 10 history vectors; learnable default for empty histories."""

    def __init__(self, hidden_dim=128, num_head=2, dropout=0.1):
        super().__init__()
        self.attention_block = SelfAttentionBlock(hidden_dim, num_head, dropout)
        self.additive_attention = AdditiveAttention(hidden_dim)
        self.default_user_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, news_embeddings, mask):
        user = self.additive_attention(self.attention_block(news_embeddings, mask), mask)
        empty = mask.all(dim=1)
        if empty.any():
            user = user.clone()
            user[empty] = self.default_user_embedding.expand(int(empty.sum()), -1)
        return user


class NewsMLP(nn.Module):
    """Scores one (user, candidate) pair from user/news representations, user features, topics and context."""

    def __init__(self, user_numerical_dense_dim, user_categorical_vocab_sizes, topic_dim=50, context_dim=4,
                 user_repr_dim=128, news_repr_dim=128, hidden_dims=(128, 64, 32), embed_dim=128):
        super().__init__()
        self.user_repr_proj = nn.Linear(user_repr_dim, embed_dim)
        self.news_repr_proj = nn.Linear(news_repr_dim, embed_dim)
        self.user_numerical_dense_proj = nn.Linear(user_numerical_dense_dim, embed_dim)
        self.user_categorical_embeddings = nn.ModuleDict(
            {col: nn.Embedding(size + 1, embed_dim) for col, size in user_categorical_vocab_sizes.items()})
        self.topic_proj = nn.Linear(topic_dim, embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        total = embed_dim * (5 + len(user_categorical_vocab_sizes))
        self.fc1 = nn.Linear(total, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.output = nn.Linear(hidden_dims[2], 1)
        self.relu = nn.ReLU()

    def forward(self, user_repr, news_repr, user_numerical_dense, user_categorical, topic_embed, context_features):
        parts = [self.relu(self.user_repr_proj(user_repr)), self.relu(self.news_repr_proj(news_repr)),
                 self.relu(self.user_numerical_dense_proj(user_numerical_dense))]
        parts += [self.user_categorical_embeddings[col](user_categorical[:, j]) for j, col in enumerate(USER_CAT_COLS)]
        parts += [self.relu(self.topic_proj(topic_embed)), self.relu(self.context_proj(context_features))]
        x = self.relu(self.fc1(torch.cat(parts, dim=1)))
        x = self.relu(self.fc3(self.relu(self.fc2(x))))
        return self.output(x).squeeze(1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
_NEWS_KEYS = ("title_abs", "subcluster_id", "kg_title", "kg_abstract", "ner_title", "ner_abstract", "flags", "topics")


class NRMSG10(nn.Module):
    def __init__(self, inputs: dict, num_heads=2, dropout=0.1, user_heads=2, user_dropout=0.1):
        super().__init__()
        news = inputs["news"]
        for k in _NEWS_KEYS:
            t = torch.tensor(np.asarray(news[k]))
            self.register_buffer(k, t.long() if k == "subcluster_id" else t.float(), persistent=False)
        imp = inputs["impressions"]
        self.register_buffer("hist_idx", torch.as_tensor(imp["hist_idx"]).long(), persistent=False)
        self.register_buffer("hist_mask", torch.as_tensor(imp["hist_mask"]).bool(), persistent=False)
        self.register_buffer("user_num", torch.as_tensor(imp["user_num"]).float(), persistent=False)
        self.register_buffer("user_cat", torch.as_tensor(imp["user_cat"]).long(), persistent=False)
        self.register_buffer("ctx", torch.as_tensor(imp["ctx"]).float(), persistent=False)

        self.news_encoder = NewsEncoder(news["title_abs"].shape[1], news["kg_title"].shape[1], news["ner_title"].shape[1],
                                        inputs["num_subclusters"], 128, num_heads, dropout)
        self.user_encoder = UserEncoder(128, user_heads, user_dropout)
        self.mlp = NewsMLP(imp["user_num"].shape[1], inputs["user_vocab_sizes"], topic_dim=news["topics"].shape[1])

    def encode_news(self, idx):
        flat = idx.reshape(-1)
        out = self.news_encoder(self.title_abs[flat], self.subcluster_id[flat], self.kg_title[flat], self.kg_abstract[flat],
                                self.ner_title[flat], self.ner_abstract[flat], self.flags[flat])
        return out.reshape(*idx.shape, -1)

    def encode_users(self, imp, news_vecs=None):
        hist = self.hist_idx[imp]
        h = self.encode_news(hist) if news_vecs is None else news_vecs[hist]
        return self.user_encoder(h, self.hist_mask[imp])

    def score_pairs(self, imp, user_vec, cand, cand_vec, recency):
        """imp [R], user_vec [R, D], cand [R], cand_vec [R, D], recency [R] -> scores [R]."""
        ctx = torch.cat([self.ctx[imp], recency.unsqueeze(1)], dim=1)
        return self.mlp(user_vec, cand_vec, self.user_num[imp], self.user_cat[imp], self.topics[cand], ctx)

    def forward(self, imp, cand, recency):
        """Training path: imp [B], cand [B, C], recency [B, C] -> scores [B, C]."""
        b, c = cand.shape
        user = self.encode_users(imp)
        cand_vec = self.encode_news(cand)
        imp_r = imp.repeat_interleave(c)
        s = self.score_pairs(imp_r, user.repeat_interleave(c, dim=0), cand.reshape(-1), cand_vec.reshape(b * c, -1),
                             recency.reshape(-1))
        return s.reshape(b, c)


# ---------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_split(model, split: dict, news_chunk=8192, imp_chunk=4096, row_chunk=262144) -> np.ndarray:
    """Score every candidate row of a split in eval mode. Returns scores aligned with split['row_id']."""
    model.eval()
    dev = model.title_abs.device
    n_news = model.title_abs.shape[0]
    news_vecs = torch.cat([model.encode_news(torch.arange(s, min(s + news_chunk, n_news), device=dev))
                           for s in range(0, n_news, news_chunk)])
    imps = np.unique(split["imp"])
    imps_t = torch.as_tensor(imps, device=dev)
    user_vecs = torch.cat([model.encode_users(imps_t[s:s + imp_chunk], news_vecs) for s in range(0, len(imps), imp_chunk)])
    u_row = torch.as_tensor(np.searchsorted(imps, split["imp"]), device=dev)
    imp_t = torch.tensor(np.asarray(split["imp"]), device=dev)
    news_t = torch.tensor(np.asarray(split["news"]), device=dev)
    rec_t = torch.tensor(np.asarray(split["recency"]), device=dev).float()
    out = []
    for s in range(0, len(imp_t), row_chunk):
        sl = slice(s, s + row_chunk)
        out.append(model.score_pairs(imp_t[sl], user_vecs[u_row[sl]], news_t[sl], news_vecs[news_t[sl]], rec_t[sl]).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train(model, samples: dict, *, epochs=10, lr=5e-4, batch_size=32, seed=4222, val_fn=None,
          select_metric="ndcg@5", log=print):
    """Train with Group 10's loop; after each epoch call val_fn(model) -> dict of validation metrics.

    Returns (history DataFrame, best_epoch). The best epoch's weights are loaded into `model`.
    """
    dev = model.title_abs.device
    imp = torch.as_tensor(samples["imp"], device=dev)
    cand = torch.as_tensor(samples["cand"], device=dev)
    target = torch.as_tensor(samples["target"], device=dev)
    rec = torch.as_tensor(samples["recency"], device=dev).float()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, best, best_state = [], -np.inf, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b in torch.randperm(len(imp), generator=gen).split(batch_size):
            b = b.to(dev)
            loss = F.cross_entropy(model(imp[b], cand[b], rec[b]), target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        row = {"epoch": epoch, "train_loss": total / n, "train_seconds": round(time.time() - t0, 1)}
        if val_fn is not None:
            row.update({f"val_{k}": v for k, v in val_fn(model).items()})
            if row[f"val_{select_metric}"] > best:
                best, best_state = row[f"val_{select_metric}"], copy.deepcopy(model.state_dict())
        hist.append(row)
        log(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    history = pd.DataFrame(hist).set_index("epoch")
    best_epoch = int(history[f"val_{select_metric}"].idxmax()) if val_fn is not None else epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch
