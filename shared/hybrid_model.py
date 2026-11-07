"""Main deep model: Group 10's NRMS-inspired network extended with engineered features and ID masking.

What is kept from Group 10 (shared/nrms_g10.py, unchanged): the news encoder (attention over
MiniLM title+abstract and KG/NER entity vectors), the user encoder (self-attention plus additive
attention over the last 10 clicks), and every input of their scoring MLP.

What is added, all fused late into the scoring MLP (Topic 6, "NMF with features"):

* a feature tower over the engineered row features (T1 own behaviour, T2 content-transferred
  behaviour, T3 explicit content, U user-article matching; optionally H content_smoothed_ctr);
* candidate-aware attention over the user's last 10 clicks (the candidate is the attention query),
  so each candidate sees the part of the user's history relevant to it (Topic 7B, attention);
* a GMF term, the element-wise product of that candidate-specific user vector and the candidate
  vector (as in NeuMF);
* an article ID embedding with ID masking (Topic 6, cold start): during training the ID vector
  of each candidate is replaced by a shared [UNKNOWN] vector with probability `id_mask_p`, so the
  network cannot rely on ID look-ups alone. At scoring time any article that never appeared in a
  training sample gets [UNKNOWN], which is exactly what a new article looks like.

Training follows Group 10's loop (one click and four sampled non-clicks from the same impression,
softmax cross-entropy, Adam) with fixed seeds and the best validation epoch restored.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import derived as D
from . import feature_groups as FG
from .nrms_g10 import NRMSG10, NewsMLP
from .nrms_inputs import USER_CAT_COLS

LOG_COLS = {"article_age_hours", "prior_exposures", "prior_clicks", "recent_exposures_6h", "recent_clicks_6h",
            "entity_count", "history_length"}
CAT_COLS = ["category_code", "subcategory_code"]
CAT_DIMS = {"category_code": 8, "subcategory_code": 16}

# Each variant is retrained from scratch. U here is Andrew's matching columns only: the attention
# user encoder (history) stays in every variant, so no variant is unpersonalised.
VARIANTS = {
    "full":                {"groups": ["T1", "T2", "T3", "U"], "id_mask_p": 0.3},
    "no_candidate_attention": {"groups": ["T1", "T2", "T3", "U"], "id_mask_p": 0.3, "candidate_attention": False},
    "no_id_masking":       {"groups": ["T1", "T2", "T3", "U"], "id_mask_p": 0.0},
    "no_t1":               {"groups": ["T2", "T3", "U"], "id_mask_p": 0.3},
    "no_t2":               {"groups": ["T1", "T3", "U"], "id_mask_p": 0.3},
    "no_t3":               {"groups": ["T1", "T2", "U"], "id_mask_p": 0.3},
    "no_u":                {"groups": ["T1", "T2", "T3"], "id_mask_p": 0.3},
    "full_plus_smoothing": {"groups": ["T1", "T2", "T3", "U", "H"], "id_mask_p": 0.3},
}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def variant_columns(groups, include_semantic=True) -> tuple[list[str], list[str]]:
    """(numeric columns, categorical columns) for a list of feature groups."""
    g = FG.group_columns(include_user_pc=False, include_semantic=include_semantic)
    num, cat = [], []
    for name in groups:
        cols = g[name] if name != "T3" else FG.T3_SCALAR   # PCs are not needed: the news encoder reads the vectors
        for c in cols:
            (cat if c in CAT_COLS else num).append(c)
    return num, cat


def raw_columns(num_cols, cat_cols) -> list[str]:
    """Columns to read from features_{split}.parquet (content_smoothed_ctr is derived, not stored)."""
    need = [c for c in num_cols if c != "content_smoothed_ctr"] + list(cat_cols)
    if "content_smoothed_ctr" in num_cols:
        need += [c for c in ("prior_clicks", "prior_exposures", "related_ctr") if c not in need]
    return need


class FeatureScaler:
    """log1p on count-like columns, then standardise with training means and deviations, clipped to ±5."""

    def __init__(self, num_cols, cat_cols, alpha=None):
        self.num_cols, self.cat_cols, self.alpha = list(num_cols), list(cat_cols), alpha

    def _num(self, df):
        out = np.empty((len(df), len(self.num_cols)), np.float32)
        for j, c in enumerate(self.num_cols):
            v = D.content_smoothed_ctr(df, self.alpha) if c == "content_smoothed_ctr" else df[c].to_numpy(np.float64)
            out[:, j] = np.log1p(np.clip(v, 0, None)) if c in LOG_COLS else v
        return out

    def fit(self, train_df):
        x = self._num(train_df)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def transform(self, df) -> tuple[np.ndarray, np.ndarray]:
        num = np.clip((self._num(df) - self.mean) / self.std, -5, 5).astype(np.float32)
        cat = df[self.cat_cols].to_numpy(np.int64) if self.cat_cols else np.zeros((len(df), 0), np.int64)
        return num, cat


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HybridScorer(NewsMLP):
    """Group 10's scoring MLP with extra late-fused inputs (GMF, feature tower, ID vector)."""

    def __init__(self, user_numerical_dense_dim, user_categorical_vocab_sizes, topic_dim, extra_dim,
                 hidden_dims=(128, 64, 32), embed_dim=128):
        super().__init__(user_numerical_dense_dim, user_categorical_vocab_sizes, topic_dim=topic_dim,
                         hidden_dims=hidden_dims, embed_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim * (5 + len(user_categorical_vocab_sizes)) + extra_dim, hidden_dims[0])

    def forward(self, user_repr, news_repr, user_numerical_dense, user_categorical, topic_embed, context_features, extra):
        parts = [self.relu(self.user_repr_proj(user_repr)), self.relu(self.news_repr_proj(news_repr)),
                 self.relu(self.user_numerical_dense_proj(user_numerical_dense))]
        parts += [self.user_categorical_embeddings[col](user_categorical[:, j]) for j, col in enumerate(USER_CAT_COLS)]
        parts += [self.relu(self.topic_proj(topic_embed)), self.relu(self.context_proj(context_features)), extra]
        x = self.relu(self.fc1(torch.cat(parts, dim=1)))
        x = self.relu(self.fc3(self.relu(self.fc2(x))))
        return self.output(x).squeeze(1)


class CandidateAttention(nn.Module):
    """Candidate-aware additive attention over the user's history (the candidate is the query).

    score_j = v · tanh(W_h h_j + W_c c). Group 10's user attention uses one fixed query for every
    candidate, initialised from a standard normal, and in our run it collapsed onto a single history
    item. Here the query depends on the candidate and v starts near zero, so weights start uniform.
    """

    def __init__(self, dim=128, hidden=64):
        super().__init__()
        self.wh = nn.Linear(dim, hidden, bias=False)
        self.wc = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.v.weight, std=0.01)

    def forward(self, h, mask, c):
        """h [R, L, D] history states, mask [R, L] (True = padding), c [R, D] candidate -> ([R, D], weights [R, L])."""
        s = self.v(torch.tanh(self.wh(h) + self.wc(c).unsqueeze(1))).squeeze(-1)
        safe = mask.clone()
        safe[mask.all(dim=1)] = False
        w = torch.softmax(s.masked_fill(safe, -1e9), dim=1).masked_fill(mask, 0.0)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)   # empty history -> all zeros -> zero vector
        return (w.unsqueeze(-1) * h).sum(dim=1), w


class HybridNRMS(NRMSG10):
    def __init__(self, inputs: dict, n_num: int, cat_sizes: list[tuple[int, int]], *, id_mask_p=0.3, id_dim=32,
                 feat_hidden=128, feat_dim=64, feat_dropout=0.1, candidate_attention=True, **g10_kwargs):
        super().__init__(inputs, **g10_kwargs)
        n_news = self.title_abs.shape[0]
        self.unk, self.id_mask_p = n_news, float(id_mask_p)
        self.id_emb = nn.Embedding(n_news + 1, id_dim)          # last row = [UNKNOWN]
        nn.init.normal_(self.id_emb.weight, std=0.01)
        with torch.no_grad():
            self.id_emb.weight[n_news].zero_()
        self.register_buffer("seen", torch.zeros(n_news, dtype=torch.bool))   # saved with the weights
        self.gmf = nn.Linear(128, 128)
        self.cat_embs = nn.ModuleList([nn.Embedding(size, dim) for size, dim in cat_sizes])
        f_in = n_num + sum(dim for _, dim in cat_sizes)
        self.feat_net = nn.Sequential(nn.Linear(f_in, feat_hidden), nn.ReLU(), nn.Dropout(feat_dropout),
                                      nn.Linear(feat_hidden, feat_dim), nn.ReLU())
        self.mlp = HybridScorer(self.user_num.shape[1], inputs["user_vocab_sizes"], self.topics.shape[1],
                                extra_dim=128 + feat_dim + id_dim + (128 if candidate_attention else 0))
        # created last, so a model without it is built from exactly the same random draws as before
        self.cand_att = CandidateAttention(128, 64) if candidate_attention else None
        self.cand_proj = nn.Linear(128, 128) if candidate_attention else None

    def user_states(self, imp, news_vecs=None):
        """Group 10's user vector [R, D] plus the self-attended history states [R, L, D] and padding mask [R, L]."""
        hist, mask = self.hist_idx[imp], self.hist_mask[imp]
        h = self.encode_news(hist) if news_vecs is None else news_vecs[hist]
        enc = self.user_encoder
        x = enc.attention_block(h, mask)
        user = enc.additive_attention(x, mask)
        empty = mask.all(dim=1)
        if empty.any():
            user = user.clone()
            user[empty] = enc.default_user_embedding.expand(int(empty.sum()), -1)
        return user, x, mask

    def mark_seen(self, cand_idx):
        self.seen[torch.as_tensor(np.unique(np.asarray(cand_idx)), device=self.seen.device)] = True

    def id_vectors(self, cand):
        idx = torch.where(self.seen[cand], cand, torch.full_like(cand, self.unk))
        if self.training and self.id_mask_p > 0:
            drop = torch.rand(cand.shape, device=cand.device) < self.id_mask_p
            idx = torch.where(drop, torch.full_like(idx, self.unk), idx)
        return self.id_emb(idx)

    def feature_vectors(self, num, cat):
        parts = [num] + [emb(cat[:, j]) for j, emb in enumerate(self.cat_embs)]
        return self.feat_net(torch.cat(parts, dim=1))

    def score_pairs(self, imp, user_vec, cand, cand_vec, recency, num, cat, hstates=None, hmask=None):
        """All inputs are per row: imp [R], user_vec [R, D], cand [R], cand_vec [R, D], recency [R], num [R, F],
        cat [R, K]; hstates [R, L, D] and hmask [R, L] are needed when candidate attention is on."""
        ctx = torch.cat([self.ctx[imp], recency.unsqueeze(1)], dim=1)
        if self.cand_att is None:
            match = [F.relu(self.gmf(user_vec * cand_vec))]
        else:
            user_c, _ = self.cand_att(hstates, hmask, cand_vec)          # the user, as seen by this candidate
            match = [F.relu(self.gmf(user_c * cand_vec)), F.relu(self.cand_proj(user_c))]
        extra = torch.cat(match + [self.feature_vectors(num, cat), self.id_vectors(cand)], dim=1)
        return self.mlp(user_vec, cand_vec, self.user_num[imp], self.user_cat[imp], self.topics[cand], ctx, extra)

    def forward(self, imp, cand, recency, num, cat):
        """Training path: imp [B], cand [B, C], recency [B, C], num [B, C, F], cat [B, C, K] -> scores [B, C]."""
        b, c = cand.shape
        user, x, mask = self.user_states(imp)
        cand_vec = self.encode_news(cand)
        s = self.score_pairs(imp.repeat_interleave(c), user.repeat_interleave(c, dim=0), cand.reshape(-1),
                             cand_vec.reshape(b * c, -1), recency.reshape(-1), num.reshape(b * c, -1),
                             cat.reshape(b * c, -1), x.repeat_interleave(c, dim=0), mask.repeat_interleave(c, dim=0))
        return s.reshape(b, c)


# ---------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------
def train(model, samples: dict, train_num: np.ndarray, train_cat: np.ndarray, *, epochs=10, lr=5e-4, batch_size=32,
          seed=4222, val_fn=None, select_metric="ndcg@10", patience=None, log=print):
    """Group 10's loop with features. train_num/train_cat are the scaled feature rows of the whole train split;
    samples["row"] [N, 5] points into them. Returns (history DataFrame, best_epoch); best weights are loaded."""
    dev = model.title_abs.device
    imp = torch.as_tensor(samples["imp"], device=dev)
    cand = torch.as_tensor(samples["cand"], device=dev)
    target = torch.as_tensor(samples["target"], device=dev)
    rec = torch.as_tensor(samples["recency"], device=dev).float()
    rows = np.asarray(samples["row"])
    num = torch.as_tensor(train_num[rows], device=dev)           # [N, 5, F], gathered once
    cat = torch.as_tensor(train_cat[rows], device=dev)           # [N, 5, K]
    model.mark_seen(samples["cand"])
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, best, best_state, since_best = [], -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b in torch.randperm(len(imp), generator=gen).split(batch_size):
            b = b.to(dev)
            loss = F.cross_entropy(model(imp[b], cand[b], rec[b], num[b], cat[b]), target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        row = {"epoch": epoch, "train_loss": total / n, "train_seconds": round(time.time() - t0, 1)}
        if val_fn is not None:
            row.update({f"val_{k}": v for k, v in val_fn(model).items()})
            if row[f"val_{select_metric}"] > best:
                best, best_state, since_best = row[f"val_{select_metric}"], copy.deepcopy(model.state_dict()), 0
            else:
                since_best += 1
        hist.append(row)
        log(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if patience is not None and since_best >= patience:
            log(f"stopping: no validation gain for {patience} epochs")
            break
    history = pd.DataFrame(hist).set_index("epoch")
    best_epoch = int(history[f"val_{select_metric}"].idxmax()) if val_fn is not None else len(history)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch


@torch.no_grad()
def encode_all(model, split: dict, news_chunk=8192, imp_chunk=4096):
    """News vectors for the catalogue; user vectors, history states and masks for the split's impressions."""
    model.eval()
    dev = model.title_abs.device
    n_news = model.title_abs.shape[0]
    news_vecs = torch.cat([model.encode_news(torch.arange(s, min(s + news_chunk, n_news), device=dev))
                           for s in range(0, n_news, news_chunk)])
    imps = np.unique(split["imp"])
    imps_t = torch.as_tensor(imps, device=dev)
    parts = [model.user_states(imps_t[s:s + imp_chunk], news_vecs) for s in range(0, len(imps), imp_chunk)]
    user_vecs, states, masks = (torch.cat([p[k] for p in parts]) for k in range(3))
    return news_vecs, imps, user_vecs, states, masks


@torch.no_grad()
def score_split(model, split: dict, num: np.ndarray, cat: np.ndarray, row_chunk=32768) -> np.ndarray:
    """Score every candidate row of a split in eval mode; num/cat are the split's scaled feature rows."""
    dev = model.title_abs.device
    news_vecs, imps, user_vecs, states, masks = encode_all(model, split)
    u_row = np.searchsorted(imps, split["imp"])
    out = []
    for s in range(0, len(split["imp"]), row_chunk):
        sl = slice(s, s + row_chunk)
        u = torch.as_tensor(u_row[sl], device=dev)
        news = torch.tensor(np.asarray(split["news"][sl]), device=dev)
        out.append(model.score_pairs(torch.tensor(np.asarray(split["imp"][sl]), device=dev), user_vecs[u], news,
                                     news_vecs[news], torch.as_tensor(np.asarray(split["recency"][sl]), device=dev).float(),
                                     torch.tensor(num[sl], device=dev), torch.tensor(cat[sl], device=dev),
                                     states[u], masks[u]).cpu())
    return torch.cat(out).numpy().astype(np.float64)


@torch.no_grad()
def candidate_attention(model, imps, cands, chunk=4096) -> np.ndarray:
    """Candidate-aware attention weights [n, 10] on each impression's last 10 clicks, with `cands` as the query."""
    model.eval()
    dev = model.title_abs.device
    out = []
    for s in range(0, len(imps), chunk):
        i = torch.as_tensor(np.asarray(imps[s:s + chunk]), device=dev)
        c = torch.as_tensor(np.asarray(cands[s:s + chunk]), device=dev)
        _, x, mask = model.user_states(i)
        out.append(model.cand_att(x, mask, model.encode_news(c))[1].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def history_attention(model, imps) -> tuple[np.ndarray, np.ndarray]:
    """Additive-attention weight on each of the last 10 history clicks, for interpretation.

    Returns (weights [n, 10], history news indices [n, 10]); padded slots have weight 0.
    """
    model.eval()
    dev = model.title_abs.device
    imps = torch.as_tensor(np.asarray(imps), device=dev)
    hist, mask = model.hist_idx[imps], model.hist_mask[imps]
    x = model.user_encoder.attention_block(model.encode_news(hist), mask)
    att = model.user_encoder.additive_attention
    scores = torch.tanh(att.proj(x)) @ att.query
    safe = mask.clone()
    safe[mask.all(dim=1)] = False
    w = torch.softmax(scores.masked_fill(safe, -1e9), dim=1).masked_fill(mask, 0.0)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return w.cpu().numpy(), hist.cpu().numpy()
