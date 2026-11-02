"""Inputs for the corrected reimplementation of Group 10's NRMS-inspired hybrid model.

Group 10 (github.com/bt422210/BT4222-Group-10, commit 3007ee3) built these inputs
across 1_data_preprocess, 1.1_news_embeddings, 1.2_entity_embeddings and
3_feature_engineering. This module rebuilds the same inputs with the same
recipes, and changes only what was a bug or a split-boundary artefact:

* every input is joined by impression_id / news index, never by row position;
* recency uses the article's first exposure anywhere in the log before t
  (Group 10: first exposure among sampled training rows only);
* habit_score uses the user's previous impression anywhere in the log before t
  (Group 10: previous impression within the same split).

Content-only steps (category merging, sub-clustering, TF-IDF topics) are fitted
on the whole news catalogue as Group 10 did; they use no labels or behaviour.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import schema as S

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
HIST_LEN = 10
N_NEG = 4
USER_NUM_COLS = ["history_length", "diversity_ratio", "concentration_ratio"]
USER_CAT_COLS = ["recent_fav_cat", "top_cat_1", "top_cat_2", "top_cat_3", "diversity_cluster", "interest_cluster"]
COLD_START = "cold_start"


# ---------------------------------------------------------------------------
# Raw MIND files
# ---------------------------------------------------------------------------
def _entity_ids(raw) -> list[str]:
    """Unique WikidataIds in order of appearance, as Group 10's extract_entity_ids."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    out, seen = [], set()
    for ent in parsed if isinstance(parsed, list) else []:
        wid = str(ent.get("WikidataId", "")).strip() if isinstance(ent, dict) else ""
        if wid and wid not in seen:
            seen.add(wid)
            out.append(wid)
    return out


def load_raw(raw_dir: str):
    """Return (news, behaviors, entity_lookup) from a MIND-format folder."""
    news = pd.read_csv(os.path.join(raw_dir, "news.tsv"), sep="\t", header=None, names=NEWS_COLS)
    news["kg_title_ids"] = news["title_entities"].apply(_entity_ids)
    news["kg_abstract_ids"] = news["abstract_entities"].apply(_entity_ids)
    beh = pd.read_csv(os.path.join(raw_dir, "behaviors.tsv"), sep="\t", header=None, names=BEHAVIOR_COLS,
                      usecols=["impression_id", "user_id", "time", "history"])
    beh["time"] = pd.to_datetime(beh["time"], format=S.MIND_TIME_FORMAT)
    beh["history"] = beh["history"].apply(lambda h: h.split() if isinstance(h, str) else [])
    lookup = {}
    with open(os.path.join(raw_dir, "entity_embedding.vec"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lookup[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return news, beh, lookup


# ---------------------------------------------------------------------------
# Text and entity embeddings (1.1_news_embeddings, 1.2_entity_embeddings)
# ---------------------------------------------------------------------------
def _l2(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0).astype(np.float32)


@lru_cache(maxsize=1)
def _minilm(device):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def encode_texts(texts, method: str = "minilm", device=None) -> np.ndarray:
    """L2-normalised 384-d sentence vectors.

    minilm    : all-MiniLM-L6-v2, the encoder Group 10 used for titles, abstracts and NER phrases.
    tfidf_svd : offline stand-in used only for synthetic test runs (no model download).
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    if method == "minilm":
        vecs = _minilm(device).encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
    elif method == "tfidf_svd":
        x = TfidfVectorizer().fit_transform(texts)
        k = min(384, x.shape[1] - 1, x.shape[0] - 1)
        vecs = np.zeros((len(texts), 384), np.float32)
        vecs[:, :k] = TruncatedSVD(k, random_state=0).fit_transform(x)
    else:
        raise ValueError(method)
    return _l2(np.asarray(vecs, np.float32))


def title_abstract_embeddings(news, method="minilm", device=None) -> np.ndarray:
    """[N, 768]: title vector and abstract vector (title when abstract missing), concatenated and renormalised."""
    t = encode_texts(news["title"].fillna("").tolist(), method, device)
    a = encode_texts(news["abstract"].fillna(news["title"]).tolist(), method, device)
    return _l2(np.hstack([t, a]))


def _clean_phrases(phrases):
    out, seen = [], set()
    for p in phrases:
        p = str(p).strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def ner_phrases(texts, method: str = "spacy") -> list[list[str]]:
    """spaCy en_core_web_sm entity strings per text, de-duplicated case-insensitively."""
    if method == "none":
        return [[] for _ in texts]
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    docs = nlp.pipe((t.strip() if isinstance(t, str) else "" for t in texts), batch_size=512)
    return [_clean_phrases([e.text for e in d.ents]) for d in docs]


def entity_branches(news, lookup, ner_method="spacy", text_method="minilm", device=None) -> dict:
    """Four entity tokens per article (KG title/abstract 100-d, NER title/abstract 384-d) and presence flags."""
    out = {}
    for field in ("title", "abstract"):
        ids = news[f"kg_{field}_ids"]
        kg = np.zeros((len(news), S.ENTITY_DIM), np.float32)
        flag = np.zeros(len(news), np.float32)
        for i, l in enumerate(ids):
            vecs = [lookup[e] for e in l if e in lookup]
            if vecs:
                kg[i] = np.mean(vecs, axis=0)
                flag[i] = 1.0
        out[f"kg_{field}"] = _l2(kg)
        out[f"kg_{field}_flag"] = flag

        phrases = ner_phrases(news[field].tolist(), ner_method)
        uniq = sorted({p for l in phrases for p in l})
        ner = np.zeros((len(news), 384), np.float32)
        if uniq:
            pv = encode_texts(uniq, text_method, device)
            p_idx = {p: i for i, p in enumerate(uniq)}
            for i, l in enumerate(phrases):
                if l:
                    ner[i] = pv[[p_idx[p] for p in l]].mean(0)
        out[f"ner_{field}"] = _l2(ner)
        out[f"ner_{field}_flag"] = np.array([1.0 if l else 0.0 for l in phrases], np.float32)
    return out


# ---------------------------------------------------------------------------
# Categories (1_data_preprocess sections 2.2-2.3) and topics (3_feature_engineering 2.3)
# ---------------------------------------------------------------------------
def g10_categories(news, text_emb, subcat_threshold=None, merge_threshold=100, n_top=8, n_sub=3, seed=42):
    """Return (category, subclustered_category) as Group 10 built them.

    1. A (category, subcategory) pair keeps its own label if it has at least `subcat_threshold`
       articles (Group 10: 75th percentile of subcategory sizes = 110), else becomes "<category>_general".
    2. Labels with fewer than `merge_threshold` articles become "miscellaneous".
    3. The `n_top` largest labels are split into `n_sub` KMeans clusters on the title+abstract vectors.
    """
    if subcat_threshold is None:
        subcat_threshold = int(news.groupby("subcategory")["news_id"].count().quantile(0.75))
    pair = news.groupby(["category", "subcategory"])["news_id"].transform("count").to_numpy()
    cat = np.where(pair >= subcat_threshold, news["subcategory"], news["category"] + "_general").astype(object)
    size = pd.Series(cat).map(pd.Series(cat).value_counts()).to_numpy()
    cat = np.where(size >= merge_threshold, cat, "miscellaneous").astype(object)
    sub = cat.copy()
    for c in pd.Series(cat).value_counts().index[:n_top]:
        idx = np.flatnonzero(cat == c)
        if len(idx) >= n_sub:
            labels = KMeans(n_clusters=n_sub, random_state=seed, n_init=10).fit_predict(text_emb[idx])
            sub[idx] = [f"{c}_subcluster_{l}" for l in labels]
    return pd.Series(cat, index=news.index), pd.Series(sub, index=news.index), subcat_threshold


def latent_topics(news, n_components=50) -> np.ndarray:
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
    x = tfidf.fit_transform(news["title"].fillna(""))
    k = min(n_components, x.shape[1] - 1)
    out = np.zeros((len(news), n_components), np.float32)
    out[:, :k] = TruncatedSVD(n_components=k, random_state=42).fit_transform(x)
    return out


# ---------------------------------------------------------------------------
# User features (3_feature_engineering section 3)
# ---------------------------------------------------------------------------
def _user_features_one(hist, cat_of, sub_of):
    subs = [sub_of[h] for h in hist if h in sub_of]
    cats = [cat_of[h] for h in hist if h in cat_of]
    top = [c for c, _ in Counter(subs).most_common(3)] if subs else []
    top += [COLD_START] * (3 - len(top))
    recent = [sub_of[h] for h in hist[-10:] if h in sub_of]
    recent_fav = Counter(recent).most_common(1)[0][0] if recent else COLD_START
    if cats:
        div = len(set(cats)) / len(cats)
        conc = Counter(cats).most_common(1)[0][1] / len(cats)
    else:
        div = conc = 0.0
    return len(hist), div, conc, recent_fav, *top


def user_feature_table(impressions, cat_of, sub_of) -> pd.DataFrame:
    vals = [_user_features_one(h, cat_of, sub_of) for h in impressions["history"]]
    cols = ["history_length", "diversity_ratio", "concentration_ratio", "recent_fav_cat", "top_cat_1", "top_cat_2", "top_cat_3"]
    return pd.DataFrame(vals, columns=cols, index=impressions.index)


class UserFeatureEncoder:
    """Clusters, category codes and scaling fitted on training rows only, then applied to every split."""

    def fit(self, train_rows: pd.DataFrame, all_top_cats: pd.DataFrame):
        active = train_rows.loc[train_rows["history_length"] != 0]
        users = active.drop_duplicates("user_id")
        self.div_scaler = StandardScaler().fit(users[["diversity_ratio", "concentration_ratio"]])
        self.div_km = KMeans(n_clusters=3, random_state=0).fit(self.div_scaler.transform(users[["diversity_ratio", "concentration_ratio"]]))
        cats = all_top_cats[["top_cat_1", "top_cat_2", "top_cat_3"]].stack().dropna().unique()
        self.cat_to_idx = {c: i for i, c in enumerate(cats)}
        self.int_km = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1024).fit(self._top3(train_rows))
        full = self._add_clusters(train_rows)
        self.mappers = {c: {v: i for i, v in enumerate(pd.unique(full[c].astype(str)))} for c in USER_CAT_COLS}
        self.vocab_sizes = {c: len(m) for c, m in self.mappers.items()}
        self.num_scaler = StandardScaler().fit(self._num(full))
        return self

    def _top3(self, df):
        return np.stack([df[c].map(self.cat_to_idx).fillna(-1).to_numpy() for c in ("top_cat_1", "top_cat_2", "top_cat_3")], 1).astype(np.int16)

    def _add_clusters(self, df):
        df = df.copy()
        active = df["history_length"] != 0
        df["diversity_cluster"] = -1
        if active.any():
            df.loc[active, "diversity_cluster"] = self.div_km.predict(
                self.div_scaler.transform(df.loc[active, ["diversity_ratio", "concentration_ratio"]]))
        df["interest_cluster"] = self.int_km.predict(self._top3(df))
        return df

    @staticmethod
    def _num(df):
        x = df[USER_NUM_COLS].to_numpy(np.float32).copy()
        x[:, 0] = np.log1p(x[:, 0])
        return x

    def transform(self, df) -> tuple[np.ndarray, np.ndarray]:
        df = self._add_clusters(df)
        num = self.num_scaler.transform(self._num(df)).astype(np.float32)
        cat = np.stack([df[c].astype(str).map(self.mappers[c]).fillna(len(self.mappers[c])).to_numpy()
                        for c in USER_CAT_COLS], 1).astype(np.int64)
        return num, cat


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def habit_scores(beh: pd.DataFrame) -> np.ndarray:
    """cos(2π·Δh/24) with Δh = hours since the user's previous impression strictly earlier in the log; 0 if none."""
    distinct = beh[["user_id", "time"]].drop_duplicates().sort_values(["user_id", "time"], kind="mergesort")
    distinct["prev_time"] = distinct.groupby("user_id")["time"].shift(1)  # last strictly earlier timestamp
    b = beh[["user_id", "time"]].merge(distinct, on=["user_id", "time"], how="left")
    hours = (b["time"] - b["prev_time"]).dt.total_seconds() / 3600
    return np.cos(2 * np.pi * hours / 24).fillna(0.0).to_numpy(np.float32)


def build_inputs(raw_dir: str, meta: dict, text_method="minilm", ner_method="spacy", device=None,
                 cache_dir: str | None = None, cat_kwargs: dict | None = None, seed: int = 4222) -> dict:
    """Everything the model needs, as aligned numpy arrays.

    meta: {split: DataFrame with row_id, impression_id, news_id, time, label} from meta_{split}.parquet.
    """
    news, beh, lookup = load_raw(raw_dir)
    nid2idx = pd.Series(np.arange(len(news)), index=news["news_id"])

    def cached(name, fn):
        path = os.path.join(cache_dir, f"{name}.npz") if cache_dir else None
        if path and os.path.exists(path):
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        val = fn()
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, **val)
        return val

    text = cached(f"title_abstract_{text_method}", lambda: {"v": title_abstract_embeddings(news, text_method, device)})["v"]
    ent = cached(f"entities_{ner_method}_{text_method}", lambda: entity_branches(news, lookup, ner_method, text_method, device))
    cat, sub, threshold = g10_categories(news, text, **(cat_kwargs or {}))
    sub_codes, sub_uniques = pd.factorize(sub)
    news_arrays = {
        "title_abs": text,
        "subcluster_id": sub_codes.astype(np.int64),
        "kg_title": ent["kg_title"], "kg_abstract": ent["kg_abstract"],
        "ner_title": ent["ner_title"], "ner_abstract": ent["ner_abstract"],
        "flags": np.stack([ent["kg_title_flag"], ent["kg_abstract_flag"], ent["ner_title_flag"], ent["ner_abstract_flag"]], 1),
        "topics": latent_topics(news),
    }
    cat_of = dict(zip(news["news_id"], cat))
    sub_of = dict(zip(news["news_id"], sub))

    # impression-level inputs for every impression in the log
    beh = beh.sort_values("time", kind="mergesort").reset_index(drop=True)
    beh["habit_score"] = habit_scores(beh)
    hour = beh["time"].dt.hour
    beh["hour_sin"], beh["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    uf = user_feature_table(beh, cat_of, sub_of)
    beh = pd.concat([beh, uf], axis=1)
    hist_idx = np.zeros((len(beh), HIST_LEN), np.int64)
    hist_mask = np.ones((len(beh), HIST_LEN), bool)   # True = padding, as Group 10
    for i, h in enumerate(beh["history"]):
        valid = [nid2idx[x] for x in h if x in nid2idx.index][-HIST_LEN:]
        if valid:
            hist_idx[i, HIST_LEN - len(valid):] = valid
            hist_mask[i, HIST_LEN - len(valid):] = False
    imp_pos = pd.Series(np.arange(len(beh)), index=beh["impression_id"])

    # recency per candidate row: first exposure anywhere in the log strictly before t
    all_rows = pd.concat([m[["impression_id", "news_id", "time"]] for m in meta.values()], ignore_index=True)
    first_seen = all_rows.groupby("news_id")["time"].min()

    def recency(m):
        fs = m["news_id"].map(first_seen)
        age = (m["time"] - fs).dt.total_seconds() / 3600
        return np.where(fs < m["time"], 1.0 / (1.0 + age), 1.0).astype(np.float32)

    # training samples: each click with 4 non-clicked candidates from the same impression
    rng = np.random.default_rng(seed)
    tr = meta["train"]
    s_imp, s_cand, s_target, s_rec = [], [], [], []
    tr_rec = recency(tr)
    for imp_id, g in tr.assign(rec=tr_rec).groupby("impression_id", sort=False):
        pos, neg = g[g["label"] == 1], g[g["label"] == 0]
        if len(neg) == 0:
            continue
        for _, p in pos.iterrows():
            pick = rng.choice(len(neg), N_NEG, replace=len(neg) < N_NEG)
            ids = [p["news_id"]] + neg["news_id"].iloc[pick].tolist()
            rec = [p["rec"]] + neg["rec"].iloc[pick].tolist()
            perm = rng.permutation(N_NEG + 1)
            s_imp.append(imp_pos[imp_id])
            s_cand.append(nid2idx[[ids[j] for j in perm]].to_numpy())
            s_rec.append(np.asarray(rec, np.float32)[perm])
            s_target.append(int(np.flatnonzero(perm == 0)[0]))
    samples = {"imp": np.asarray(s_imp), "cand": np.stack(s_cand), "target": np.asarray(s_target), "recency": np.stack(s_rec)}

    # user features: fitted on training samples (one row per click, as Group 10), applied to all impressions
    train_rows = beh.iloc[samples["imp"]].reset_index(drop=True)
    enc = UserFeatureEncoder().fit(train_rows, beh)
    user_num, user_cat = enc.transform(beh)

    splits = {}
    for split, m in meta.items():
        splits[split] = {
            "row_id": m["row_id"].to_numpy(),
            "imp": imp_pos[m["impression_id"]].to_numpy(),
            "news": nid2idx[m["news_id"]].to_numpy(),
            "label": m["label"].to_numpy(),
            "recency": recency(m),
        }
    return {
        "news": news_arrays,
        "num_subclusters": len(sub_uniques),
        "impressions": {"impression_id": beh["impression_id"].to_numpy(), "hist_idx": hist_idx, "hist_mask": hist_mask,
                        "user_num": user_num, "user_cat": user_cat,
                        "ctx": beh[["hour_sin", "hour_cos", "habit_score"]].to_numpy(np.float32)},
        "train_samples": samples,
        "splits": splits,
        "user_vocab_sizes": enc.vocab_sizes,
        "info": {"subcat_threshold": threshold, "n_categories": int(pd.Series(cat).nunique()),
                 "n_subclusters": len(sub_uniques), "n_train_samples": len(s_imp),
                 "text_method": text_method, "ner_method": ner_method},
    }
