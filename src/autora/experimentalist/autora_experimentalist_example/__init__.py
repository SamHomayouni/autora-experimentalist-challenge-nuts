from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Union, List

def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    num_samples: int = 1,
    random_state: Union[int, None] = None,
    *,
    # --- Discrete repooling options (manual) ---
    discrete_repool: bool = False,                 # manually force repooling
    discrete_cols: Union[List[str], None] = None,  # columns that define discrete buckets (e.g., ["n"])
    balance_alpha: float = 1.0,                    # inverse-frequency exponent
    balance_uniform_mix: float = 0.15,             # uniform floor (0..1)
    balance_gain: float = 0.35,                    # weight of rebalance bonus in Stage B (0..1)
    # --- Auto toggle for small discrete spaces ---
    auto_repool: bool = True,                      # detect small discrete spaces automatically
    space_thresh: int = 20,                        # max unique combos to consider "small"
    col_unique_thresh: int = 15,                   # per-column absolute uniqueness cutoff
    col_ratio_thresh: float = 0.2,                 # or <=20% unique/rows => column treated as discrete
) -> pd.DataFrame:
    """
    Annealed hybrid sampler with memory-safe scoring:
    - Stage A: Random vs. Randomizer (annealed ε with coverage guard)
    - Stage B: Falsification / Uncertainty / Novelty chosen via softmax of signals
    - Diversity: farthest-first on a shortlist
    - NEW: cap candidate/reference sizes to avoid O(N*M) memory blowups; novelty via NearestNeighbors.
    - NEW: discrete repooling with inverse-frequency weighting (Stage A & Stage B bonus).
    - NEW: auto-detect small discrete IV spaces and enable repooling automatically.
    """
    # ----------------- Tunables for scale -----------------
    REF_CAP  = 5000
    CAND_CAP = 10000
    SHORTLIST_MULT = 3
    SHORTLIST_MIN = 30

    # ----------------- Helpers -----------------
    def _safe_df(df_like, like_cols=None) -> pd.DataFrame:
        if isinstance(df_like, pd.DataFrame):
            return df_like
        return pd.DataFrame(df_like, columns=(list(like_cols) if like_cols is not None else None))

    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size == 0:
            return v
        mn, mx = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
            out = np.zeros_like(v)
        else:
            out = (v - mn) / (mx - mn)
        out[~np.isfinite(out)] = 0.0
        return out

    def _normalize01(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mn, mx = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
            return np.zeros_like(x)
        out = (x - mn) / (mx - mn)
        out[~np.isfinite(out)] = 0.0
        return out

    def _predict_probs(model, X: pd.DataFrame) -> Union[np.ndarray, None]:
        if hasattr(model, "predict_proba"):
            try:
                p = model.predict_proba(X)
                return np.clip(np.atleast_2d(p), 1e-9, 1 - 1e-9)
            except Exception:
                return None
        return None

    def _predict_regr(model, X: pd.DataFrame) -> Union[np.ndarray, None]:
        if hasattr(model, "predict"):
            try:
                y = np.asarray(model.predict(X)).reshape(-1, 1)
                return y
            except Exception:
                return None
        return None

    def _uncertainty_scores(cands: pd.DataFrame) -> np.ndarray:
        if len(models) == 0:
            return np.zeros(len(cands))
        # Classification path
        probs = []
        for m in models:
            p = _predict_probs(m, cands)
            if p is not None:
                probs.append(p)
        if probs:
            p_best = probs[0]
            lc = 1.0 - np.max(p_best, axis=1)
            if len(probs) > 1:
                P = np.stack(probs, axis=0)               # (M,N,C)
                var_ens = np.mean(np.var(P, axis=0),  axis=1)
                lc = 0.7 * _normalize(lc) + 0.3 * _normalize(var_ens)
            return _normalize(lc)
        # Regression path (ensemble variance if >=2 models)
        preds = []
        for m in models:
            y = _predict_regr(m, cands)
            if y is not None:
                preds.append(y)
        if len(preds) >= 2:
            Y = np.stack(preds, axis=0)                   # (M,N,1)
            var = np.var(Y.squeeze(-1), axis=0)
            return _normalize(var)
        return np.zeros(len(cands))

    def _falsification_scores(cands: pd.DataFrame) -> np.ndarray:
        # Use model disagreement as falsification proxy (no giant surrogate)
        if len(models) == 0:
            return np.zeros(len(cands))
        probs = []
        for m in models:
            p = _predict_probs(m, cands)
            if p is not None:
                probs.append(p)
        if len(probs) >= 2:
            P = np.stack(probs, axis=0)                   # (M,N,C)
            var_across = np.mean(np.var(P, axis=0), axis=1)
            return _normalize(var_across)
        # Regression ensemble variance
        preds = []
        for m in models:
            y = _predict_regr(m, cands)
            if y is not None:
                preds.append(y)
        if len(preds) >= 2:
            Y = np.stack(preds, axis=0)                   # (M,N,1)
            var = np.var(Y.squeeze(-1), axis=0)
            return _normalize(var)
        # Fallback to uncertainty
        return _uncertainty_scores(cands)

    def _novelty_scores(cands: pd.DataFrame, refX: pd.DataFrame) -> np.ndarray:
        """Distance to nearest explored/reference point using NN (memory-safe)."""
        if refX is None or len(refX) == 0:
            return np.ones(len(cands))
        from sklearn.neighbors import NearestNeighbors
        both = pd.concat([cands, refX], axis=0)
        rng = both.max(axis=0) - both.min(axis=0)
        rng[rng < 1e-12] = 1.0
        A = (cands / rng).to_numpy(dtype=float)
        B = (refX  / rng).to_numpy(dtype=float)
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(B)
        dists, _ = nn.kneighbors(A, n_neighbors=1, return_distance=True)
        return _normalize(dists.reshape(-1))

    def _farthest_first(df: pd.DataFrame, k: int) -> pd.Index:
        if len(df) <= k:
            return df.index
        X = df.to_numpy(dtype=float)
        sel = [0]
        d2 = np.sum((X - X[0])**2, axis=1)
        for _ in range(1, k):
            nxt = int(np.argmax(d2))
            sel.append(nxt)
            d2 = np.minimum(d2, np.sum((X - X[nxt])**2, axis=1))
        return df.iloc[sel].index

    # ---- Discrete-space auto-detection helpers ----
    def _bucket_keys(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        if not cols:
            return np.array(["_all_"] * len(df))
        sub = df[cols].astype(object).where(pd.notnull(df[cols]), None)
        return sub.apply(lambda r: tuple(r.values.tolist()), axis=1).to_numpy()

    def _inverse_freq_weights(
        cands: pd.DataFrame,
        refX: pd.DataFrame,
        cols: List[str],
        alpha: float = 1.0,
        uniform_mix: float = 0.15,
    ) -> np.ndarray:
        """
        Weight ~ (count + 1)^(-alpha), normalized and mixed with uniform floor.
        If no refs, returns uniform.
        """
        n = len(cands)
        if n == 0:
            return np.array([])
        if refX is None or len(refX) == 0 or not cols:
            return np.ones(n) / max(1, n)

        ref_keys = _bucket_keys(refX, cols)
        vc = pd.Series(ref_keys).value_counts()
        counts = vc.to_dict()

        cand_keys = _bucket_keys(cands, cols)
        c = np.array([counts.get(k, 0) for k in cand_keys], dtype=float)

        w = (c + 1.0) ** (-alpha)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            w = np.ones_like(w) / max(1, n)
        else:
            w = w / s

        if uniform_mix > 0:
            u = np.ones(n) / n
            w = (1.0 - uniform_mix) * w + uniform_mix * u

        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        return w

    def _infer_discrete_cols(df: pd.DataFrame,
                             col_unique_thresh: int,
                             col_ratio_thresh: float) -> List[str]:
        """Pick columns that behave discrete (few uniques or small unique/rows)."""
        n = max(1, len(df))
        cols = []
        for c in df.columns:
            u = df[c].nunique(dropna=True)
            if u <= col_unique_thresh or (u / n) <= col_ratio_thresh:
                cols.append(c)
        return cols

    def _num_unique_combos(df: pd.DataFrame, cols: List[str]) -> int:
        if not cols:
            return 0
        return df[cols].astype(object).where(pd.notnull(df[cols]), None) \
                 .apply(lambda r: tuple(r.values.tolist()), axis=1) \
                 .nunique(dropna=True)

    # ----------------- Normalize inputs -----------------
    conditions_df = _safe_df(conditions)
    ref_df = _safe_df(reference_conditions, like_cols=conditions_df.columns) if reference_conditions is not None else pd.DataFrame(columns=conditions_df.columns)

    if num_samples is None or num_samples <= 0:
        num_samples = max(1, len(conditions_df))
    if len(conditions_df) == 0:
        return conditions_df

    # ----------------- Auto-toggle discrete repooling -----------------
    if auto_repool:
        inferred_cols = discrete_cols
        if not inferred_cols:
            inferred_cols = _infer_discrete_cols(
                conditions_df,
                col_unique_thresh=col_unique_thresh,
                col_ratio_thresh=col_ratio_thresh
            )
        if inferred_cols:
            combos = _num_unique_combos(conditions_df, inferred_cols)
            if combos > 0 and combos <= space_thresh:
                discrete_repool = True
                discrete_cols = inferred_cols

    # ----------------- Subsample for safety -----------------
    ref_len = 0 if reference_conditions is None else (len(reference_conditions) if not isinstance(reference_conditions, np.ndarray) else reference_conditions.shape[0])
    seed_iter = (random_state or 0) * 1000003 + int(ref_len)
    rng = np.random.default_rng(seed_iter)

    if len(conditions_df) > CAND_CAP:
        cand_idx = rng.choice(conditions_df.index.values, size=CAND_CAP, replace=False)
        cands_view = conditions_df.loc[cand_idx]
    else:
        cands_view = conditions_df

    if len(ref_df) > REF_CAP:
        ref_idx = rng.choice(ref_df.index.values, size=REF_CAP, replace=False)
        ref_view = ref_df.loc[ref_idx]
    else:
        ref_view = ref_df

    # ----------------- Stage A: Random vs Randomizer -----------------
    ref_n = len(ref_view)

    eps0, tau, eps_min = 0.4, 80.0, 0.05
    eps = max(eps_min, eps0 * math.exp(-ref_n / max(1.0, tau)))
    try:
        nov_all = _novelty_scores(cands_view, ref_view)
        coverage = float(np.mean(nov_all))
    except Exception:
        coverage = 0.5
    if coverage < 0.2:
        eps = min(1.0, eps + 0.1)

    use_random = (rng.random() < eps) or (len(models) == 0)

    if use_random:
        if discrete_repool and discrete_cols is not None and len(conditions_df) > 0:
            wA = _inverse_freq_weights(
                cands=conditions_df,
                refX=ref_view,
                cols=discrete_cols,
                alpha=balance_alpha,
                uniform_mix=balance_uniform_mix,
            )
            chosen_idx = rng.choice(
                conditions_df.index.values,
                size=min(num_samples, len(conditions_df)),
                replace=False,
                p=wA
            )
        else:
            chosen_idx = rng.choice(
                conditions_df.index.values,
                size=min(num_samples, len(conditions_df)),
                replace=False
            )
        return conditions_df.loc[chosen_idx]

    # ----------------- Stage B: pick strategy & score (on the view) -----------------
    sF = _falsification_scores(cands_view)
    sU = _uncertainty_scores(cands_view)
    sN = _novelty_scores(cands_view, ref_view)

    T0, tauT, Tmin = 1.0, 120.0, 0.25
    T = max(Tmin, T0 * math.exp(-ref_n / max(1.0, tauT)))
    q = 0.2

    def topq_mean(a: np.ndarray, q: float) -> float:
        if a.size == 0: return 0.0
        k = max(1, int(round(q * a.size)))
        return float(np.mean(np.sort(a)[-k:]))

    SF, SU, SN = topq_mean(sF, q), topq_mean(sU, q), topq_mean(sN, q)

    prior = np.array([0.05, 0.05, 0.05], dtype=float)
    logits = (np.array([SF, SU, SN]) + prior) / max(1e-6, T)
    logits -= np.max(logits)
    w = np.exp(logits); w /= np.sum(w)

    has_classifier = any(hasattr(m, "predict_proba") for m in models)
    labels_all = np.array(["F", "U", "N"])
    weights_all = w

    if has_classifier:
        labels = labels_all
        weights = weights_all
    else:
        labels = np.array(["U", "N"])
        weights = weights_all[1:]

    strat = labels[int(np.argmax(weights))]
    scores = {"F": sF, "U": sU, "N": sN}[strat]

    # --- Discrete rebalance bonus in Stage B ---
    if discrete_repool and discrete_cols is not None:
        wB = _inverse_freq_weights(
            cands=cands_view,
            refX=ref_view,
            cols=discrete_cols,
            alpha=balance_alpha,
            uniform_mix=balance_uniform_mix,
        )
        bonus = _normalize01(wB)  # higher for under-sampled buckets
        scores = (1.0 - balance_gain) * scores + balance_gain * bonus

    # ----------------- Shortlist + Diversity (on the view) -----------------
    M = min(len(cands_view), max(num_samples * SHORTLIST_MULT, SHORTLIST_MIN))
    order = np.argsort(scores)[::-1]
    shortlist_idx = cands_view.index.values[order[:M]]
    shortlist = conditions_df.loc[shortlist_idx]

    shortlist_scored = shortlist.assign(_score=np.asarray(scores)[order[:M]])
    shortlist_scored = shortlist_scored.sort_values("_score", ascending=False).drop(columns="_score")

    pick_idx = _farthest_first(shortlist_scored, num_samples)
    return conditions_df.loc[pick_idx]
