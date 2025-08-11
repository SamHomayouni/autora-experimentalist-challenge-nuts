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
        ) -> pd.DataFrame:
    """
    Annealed hybrid sampler with memory-safe scoring:
    - Stage A: Random vs. Randomizer (annealed ε with coverage guard)
    - Stage B: Falsification / Uncertainty / Novelty chosen via softmax of signals
    - Diversity: farthest-first on a shortlist
    - NEW: cap candidate/reference sizes to avoid O(N*M) memory blowups; novelty via NearestNeighbors.
    """
    # ----------------- Tunables for scale -----------------
    # Max candidates to score each iteration (subset of pool)
    #CAND_CAP = 5000                      # try 5k; lower if you still see crashes
    # Maxx reference points used for novelty
    REF_CAP  = 5000
    # Shortlist multiplier before diversity pick
    #SHORTLIST_MULT = 5                   # shortlist size ≈ num_samples * SHORTLIST_MULT
    #SHORTLIST_MIN = 50
    CAND_CAP = 10000
    SHORTLIST_MULT = 3
    SHORTLIST_MIN = 30


    # ----------------- Helpers -----------------
    def _to_df(X, like: pd.DataFrame | None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = list(like.columns) if isinstance(like, pd.DataFrame) else None
        return pd.DataFrame(X, columns=cols)

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

    def _predict_probs(model, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            try:
                p = model.predict_proba(X)
                return np.clip(np.atleast_2d(p), 1e-9, 1 - 1e-9)
            except Exception:
                return None
        return None

    def _predict_regr(model, X: pd.DataFrame) -> np.ndarray | None:
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
        # Standardize by range to prevent scale dominance
        both = pd.concat([cands, refX], axis=0)
        rng = both.max(axis=0) - both.min(axis=0)
        rng[rng < 1e-12] = 1.0
        A = (cands / rng).to_numpy(dtype=float)
        B = (refX  / rng).to_numpy(dtype=float)
        # Fit NN on (possibly capped) reference
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(B)
        dists, _ = nn.kneighbors(A, n_neighbors=1, return_distance=True)
        return _normalize(dists.reshape(-1))

    def _farthest_first(df: pd.DataFrame, k: int) -> pd.Index:
        if len(df) <= k:
            return df.index
        X = df.to_numpy(dtype=float)
        sel = [0]
        # incremental min squared distance to selected set
        d2 = np.sum((X - X[0])**2, axis=1)
        for _ in range(1, k):
            nxt = int(np.argmax(d2))
            sel.append(nxt)
            d2 = np.minimum(d2, np.sum((X - X[nxt])**2, axis=1))
        return df.iloc[sel].index

    # ----------------- Normalize inputs -----------------
    conditions_df = _to_df(conditions, like=None)
    ref_df = _to_df(reference_conditions, like=conditions_df) if reference_conditions is not None else pd.DataFrame(columns=conditions_df.columns)

    if num_samples is None or num_samples <= 0:
        num_samples = max(1, len(conditions_df))
    if len(conditions_df) == 0:
        return conditions_df

    # ----------------- Subsample for safety -----------------
    #rng = np.random.default_rng()
    #if len(conditions_df) > CAND_CAP:
     #   cand_idx = rng.choice(conditions_df.index.values, size=CAND_CAP, replace=False)
     #   cands_view = conditions_df.loc[cand_idx]
    #else:
     #   cands_view = conditions_df
    # Per-iteration seed: base seed mixed with how many reference points we have
    ref_len = 0 if reference_conditions is None else (len(reference_conditions) if not isinstance(reference_conditions, np.ndarray) else reference_conditions.shape[0])
    seed_iter = (random_state or 0) * 1000003 + int(ref_len)
    rng = np.random.default_rng(seed_iter)   # <— use this rng everywhere below
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
    #ref_n = len(ref_df)
    ref_n = len(ref_view)

    #eps0, tau, eps_min = 0.6, 200.0, 0.1
    #eps = max(eps_min, eps0 * math.exp(-ref_n / max(1.0, tau)))
    eps0, tau, eps_min = 0.4, 80.0, 0.05
    eps = max(eps_min, eps0 * math.exp(-ref_n / max(1.0, tau)))
    # Coverage guard from novelty on the (subsampled) view
    try:
        nov_all = _novelty_scores(cands_view, ref_view)
        coverage = float(np.mean(nov_all))
    except Exception:
        coverage = 0.5
    #if coverage < 0.2:
    #    eps = min(1.0, eps + 0.2)
    if coverage < 0.2:
        eps = min(1.0, eps + 0.1)

    use_random = (rng.random() < eps) or (len(models) == 0)

    if use_random:
        chosen_idx = rng.choice(conditions_df.index.values, size=min(num_samples, len(conditions_df)), replace=False)
        return conditions_df.loc[chosen_idx]

    # ----------------- Stage B: pick strategy & score (on the view) -----------------
    sF = _falsification_scores(cands_view)
    sU = _uncertainty_scores(cands_view)
    sN = _novelty_scores(cands_view, ref_view)

    # Route by softmax over aggregated signals
    #T0, tauT, Tmin = 1.0, 300.0, 0.3
    #T = max(Tmin, T0 * math.exp(-ref_n / max(1.0, tauT)))
    #q = 0.1
    # Commit earlier (lower temperature faster) and use a slightly larger tail
    T0, tauT, Tmin = 1.0, 120.0, 0.25
    T = max(Tmin, T0 * math.exp(-ref_n / max(1.0, tauT)))
    q = 0.2


    def topq_mean(a: np.ndarray, q: float) -> float:
        if a.size == 0: return 0.0
        k = max(1, int(round(q * a.size)))
        return float(np.mean(np.sort(a)[-k:]))

    SF, SU, SN = topq_mean(sF, q), topq_mean(sU, q), topq_mean(sN, q)
    #prior = np.array([0.05, 0.05, 0.05], dtype=float)
    #logits = (np.array([SF, SU, SN]) + prior) / max(1e-6, T)
    #logits -= np.max(logits)
    #w = np.exp(logits); w /= np.sum(w)

    # Soft scores (still computed with temperature), but pick deterministically
    prior = np.array([0.05, 0.05, 0.05], dtype=float)
    logits = (np.array([SF, SU, SN]) + prior) / max(1e-6, T)
    logits -= np.max(logits)
    w = np.exp(logits); w /= np.sum(w)
    #strat = rng.choice(np.array(["F", "U", "N"]), p=w)
    # Determine if any classifier is present (has predict_proba)
    has_classifier = any(hasattr(m, "predict_proba") for m in models)

    labels_all = np.array(["F", "U", "N"])
    weights_all = w  # corresponds to [F, U, N]

    if has_classifier:
        labels = labels_all
        weights = weights_all
    else:
        # Drop F when only regressors: F≈U; avoid noisy routing
        labels = np.array(["U", "N"])
        weights = weights_all[1:]  # keep U,N

    # Deterministic routing: pick the highest-weight strategy
    strat = labels[int(np.argmax(weights))]


    scores = {"F": sF, "U": sU, "N": sN}[strat]

    # ----------------- Shortlist + Diversity (on the view) -----------------
    M = min(len(cands_view), max(num_samples * SHORTLIST_MULT, SHORTLIST_MIN))
    order = np.argsort(scores)[::-1]
    shortlist_idx = cands_view.index.values[order[:M]]
    shortlist = conditions_df.loc[shortlist_idx]  # keep original frame for consistent dtype/cols

    shortlist_scored = shortlist.assign(_score=np.asarray(scores)[order[:M]])
    shortlist_scored = shortlist_scored.sort_values("_score", ascending=False).drop(columns="_score")

    pick_idx = _farthest_first(shortlist_scored, num_samples)

    return conditions_df.loc[pick_idx]
