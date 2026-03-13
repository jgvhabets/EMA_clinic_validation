
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomTreesEmbedding, IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import spearmanr, kruskal

from utils.pred_utils import logtransform_skewed_feats, check_skewness


# ─────────────────────────────────────────────
# 1. Preprocessing
# ─────────────────────────────────────────────

def prepare_X_for_clustering(
    X, y=None, skew_threshold=None,
    do_not_return_scaler=True,
    do_not_return_nan_bool=True,
    verbose=False,
):
    """
    Checks nan's, log-transform skewed features, then z-score.
    optionally keeps y aligned with X, and returns scaler and nan row bool array.
    
    Returns
    -------
    X_scaled : ndarray, shape (n_samples, n_feats)
    y_aligned : ndarray, shape (n_samples,) if y provided
    scaler   : fitted StandardScaler (for inverse_transform if needed)
    nan_rows : boolean array indicating rows with NaNs

    if scaler and nan_bool are returned, order is (X_scaled, scaler, nan_rows)
    """
    # check nans
    if verbose: print('any nans in X before prep?', np.any(np.isnan(X)), X.shape)
    nan_rows = np.any(np.isnan(X), axis=1)
    X = X[~nan_rows]
    if verbose: print('any nans in X after removing NaNs?', np.any(np.isnan(X)), X.shape)

    if skew_threshold is not None:
        X = logtransform_skewed_feats(X.copy(), skew_threshold=skew_threshold)
        n_skewed = sum(
            check_skewness(X[:, i], threshold=skew_threshold)[0]
            for i in range(X.shape[1])
        )
        if verbose: print(f'[prepare] {n_skewed} features log-transformed (threshold={skew_threshold})')


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    output = [X_scaled]

    if y is not None: output.append(y[~nan_rows])

    if not do_not_return_scaler: output.append(scaler)

    if not do_not_return_nan_bool: output.append(nan_rows)
    
    return tuple(output)


# ─────────────────────────────────────────────
# 2. Outlier removal (z-score based)
# ─────────────────────────────────────────────

def remove_outliers_zscore(
    X, y=None, threshold=3.0, do_not_return_outl_mask=True, verbose=False):
    """Flag and remove samples where any feature z-score exceeds threshold.

    Parameters
    ----------
    X         : ndarray, already z-scored (output of prepare_X_for_clustering)
    y         : optional array aligned with X rows
    threshold : float, default 3.0

    Returns
    -------
    X_clean, y_clean (if y provided), outlier_mask (bool, True = outlier)
    """
    outlier_mask = np.any(np.abs(X) > threshold, axis=1)

    if verbose:
        print(f'[outliers] {outlier_mask.sum()} / {len(X)} samples removed '
              f'(|z| > {threshold})')

    X_clean = X[~outlier_mask]

    output = [X_clean]

    if y is not None: output.append(y[~outlier_mask])
    
    if not do_not_return_outl_mask: output.append(outlier_mask)

    return tuple(output)


# ─────────────────────────────────────────────
# 3a. DBSCAN clustering
# ─────────────────────────────────────────────

def cluster_dbscan(X, eps=1.5, min_samples=5, verbose=False):
    """DBSCAN clustering on z-scored feature matrix.

    Returns
    -------
    labels : ndarray of int  (-1 = noise/outlier)
    n_clusters : int  (excluding noise)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    if verbose:
        print(f'[DBSCAN] eps={eps}, min_samples={min_samples} → '
              f'{n_clusters} clusters, {n_noise} noise points')

    return labels, n_clusters


# ─────────────────────────────────────────────
# 2b. Outlier removal (Isolation Forest)
# ─────────────────────────────────────────────

def remove_outliers_isoforest(X, y=None, contamination=0.05,
                               random_state=42, verbose=False):
    """Flag and remove outliers using Isolation Forest.

    Isolation Forest isolates samples by random splits; outliers are isolated
    faster (shorter average path length). Does not rely on density assumptions,
    making it suitable for sparse high-dimensional feature spaces where DBSCAN
    fails.

    Parameters
    ----------
    X             : ndarray, z-scored features
    y             : optional array aligned with X rows
    contamination : float in (0, 0.5], expected fraction of outliers (default 0.05)

    Returns
    -------
    X_clean, y_clean (if y provided)
    """
    iso = IsolationForest(contamination=contamination, random_state=random_state,
                          n_jobs=-1)
    outlier_pred = iso.fit_predict(X)  # +1 = inlier, -1 = outlier
    inlier_mask = outlier_pred == 1

    if verbose:
        n_removed = (~inlier_mask).sum()
        print(f'[IsoForest] {n_removed} / {len(X)} samples removed '
              f'(contamination={contamination})')

    output = [X[inlier_mask]]
    if y is not None:
        output.append(y[inlier_mask])
    return tuple(output)


# ─────────────────────────────────────────────
# 3a. KMeans clustering
# ─────────────────────────────────────────────

def cluster_kmeans(X, n_clusters=3, random_state=42, verbose=False):
    """KMeans clustering on z-scored feature matrix.

    Parameters
    ----------
    X          : ndarray, z-scored features
    n_clusters : int, number of clusters (default 3)

    Returns
    -------
    labels   : ndarray of int
    inertia  : float, sum of squared distances to cluster centres (lower = tighter)
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X)

    if verbose:
        counts = {c: (labels == c).sum() for c in np.unique(labels)}
        print(f'[KMeans] {n_clusters} clusters, inertia={km.inertia_:.2f}')
        print(f'  cluster sizes: {counts}')

    return labels, km.inertia_


# ─────────────────────────────────────────────
# 3b. Forest-based clustering
# ─────────────────────────────────────────────

def cluster_forest(X, y=None, n_clusters=3, n_estimators=200, random_state=42,
                   return_embedder_and_clusterer=False, verbose=False):
    """Forest-based clustering via RandomTreesEmbedding + AgglomerativeClustering.

    RandomTreesEmbedding maps each sample to the set of leaf nodes it falls
    into across all trees. Two samples that co-occur in the same leaves are
    considered similar. AgglomerativeClustering then partitions the resulting
    sparse similarity space into n_clusters groups.

    Parameters
    ----------
    X                 : ndarray, z-scored features
    y                 : optional array aligned with X; kept aligned after outlier removal
    n_clusters        : int, number of clusters (default 3)
    n_estimators      : int, number of trees in the forest
    outlier_removal   : None | 'isoforest'  — method to remove outliers before clustering
    iso_contamination : float, contamination parameter for IsolationForest (default 0.05)

    Returns
    -------
    labels       : ndarray of int, aligned to X_clean
    X_clean      : ndarray, X after optional outlier removal (= X if disabled)
    y_clean      : ndarray or None, y aligned to X_clean
    embedder     : fitted RandomTreesEmbedding  (only if return_embedder_and_clusterer)
    clusterer    : fitted AgglomerativeClustering  (only if return_embedder_and_clusterer)
    """
    X_clean = X.copy()
    y_clean = y.copy() if y is not None else None

    embedder = RandomTreesEmbedding(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    X_embedded = embedder.fit_transform(X_clean)  # sparse binary leaf-indicator matrix

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward',
    )
    labels = clusterer.fit_predict(X_embedded.toarray())

    if verbose:
        print(f'[forest] {n_estimators} trees → {n_clusters} clusters via '
              f'AgglomerativeClustering')
        counts = {c: (labels == c).sum() for c in np.unique(labels)}
        print(f'  cluster sizes: {counts}')

    if return_embedder_and_clusterer:
        return labels, X_clean, y_clean, embedder, clusterer
    else:
        return labels, X_clean, y_clean


# ─────────────────────────────────────────────
# 4. Cluster validation with EMA scores
# ─────────────────────────────────────────────

def validate_clusters(labels, y, X=None, verbose=True):
    """Validate cluster assignments against EMA ground-truth scores (y).

    Metrics
    -------
    - Mean ± SD of y per cluster
    - Kruskal-Wallis H test (non-parametric ANOVA across clusters)
    - Spearman ρ between cluster label and y
    - Silhouette score (if X is provided)

    Parameters
    ----------
    labels : ndarray of int  (-1 noise points are excluded)
    y      : ndarray of EMA scores, aligned with labels
    X      : optional ndarray of features; if provided, silhouette score is computed

    Returns
    -------
    results : dict with keys 'per_cluster', 'kruskal_H', 'kruskal_p',
              'spearman_rho', 'spearman_p', 'silhouette' (nan if X not provided)
    """
    valid_mask = labels != -1  # exclude DBSCAN noise if present
    labels_v = labels[valid_mask]
    y_v = y[valid_mask]

    unique_clusters = np.unique(labels_v)
    per_cluster = {}
    groups = []

    for c in unique_clusters:
        sel = labels_v == c
        vals = y_v[sel]
        per_cluster[c] = {
            'n': sel.sum(),
            'mean': np.mean(vals),
            'median': np.median(vals),
            'sd': np.std(vals),
        }
        groups.append(vals)

    H, p_kruskal = kruskal(*groups)
    rho, p_spear = spearmanr(labels_v.astype(float), y_v)

    # silhouette score (internal quality, needs feature matrix)
    if X is not None and valid_mask.sum() > 1 and len(unique_clusters) > 1:
        sil = silhouette_score(X[valid_mask], labels_v)
    else:
        sil = np.nan

    # sort clusters by mean EMA (proof of concept only — not a validation metric)
    per_cluster_sorted = dict(
        sorted(per_cluster.items(), key=lambda item: item[1]['mean'])
    )

    results = {
        'per_cluster': per_cluster_sorted,
        'kruskal_H': H,
        'kruskal_p': p_kruskal,
        'spearman_rho': rho,
        'spearman_p': p_spear,
        'silhouette': sil,
    }

    if verbose:
        print('\n[validation] EMA scores per cluster (sorted by mean EMA — proof of concept):')
        print('  NOTE: sorting by mean EMA is for visual inspection only, not a validation metric.')
        for c, s in per_cluster_sorted.items():
            print(f'  cluster {c} (n={s["n"]}): '
                  f'mean={s["mean"]:.2f}, median={s["median"]:.2f}, sd={s["sd"]:.2f}')
        print(f'  Kruskal-Wallis: H={H:.2f}, p={p_kruskal:.4f}')
        print(f'  Spearman rho={rho:.2f}, p={p_spear:.4f}')
        if not np.isnan(sil):
            print(f'  Silhouette score={sil:.3f}')

    return results


def validate_clusters_silhouette(X, labels):
    """
    Silhouette score for internal cluster quality (requires feature matrix).

    Excludes noise points (label == -1).
    """
    valid_mask = labels != -1
    if valid_mask.sum() < 2 or len(np.unique(labels[valid_mask])) < 2:
        return np.nan
    score = silhouette_score(X[valid_mask], labels[valid_mask])
    return score


def compare_cluster_solutions(labels_a, labels_b):
    """Adjusted Rand Index between two cluster label arrays.

    Useful for comparing DBSCAN vs forest solutions.
    Excludes noise from both arrays.
    """
    valid = (labels_a != -1) & (labels_b != -1)
    if valid.sum() < 2:
        return np.nan
    return adjusted_rand_score(labels_a[valid], labels_b[valid])
