
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomTreesEmbedding, IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import spearmanr, kruskal
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


from utils.pred_utils import (
    logtransform_skewed_feats,
    check_skewness,
    remove_outliers_zscore
)



def run_clustering(
    X, y=None, n_clusters=3, cluster_method='forest',
    USE_PCA=False, n_comp=6, plot_pca=False,
    outlier_removal=None, transform_y=None, skew_thresh=1.5,
    return_nanrow_bool=False,
    return_fitted_params=False,
    apply_fitted_skewed_fts=None,
    apply_fitted_pca=None,
    apply_fitted_kmeans=None,
    verbose=False,
):
    assert cluster_method in ['forest', 'kmeans', 'both'], (
        'Invalid cluster method. Choose "forest", "kmeans", or "both".'
    )
    # returns z-scored features
    result = prepare_X_for_clustering(
        X=X, y=y, skew_threshold=skew_thresh,
        return_skewed_ft_list=return_fitted_params,
        apply_skewed_ft_bool=apply_fitted_skewed_fts,
        return_nan_bool=return_nanrow_bool,
        verbose=verbose,
    )
    X = result['X']
    if y is not None:
        y = result['y']
    if return_nanrow_bool:
        nanrow_bool = result['nan_rows']
    
    if outlier_removal == 'isoforest':
        X, y = remove_outliers_isoforest(
            X, y=y, verbose=verbose,
        )
    
    elif outlier_removal == 'zscore':
        outlier_result = remove_outliers_zscore(
            X=X, y=y, threshold_n_sd=3.0, verbose=verbose,
            return_outl_mask=return_nanrow_bool,
        )
        X = outlier_result['X']
        if y is not None:
            y = outlier_result['y']
        if return_nanrow_bool:
            nanrow_bool[~result['nan_rows']] = outlier_result['outl_mask']
            
    else:
        pass

    if transform_y is not None and y is not None:
        for new_y, old_y_list in transform_y.items():
            y[np.isin(y, old_y_list)] = new_y


    if USE_PCA:
        if apply_fitted_pca is None:
            fitted_pca = PCA(n_components=n_comp).fit(X)
        else:
            fitted_pca = apply_fitted_pca
        X = fitted_pca.transform(X)

        if plot_pca:
            # plot PC-1 and -2, color-code for tremor score
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for i_ax, pc2 in enumerate([1, 2]):
                ax = axes[i_ax]
                im = ax.scatter(X[:, 0], X[:, pc2], c=y, cmap='viridis', edgecolor='k')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel(f'Principal Component {pc2+1}')
                ax.set_title(f'PC-1 vs PC-{pc2+1}')
            fig.colorbar(im, label='Tremor score',)
            plt.tight_layout()
            plt.show()
    
    output = {}

    if return_fitted_params:
        output['fitted_model'] = {}
        output['fitted_model']['pca'] = fitted_pca if USE_PCA else None
        output['fitted_model']['skewed_ft_bool'] = result['skewed_bool'] if skew_thresh is not None else None

    if cluster_method in ['forest', 'both']:
        labels_forest, X, y = cluster_forest(
            X, y=y, n_clusters=n_clusters, verbose=verbose,
        )
        if verbose: print(f'\n\n#####validate forest clusters')
        forest_results = validate_clusters(labels_forest, y, X=X, verbose=verbose,)
        output['forest'] = forest_results
        output['forest']['clust_labels'] = labels_forest
        output['forest']['true_y'] = y
        
    if cluster_method in ['kmeans', 'both']:
        kmeans_output = cluster_kmeans(
            X, n_clusters=n_clusters,
            return_fitted_km=return_fitted_params,
            apply_fitted_kmeans=apply_fitted_kmeans,
            verbose=verbose,
        )
        labels_kmeans = kmeans_output['labels']
        if return_fitted_params: fitted_km = kmeans_output['fitted_km']
        
        if apply_fitted_kmeans is None:
            km_results = validate_clusters(labels_kmeans, y, X=X, verbose=verbose,)
            output['kmeans'] = km_results
        else:
            output['kmeans'] = {}
        output['kmeans']['clust_labels'] = labels_kmeans

        if y is not None:
            output['kmeans']['true_y'] = y
        if return_nanrow_bool:
            output['kmeans']['nan_row_bool'] = nanrow_bool
        # add kmeans cluster info to output
        if return_fitted_params:
            output['fitted_model']['fitted_km'] = fitted_km
            output['fitted_model']['kmeans_centroids'] = fitted_km.cluster_centers_
            output['fitted_model']['cluster_scores'] = {
                c: res['mean'] for c, res in
                km_results['per_cluster'].items()
            }
    
    if cluster_method == 'both':
        if verbose: print('\n##### compare forest vs kmeans clusters:')
        if verbose: print('ARI:', compare_cluster_solutions(labels_kmeans, labels_forest))
    
    return output



# ─────────────────────────────────────────────
# 1. Preprocessing
# ─────────────────────────────────────────────

def prepare_X_for_clustering(
    X, y=None, skew_threshold=None,
    do_not_return_scaler=True,
    return_nan_bool=False,
    return_skewed_ft_list=False,
    apply_skewed_ft_bool=None,
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
        X = logtransform_skewed_feats(X.copy(), skew_threshold=skew_threshold,
                                      apply_skewed_ft_bool=apply_skewed_ft_bool,)
        skewed_bool = [
            check_skewness(X[:, i], threshold=skew_threshold)[0]
            for i in range(X.shape[1])
        ]
        if verbose: print(f'[prepare] {sum(skewed_bool)} features log-transformed (threshold={skew_threshold})')


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    output = {'X': X_scaled}

    if y is not None: output['y'] = y[~nan_rows]

    if not do_not_return_scaler: output['scaler'] = scaler

    if return_nan_bool: output['nan_rows'] = nan_rows

    if return_skewed_ft_list and skew_threshold is not None:
        output['skewed_bool'] = skewed_bool
    
    return output


# ─────────────────────────────────────────────
# 2. Outlier removal (z-score based) -> moved to pred_utils.py
# ─────────────────────────────────────────────



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

def cluster_kmeans(
    X,
    n_clusters=3,
    random_state=42,
    return_fitted_km=False,
    apply_fitted_kmeans=None,
    verbose=False
):
    """KMeans clustering on z-scored feature matrix.

    Parameters
    ----------
    X                : ndarray, z-scored features
    n_clusters       : int, number of clusters (default 3)
    return_fitted_km      : bool, if True also return fitted KMeans model (default False)

    Returns
    -------
    labels    : ndarray of int
    inertia   : float, sum of squared distances to cluster centres (lower = tighter)
    centroids : ndarray, shape (n_clusters, n_features) — only if return_fitted_km=True
    """
    # fit or allocate fitted k-means model
    if apply_fitted_kmeans is None:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        fitted_km = km.fit(X)
    else:
        fitted_km = apply_fitted_kmeans
    # apply fitted kmeans model on X to get predictions
    output = {'labels': fitted_km.predict(X)}

    if verbose:
        counts = {c: (output['labels'] == c).sum()
                  for c in np.unique(output['labels'])}
        print(f'[KMeans] {n_clusters} clusters, inertia={fitted_km.inertia_:.2f}')
        print(f'  cluster sizes: {counts}')

    if return_fitted_km:
        output['fitted_km'] = fitted_km

    return output


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
