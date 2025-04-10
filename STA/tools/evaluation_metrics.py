import scanpy as sc
import pandas as pd
import squidpy as sq
import numpy as np

from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, \
    silhouette_score


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def _compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)


def marker_score(adata, domain_key, top_n=5):
    adata = adata.copy()
    count_dict = adata.obs[domain_key].value_counts()
    adata = adata[adata.obs[domain_key].isin(count_dict.keys()[count_dict > 3].values)]
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=domain_key)
    selected_genes = []
    for i in range(top_n):
        toadd = list(adata.uns['rank_genes_groups']['names'][i])
        selected_genes.extend(toadd)
    selected_genes = np.unique(selected_genes)
    selected_genes = list(set(selected_genes) & set(adata.var.index))
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=selected_genes,
        n_perms=100,
        # n_jobs=1,
    )
    sq.gr.spatial_autocorr(
        adata,
        mode="geary",
        genes=selected_genes,
        n_perms=100,
        # n_jobs=1,
    )
    moranI = np.median(adata.uns["moranI"]['I'])
    gearyC = np.median(adata.uns["gearyC"]['C'])
    return moranI, gearyC


def evaluate_all(adata, gt_key, pred_key, spatial_key='spatial'):
    """
    Parameters
    ----------
    adata: adata
    gt_key: ground truth or manual annotation
    pred_key: prediction label
    spatial_key: spatial_key in adata.obsm['']

    Returns
    -------

    """
    ARI = adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])
    NMI = normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])
    HOM = homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])
    COM = completeness_score(adata.obs[gt_key], adata.obs[pred_key])

    CHAOS = _compute_CHAOS(adata.obs[pred_key], adata.obsm[spatial_key])
    ASW = silhouette_score(X=squareform(pdist(adata.obsm[spatial_key])), labels=adata.obs[pred_key],
                           metric='precomputed')
    PAS = _compute_PAS(adata.obs[pred_key], adata.obsm[spatial_key])

    MoranI, GearyC = marker_score(adata, gt_key, top_n=5)
    results_df = pd.DataFrame([[ARI, NMI, HOM, COM, CHAOS, ASW, PAS, MoranI, GearyC]],
                              columns=['ARI', 'NMI', 'HOM', 'COM', 'CHAOS', 'ASW', 'PAS', 'MoranI', 'GearyC'])
    return results_df
