import ot
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import fowlkes_mallows_score
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error
from copy import deepcopy


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type


def mclust_R(adata, n_clusters, model_names='EEE', used_embedding='X_pca', radius=50, random_seed=42, smooth=True):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_embedding]), n_clusters, model_names)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    adata.obsm['mclust_prob'] = np.array(res[-3])

    if smooth:
        adata.obs['mclust'] = refine_label(adata, radius=radius, key='mclust')

    return adata


def kmeans_cluster(adata, n_clusters=10, used_embedding='X_pca', mode='KMeans'):
    if mode == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(adata.obsm[used_embedding])
    elif mode == 'MiniBatchKMeans':
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(
            adata.obsm[used_embedding])
    else:
        print('mode in [KMeans, MiniBatchKMeans]')
        raise NotImplementedError
    cell_label = model.labels_
    adata.obs.loc[:, 'kmeans'] = cell_label
    adata.obs.loc[:, 'kmeans'] = adata.obs.loc[:, 'kmeans'].astype(str)
    return adata


def mirror_stability(n_clusters, stability):
    stability = [
        stability[i: i + len(n_clusters) - 1] for i in range(0, len(stability), len(n_clusters) - 1)
    ]
    stability = list(map(list, zip(*stability)))
    return np.array([stability[i] + stability[i - 1] for i in range(1, len(stability))])


def update_labels(labels, new_labels):
    for k, new_l in new_labels.items():
        labels[k].append(new_l)


def compute_similarity(pair, similarity_function):
    return similarity_function(*pair)


def cluster_stability(
        adata,
        n_clusters: tuple,
        use_rep: str = 'X_emb',
        max_runs: int = 10,
        convergence_tol: float = 1e-2,
        similarity_function: callable = None):
    """
    Varrone M, Tavernari D, Santamaria-Martínez A, et al. CellCharter reveals spatial cell niches associated
    with tissue remodeling and cell plasticity[J]. Nature genetics, 2024, 56(1): 74-84.

    Qian J, Shao X, Bao H, et al. Identification and characterization of cell niches in tissue from spatial omics data
    at single-cell resolution[J]. Nature Communications, 2025, 16(1): 1693.

    :param adata:
    :param n_clusters:
    :param use_rep:
    :param max_runs:
    :param convergence_tol:
    :param similarity_function:
    :return:
    """
    n_clusters = list(range(*(max(1, n_clusters[0] - 1), n_clusters[1] + 2)))
    X = adata.obsm[use_rep]
    random_state = 0
    labels = defaultdict(list)
    stability = []
    if similarity_function is None:
        similarity_function = fowlkes_mallows_score

    previous_stability = None
    for i in range(max_runs):
        new_labels = {}

        pbar = tqdm(n_clusters)
        for k in pbar:
            clustering = KMeans(n_clusters=k, random_state=i + random_state)
            new_labels[k] = clustering.fit_predict(X)
            pbar.set_description(f"Iteration {i + 1}/{max_runs}")

        if i > 0:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                pairs = [
                    (new_labels[k], labels[k + 1][i])
                    for i in range(len(list(labels.values())[0]))
                    for k in list(labels.keys())[:-1]
                ]
                stability.extend(list(executor.map(lambda x: similarity_function(*x), pairs)))

            if previous_stability is not None:
                current_stability = mirror_stability(n_clusters, stability)
                previous_stability_mean = np.mean(mirror_stability(n_clusters, previous_stability), axis=1)
                current_stability_mean = np.mean(current_stability, axis=1)
                stability_change = mean_absolute_percentage_error(previous_stability_mean, current_stability_mean)

                if stability_change < convergence_tol:
                    update_labels(labels, new_labels)
                    print(
                        f"Convergence with a change in stability of {stability_change} reached after {i + 1} iterations"
                    )
                    break

            previous_stability = deepcopy(stability)

        update_labels(labels, new_labels)

    if max_runs > 1:
        stability = mirror_stability(n_clusters, stability)
    else:
        stability = None

    # best_k
    if max_runs <= 1:
        raise ValueError("Cannot compute stability with max_runs <= 1")
    stability_mean = np.array([np.mean(stability[k]) for k in range(len(n_clusters[1:-1]))])
    best_idx = np.argmax(stability_mean)
    best_k = n_clusters[best_idx + 1]

    robustness_df = pd.melt(
        pd.DataFrame.from_dict({k: stability[i] for i, k in enumerate(n_clusters[1:-1])}, orient="columns"),
        var_name="K",
        value_name="Stability",
    )

    adata.uns['robustness_df'] = robustness_df
    adata.uns['best_k'] = best_k
