import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import entropy
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def softmax(adata_ref,
            adata_query,
            T=0.5,
            n_top=None,
            percentile=0):
    """
    ref: utilized SIMBA's work
    Chen H, Ryu J, Vinyard M E, et al. SIMBA: single-cell embedding along with features[J].
    Nature Methods, 2024, 21(6): 1003-1013. https://doi.org/10.1038/s41592-023-01899-8
    Softmax-based transformation
    This will transform query data to reference-comparable data
    Parameters
    ----------
    adata_ref: `AnnData`
        Reference anndata.
    adata_query: `list`
        Query anndata objects
    T: `float`
        Temperature parameter.
        It controls the output probability distribution.
        When T goes to inf, it becomes a discrete uniform distribution,
        each query becomes the average of reference;
        When T goes to zero, softargmax converges to arg max,
        each query is approximately the best of reference.
    n_top: `float`
    percentile: `float`
    """

    scores_ref_query = np.matmul(adata_ref.X, adata_query.X.T)
    # avoid overflow encountered
    scores_ref_query = scores_ref_query - scores_ref_query.max()
    scores_softmax = np.exp(scores_ref_query / T) / (np.exp(scores_ref_query / T).sum(axis=0))[None, :]
    if n_top is None:
        thresh = np.percentile(scores_softmax, q=percentile, axis=0)
    else:
        thresh = (np.sort(scores_softmax, axis=0)[::-1, :])[n_top - 1,]
    mask = scores_softmax < thresh[None, :]
    scores_softmax[mask] = 0
    # rescale to make scores add up to 1
    scores_softmax = scores_softmax / scores_softmax.sum(axis=0, keepdims=1)
    X_query = np.dot(scores_softmax.T, adata_ref.X)
    adata_query.layers['softmax'] = X_query


def gene_embedding(adata, cell_key='X_STAX', gene_key='gene_embedding', T=1e-1):
    adata_cell = sc.AnnData(adata.obsm[cell_key], obs=adata.obs)
    adata_cell.obs.loc[:, 'cell_gene'] = 'cell'
    adata_gene = sc.AnnData(adata.varm[gene_key], obs=adata.var)
    adata_gene.obs.loc[:, 'cell_gene'] = 'gene'
    adata_cell_gene = embed(adata_cell, [adata_gene], T=T)
    return adata_cell_gene


def embed(adata_ref,
          list_adata_query,
          use_precomputed=False,
          T=1e-1,
          list_T=None,
          n_top=None,
          percentile=0,
          list_percentile=None,
          ):
    """
    ref: utilized SIMBA's work
    SIMBA: single-cell embedding along with features, https://doi.org/10.1038/s41592-023-01899-8
    Embed a list of query datasets along with reference dataset into the same space
    Returns
    -------
    adata_all: `AnnData`
        Store #entities × #dimensions.
    """
    X_all = adata_ref.X.copy()
    obs_all = adata_ref.obs.copy()
    obs_all['id_dataset'] = ['ref'] * adata_ref.shape[0]
    for i, adata_query in enumerate(list_adata_query):
        if list_T is not None:
            param_T = list_T[i]
        else:
            param_T = T
        if list_percentile is not None:
            param_percentile = list_percentile[i]
        else:
            param_percentile = percentile
        if use_precomputed:
            if 'softmax' in adata_query.layers.keys():
                print(f'Reading in precomputed softmax-transformed matrix '
                      f'for query data {i};')
            else:
                print(f'No softmax-transformed matrix exists '
                      f'for query data {i}')
                print("Performing softmax transformation;")
                softmax(
                    adata_ref,
                    adata_query,
                    T=param_T,
                    percentile=param_percentile,
                    n_top=n_top,
                )
        else:
            print(f'Performing softmax transformation '
                  f'for query data {i};')
            softmax(
                adata_ref,
                adata_query,
                T=param_T,
                percentile=param_percentile,
                n_top=n_top,
            )
        X_all = np.vstack((X_all, adata_query.layers['softmax']))
        obs_query = adata_query.obs.copy()
        obs_query['id_dataset'] = [f'query_{i}'] * adata_query.shape[0]
        # obs_all = obs_all.append(obs_query, ignore_index=False)
        obs_all = pd.concat([obs_all, obs_query], axis=0, ignore_index=False)
    adata_all = sc.AnnData(X=X_all, obs=obs_all)
    return adata_all


def gene_SIMBA_metrics(adata, cell_key='X_STAX', gene_key='gene_embedding', T=1e-1, n_top_cells=50):
    adata_cell = sc.AnnData(adata.obsm[cell_key], obs=adata.obs)
    adata_cell.obs.loc[:, 'cell_gene'] = 'cell'
    adata_gene = sc.AnnData(adata.varm[gene_key], obs=adata.var)
    adata_gene.obs.loc[:, 'cell_gene'] = 'gene'
    adata_gene_metrics = compare_entities(adata_ref=adata_cell, adata_query=adata_gene, T=T, n_top_cells=n_top_cells)
    return adata_gene_metrics


def compare_entities(adata_ref,
                     adata_query,
                     n_top_cells=50,
                     T=1e-1,
                     ):
    """
    ref: utilized SIMBA's work
    SIMBA: single-cell embedding along with features, https://doi.org/10.1038/s41592-023-01899-8
    Embed a list of query datasets along with reference dataset into the same space
    Returns
    -------
    adata_all: `AnnData`
        Store #entities × #dimensions.
    """
    X_ref = adata_ref.X
    X_query = adata_query.X
    X_cmp = np.matmul(X_ref, X_query.T)
    adata_cmp = sc.AnnData(X=X_cmp,
                           obs=adata_ref.obs,
                           var=adata_query.obs)
    adata_cmp.layers['norm'] = X_cmp - np.log(np.exp(X_cmp).mean(axis=0)).reshape(1, -1)
    adata_cmp.layers['softmax'] = np.exp(X_cmp / T) / np.exp(X_cmp / T).sum(axis=0).reshape(1, -1)
    adata_cmp.var['max'] = np.clip(np.sort(adata_cmp.layers['norm'], axis=0)[-n_top_cells:, ],
                                   a_min=0, a_max=None).mean(axis=0)
    adata_cmp.var['std'] = np.std(X_cmp, axis=0, ddof=1)
    adata_cmp.var['gini'] = np.array([_gini(adata_cmp.layers['softmax'][:, i])
                                      for i in np.arange(X_cmp.shape[1])])
    adata_cmp.var['entropy'] = entropy(adata_cmp.layers['softmax'])
    return adata_cmp


def _gini(array):
    array = array.flatten().astype(float)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def clinical_feature_transform(adata, sample_label='sample', domain_label='domain', clinical_label=None):
    obs_df = adata.obs
    domain_proportions = obs_df.groupby(sample_label)[domain_label].value_counts(normalize=True).unstack(fill_value=0)
    sample_metadata = obs_df.drop_duplicates(sample_label).set_index(sample_label)
    new_obs = sample_metadata.reindex(domain_proportions.index)
    adata_proportions = sc.AnnData(X=domain_proportions.values, obs=new_obs,
                                   var=pd.DataFrame(index=domain_proportions.columns))
    adata_proportions.var_names = domain_proportions.columns
    adata_proportions.obs_names = domain_proportions.index
    return adata_proportions.to_df(), adata_proportions.obs.loc[:, clinical_label]


def gene_feature_transform(adata, cell_type_label='cell_type_sub', domain_label='domain', sample_label='sample'):
    obs_df = adata.obs[[cell_type_label, domain_label, sample_label]]
    grouped = obs_df.groupby([cell_type_label, domain_label, sample_label]).size().reset_index(name="count")
    total_counts = grouped.groupby([sample_label, domain_label])['count'].transform("sum")
    grouped["proportion"] = grouped["count"] / total_counts
    pivot_table = grouped.pivot(index=[sample_label, domain_label], columns=cell_type_label,
                                values='proportion').fillna(0)
    pivot_table.index = pivot_table.index.map(lambda x: f"{x[0]}_{x[1]}")
    obs = pd.DataFrame(index=pivot_table.index)  # 合并后的 obs
    var = pd.DataFrame(index=pivot_table.columns)  # cell_type_sub 作为 var
    adata_temp_concat = sc.AnnData(X=pivot_table.values, obs=obs, var=var)
    adata_temp_concat.obs.loc[:, domain_label] = [item.split('_')[-1] for item in adata_temp_concat.obs.index]
    return adata_temp_concat.to_df(), adata_temp_concat.obs.loc[:, domain_label]


def calculate_feature_differences(feature_matrix, label_matrix):
    """
    """
    # 获取特征名称和类别
    feature_names = feature_matrix.columns
    categories = label_matrix.unique()
    n_categories = len(categories)

    # 初始化存储矩阵
    p_value_matrix = pd.DataFrame(np.zeros((len(feature_names), n_categories)),
                                  index=feature_names,
                                  columns=categories)  # 直接使用类别名称作为列名
    lfc_matrix = pd.DataFrame(np.zeros((len(feature_names), n_categories)),
                              index=feature_names,
                              columns=categories)  # 直接使用类别名称作为列名

    # 对每个类别进行分析
    for category in categories:
        # 获取当前类别和其他类别的样本索引
        current_indices = label_matrix == category
        other_indices = label_matrix != category

        # 对每个特征进行计算
        for feature_name in feature_names:
            # 提取当前类别和其他类别的特征值
            current_values = feature_matrix.loc[current_indices, feature_name]
            other_values = feature_matrix.loc[other_indices, feature_name]

            # Mann-Whitney U 检验
            u_stat, p_value = mannwhitneyu(current_values, other_values, alternative='greater')
            p_value_matrix.loc[feature_name, category] = p_value

            # Log Fold Change (LFC)
            median_current = np.median(current_values)
            median_other = np.median(other_values)
            if median_other == 0:
                median_other = 1e-5  # 避免除零错误
            lfc = np.log2(median_current / median_other)
            lfc_matrix.loc[feature_name, category] = lfc

        # 对当前类别的 p值进行多重检验校正（FDR校正）
        rejected, corrected_p_values, _, _ = multipletests(p_value_matrix[category], method='fdr_bh')
        p_value_matrix[category] = corrected_p_values

    return p_value_matrix, lfc_matrix


def label_transfer(ref, query, rep='X_STAX', label='cell_type', n_neighbors=10, threshold=0.5):
    """
    Inputs:
    ref
        reference containing the projected representations and labels
    query
        query containing the projected representations
    rep
        keys of the embeddings in adata.obsm
    labels
        label name in adata.obs

    Return:
    transferred label: np.array

    Examples:
        adata_query.obs['celltype_transfer']=label_transfer(adata_ref, adata_query, rep='latent', label='celltype')

    """

    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    X_train = ref.obsm[rep]
    y_train = ref.obs.loc[:, label].to_numpy()
    X_test = query.obsm[rep]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    if threshold == 0:
        return y_pred, y_prob
    else:
        max_probs = np.max(y_prob, axis=1)
        unknown_indices = np.where(max_probs <= threshold)
        y_pred = y_pred.astype('object')
        y_pred[unknown_indices] = 'unknown'
        return y_pred, y_prob