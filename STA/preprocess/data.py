#!/usr/bin/env python
import numpy as np
import pandas as pd
import copy
import torch
import scanpy as sc
import scipy.sparse as sp
import cv2
import scipy
import sklearn
from typing import Any
from scipy.sparse import csr, csr_matrix
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from anndata import AnnData
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors


def minmax_scale(adata):
    data_min = adata.X.min(axis=1).toarray().flatten()
    data_max = adata.X.max(axis=1).toarray().flatten()
    scale_factor = data_max - data_min
    scale_factor[scale_factor == 0] = 1
    row_indices = adata.X.nonzero()[0]
    adata.X.data = 10 * (adata.X.data - data_min[row_indices]) / scale_factor[row_indices]
    return adata


def process_adata(data_list,
                  label='batch',
                  keys=None,
                  join='inner',
                  n_top_features=None,
                  spatial_key='spatial',
                  coordinate_dimension=2,
                  filter_hk_genes=False,
                  norm=True,
                  target_sum=1e4,
                  scale='log',
                  ):
    """
    :param data_list:
    :param label:
    :param keys:
    :param join:
    :param n_top_features:
    :param spatial_key:
    :param coordinate_dimension:
    :param filter_hk_genes:
    :param norm:
    :param target_sum:
    :param scale:
    :return:
    """
    if keys is None:
        keys = list(range(len(data_list)))
        keys = ['batch' + key for key in keys]
    for i, temp in enumerate(data_list):
        temp.obs_names_make_unique()
        temp.var_names_make_unique()
        # reset index
        temp.obs = temp.obs.reset_index(drop=True)
        temp.obs.loc[:, 'original_index'] = temp.obs.index
        temp.obs.index = [str(keys[i]) + '_' + str(j) for j in temp.obs.index]
        if not isinstance(temp.X, csr.csr_matrix):
            data_list[i].X = csr_matrix(temp.X)
        if filter_hk_genes:
            temp = temp[:, [gene for gene in temp.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
        if 'spot_quality' not in temp.obs.columns:
            temp.obs.loc[:, 'spot_quality'] = 'real'
        if spatial_key in temp.obsm.keys():
            pass
            # coordinates = temp.obsm['spatial']
            # max_value = np.max(coordinates)
            # normalized_coordinates = coordinates / max_value
            # temp.obsm['spatial'] = normalized_coordinates + i
        else:
            num_points = len(temp)
            x = np.random.uniform(low=0.5, high=0.6, size=num_points)
            y = np.random.uniform(low=0.5, high=0.6, size=num_points)
            pseudo_coordinates = np.vstack((x, y)).T
            if coordinate_dimension == 3:  # 3d
                z = np.random.uniform(low=0.5, high=0.6, size=num_points)
                pseudo_coordinates = np.vstack((x, y, z)).T
            temp.obsm['spatial'] = pseudo_coordinates
            print(f'warning! spatial is not in {i}th adata.obsm')

    adata = sc.concat([*data_list], label=label, keys=keys, join=join)
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    # counts
    adata.layers['counts'] = copy.deepcopy(adata.X)
    # choose real to select hvg
    if n_top_features:
        temp_real = adata[adata.obs.loc[:, 'spot_quality'] == 'real', :]
        sc.pp.highly_variable_genes(temp_real, n_top_genes=n_top_features, batch_key=label, flavor='seurat_v3')
        adata.var = temp_real.var
    if norm:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if scale == 'log':
        sc.pp.log1p(adata)
    elif scale == 'z':
        sc.pp.scale(adata, zero_center=False, max_value=10)
    elif scale == 'minmax':
        temp = sc.concat([minmax_scale(data.copy()) for data in data_list])
        adata.X = temp.X
    if norm and scale:
        adata.layers['norm_log'] = copy.deepcopy(adata.X)
    adata.raw = adata
    # hvg
    if n_top_features:
        adata = adata[:, adata.var.highly_variable]
    if not isinstance(adata.X, csr.csr_matrix):
        adata.X = csr_matrix(adata.X)
    return adata


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
        adata=None,
        n_components=50,
        use_highly_variable=None,
        random_state=42,
):
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)  # TODO difference in win10 and linux ubuntu 22.04
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, random_state=random_state)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


class SubImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 adata,
                 adata_name=None,
                 image_use='hires',
                 coordinate_key='spatial',
                 row_width=20,
                 col_width=20,
                 ):
        if image_use not in ['hires', 'lowres']:
            print("resolution should be 'hires' or 'lowres'")
            raise TypeError
        if adata_name is None:
            print("adata_name should not be None")
            raise TypeError
        self.adata = adata
        self.image = adata.uns['spatial'][adata_name]['images'][image_use]
        self.scalefactors = adata.uns['spatial'][adata_name]['scalefactors'][f'tissue_{image_use}_scalef']
        self.coordinate = adata.obsm[coordinate_key]
        self.row_width = row_width
        self.col_width = col_width

    def __getitem__(self, idx):
        # adata.obsm['spatial'][:, 0] is the col and adata.obsm['spatial'][:, 1] is row
        row, col = int(self.coordinate[idx][1] * self.scalefactors), \
                   int(self.coordinate[idx][0] * self.scalefactors)
        sub_image = self.image[(row - self.row_width):(row + self.row_width),
                    (col - self.col_width):(col + self.col_width)]
        if row != 112 or col != 112:
            resized_image = cv2.resize(sub_image, (224, 224))
        else:
            resized_image = sub_image
        # resized_image = sub_image if row == 112 and col == 112 else cv2.resize(sub_image, (224, 224))  # TODO
        return resized_image, torch.tensor(resized_image).permute(2, 0, 1).float(), idx

    def __len__(self):
        return len(self.adata)


def generate_image_embedding(adata,
                             adata_name,
                             image_use='hires',
                             coordinate_key='spatial',
                             row_width=112,
                             col_width=112,
                             batch_size=1,
                             device='cpu',
                             pca_components=50,
                             ):
    import timm
    from tqdm import tqdm
    from sklearn.decomposition import PCA

    dataset = SubImageDataset(adata, adata_name=adata_name, image_use=image_use, coordinate_key=coordinate_key,
                              row_width=row_width, col_width=col_width)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    print('load vit model')
    model.load_state_dict(torch.load('./model_UNI/pytorch_model.bin', map_location="cpu"), strict=True)
    model.eval()
    device = device
    model.to(device)
    output = []
    print('start infer')
    with torch.inference_mode():
        for sub_image, inputs, idx in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            hidden = model(inputs)
            output.append(hidden.detach().cpu().numpy())
    model.to('cpu')
    torch.cuda.empty_cache()
    del model
    output = np.concatenate(output, axis=0)
    # PCA
    n_components = pca_components
    pca = PCA(n_components=n_components)
    reduced_output = pca.fit_transform(output)
    adata.obsm['image_embedding'] = output
    adata.obsm['image_embedding_PCA'] = reduced_output

    return adata


def cal_spatial_net(adata: AnnData,
                    cutoff: [int, float] = None,
                    max_neigh: int = 100,
                    metric='euclidean',
                    model: str = 'Radius',
                    spatial_key: str = 'spatial',
                    verbose: bool = True) -> tuple[AnnData, Any]:
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata : AnnData
        AnnData object of scanpy package.
    cutoff : float, optional
        Radius cutoff when model='Radius'.
    max_neigh: int
        The max neighbors of KNN
    metric: str
        euclidean or cosine
    model : str
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less
        than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    spatial_key: str
        The key of spatial coordinates in adata.obsm
    verbose : bool
        Whether to print progress messages.

    """
    assert model in ['Radius', 'KNN'], "Model must be either 'Radius' or 'KNN'"
    from sklearn.neighbors import NearestNeighbors

    if verbose:
        print('------Calculating spatial graph...')

    coor = pd.DataFrame(adata.obsm[spatial_key], index=adata.obs.index)
    nbrs = NearestNeighbors(n_neighbors=max_neigh + 1, algorithm='ball_tree', metric=metric).fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    if model == 'KNN':
        indices = indices[:, 1:cutoff + 1]
        distances = distances[:, 1:cutoff + 1]
    else:  # model == 'Radius'
        mask = distances[:, 1:] < cutoff
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        indices[~mask] = -1
        distances[~mask] = -1

    valid_mask = indices.flatten() != -1
    KNN_df = pd.DataFrame({
        'Cell1': np.repeat(np.arange(coor.shape[0]), indices.shape[1])[valid_mask],
        'Cell2': indices.flatten()[valid_mask],
        'Distance': distances.flatten()[valid_mask]
    })

    id_cell_trans = dict(enumerate(coor.index))
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)

    if model == 'Radius':
        Spatial_Net = KNN_df[KNN_df['Distance'] < cutoff]
    else:
        Spatial_Net = KNN_df

    if verbose:
        print(f'The graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{Spatial_Net.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')

    # adata.uns['Spatial_Net'] = Spatial_Net

    # Create adjacency matrix
    cell_indices = pd.Series(range(adata.n_obs), index=adata.obs.index)
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cell_indices)
    G_df['Cell2'] = G_df['Cell2'].map(cell_indices)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # Add self-loops

    adata.uns['adj'] = G

    return adata, G


def process_graph(adata, data_list, cutoff_list=None, max_neigh=100, spatial_key='spatial', metric='euclidean',
                  model='Radius', verbose=True):
    if adata.obsm[spatial_key].shape[1] == 2:  # 2d, construct graph in each data_list
        from scipy.sparse import block_diag
        print('cal spatial net in data_list')
        data_list = [
            cal_spatial_net(adata_temp, cutoff=cutoff_list[i], max_neigh=max_neigh, spatial_key=spatial_key,
                            metric=metric, model=model, verbose=verbose)[0]
            for i, adata_temp in enumerate(data_list)]
        adj_list = [item.uns['adj'] for item in data_list]
        adj_concat = block_diag(adj_list)
        adata.uns['adj'] = adj_concat
    elif adata.obsm[spatial_key].shape[1] >= 3:  # 3d, construct graph in one adata
        print('cal spatial net in one adata')
        adata, adj_concat = cal_spatial_net(adata, cutoff=cutoff_list[0], max_neigh=max_neigh, spatial_key=spatial_key,
                                            metric=metric, model=model, verbose=verbose)
    else:
        raise NotImplementedError
    return adata, adj_concat


def make_fake_spot(adata, spatial_key='spatial', x_density=50, y_density=50, z_slices=2, max_neigh=10):
    from sklearn.neighbors import KDTree
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if 'spot_quality' not in adata.obs.columns:
        adata.obs.loc[:, 'spot_quality'] = 'real'
    # if spatial_key in ['spatial', 'spatial_2d']:
    if adata.obsm[spatial_key].shape[1] == 2:
        X_min, X_max = adata.obsm[spatial_key][:, 0].min(), adata.obsm[spatial_key][:, 0].max()
        Y_min, Y_max = adata.obsm[spatial_key][:, 1].min(), adata.obsm[spatial_key][:, 1].max()
        X = np.round(np.linspace(X_min, X_max, x_density), 5)
        Y = np.round(np.linspace(Y_min, Y_max, y_density), 5)
        locations = np.array([[x, y] for y in Y for x in X]).astype(np.float32)
        adata_fake = sc.AnnData(np.zeros([len(locations), len(adata.var)]), var=adata.var)
        adata_fake.obsm[spatial_key] = locations
        adata_fake.obs.loc[:, 'spot_quality'] = 'fake'
        adata_fake.obs.loc[:, 'Region'] = 'fake'
        # 2，选择部分点作为可用点
        # 构建KDTree
        tree = KDTree(adata_fake.obsm[spatial_key])
        distances, indices = tree.query(adata.obsm[spatial_key], k=max_neigh)
    # elif spatial_key == 'spatial_3d':
    elif adata.obsm[spatial_key].shape[1] == 3:
        # 1，产生一定密度的点
        X_min, X_max = adata.obsm[spatial_key][:, 0].min(), adata.obsm[spatial_key][:, 0].max()
        Y_min, Y_max = adata.obsm[spatial_key][:, 1].min(), adata.obsm[spatial_key][:, 1].max()
        X = np.round(np.linspace(X_min, X_max, x_density), 5)
        Y = np.round(np.linspace(Y_min, Y_max, y_density), 5)
        Z_min, Z_max = adata.obsm[spatial_key][:, 2].min(), adata.obsm[spatial_key][:, 2].max()
        # num_z_slice = len(np.unique(adata.obsm[spatial_key][:, 2]))
        Z = np.round(np.linspace(Z_min, Z_max, z_slices + 2), 5)
        Z = Z[1: -1]  # discard the first and the last slice
        locations = np.array([[x, y, z] for z in Z for y in Y for x in X]).astype(np.float32)
        # adata_fake = sc.AnnData(np.zeros([len(locations), len(adata.var)]), var=adata.var)
        adata_fake = sc.AnnData(csr_matrix((len(locations), len(adata.var))), var=adata.var)
        adata_fake.obsm[spatial_key] = locations
        adata_fake.obs.loc[:, 'spot_quality'] = 'fake'
        adata_fake.obs.loc[:, 'Region'] = 'fake'
        # 2，选择部分点作为可用点
        # 构建KDTree
        tree = KDTree(adata_fake.obsm[spatial_key])
        distances, indices = tree.query(adata.obsm[spatial_key], k=max_neigh)
    else:
        print('correct dimension of spatial coordinates')
        raise NotImplementedError
    new_indices = np.unique(indices.reshape(-1, 1).squeeze())
    adata_fake_sub = adata_fake[new_indices, :]
    adata_new = sc.concat([adata, adata_fake_sub], join='outer')
    if adata_new.obsm[spatial_key].shape[1] == 3:
        adata_new.obs.loc[:, 'slices'] = adata_new.obsm[spatial_key][:, 2].astype(str)
    return adata_new


def make_fake_spot_v2(adata, spatial_key='spatial', step=None, max_neigh=10):
    from sklearn.neighbors import KDTree
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if 'spot_quality' not in adata.obs.columns:
        adata.obs.loc[:, 'spot_quality'] = 'real'
    # if spatial_key in ['spatial', 'spatial_2d']:
    if adata.obsm[spatial_key].shape[1] == 2:
        X_min, X_max = adata.obsm[spatial_key][:, 0].min(), adata.obsm[spatial_key][:, 0].max()
        Y_min, Y_max = adata.obsm[spatial_key][:, 1].min(), adata.obsm[spatial_key][:, 1].max()
        X = np.round(np.arange(X_min - step, X_max + step, step), 5)
        Y = np.round(np.arange(Y_min - step, Y_max + step, step), 5)
        locations = np.array([[x, y] for y in Y for x in X]).astype(np.float32)
        adata_fake = sc.AnnData(np.zeros([len(locations), len(adata.var)]), var=adata.var)
        adata_fake.obsm[spatial_key] = locations
        adata_fake.obs.loc[:, 'spot_quality'] = 'fake'
        adata_fake.obs.loc[:, 'Region'] = 'fake'
        # 2，选择部分点作为可用点
        # 构建KDTree
        tree = KDTree(adata_fake.obsm[spatial_key])
        distances, indices = tree.query(adata.obsm[spatial_key], k=max_neigh)
    # elif spatial_key == 'spatial_3d':
    elif adata.obsm[spatial_key].shape[1] == 3:
        # 1，产生一定密度的点
        X_min, X_max = adata.obsm[spatial_key][:, 0].min(), adata.obsm[spatial_key][:, 0].max()
        Y_min, Y_max = adata.obsm[spatial_key][:, 1].min(), adata.obsm[spatial_key][:, 1].max()
        Z_min, Z_max = adata.obsm[spatial_key][:, 2].min(), adata.obsm[spatial_key][:, 2].max()
        X = np.round(np.arange(X_min - step, X_max + step, step), 5)
        Y = np.round(np.arange(Y_min - step, Y_max + step, step), 5)
        Z = np.round(np.arange(Z_min - step, Z_max + step, step), 5)
        X = X[1: -1]  # discard the first and the last slice
        Y = Y[1: -1]  # discard the first and the last slice
        Z = Z[1: -1]  # discard the first and the last slice
        locations = np.array([[x, y, z] for z in Z for y in Y for x in X]).astype(np.float32)
        # adata_fake = sc.AnnData(np.zeros([len(locations), len(adata.var)]), var=adata.var)
        adata_fake = sc.AnnData(csr_matrix((len(locations), len(adata.var))), var=adata.var)
        adata_fake.obsm[spatial_key] = locations
        adata_fake.obs.loc[:, 'spot_quality'] = 'fake'
        adata_fake.obs.loc[:, 'Region'] = 'fake'
        # 2，选择部分点作为可用点
        # 构建KDTree
        tree = KDTree(adata_fake.obsm[spatial_key])
        distances, indices = tree.query(adata.obsm[spatial_key], k=max_neigh)
    else:
        print('correct dimension of spatial coordinates')
        raise NotImplementedError
    new_indices = np.unique(indices.reshape(-1, 1).squeeze())
    adata_fake_sub = adata_fake[new_indices, :]
    adata_new = sc.concat([adata, adata_fake_sub], join='outer')
    if adata_new.obsm[spatial_key].shape[1] == 3:
        adata_new.obs.loc[:, 'slices'] = adata_new.obsm[spatial_key][:, 2].astype(str)
    return adata_new


def remove_close_nodes_knn(coords, threshold):
    coords = np.array(coords)
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    mask = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if mask[i]:
            neighbor_idx = indices[i][1]
            if distances[i][1] < threshold:
                mask[neighbor_idx] = False

    return coords[mask]


def make_fake_spot_visium(adata, cutoff=None, metric='euclidean', model='Radius', spatial_key='spatial', verbose=True):
    from sklearn.neighbors import KDTree
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if 'spot_quality' not in adata.obs.columns:
        adata.obs.loc[:, 'spot_quality'] = 'real'
    # if spatial_key in ['spatial', 'spatial_2d']:
    if adata.obsm[spatial_key].shape[1] == 2:
        cal_spatial_net(adata, cutoff=cutoff, metric=metric, model=model, spatial_key=spatial_key, verbose=verbose)
        adj_coo = adata.uns['adj'].tocoo()
        edge_array = np.column_stack((adj_coo.row, adj_coo.col))
        coordinates = adata.obsm[spatial_key]
        location_fake = 0.5 * (coordinates[edge_array[:, 0]] + coordinates[edge_array[:, 1]])
        location_fake = np.unique(location_fake, axis=0)
        location_fake = remove_close_nodes_knn(location_fake, threshold=1)
        # remove real in fake
        # location_real = adata.obsm[spatial_key]
        # mask = ~np.any(np.all(location_fake[:, None] == location_real, axis=2), axis=1)
        # location_fake = location_fake[mask]
        adata_fake = sc.AnnData(np.zeros([len(location_fake), len(adata.var)]), var=adata.var)
        adata_fake.obsm[spatial_key] = location_fake
        adata_fake.obs.loc[:, 'spot_quality'] = 'fake'
        # adata_fake.obs.loc[:, 'Region'] = 'fake'
        # 2，move overlap, TODO in the future
    # elif spatial_key == 'spatial_3d':
    elif adata.obsm[spatial_key].shape[1] == 3:
        print('TODO in the future')
        raise NotImplementedError
    else:
        print('correct dimension of spatial coordinates')
        raise NotImplementedError
    adata_fake_sub = adata_fake
    adata_new = sc.concat([adata, adata_fake_sub], join='outer')
    if adata_new.obsm[spatial_key].shape[1] == 3:
        adata_new.obs.loc[:, 'slices'] = adata_new.obsm[spatial_key][:, 2].astype(str)
    del adata.uns['adj']
    return adata_new


def cell_type_onehot_encoding(adata, cell_type_key='cell_type'):
    categories = adata.obs.loc[:, cell_type_key].to_numpy()
    encoder = OneHotEncoder()
    categories_reshaped = categories.reshape(-1, 1)
    one_hot_matrix = encoder.fit_transform(categories_reshaped).todense()
    one_hot_columns = encoder.categories_[0]
    adata_new = sc.AnnData(one_hot_matrix, obs=adata.obs, obsm=adata.obsm)
    adata_new.var.index = one_hot_columns
    return adata_new


def define_plane_from_three_points(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    print('v1, v2: ', v1, v2)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    print('normal: ', normal)
    d = -np.dot(normal, p1)
    return normal, d


def slice_point_cloud(points, plane_normal, plane_d, thickness=0.1):
    distances = (points @ plane_normal + plane_d) / np.linalg.norm(plane_normal)
    mask = np.abs(distances) <= thickness/2
    sliced_points = points[mask]
    distances_sliced = distances[mask]
    indices = np.where(mask)[0]
    return sliced_points, distances_sliced, indices


def adata_slicedby_points(adata, key_points, spatial_key='spatial', thickness=0.2):
    cloud_points = adata.obsm[spatial_key]
    plane_normal, plane_d = define_plane_from_three_points(key_points[0], key_points[1], key_points[2])
    sliced_points, distances, indices = slice_point_cloud(cloud_points, plane_normal, plane_d, thickness=thickness)
    print(f"Plane Equation: {plane_normal[0]:.2f}x + {plane_normal[1]:.2f}y + {plane_normal[2]:.2f}z + {plane_d:.2f} = 0")
    print(f"All points: {len(cloud_points)} Sub points: {len(sliced_points)} ")
    return adata[indices, :].copy()


def compute_global_avg_knn_distance(adata, spatial_key='spatial', k=2):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(adata.obsm[spatial_key])  # k+1因为包含自己
    distances, indices = nbrs.kneighbors(adata.obsm[spatial_key])
    individual_avgs = np.mean(distances[:, 1:], axis=1)  # 从第二列开始，因为第一列是自己
    global_avg = np.mean(individual_avgs)

    return global_avg, individual_avgs


# def permutation(feature):
#     # fix_seed(FLAGS.random_seed)
#     # node permutrated
#     # # 1
#     ids = np.arange(feature.shape[0])
#     ids = np.random.permutation(ids)
#     feature_permutated = feature[ids]
#     # 2
#     # perm = torch.randperm(feature.shape[0])
#     # feature_permutated = feature[perm]
#     # feature permutrated
#     # # 1
#     # feature_id = np.arange(feature.shape[1])
#     # feature_id = np.random.permutation(feature_id)
#     # feature_permutated = feature_permutated[:, feature_id]
#     # 2
#     # perm = torch.randperm(feature.shape[1])
#     # feature_permutated = feature_permutated[:, perm]
#     return feature_permutated
