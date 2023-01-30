import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)

    node_idx = adata.X.shape[0]
    train_, test_ = train_test_split(range(node_idx), test_size=0.2, random_state=42)
    train_, val_ = train_test_split(train_, test_size=0.2, random_state=42)
    train_mask = index_to_mask(torch.tensor(train_), size=node_idx)
    val_mask = index_to_mask(torch.tensor(val_), size=node_idx)
    test_mask = index_to_mask(torch.tensor(test_), size=node_idx)

    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X), test_mask=test_mask,
                    train_mask=train_mask, val_mask=val_mask)  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()), test_mask=test_mask,
                    train_mask=train_mask, val_mask=val_mask)  # .todense()
    return data


def get_kNN(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    Construct the kNN graph.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    if adata.obsm['spatial'].shape[1] == 2:
        coor.columns = ['imagerow', 'imagecol']
    else:
        coor.columns = ['imagerow', 'imagecol', 'imagez']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


# def Stats_Spatial_Net(adata):
#     import matplotlib.pyplot as plt
#     Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
#     Mean_edge = Num_edge/adata.shape[0]
#     plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
#     plot_df = plot_df/adata.shape[0]
#     fig, ax = plt.subplots(figsize=[3,2])
#     plt.ylabel('Percentage')
#     plt.xlabel('')
#     plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
#     ax.bar(plot_df.index, plot_df)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2022, save_obs='mclust_impute'):
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

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[save_obs] = mclust_res
    adata.obs[save_obs] = adata.obs[save_obs].astype('int')
    adata.obs[save_obs] = adata.obs[save_obs].astype('category')
    return adata
