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


def downstream_analyses(section_id_, adata_, ari, save_folder_, save_path, imputed_=0):
    # umap
    adata = adata_
    section_id = section_id_
    ARI = ari

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pp.neighbors(adata, use_rep='ade')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["mclust_impute", "Ground Truth"], title=['ADePute (ARI=%.2f)' % ARI, "Ground Truth"])
    # plt.savefig("./151675_viz/151675_ADePute_0103_umap" + str(i) + ".pdf")
    plt.savefig(os.path.join(save_folder_, save_path) + "_umap.pdf")

    # Spatial trajectory inference (PAGA)
    used_adata = adata[adata.obs['Ground Truth'] != 'nan',]
    sc.tl.paga(used_adata, groups='Ground Truth')

    plt.rcParams["figure.figsize"] = (4, 3)
    sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                       title=section_id + '_ADePute', legend_fontoutline=2, show=False)
    plt.savefig(os.path.join(save_folder_, save_path) + "_paga.pdf")

    if imputed_:
        plot_gene = 'ATP2B4'
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_' + plot_gene, vmax='p99')
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='ADePute_' + plot_gene, vmax='p99')
        plt.savefig(os.path.join(save_folder_, save_path) + plot_gene + ".pdf")

        plot_gene = 'RASGRF2'
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_' + plot_gene, vmax='p99')
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='ADePute_' + plot_gene,
                      vmax='p99')
        plt.savefig(os.path.join(save_folder_, save_path) + plot_gene + ".pdf")

        plot_gene = 'LAMP5'
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_' + plot_gene, vmax='p99')
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='ADePute_' + plot_gene,
                      vmax='p99')
        plt.savefig(os.path.join(save_folder_, save_path) + plot_gene + ".pdf")

        plot_gene = 'NEFH'
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_' + plot_gene, vmax='p99')
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='ADePute_' + plot_gene,
                      vmax='p99')
        plt.savefig(os.path.join(save_folder_, save_path) + plot_gene + ".pdf")

        plot_gene = 'NTNG2'
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_' + plot_gene, vmax='p99')
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='ADePute_' + plot_gene,
                      vmax='p99')
        plt.savefig(os.path.join(save_folder_, save_path) + plot_gene + ".pdf")

        plot_gene = 'B3GALT2'
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_' + plot_gene, vmax='p99')
        sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='ADePute_' + plot_gene,
                      vmax='p99')
        plt.savefig(os.path.join(save_folder_, save_path) + plot_gene + ".pdf")


def filter_num_calc(args_, comp_):
    if comp_ is not None:
        return comp_
    print("optimizing minimum filter number")
    input_dir = os.path.join(args_.data_dir, args_.input_data)
    adata = sc.read_visium(path=input_dir, count_file=args_.input_data + '_filtered_feature_bc_matrix.h5')

    adata.var_names_make_unique()

    for temp_count in range(5, 150):
        sc.pp.filter_genes(adata, min_counts=temp_count)
        if np.count_nonzero(adata.X.todense())/(adata.X.shape[0]*adata.X.shape[1]) > args_.filter_nzr:
            return temp_count
    return 150


def initialize(args_, gene_min_count):
    print("initializing spatial transcriptomic data")
    data_path = args_.data_dir
    data_name = args_.input_data
    if data_name in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']:
        adata_ = load_DLPFC(root_dir=data_path, section_id=data_name)
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
        if args_.use_preprocessing:
            # Normalization
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            if args_.use_hvgs != 0:
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=args_.use_hvgs)
            sc.pp.normalize_total(adata_, target_sum=1e4)
            sc.pp.log1p(adata_)
        else:
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
    elif data_name in ['section1']:
        adata_ = load_BC(root_dir=data_path, section_id=data_name)
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
        if args_.use_preprocessing:
            # Normalization
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            if args_.use_hvgs != 0:
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=args_.use_hvgs)
            sc.pp.normalize_total(adata_, target_sum=1e4)
            sc.pp.log1p(adata_)
        else:
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
    elif data_name in ['MA']:
        adata_ = load_mMAMP(root_dir=data_path, section_id=data_name)
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
        if args_.use_preprocessing:
            # Normalization
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            if args_.use_hvgs != 0:
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=args_.use_hvgs)
            sc.pp.normalize_total(adata_, target_sum=1e4)
            sc.pp.log1p(adata_)
        else:
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
    elif data_name in ['STARmap_20180505_BY3_1k.h5ad']:
        adata_ = load_mVC(root_dir=data_path, section_id=data_name)
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
        # Normalization
        # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)
    elif data_name in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control']:
        adata_ = load_mPFC(root_dir=data_path, section_id=data_name)
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)
    elif data_name in ['-0.04', '-0.09', '-0.14', '-0.19', '-0.24', '-0.29']:
        adata_ = load_mHypothalamus(root_dir=data_path, section_id=data_name)  # special, seems to have been preprocessed
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
    elif data_name in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']:
        adata_ = load_her2_tumor(root_dir=data_path, section_id=data_name)
        adata_.obs['Ground Truth'] = adata_.obs['original_clusters']
        adata_ori_ = adata_
        if args_.use_preprocessing:
            # Normalization
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
            if args_.use_hvgs != 0:
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=args_.use_hvgs)
            sc.pp.normalize_total(adata_, target_sum=1e4)
            sc.pp.log1p(adata_)
        else:
            sc.pp.filter_genes(adata_, min_counts=gene_min_count)
    else:
        print("exception in data name")
        exit(-1)
    return adata_, adata_ori_


def DE_num_calc(args_, ad):
    print("optimizing top DEs before imputation")
    out_list = []
    if ad.X.shape[0] < 3000:
        interval_ = 25
        max_ = 200
    else:
        interval_ = 50
        max_ = 500
    for de_ in range(interval_, max_, interval_):
        print("DE topk = ", de_)
        print("section id = ", args_.input_data)
        nzr_list = []
        for i in range(3):
            GAAE.get_kNN(ad, rad_cutoff=args_.radius)
            # GAAE.Stats_Spatial_Net(ad)
            nzr = GAAE.DE_nzr(ad, n_epochs=1000, num_cluster=args_.cluster_num, dif_k=de_, device_id=args_.use_gpu_id)
            nzr_list.append(nzr)
        if args_.de_nzr_min <= np.mean(nzr_list) <= args_.de_nzr_max:
            out_list.append(de_)
        if args_.de_nzr_max <= np.mean(nzr_list):
            break
        print(de_, nzr)
    return out_list


def impute(args_, adata_list_, g_union, de_top_k_list_):
    m_list = []
    for adata_ in adata_list_:
        barcode_list_ = adata_.obs.index.values.tolist()
        pred_label_list_ = adata_.obs["mclust_impute"].tolist()
        # print(pred_label_list_[:30])
        pred_label_list_ = [x - 1 for x in pred_label_list_]
        g_union = g_union.intersection(set(adata_.var.index.tolist()))
        exp_m_ = adata_[:, list(g_union)].X.toarray()  # spots by genes
        # exp_m_ = adata_.X.toarray()
        m_list.append(impute_(args_.cluster_num, exp_m_, pred_label_list_, barcode_list_))
        # print(adata_.obs.index)
        # print(adata_.X.shape)
        # print(g_union)
        # try:
        #     exp_m_ = adata_[:, list(g_union)].X.toarray()  # spots by genes
        #     # exp_m_ = adata_.X.toarray()
        #     m_list.append(impute_(args_.cluster_num, exp_m_, pred_label_list_, barcode_list_))
        # except:
        #     continue

    total_m = args_.impute_runs * len(de_top_k_list_)
    avg_m = np.zeros_like(m_list[0])
    for m in m_list:
        avg_m += m
    # final inputed matrix
    avg_m /= total_m
    # avg_m is the final output

    # h5ad_filename = os.path.join(root_d_, 'final_imputed', args_.input_data + 'de_imputed_' + str(total_m) + 'X.h5ad')
    # print("h5ad filename = ", h5ad_filename)
    adata_list_[0] = adata_list_[0][:, list(g_union)]
    adata_list_[0].X = sparse.csr_matrix(avg_m)

    # adata_list_[0].write_h5ad(h5ad_filename)
    # print("h5ad file successfully written")
    return adata_list_[0]