import warnings
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import os
from imputation.impute import impute_
import GAAE
import argparse
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import time
import seaborn as sns 
from st_loading_utils import load_DLPFC, load_BC, load_mVC, load_mPFC, load_mHypothalamus, load_her2_tumor, load_mMAMP
warnings.filterwarnings("ignore")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./", help="root dir for input data")
    parser.add_argument('--gt_dir', type=str, default="./", help="root dir for data ground truth")
    parser.add_argument('--input_data', type=str, default="151673", help="input data section id")
    parser.add_argument('--impute_cluster_num', type=str, default="7", help="diff cluster numbers for imputation")
    parser.add_argument('--cluster_num', type=int, default=7, help="input data cluster number")
    parser.add_argument('--radius', type=int, default=150, help="input data radius")
    parser.add_argument("--de_candidates", type=str, default="200", help="candidate de list for imputation, separated by comma")
    parser.add_argument('--no_de', type=int, default=0, help='switch on/off DEG selection module')
    parser.add_argument("--use_mean", type=int, default=0, help="use mean value in de list or not")
    parser.add_argument("--impute_runs", type=int, default=2, help="time of runs for imputation")
    parser.add_argument("--runs", type=int, default=20, help="total runs for the data")
    parser.add_argument('--gt', type=int, default=1, help="ground truth for the input data")
    parser.add_argument('--use_hvgs', type=int, default=3000, help="select highly variable genes before training")
    parser.add_argument('--use_preprocessing', type=int, default=1, help='use preprocessed input or raw input')
    parser.add_argument('--save_fig', type=int, default=1, help='saving output visualization')
    parser.add_argument('--filter_nzr', type=float, default=0.15, help='suggested nzr threshold for filtering')
    parser.add_argument('--filter_num', type=int, default=None, help='suggested gene threshold for filtering')
    parser.add_argument('--de_nzr_min', type=float, default=0.299, help='suggested min nzr threshold after de selection')
    parser.add_argument('--de_nzr_max', type=float, default=0.399, help='suggested max nzr threshold after de selection')
    parser.add_argument('--use_gpu_id', type=str, default='0', help='use which GPU, only applies when you have multiple gpu')
    args = parser.parse_args()
    args.impute_cluster_num = args.impute_cluster_num.split(",")  # ["5", "6", "7"]
    root_d = args.data_dir
    if args.input_data != 'starmap':
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        de_top_k_list = DE_num_calc(args, adata)
        print("optimized de list = ", de_top_k_list)
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []
    if args.no_de == 1:
        print("none DE selection mode")
        print("section id = ", args.input_data)
        ari_doc_ini = []
        ari_doc_final = []

        for i in range(args.impute_runs):
            for cluster_n in args.impute_cluster_num:
                print("cluster_n = ", cluster_n)
                GAAE.Cal_Spatial_Net(adata, rad_cutoff=args.radius)
                GAAE.Stats_Spatial_Net(adata)

                ari_ini, adata_out = GAAE.train_STA(adata, n_epochs=1000,
                                                       num_cluster=int(cluster_n),
                                                       device_id=args.use_gpu_id)
                ari_doc_ini.append(ari_ini)
                adata_list.append(adata_out)

                if args.save_fig:
                    if args.input_data == 'starmap':
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(adata_out, color=["mclust_impute", "Ground Truth"],
                                      title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=95)
                        plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    if args.input_data in ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']:
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(adata_out, color=["mclust_impute", "Ground Truth"],
                                      title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=55)
                        plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    if args.input_data == 'sedr':
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(adata_out, color=["mclust_impute", "Ground Truth"],
                                      title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=150, color_map='viridis')
                        print(adata_out.uns)
                        print(adata_out.uns['mclust_impute_colors'])
                        adata_out.uns['mclust_impute_colors'] = ['#440154', '#481467', '#482576', '#453781', '#404688',
                                                             '#39558c', '#33638d', '#2d718e', '#287d8e', '#238a8d',
                                                             '#1f968b', '#20a386', '#29af7f', '#3dbc74', '#56c667',
                                                             '#75d054', '#95d840', '#bade28', '#dde318', '#fde725']
                        print(adata_out.uns['mclust_impute_colors'])
                        sc.pl.spatial(adata_out, color=["mclust_impute", "Ground Truth"],
                                      title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=150, color_map='viridis')
                        plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))

        print(ari_doc_ini)
        print(np.mean(ari_doc_ini))
        print(np.std(ari_doc_ini))

        print(ari_doc_final)
        print(np.mean(ari_doc_final))
        print(np.std(ari_doc_final))

        g_union = list(adata_list[0].var.index.values)
        imputed_ad = impute(args, adata_list, g_union, de_top_k_list)


        """result of imputed data"""
        GAAE.Cal_Spatial_Net(imputed_ad, rad_cutoff=args.radius)
        ari_doc_ini = []
        for i in range(args.runs):
            ari_ini, adata = GAAE.train_STA(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num,
                                               device_id=args.use_gpu_id)
            ari_doc_ini.append(ari_ini)
            if args.save_fig:
                if args.input_data == 'starmap':
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata, color=["mclust", "Ground Truth"],
                                  title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=95)
                    plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    downstream_analyses(args.input_data, adata, ari_ini, root_d, args.input_data + "_" + timestr)
                if args.input_data in ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata, color=["mclust", "Ground Truth"],
                                  title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=55)
                    plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    downstream_analyses(args.input_data, adata, ari_ini, root_d, args.input_data + "_" + timestr)
                if args.input_data == 'sedr':
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata, color=["mclust_impute", "Ground Truth"],
                                  title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=150)
                    # print(adata_out.uns['mclust_impute_colors'])
                    adata.uns['mclust_impute_colors'] = ['#440154', '#481467', '#482576', '#453781', '#404688',
                                                         '#39558c', '#33638d', '#2d718e', '#287d8e', '#238a8d',
                                                         '#1f968b', '#20a386', '#29af7f', '#3dbc74', '#56c667',
                                                         '#75d054', '#95d840', '#bade28', '#dde318', '#fde725']
                    # print(adata_out.uns['mclust_impute_colors'])
                    sc.pl.spatial(adata, color=["mclust_impute", "Ground Truth"],
                                  title=['Our method (ARI=%.2f)' % ari_ini, "Ground Truth"], spot_size=150)
                    plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    downstream_analyses(args.input_data, adata, ari_ini, root_d, args.input_data + "_" + timestr)
        print(ari_doc_ini)
        print(np.mean(ari_doc_ini))
        print(np.std(ari_doc_ini))
    else:
        # exit(-1)

        if args.use_mean:
            de_top_k_list = [np.mean(de_top_k_list)]

        de_list_epoch = []
        for de_ in de_top_k_list:
            print("DE topk = ", de_)
            print("section id = ", args.input_data)
            ari_doc_ini = []
            ari_doc_final = []

            for i in range(args.impute_runs):
                for cluster_n in args.impute_cluster_num:
                    print("cluster_n = ", cluster_n)
                    GAAE.get_kNN(adata, rad_cutoff=args.radius)

                    ari_ini, ari_final, de_list, adata_out = GAAE.train_STA_use_DE(adata, n_epochs=1000,
                                                                                      num_cluster=int(cluster_n),
                                                                                      dif_k=de_, device_id=args.use_gpu_id)
                    ari_doc_ini.append(ari_ini)
                    ari_doc_final.append(ari_final)
                    de_list_epoch.append(de_list)
                    adata_list.append(adata_out)
                    if args.save_fig:
                        if args.input_data == 'starmap':
                            timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.rcParams["figure.figsize"] = (3, 3)
                            sc.pl.spatial(adata_out, color=["mclust_impute", "Ground Truth"],
                                          title=['Our method (ARI=%.2f)' % ari_final, "Ground Truth"], spot_size=95)
                            plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                            downstream_analyses(args.input_data, adata, ari_ini, root_d,
                                                args.input_data + "_" + timestr)
                        if args.input_data in ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']:
                            timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.rcParams["figure.figsize"] = (3, 3)
                            sc.pl.spatial(adata_out, color=["mclust", "Ground Truth"],
                                          title=['Our method (ARI=%.2f)' % ari_final, "Ground Truth"], img_key=None)
                            plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                            downstream_analyses(args.input_data, adata, ari_ini, root_d,
                                                args.input_data + "_" + timestr)

            print(ari_doc_ini)
            print(np.mean(ari_doc_ini))
            print(np.std(ari_doc_ini))

            print(ari_doc_final)
            print(np.mean(ari_doc_final))
            print(np.std(ari_doc_final))
        g_union = set.union(*de_list_epoch)
        imputed_ad = impute(args, adata_list, g_union)

        """result of imputed data"""
        GAAE.Cal_Spatial_Net(imputed_ad, rad_cutoff=args.radius)
        ari_doc_ini = []
        ari_doc_final = []
        for i in range(args.runs):
            ari_ini, ari_final, de_list, adata_out = GAAE.train_STA_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)
            ari_doc_ini.append(ari_ini)
            ari_doc_final.append(ari_final)
            if args.save_fig:
                if args.input_data == 'starmap':
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['Our method (ARI=%.2f)' % ari_final, "Ground Truth"], spot_size=95)
                    plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    downstream_analyses(args.input_data, adata_out, ari_ini, root_d, args.input_data + "_" + timestr, imputed_=1)
                if args.input_data in ['151507', '151508', '151509', '151510', '151673', '151674', '151675', '151676']:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['Our method (ARI=%.2f)' % ari_final, "Ground Truth"], spot_size=55)
                    plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    downstream_analyses(args.input_data, adata_out, ari_ini, root_d, args.input_data + "_" + timestr, imputed_=1)
                if args.input_data == 'BC1':
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['Our method (ARI=%.2f)' % ari_final, "Ground Truth"], spot_size=150)
                    plt.savefig(os.path.join(root_d, args.input_data + '_viz', "_" + timestr + "_" + str(i) + ".pdf"))
                    downstream_analyses(args.input_data, adata_out, ari_ini, root_d, args.input_data + "_" + timestr, imputed_=1)
        print(ari_doc_ini)
        print(np.mean(ari_doc_ini))
        print(np.std(ari_doc_ini))

        print(ari_doc_final)
        print(np.mean(ari_doc_final))
        print(np.std(ari_doc_final))
