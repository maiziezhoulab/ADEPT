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
from GAAE.utils import impute, DE_num_calc, initialize, filter_num_calc, downstream_analyses 
warnings.filterwarnings("ignore")


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
parser.add_argument('--use_gpu_id', type=str, default='1', help='use which GPU, only applies when you have multiple gpu')
args = parser.parse_args()
args.impute_cluster_num = args.impute_cluster_num.split(",")  # ["5", "6", "7"]


# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + args.use_gpu_id)
# params.device = device


iters=20
# """DLPFC"""
# # the number of clusters
# setting_combinations = [[7, '151507'], [7, '151508'], [7, '151509'], [7, '151510'], [5, '151669'], [5, '151670'], [5, '151671'], [5, '151672'], [7, '151673'], [7, '151674'], [7, '151675'], [7, '151676']]
# # setting_combinations = [[7, '151674'], [7, '151675'], [7, '151676']]
# for setting_combi in setting_combinations:
#     args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12'
#     args.de_candidates = "None"
#     dataset = args.input_data = setting_combi[1]
#     args.cluster_num = setting_combi[0]
#     args.impute_cluster_num = [setting_combi[0]]
#     args.radius = 150
#     args.use_preprocessing = 1
#     args.use_hvgs = 0
#     aris = []
    
#     if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
#         filter_num = filter_num_calc(args, args.filter_num)
#         print("optimized filter number = ", filter_num)
#     else:
#         filter_num = 0
#     adata, adata_ori = initialize(args, filter_num)
#     if args.de_candidates == "None":
#         if os.path.exists('./cache/DLPFC' + dataset + '.txt'):
#             with open('./cache/DLPFC' + dataset + '.txt', 'r') as fp:
#                 line = fp.readlines()[0]
#                 split_ = line.strip().split(",")
#                 de_top_k_list = []
#                 for e in split_:
#                     de_top_k_list.append(int(e))
#             print("previously cached de list = ", de_top_k_list)
#         else:
#             de_top_k_list = DE_num_calc(args, adata)
#             print("optimized de list = ", de_top_k_list)
#             with open('./cache/DLPFC' + dataset + '.txt', 'a+') as fp:
#                 # fp.write('de list: ')
#                 fp.write(','.join([str(i) for i in de_top_k_list]))
#                 # fp.write('\n')
#     else:
#         split_ = args.de_candidates.strip().split(",")
#         de_top_k_list = []
#         for e in split_:
#             de_top_k_list.append(int(e))
#         print("manually defined de list = ", de_top_k_list)
#     adata_list = []
    

#     for iter_ in range(iters):
#         de_list_epoch = []
#         for de_ in de_top_k_list:
#             for cluster_n in args.impute_cluster_num:
#                 print("cluster_n = ", cluster_n)
#                 GAAE.get_kNN(adata, rad_cutoff=args.radius)

#                 ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
#                                                                                num_cluster=int(cluster_n),
#                                                                                dif_k=de_, device_id=args.use_gpu_id)
#                 de_list_epoch.append(de_list)
#                 adata_list.append(adata_out)
#         g_union = set.union(*de_list_epoch)
#         imputed_ad = impute(args, adata_list, g_union, de_top_k_list)

#         """result of imputed data"""
#         GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
#         ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

#         print('Dataset:', dataset)
#         print('ARI:', ARI)
#         aris.append(ARI)
#         print(aris)
#     print('Dataset:', dataset)
#     print(aris)
#     print(np.mean(aris))
#     with open('adept_aris.txt', 'a+') as fp:
#         fp.write('DLPFC' + dataset + ' ')
#         fp.write(' '.join([str(i) for i in aris]))
#         fp.write('\n')


"""BC"""
# the number of clusters
setting_combinations = [[20, 'section1']]
for setting_combi in setting_combinations:
    args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/BC'
    args.de_candidates = "None"
    dataset = args.input_data = setting_combi[1]
    args.cluster_num = setting_combi[0]
    args.impute_cluster_num = [setting_combi[0]]
    args.radius = 450
    args.use_preprocessing = 1
    args.use_hvgs = 0
    aris = []
    
    if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        if os.path.exists('./cache/BC' + dataset + '.txt'):
            with open('./cache/BC' + dataset + '.txt', 'r') as fp:
                line = fp.readlines()[0]
                split_ = line.strip().split(",")
                de_top_k_list = []
                for e in split_:
                    de_top_k_list.append(int(e))
            print("previously cached de list = ", de_top_k_list)
        else:
            de_top_k_list = DE_num_calc(args, adata)
            print("optimized de list = ", de_top_k_list)
            with open('./cache/BC' + dataset + '.txt', 'a+') as fp:
                # fp.write('de list: ')
                fp.write(','.join([str(i) for i in de_top_k_list]))
                # fp.write('\n')
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []

    for iter_ in range(iters):
        de_list_epoch = []
        if de_top_k_list != []:
            print("performing DEGs selection")
            for de_ in de_top_k_list:
                for cluster_n in args.impute_cluster_num:
                    print("cluster_n = ", cluster_n)
                    GAAE.get_kNN(adata, rad_cutoff=args.radius)

                    ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
                                                                                num_cluster=int(cluster_n),
                                                                                dif_k=de_, device_id=args.use_gpu_id)
                    de_list_epoch.append(de_list)
                    adata_list.append(adata_out)
            g_union = set.union(*de_list_epoch)
            imputed_ad = impute(args, adata_list, g_union, de_top_k_list)
        else:
            print("skip performing DEGs selection")
            imputed_ad = adata

        """result of imputed data"""
        if de_top_k_list != []:
            GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
            ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)
        else:
            GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
            ARI, adata_out = GAAE.train_ADEPT(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

        print('Dataset:', dataset)
        print('ARI:', ARI)
        aris.append(ARI)
        print(aris)
    print('Dataset:', dataset)
    print(aris)
    print(np.mean(aris))
    with open('adept_aris.txt', 'a+') as fp:
        fp.write('BC' + dataset + ' ')
        fp.write(' '.join([str(i) for i in aris]))
        fp.write('\n')


"""MA"""
setting_combinations = [[52, 'MA']]
for setting_combi in setting_combinations:
    args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/mMAMP'
    args.de_candidates = "None"
    dataset = args.input_data = setting_combi[1]
    args.cluster_num = setting_combi[0]
    args.impute_cluster_num = [setting_combi[0]]
    args.radius = 450
    args.use_preprocessing = 1
    args.use_hvgs = 0
    aris = []
    
    if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        if os.path.exists('./cache/MA' + dataset + '.txt'):
            with open('./cache/MA' + dataset + '.txt', 'r') as fp:
                line = fp.readlines()[0]
                split_ = line.strip().split(",")
                de_top_k_list = []
                for e in split_:
                    de_top_k_list.append(int(e))
            print("previously cached de list = ", de_top_k_list)
        else:
            de_top_k_list = DE_num_calc(args, adata)
            print("optimized de list = ", de_top_k_list)
            with open('./cache/DLPFC' + dataset + '.txt', 'a+') as fp:
                # fp.write('de list: ')
                fp.write(','.join([str(i) for i in de_top_k_list]))
                # fp.write('\n')
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []

    for iter_ in range(iters):
        de_list_epoch = []
        for de_ in de_top_k_list:
            for cluster_n in args.impute_cluster_num:
                print("cluster_n = ", cluster_n)
                GAAE.get_kNN(adata, rad_cutoff=args.radius)

                ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
                                                                               num_cluster=int(cluster_n),
                                                                               dif_k=de_, device_id=args.use_gpu_id)
                de_list_epoch.append(de_list)
                adata_list.append(adata_out)
        g_union = set.union(*de_list_epoch)
        imputed_ad = impute(args, adata_list, g_union, de_top_k_list)

        """result of imputed data"""
        GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
        ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

        print('Dataset:', dataset)
        print('ARI:', ARI)
        aris.append(ARI)
        print(aris)
    print('Dataset:', dataset)
    print(aris)
    print(np.mean(aris))
    with open('adept_aris.txt', 'a+') as fp:
        fp.write('mAB' + dataset + ' ')
        fp.write(' '.join([str(i) for i in aris]))
        fp.write('\n')


"""Her2st"""
setting_combinations = [[6, 'A1'], [5, 'B1'], [4, 'C1'], [4, 'D1'], [4, 'E1'], [4, 'F1'], [7, 'G2'], [7, 'H1']]
for setting_combi in setting_combinations:
    args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/Her2_tumor'
    args.de_candidates = "None"
    dataset = args.input_data = setting_combi[1]
    args.cluster_num = setting_combi[0]
    args.impute_cluster_num = [setting_combi[0]]
    args.radius = 150
    args.use_preprocessing = 1
    args.use_hvgs = 0
    aris = []
    
    if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        if os.path.exists('./cache/DLPFC' + dataset + '.txt'):
            with open('./cache/DLPFC' + dataset + '.txt', 'r') as fp:
                line = fp.readlines()[0]
                split_ = line.strip().split(",")
                de_top_k_list = []
                for e in split_:
                    de_top_k_list.append(int(e))
            print("previously cached de list = ", de_top_k_list)
        else:
            de_top_k_list = DE_num_calc(args, adata)
            print("optimized de list = ", de_top_k_list)
            with open('./cache/DLPFC' + dataset + '.txt', 'a+') as fp:
                # fp.write('de list: ')
                fp.write(','.join([str(i) for i in de_top_k_list]))
                # fp.write('\n')
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []

    for iter_ in range(iters):
        de_list_epoch = []
        for de_ in de_top_k_list:
            for cluster_n in args.impute_cluster_num:
                print("cluster_n = ", cluster_n)
                GAAE.get_kNN(adata, rad_cutoff=args.radius)

                ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
                                                                               num_cluster=int(cluster_n),
                                                                               dif_k=de_, device_id=args.use_gpu_id)
                de_list_epoch.append(de_list)
                adata_list.append(adata_out)
        g_union = set.union(*de_list_epoch)
        imputed_ad = impute(args, adata_list, g_union, de_top_k_list)

        """result of imputed data"""
        GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
        ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

        print('Dataset:', dataset)
        print('ARI:', ARI)
        aris.append(ARI)
        print(aris)
    print('Dataset:', dataset)
    print(aris)
    print(np.mean(aris))
    with open('adept_aris.txt', 'a+') as fp:
        fp.write('Her2tumor' + dataset + ' ')
        fp.write(' '.join([str(i) for i in aris]))
        fp.write('\n')


"""mVC"""
setting_combinations = [[7, 'STARmap_20180505_BY3_1k.h5ad']]
for setting_combi in setting_combinations:
    args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/STARmap_mouse_visual_cortex'
    args.de_candidates = "None"
    dataset = args.input_data = setting_combi[1]
    args.cluster_num = setting_combi[0]
    args.impute_cluster_num = [setting_combi[0]]
    args.radius = 450
    aris = []
    
    if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        if os.path.exists('./cache/DLPFC' + dataset + '.txt'):
            with open('./cache/DLPFC' + dataset + '.txt', 'r') as fp:
                line = fp.readlines()[0]
                split_ = line.strip().split(",")
                de_top_k_list = []
                for e in split_:
                    de_top_k_list.append(int(e))
            print("previously cached de list = ", de_top_k_list)
        else:
            de_top_k_list = DE_num_calc(args, adata)
            print("optimized de list = ", de_top_k_list)
            with open('./cache/DLPFC' + dataset + '.txt', 'a+') as fp:
                # fp.write('de list: ')
                fp.write(','.join([str(i) for i in de_top_k_list]))
                # fp.write('\n')
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []

    for iter_ in range(iters):
        de_list_epoch = []
        for de_ in de_top_k_list:
            for cluster_n in args.impute_cluster_num:
                print("cluster_n = ", cluster_n)
                GAAE.get_kNN(adata, rad_cutoff=args.radius)

                ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
                                                                               num_cluster=int(cluster_n),
                                                                               dif_k=de_, device_id=args.use_gpu_id)
                de_list_epoch.append(de_list)
                adata_list.append(adata_out)
        g_union = set.union(*de_list_epoch)
        imputed_ad = impute(args, adata_list, g_union, de_top_k_list)

        """result of imputed data"""
        GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
        ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

        print('Dataset:', dataset)
        print('ARI:', ARI)
        aris.append(ARI)
        print(aris)
    print('Dataset:', dataset)
    print(aris)
    print(np.mean(aris))
    with open('adept_aris.txt', 'a+') as fp:
        fp.write('mVC ')
        fp.write(' '.join([str(i) for i in aris]))
        fp.write('\n')


"""mPFC"""
# the number of clusters
setting_combinations = [[4, '20180417_BZ5_control'], [4, '20180419_BZ9_control'], [4, '20180424_BZ14_control']]
for setting_combi in setting_combinations:
    args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/STARmap_mouse_PFC'
    args.de_candidates = "None"
    dataset = args.input_data = setting_combi[1]
    args.cluster_num = setting_combi[0]
    args.impute_cluster_num = [setting_combi[0]]
    args.radius = 450
    aris = []
    
    if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        if os.path.exists('./cache/DLPFC' + dataset + '.txt'):
            with open('./cache/DLPFC' + dataset + '.txt', 'r') as fp:
                line = fp.readlines()[0]
                split_ = line.strip().split(",")
                de_top_k_list = []
                for e in split_:
                    de_top_k_list.append(int(e))
            print("previously cached de list = ", de_top_k_list)
        else:
            de_top_k_list = DE_num_calc(args, adata)
            print("optimized de list = ", de_top_k_list)
            with open('./cache/DLPFC' + dataset + '.txt', 'a+') as fp:
                # fp.write('de list: ')
                fp.write(','.join([str(i) for i in de_top_k_list]))
                # fp.write('\n')
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []

    for iter_ in range(iters):
        de_list_epoch = []
        for de_ in de_top_k_list:
            for cluster_n in args.impute_cluster_num:
                print("cluster_n = ", cluster_n)
                GAAE.get_kNN(adata, rad_cutoff=args.radius)

                ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
                                                                               num_cluster=int(cluster_n),
                                                                               dif_k=de_, device_id=args.use_gpu_id)
                de_list_epoch.append(de_list)
                adata_list.append(adata_out)
        g_union = set.union(*de_list_epoch)
        imputed_ad = impute(args, adata_list, g_union, de_top_k_list)

        """result of imputed data"""
        GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
        ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

        print('Dataset:', dataset)
        print('ARI:', ARI)
        aris.append(ARI)
        print(aris)
    print('Dataset:', dataset)
    print(aris)
    print(np.mean(aris))
    with open('adept_aris.txt', 'a+') as fp:
        fp.write('mPFC' + dataset + ' ')
        fp.write(' '.join([str(i) for i in aris]))
        fp.write('\n')


"""mHypo"""
setting_combinations = [[8, '-0.04'], [8, '-0.09'], [8, '-0.14'], [8, '-0.19'], [8, '-0.24'], [8, '-0.29']]
for setting_combi in setting_combinations:
    args.data_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus'
    args.de_candidates = "None"
    dataset = args.input_data = setting_combi[1]
    args.cluster_num = setting_combi[0]
    args.impute_cluster_num = [setting_combi[0]]
    args.radius = 150
    aris = []
    
    if args.input_data not in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control', 'STARmap_20180505_BY3_1k.h5ad'] :
        filter_num = filter_num_calc(args, args.filter_num)
        print("optimized filter number = ", filter_num)
    else:
        filter_num = 0
    adata, adata_ori = initialize(args, filter_num)
    if args.de_candidates == "None":
        if os.path.exists('./cache/DLPFC' + dataset + '.txt'):
            with open('./cache/DLPFC' + dataset + '.txt', 'r') as fp:
                line = fp.readlines()[0]
                split_ = line.strip().split(",")
                de_top_k_list = []
                for e in split_:
                    de_top_k_list.append(int(e))
            print("previously cached de list = ", de_top_k_list)
        else:
            de_top_k_list = DE_num_calc(args, adata)
            print("optimized de list = ", de_top_k_list)
            with open('./cache/DLPFC' + dataset + '.txt', 'a+') as fp:
                # fp.write('de list: ')
                fp.write(','.join([str(i) for i in de_top_k_list]))
                # fp.write('\n')
    else:
        split_ = args.de_candidates.strip().split(",")
        de_top_k_list = []
        for e in split_:
            de_top_k_list.append(int(e))
        print("manually defined de list = ", de_top_k_list)
    adata_list = []

    for iter_ in range(iters):
        de_list_epoch = []
        for de_ in de_top_k_list:
            for cluster_n in args.impute_cluster_num:
                print("cluster_n = ", cluster_n)
                GAAE.get_kNN(adata, rad_cutoff=args.radius)

                ari_ini, ari_final, de_list, adata_out = GAAE.train_ADEPT_use_DE(adata, n_epochs=1000,
                                                                               num_cluster=int(cluster_n),
                                                                               dif_k=de_, device_id=args.use_gpu_id)
                de_list_epoch.append(de_list)
                adata_list.append(adata_out)
        g_union = set.union(*de_list_epoch)
        imputed_ad = impute(args, adata_list, g_union, de_top_k_list)

        """result of imputed data"""
        GAAE.get_kNN(imputed_ad, rad_cutoff=args.radius)
        ari_ini, ARI, de_list, adata_out = GAAE.train_ADEPT_use_DE(imputed_ad, n_epochs=1000, num_cluster=args.cluster_num, device_id=args.use_gpu_id)

        print('Dataset:', dataset)
        print('ARI:', ARI)
        aris.append(ARI)
    print('Dataset:', dataset)
    print(aris)
    print(np.mean(aris))
    print(aris)
    with open('adept_aris.txt', 'a+') as fp:
        fp.write('mHypothalamus' + dataset + ' ')
        fp.write(' '.join([str(i) for i in aris]))
        fp.write('\n')
