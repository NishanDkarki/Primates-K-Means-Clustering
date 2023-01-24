from datetime import date
from matplotlib import colors
from sklearn import preprocessing
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import kmeans_plusplus

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# generate dataframe with hormone and immune data during induction
def generate_data(hdf, idf):
    return hdf.merge(idf, how='outer')


# plot results with colors for all drinking categories
def plot_all_categories(df, atype, out_path, timepoint, suffix=''):
    if atype == 'MDS' or atype == 'TSNE':
        x, y = 'Dimension 1', 'Dimension 2'
    elif atype == 'PCA':
        x, y = 'PC1', 'PC2'
    else:
        return 'ERROR: Analysis type not found'

    sns.relplot(x=x, y=y, hue='Drinking Category', hue_order=['LD', 'BD', 'HD', 'VHD'],
                palette=['tab:green', 'tab:blue', 'tab:orange', 'tab:red'], data=df, kind='scatter', aspect=1)
    plt.savefig('{}/{}_{}_{}_all-drink-cat{}.pdf'.format(out_path, str(date.today()), atype, timepoint, suffix))
    plt.close()


# plot results with colors for only High v. Low drinking categories
def plot_two_categories(df, atype, out_path, timepoint, suffix=''):
    if atype == 'MDS' or atype == 'TSNE':
        x, y = 'Dimension 1', 'Dimension 2'
    elif atype == 'PCA':
        x, y = 'PC1', 'PC2'
    else:
        return 'ERROR: Analysis type not found'

    sns.relplot(x=x, y=y, hue='two_drink_cat', hue_order=['Low', 'High'], palette=['#0571b0', '#ca0020'],
                data=df, kind='scatter')
    plt.savefig('{}/{}_{}_{}_two-drink-cat{}.pdf'.format(out_path, str(date.today()), atype, timepoint, suffix))
    plt.close()


# fit PCA and plot results for scaled dataframe
def run_pca(sdf, path, suffix=''):
    scaled_hilo = sdf[(sdf['Drinking Category'] == "LD") | (sdf['Drinking Category'] == "VHD")]

    # run PCA for all categories and LD/BD v. HD/VHD
    pca_res = fit_pca(sdf)
    plot_all_categories(pca_res, 'PCA', out_path=path, timepoint='induction', suffix=suffix)
    plot_two_categories(pca_res, 'PCA', out_path=path, timepoint='induction', suffix=suffix)

    # run MDS for hi-lo categories only LD v. VHD
    res_hilo = fit_pca(scaled_hilo)
    plot_all_categories(res_hilo, 'PCA', out_path=path, timepoint='induction', suffix='_hilo' + suffix)


# run all dimensionality reduction on individuals with hormone data only, n = 31
def run_hormone_only(fig_path):
    # generate dataset and summarize by individual
    tmp = '/Users/nises/Documents/RA-Dr.Benton/empirical_clustering/data/induction_hormone_df.tsv'
    mhd = pd.read_table(tmp)

    mhd_grp = mhd.groupby(['ID', 'Drinking Category', 'Species', 'Sex', 'Age'], as_index=False).mean()

    # scale the hormone data
    names = ['Cortisol', 'ACTH', 'Testosterone', 'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX']
    scaler = preprocessing.StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(mhd_grp.filter(names)), columns=names)
    scaled_mhd_grp = pd.concat([mhd_grp.filter(['ID', 'Drinking Category', 'Species', 'Sex', 'Age']), scaled_df],
                               axis=1)
    run_pca(scaled_mhd_grp, path=fig_path)


# run all dimensionality reduction on individuals with both immune and hormone data, n = 22
def run_hormone_plus_immune(fig_path):
    tmpi = '/Users/nises/Documents/RA-Dr.Benton/empirical_clustering/data/induction_immune_df.tsv'
    tmph = '/Users/nises/Documents/RA-Dr.Benton/empirical_clustering/data/induction_hormone_df.tsv'
    hdf = pd.read_table(tmph)
    idf = pd.read_table(tmpi)
    mhd = generate_data(hdf, idf)

    immune_grp = (mhd.groupby(['ID', 'Drinking Category', 'Species', 'Sex', 'Age'], as_index=False).mean()
                  .drop(columns='MIF')
                  .dropna()
                  .reset_index()
                  )
    print(immune_grp)

    names = ['Cortisol', 'ACTH', 'Testosterone', 'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX',
             'FGF_basic', 'IL_1b', 'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
             'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
             'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23']
    scaler = preprocessing.StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(immune_grp.filter(names)), columns=names)
    scaled_immune = pd.concat([immune_grp.filter(['ID', 'Drinking Category', 'Species', 'Sex', 'Age']), scaled_df],
                              axis=1)
    print(scaled_immune)
    run_mds(immune_grp, scaled_immune, path=fig_path, suffix='_immune-subset')
    run_pca(scaled_immune, path=fig_path, suffix='_immune-subset')
    run_tsne(scaled_immune, path=fig_path, suffix='_immune-subset')


def run_kmeans_clustering():
    # baseline_data = '/Users/nises/Desktop/RA/2021-10-05_baseline_hormone_immune.tsv'
    # induction_data = '/Users/nises/Desktop/RA/2021-09-09_induction_hormone_immune.tsv'
    necropsy_data = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    # tmpi = '/Users/nises/Documents/RA-Dr.Benton/empirical_clustering/data/induction_immune_df.tsv'
    # tmph = '/Users/nises/Documents/RA-Dr.Benton/empirical_clustering/data/induction_hormone_df.tsv'
    # hdf = pd.read_table(tmph)
    # idf = pd.read_table(tmpi)
    # mhd = generate_data(hdf, idf)

    # names_h = ['ID', 'Drinking Category', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Testosterone',
    #            'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX']
    # names_h_nec = ['ID', 'Drinking Category', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Deoxycorticosterone',
    #                'DHEAS', 'Osteocalcin', 'CTX']
    names_i = ['ID', 'Drinking Category', 'Species', 'Sex', 'Age', 'FGF_basic', 'IL_1b', 'G_CSF', 'IL_10', 'IL_6',
               'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a', 'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5',
               'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC', 'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG',
               'IL_4', 'IL_8']

    # names_hi = ['ID', 'Drinking Category', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Testosterone',
    #             'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX', 'FGF_basic', 'IL_1b',
    #             'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #             'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #             'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23', 'VEGF_D']

    # names_hi_nec = ['ID', 'Drinking Category', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH',
    #             'Deoxycorticosterone', 'DHEAS', 'Osteocalcin', 'CTX', 'FGF_basic', 'IL_1b',
    #             'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #             'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #             'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23', 'VEGF_D']

    hdf = pd.read_table(necropsy_data)
    # print(hdf.shape)
    mhd = pd.DataFrame(hdf.filter(names_i), columns=names_i)
    # print(mhd.shape)

    immune_grp = (mhd.groupby(['ID', 'Drinking Category', 'Species', 'Sex', 'Age'], as_index=True)
                  .mean()
                  .dropna()
                  .reset_index()
                  )
    # print(immune_grp)
    # print(immune_grp.shape)

    # names = ['Age', 'Cortisol', 'ACTH', 'Deoxycorticosterone', 'DHEAS', 'Osteocalcin', 'CTX']

    # names = ['Age', 'Cortisol', 'ACTH', 'Testosterone', 'Deoxycorticosterone', 'Aldosterone', 'DHEAS',
    #          'Osteocalcin', 'CTX']

    names = ['Age', 'FGF_basic', 'IL_1b', 'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
             'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
             'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8']

    # names = ['Age', 'Cortisol', 'ACTH', 'Testosterone',
    #          'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX', 'FGF_basic', 'IL_1b',
    #          'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #          'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #          'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23', 'VEGF_D']

    # names = ['Age', 'Cortisol', 'ACTH', 'Deoxycorticosterone', 'DHEAS', 'Osteocalcin', 'CTX', 'FGF_basic', 'IL_1b',
    #          'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #          'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #          'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23', 'VEGF_D']

    scaler = preprocessing.StandardScaler()

    filtered_no = pd.DataFrame(immune_grp.filter(names), columns=names)
    # plt.figure(figsize=(10, 10))
    # corr = filtered_no.corr()
    # heatmap = sns.heatmap(corr, center=0, mask=np.triu(np.ones_like(corr, dtype=bool)),
    #                       cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, annot=True, square=True)
    # heatmap.set_title('Fig 1.3 Necropsy hormone data correlation heatmap', fontdict={'fontsize': 12}, pad=12)
    # heatmap.figure.tight_layout()
    # plt.show()

    # Compute the correlation matrix
    # corr = d.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    filter_shape = (immune_grp.filter(names))
    print(filter_shape.shape)
    scaled_df = pd.DataFrame(scaler.fit_transform(immune_grp.filter(names)), columns=names)
    # scaled_immune = pd.concat([immune_grp.filter(['ID', 'Drinking Category', 'Species', 'Sex', 'Age']), scaled_df],
    #                           axis=1)
    # print(scaled_df.shape)
    # print(scaled_immune)

    # Run PCA
    pca_2 = PCA()
    pca_2.fit(scaled_df)
    print(pca_2.explained_variance_ratio_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(0, 22), pca_2.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')

    pca_2 = PCA(n_components=7)
    pca_2_result = pca_2.fit_transform(scaled_df)

    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
    print('Cumulative variance explained by principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))
    print(pca_2_result)

    # df = pd.DataFrame(data=pca_2_result, index=["row1", "row2"], columns=["column1", "column2"])
    # print(df)
    #
    # df_p = pd.DataFrame(columns=[1])
    # df_p[1] = df_p[1].astype(object)
    # df_p.loc[1, 1] = pca_2_result
    # print(df_p)
    # pca_df = pd.concat([immune_grp.filter(['ID', 'Drinking Category']), df_p], axis=1)
    # print(pca_df)

    # Run the Kmeans algorithm and get the index of data points clusters
    # sse = []
    # list_k = list(range(1, 10))
    # for k in list_k:
    #     km = KMeans(n_clusters=k)
    #     km.fit(pca_2_result)
    #     sse.append(km.inertia_)
    # # Plot sse against k
    # plt.figure(figsize=(6, 6))
    # plt.plot(list_k, sse, '-o')
    # plt.xlabel(r'Number of clusters *k*')
    # plt.ylabel('Sum of squared distance')
    # plt.show()

    kmeans = KMeans(n_clusters=2)
    label = kmeans.fit_predict(pca_2_result)
    centroids = kmeans.cluster_centers_

    print(label)

    u_labels = np.unique(label)
    for i in u_labels:
        print(i)
        # print(pca_2_result[i, 0])
        # print(pca_2_result[i, 1])
        plt.scatter(pca_2_result[label == i, 0], pca_2_result[label == i, 1], label='cluster' + str(i))
    # plt.text(pca_2_result[label == i, 0], pca_2_result[label == i, 1], 1)

    # plt.scatter(pca_2_result[label == 0, 0], pca_2_result[label == 0, 1], label='cluster 0', c='blue')
    # plt.text(pca_2_result[label == i, 0], pca_2_result[label == i, 1], 'drinking category')
    # plt.scatter(pca_2_result[label == 1, 0], pca_2_result[label == 1, 1], label='cluster 1', c='red')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')
    plt.title("Fig. 2.2 K-means clustering on induction hormone data")

    plt.legend()
    plt.show()

    labels = pd.DataFrame(kmeans.labels_)
    cluster_group1 = pd.concat([immune_grp.filter(['ID', 'Drinking Category', 'Species', 'Sex', 'Age']), labels],
                               axis=1)

    cluster_group = cluster_group1.rename({0: 'Cluster'}, axis=1)
    cluster_group = cluster_group.sort_values(['Cluster'])

    print(cluster_group['Cluster'].value_counts())
    # print(cluster_group)
    LD = cluster_group[(cluster_group['Drinking Category'] == 'LD')]
    print('LD ', LD['Cluster'].value_counts())
    BD = cluster_group[(cluster_group['Drinking Category'] == 'BD')]
    print('BD ', BD['Cluster'].value_counts())
    HD = cluster_group[(cluster_group['Drinking Category'] == 'HD')]
    print('HD ', HD['Cluster'].value_counts())
    VHD = cluster_group[(cluster_group['Drinking Category'] == 'VHD')]
    print('VHD ', VHD['Cluster'].value_counts())


def main():
    #     FIG_PATH = '/Users/nises/PycharmProjects/pythonProject/Outputs'
    #     baseline_data = '/Users/nises/Desktop/RA/2021-10-05_baseline_hormone_immune.tsv'
    #     necropsy_data = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'
    #     induction_data = '/Users/nises/Desktop/RA/2021-09-09_induction_hormone_immune.tsv'

    run_kmeans_clustering()


if __name__ == "__main__":
    main()
