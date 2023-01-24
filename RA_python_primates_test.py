from datetime import date
from matplotlib import colors
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def run_kmeans_clustering():
    # baseline_data = '/Users/nises/Desktop/RA/2021-10-05_baseline_hormone_immune.tsv'
    # induction_data = '/Users/nises/Desktop/RA/2021-09-09_induction_hormone_immune.tsv'
    necropsy_data = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    # names_h = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Testosterone',
    #            'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX']
    # names_h_nec = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Deoxycorticosterone',
    #                'DHEAS', 'Osteocalcin', 'CTX']
    names_i = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'FGF_basic', 'IL_1b', 'G_CSF', 'IL_10', 'IL_6',
               'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a', 'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5',
               'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC', 'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG',
               'IL_4', 'IL_8']

    # names_hi = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Testosterone',
    #             'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX', 'FGF_basic', 'IL_1b',
    #             'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #             'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #             'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23', 'VEGF_D']

    # names_hi_nec = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH',
    #             'Deoxycorticosterone', 'DHEAS', 'Osteocalcin', 'CTX', 'FGF_basic', 'IL_1b',
    #             'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #             'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #             'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8', 'IL_23', 'VEGF_D']

    hdf = pd.read_table(necropsy_data)
    # print(hdf.shape)
    mhd = pd.DataFrame(hdf.filter(names_i), columns=names_i)
    # mhd.columns = mhd.columns.str.replace(' ', '_')
    # print(mhd)
    # print(mhd.shape)

    #   ID not in [10077, 10059, 10065]
    immune_grp1 = mhd.filter(names_i).query("DrinkingCategory in ['LD', 'VHD']").dropna().reset_index(drop=True)
    #     immune_grp1 = mhd.filter(names_h).dropna().reset_index(drop=True)
    immune_grp = pd.DataFrame(immune_grp1)
    # print(immune_grp)

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

    filtered_no = pd.DataFrame(immune_grp.filter(names), columns=names)
    print(filtered_no.shape)

    plt.figure(figsize=(20, 20))
    corr = filtered_no.corr()
    heatmap = sns.heatmap(corr, center=0, mask=np.triu(np.ones_like(corr, dtype=bool)),
                          cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, annot=True, square=True)
    heatmap.set_title('Fig 1.3 Necropsy hormone data correlation heatmap', fontdict={'fontsize': 12}, pad=12)
    heatmap.figure.tight_layout()
    plt.show()

    scaler = preprocessing.StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(filtered_no))
    # print(scaled_df.shape)

    pca = PCA()
    pca.fit(scaled_df)
    # print(pca.explained_variance_ratio_)

    # plt.figure(figsize=(10, 8))
    # plt.plot(range(0, 20), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    # plt.show()

    # Run PCA
    pca_2 = PCA(n_components=4)
    pca_2_result = pca_2.fit_transform(scaled_df)
    print('Cumulative variance explained by principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))
    # print(pca_2_result)

    # Run the Kmeans algorithm and get the index of data points clusters
    sse = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k)
        km.fit(pca_2_result)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 10), sse, '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distance')
    plt.show()

    kmeans = KMeans(n_clusters=2)
    label = kmeans.fit_predict(pca_2_result)
    centroids = kmeans.cluster_centers_

    # print(immune_grp)
    pca_df = pd.concat([immune_grp, pd.DataFrame(pca_2_result)], axis=1)
    pca_df['K-mensLabels'] = label
    d_cat = pca_df['DrinkingCategory']

    names_print = ['ID', 'Age', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'K-mensLabels']
    pca_print = pd.DataFrame(pca_df.filter(names_print), columns=names_print)
    print(pca_print)
    print(label)

    u_labels = np.unique(label)

    for index, item in enumerate(u_labels):
        print(index, item)

    for i in u_labels:
        print(pca_2_result)
        # print(pca_2_result[i])
        # print(pca_2_result[i, 0])
        # print(pca_2_result[i, 1])
        plt.scatter(pca_2_result[label == i, 0], pca_2_result[label == i, 1], label='cluster' + str(i))

    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')
    plt.title("Fig. 2.3 K-means clustering on necropsy immune data")

    plt.legend()
    plt.show()

    labels = pd.DataFrame(kmeans.labels_)
    cluster_group1 = pd.concat([immune_grp.filter(['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age']), labels],
                               axis=1)

    cluster_group = cluster_group1.rename({0: 'Cluster'}, axis=1)
    cluster_group = cluster_group.sort_values(['Cluster'])

    print(cluster_group['Cluster'].value_counts())
    # print(cluster_group)
    LD = cluster_group[(cluster_group['DrinkingCategory'] == 'LD')]
    print('LD ', LD['Cluster'].value_counts())
    BD = cluster_group[(cluster_group['DrinkingCategory'] == 'BD')]
    print('BD ', BD['Cluster'].value_counts())
    HD = cluster_group[(cluster_group['DrinkingCategory'] == 'HD')]
    print('HD ', HD['Cluster'].value_counts())
    VHD = cluster_group[(cluster_group['DrinkingCategory'] == 'VHD')]
    print('VHD ', VHD['Cluster'].value_counts())


def b_data_process(data):
    # print(data)
    raw_filtered_df = data
    names_all = ['monkey', 'cortical thickness', 'cortical volume', 'cross sectional volume',
                 'cross sectional volume stdev', 'marrow volume', 'marrow volume stdev', 'pmoi', 'pmoi stdev']
    names_no_dev = ['monkey', 'cortical thickness', 'cortical volume', 'cross sectional volume', 'marrow volume',
                    'pmoi']

    names_all_fe = ['cortical thickness', 'cortical volume', 'cross sectional volume',
                    'cross sectional volume stdev', 'marrow volume', 'marrow volume stdev', 'pmoi', 'pmoi stdev']

    filtered_data_i = pd.DataFrame(raw_filtered_df.dropna().reset_index(drop=True))

    filtered_data_id = pd.DataFrame(filtered_data_i.filter(items=['monkey']))

    filtered_data = pd.DataFrame(filtered_data_i.filter(names_all_fe), columns=names_all_fe)
    # print(filtered_data_id)filtered_data_i

    plt.figure(figsize=(20, 20))
    corr = filtered_data.corr()
    heatmap = sns.heatmap(corr, center=0, mask=np.triu(np.ones_like(corr, dtype=bool)),
                          cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, annot=True, square=True)
    heatmap.set_title('Fig 2.1 Bone cortisol heatmap', fontdict={'fontsize': 12}, pad=12)
    heatmap.figure.tight_layout()
    plt.show()

    scaler = preprocessing.StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(filtered_data))
    # print(scaled_data)

    return scaled_data, filtered_data_id


def do_PCA(scaled_data, filtered_data_id):
    pca = PCA()
    pca.fit(scaled_data)

    # Run PCA
    pca_2 = PCA(n_components=3)
    pca_2_result = pca_2.fit_transform(scaled_data)
    print('Cumulative variance explained by principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))

    sse = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k)
        km.fit(pca_2_result)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 10), sse, '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distance')
    plt.show()

    return pca_2_result, filtered_data_id


def run_kmeans(pca_2_result, filtered_data_id):
    kmeans = KMeans(n_clusters=4)
    label = kmeans.fit_predict(pca_2_result)
    centroids = kmeans.cluster_centers_

    u_labels = np.unique(label)

    for i in u_labels:
        # print(pca_2_result)
        plt.scatter(pca_2_result[label == i, 0], pca_2_result[label == i, 1], label='cluster' + str(i))

    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')
    plt.title("Fig. K-means clustering")

    plt.legend()
    plt.show()

    # labels = pd.DataFrame(kmeans.labels_)
    # cluster_group1 = pd.concat([immune_grp.filter(['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age']), labels],
    #                            axis=1)
    #
    # cluster_group = cluster_group1.rename({0: 'Cluster'}, axis=1)
    # cluster_group = cluster_group.sort_values(['Cluster'])
    #
    # print(cluster_group['Cluster'].value_counts())
    # # print(cluster_group)
    # LD = cluster_group[(cluster_group['DrinkingCategory'] == 'LD')]
    # print('LD ', LD['Cluster'].value_counts())
    # BD = cluster_group[(cluster_group['DrinkingCategory'] == 'BD')]
    # print('BD ', BD['Cluster'].value_counts())
    # HD = cluster_group[(cluster_group['DrinkingCategory'] == 'HD')]
    # print('HD ', HD['Cluster'].value_counts())
    # VHD = cluster_group[(cluster_group['DrinkingCategory'] == 'VHD')]
    # print('VHD ', VHD['Cluster'].value_counts())


def main():
    bone_cortical = '/Users/nises/Desktop/RA/bone_cortical.csv'
    necropsy_data = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    hdf = pd.read_table(necropsy_data)

    hd_df = pd.read_csv(bone_cortical, index_col=False)
    scaled_data, filtered_data_id = b_data_process(hd_df)
    pca_2_result, filtered_data_id = do_PCA(scaled_data, filtered_data_id)
    run_kmeans(pca_2_result, filtered_data_id)

    # run_kmeans_clustering()


if __name__ == "__main__":
    main()
