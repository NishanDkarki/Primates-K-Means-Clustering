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
import adjustText as ad


def run_kmeans_clustering():
    # baseline_data = '/Users/nises/Desktop/RA/2021-10-05_baseline_hormone_immune.tsv'
    # induction_data = '/Users/nises/Desktop/RA/2021-09-09_induction_hormone_immune.tsv'
    # necropsy_data = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    cortical_data = '/Users/nises/Desktop/RA/bone_cortical.csv'
    # cancellous_data = '/Users/nises/Desktop/RA/03/bone_cancellous.tsv'

    names_cor = ['corticalthickness', 'corticalthicknessstdev', 'corticalvolume', 'corticalvolumestdev',
                 'crosssectionalvolume',
                 'crosssectionalvolumestdev', 'marrowvolume', 'marrowvolumestdev', 'monkey', 'pmoi', 'pmoistdev']

    # names_h = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Testosterone',
    #            'Deoxycorticosterone', 'Aldosterone', 'DHEAS', 'Osteocalcin', 'CTX']
    # names_h_nec = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'Cortisol', 'ACTH', 'Deoxycorticosterone',
    #                'DHEAS', 'Osteocalcin', 'CTX']

    # names_i = ['ID', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'IL_1b', 'GM_CSF', 'MIP_1b', 'EGF', 'VEGF', 'I_TAC',
    #            'MIG']

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

    # hdf = pd.read_table(induction_data)

    hdf = pd.read_csv(cortical_data)
    # print(hdf)
    # mhd = pd.DataFrame(hdf.filter(names_cor), columns=names_cor)
    # mhd = pd.DataFrame(hdf.filter(names_i), columns=names_i)
    # mhd.columns = mhd.columns.str.replace(' ', '_')
    # print(mhd)
    # print(mhd.shape)

    #   ID not in [10077, 10059, 10065]
    # immune_grp1 = mhd.filter(names_cor).query("DrinkingCategory in ['LD', 'VHD']").dropna().reset_index(drop=True)
    immune_grp1 = hdf.filter(names_cor).dropna().reset_index(drop=True)
    # immune_grp1 = immune_grp1.filter(names_i).query("ID not in [10077]").dropna().reset_index(drop=True)
    # immune_grp1 = mhd.filter(names_i).dropna().reset_index(drop=True)
    immune_grp = pd.DataFrame(immune_grp1)
    # print(immune_grp)

    names = ['corticalthickness', 'corticalthicknessstdev', 'corticalvolume', 'corticalvolumestdev',
             'crosssectionalvolume', 'crosssectionalvolumestdev', 'marrowvolume', 'marrowvolumestdev', 'pmoi', 'pmoistdev']

    # names = ['Age', 'Cortisol', 'ACTH', 'Deoxycorticosterone', 'DHEAS', 'Osteocalcin', 'CTX']
    # names = ['ACTH', 'Testosterone', 'Deoxycorticosterone', 'Aldosterone']

    # names = ['Age', 'FGF_basic', 'IL_1b', 'G_CSF', 'IL_10', 'IL_6', 'IL_12', 'RANTES', 'Eotaxin', 'IL_17', 'MIP_1a',
    #          'GM_CSF', 'MIP_1b', 'MCP_1', 'IL_15', 'EGF', 'IL_5', 'HGF', 'VEGF', 'IFN_y', 'MDC', 'I_TAC',
    #          'MIFAnalyte_46', 'IL_1RA', 'TNF_a', 'IL_2', 'IP_10', 'MIG', 'IL_4', 'IL_8']

    # names = ['IL_1b', 'GM_CSF', 'MIP_1b', 'EGF', 'VEGF', 'I_TAC', 'MIG']

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
    # print(filtered_no.shape)

    # print("Variances:")
    # print(filtered_no.var())
    #
    # print("Correlation:")
    # print(filtered_no.corr())

    plt.figure(figsize=(10, 10))
    corr = filtered_no.corr()
    heatmap = sns.heatmap(corr, center=0, mask=np.triu(np.ones_like(corr, dtype=bool)),
                          cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, square=True)

    # heatmap = sns.heatmap(corr,
    #                       cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, annot=True, square=True)
    heatmap.set_title('Fig. 1 Immune data correlation heatmap', fontdict={'fontsize': 12}, pad=12)
    heatmap.figure.tight_layout()
    plt.savefig("/Users/nises/Desktop/heatmap.pdf")
    plt.show()

    scaler = preprocessing.StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(filtered_no))
    print(scaled_df)

    # pca = PCA()
    # pca.fit(scaled_df)
    # print(pca.explained_variance_ratio_)

    # plt.figure(figsize=(10, 8))
    # plt.plot(range(0, 20), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    # plt.show()

    # Run PCA
    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(scaled_df)
    print('Cumulative variance explained by principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))
    # print(pca_2_result)
    print(pca_2.explained_variance_ratio_)
    print(abs(pca_2.components_))

    # Plot the explained variances
    features = range(pca_2.n_components_)
    plt.bar(features, pca_2.explained_variance_ratio_, color='blue')
    plt.xlabel('PCA components')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.title("Fig. 2.1 PCA variance for each component")
    plt.savefig("/Users/nises/Desktop/PCA_variance.pdf")
    plt.show()

    PCA_components = pd.DataFrame(pca_2_result)
    plt.scatter(PCA_components[0], PCA_components[1], color='blue')
    plt.xlabel('PCA 0')
    plt.ylabel('PCA 1')
    plt.title("Fig. 2.2 Scatter plot for first two PCA components")
    plt.savefig("/Users/nises/Desktop/PCA_plot.pdf")
    plt.show()

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
    plt.title("Fig. 3 Elbow method for optimal value of $k$ (number of centroids)")
    plt.savefig("/Users/nises/Desktop/Elbow.pdf")
    plt.show()

    scores = [0]
    for i in range(2, 11):
        fitx = KMeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(pca_2_result)
        score = silhouette_score(pca_2_result, fitx.labels_)
        scores.append(score)
    print("Optimized at", max(range(len(scores)), key=scores.__getitem__) + 1, "clusters")
    plt.figure(figsize=(11, 8.5))
    plt.plot(range(1, 11), np.array(scores), 'bx-')
    plt.xlabel('Number of clusters $k$')
    plt.ylabel('Average Silhouette')
    plt.title('The Silhouette Method showing the optimal $k$ (number of centroids)')
    plt.show()

    kmeans = KMeans(n_clusters=2)
    label = kmeans.fit_predict(pca_2_result)
    centroids = kmeans.cluster_centers_

    # print(immune_grp)
    pca_df = pd.concat([immune_grp, pd.DataFrame(pca_2_result)], axis=1)
    pca_df['K-mensLabels'] = label
    # d_cat = pca_df['DrinkingCategory']

    names_print = ['ID', 'Age', 'DrinkingCategory', 'Species', 'Sex', 'Age', 'K-mensLabels']
    pca_print = pd.DataFrame(pca_df.filter(names_print), columns=names_print)
    print(pca_print)
    # print(label)
    id_array = pca_print['ID'].tolist()
    id_array = list(map(str, id_array))
    id_array = [x[-2:] + ',' for x in id_array]

    cat_array = pca_print['DrinkingCategory'].tolist()
    cat_array = list(map(str, cat_array))

    id_array = np.char.add(id_array, cat_array)
    print(id_array)
    id_array = [x[-6:] for x in id_array]

    pca1_scores = pca_2_result[:, 0]
    pca2_scores = pca_2_result[:, 1]

    fig, ax = plt.subplots()
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(pca_2_result[label == i, 0], pca_2_result[label == i, 1], label='cluster' + str(i))
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')

    texts = []
    for i, txt in enumerate(id_array):
        texts.append(ax.text(pca1_scores[i], pca2_scores[i], id_array[i]))
    ad.adjust_text(texts)
    # for i in range(len(pca_2_result)):
    #     plt.annotate(id_array[i], (pca1_scores[i], pca2_scores[i]), fontsize=5)

    plt.title("Fig. 4.2 K-means clustering on induction immune data")
    plt.legend()
    plt.savefig("/Users/nises/Desktop/kmeans.pdf")
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


# def b_data_process(data):
#     # print(data)
#     raw_filtered_df = data
#     names_all = ['monkey', 'cortical thickness', 'cortical volume', 'cross sectional volume',
#                  'cross sectional volume stdev', 'marrow volume', 'marrow volume stdev', 'pmoi', 'pmoi stdev']
#     names_no_dev = ['monkey', 'cortical thickness', 'cortical volume', 'cross sectional volume', 'marrow volume',
#                     'pmoi']
#
#     names_all_fe = ['cortical thickness', 'cortical volume', 'cross sectional volume',
#                     'cross sectional volume stdev', 'marrow volume', 'marrow volume stdev', 'pmoi', 'pmoi stdev']
#
#     filtered_data_i = pd.DataFrame(raw_filtered_df.dropna().reset_index(drop=True))
#
#     filtered_data_id = pd.DataFrame(filtered_data_i.filter(items=['monkey']))
#
#     filtered_data = pd.DataFrame(filtered_data_i.filter(names_all_fe), columns=names_all_fe)
#     # print(filtered_data_id)filtered_data_i
#
#     plt.figure(figsize=(20, 20))
#     corr = filtered_data.corr()
#     heatmap = sns.heatmap(corr, center=0, mask=np.triu(np.ones_like(corr, dtype=bool)),
#                           cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, annot=True, square=True)
#     heatmap.set_title('Fig 2.1 Bone cortisol heatmap', fontdict={'fontsize': 12}, pad=12)
#     heatmap.figure.tight_layout()
#     plt.show()
#
#     scaler = preprocessing.StandardScaler()
#     scaled_data = pd.DataFrame(scaler.fit_transform(filtered_data))
#     # print(scaled_data)
#
#     return scaled_data, filtered_data_id
#
#
# def do_PCA(scaled_data, filtered_data_id):
#     pca = PCA()
#     pca.fit(scaled_data)
#
#     # Run PCA
#     pca_2 = PCA(n_components=3)
#     pca_2_result = pca_2.fit_transform(scaled_data)
#     print('Cumulative variance explained by principal components: {:.2%}'.format(
#         np.sum(pca_2.explained_variance_ratio_)))
#
#     sse = []
#     for k in range(1, 10):
#         km = KMeans(n_clusters=k)
#         km.fit(pca_2_result)
#         sse.append(km.inertia_)
#
#     # Plot sse against k
#     plt.figure(figsize=(10, 10))
#     plt.plot(range(1, 10), sse, '-o')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Sum of squared distance')
#     plt.show()
#
#     return pca_2_result, filtered_data_id
#
#
# def run_kmeans(pca_2_result, filtered_data_id):
#     kmeans = KMeans(n_clusters=4)
#     label = kmeans.fit_predict(pca_2_result)
#     centroids = kmeans.cluster_centers_
#
#     u_labels = np.unique(label)
#
#     for i in u_labels:
#         # print(pca_2_result)
#         plt.scatter(pca_2_result[label == i, 0], pca_2_result[label == i, 1], label='cluster' + str(i))
#
#     plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')
#     plt.title("Fig. K-means clustering")
#
#     plt.legend()
#     plt.show()

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
    # bone_cortical = '/Users/nises/Desktop/RA/bone_cortical.csv'
    # necropsy_data = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    # hdf = pd.read_table(necropsy_data)

    # hd_df = pd.read_csv(bone_cortical, index_col=False)
    # scaled_data, filtered_data_id = b_data_process(hd_df)
    # pca_2_result, filtered_data_id = do_PCA(scaled_data, filtered_data_id)
    # run_kmeans(pca_2_result, filtered_data_id)

    run_kmeans_clustering()


if __name__ == "__main__":
    main()
