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


def feature_selector(data_path):
    features_cor = ['ID', 'corticalthickness', 'corticalvolume', 'crosssectionalvolume', 'marrow volume', 'pmoi']
    org_data = pd.read_csv(data_path)
    org_data = org_data.filter(features_cor).dropna().reset_index(drop=True)
    return org_data


def target_value(target_path):
    features_tar = ['ID', 'DrinkingCategory']
    target_data = pd.read_table(target_path)
    target_data = target_data.filter(features_tar).dropna().reset_index(drop=True)
    return target_data


def processor():
    data_path = '/Users/nises/Desktop/RA/bone_cortical.csv'
    target_path = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    data_process = feature_selector(data_path)
    target_variables = target_value(target_path)

    print(data_process)
    print(target_variables)


def path_grabber():
    path = '/Users/nises/Desktop/RA/bone_cortical.csv'
    data = pd.read_csv(path)

    return data


def data_processor(data):
    names = ['corticalthickness', 'corticalthicknessstdev', 'corticalvolume', 'corticalvolumestdev',
             'crosssectionalvolume', 'crosssectionalvolumestdev', 'marrowvolume', 'marrowvolumestdev',
             'ID', 'pmoi', 'pmoistdev', ]

    names_cor = ['corticalthickness', 'corticalthicknessstdev', 'corticalvolume', 'corticalvolumestdev',
                 'crosssectionalvolume', 'crosssectionalvolumestdev', 'marrowvolume', 'marrowvolumestdev',
                 'pmoi', 'pmoistdev']

    # names_cor = ['corticalthickness', 'corticalvolume', 'crosssectionalvolume', 'marrow volume', 'pmoi']

    original_data = data.filter(names).dropna().reset_index(drop=True)
    # print(original_data)

    filtered_data = original_data.filter(names_cor)
    # print(filtered_data)

    # filtered_data_id = pd.DataFrame(filtered_data_i.filter(items=['monkey']))

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

    return scaled_data, original_data


def run_PCA(scaled_data):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    print('Cumulative variance explained by principal components: {:.2%}'.format(
        np.sum(pca.explained_variance_ratio_)))
    print(pca.explained_variance_ratio_)
    # print(abs(pca.components_))

    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='blue')
    plt.xlabel('PCA components')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.title("Fig. 2.1 PCA variance for each component")
    plt.savefig("/Users/nises/Desktop/PCA_variance.pdf")
    plt.show()

    PCA_components = pd.DataFrame(pca_result)
    plt.scatter(PCA_components[0], PCA_components[1], color='blue')
    plt.xlabel('PCA 0')
    plt.ylabel('PCA 1')
    plt.title("Fig. 2.2 Scatter plot for first two PCA components")
    plt.savefig("/Users/nises/Desktop/PCA_plot.pdf")
    plt.show()

    return pca_result


def run_kmeans(pca_2_result):
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

    fig, ax = plt.subplots()
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(pca_2_result[label == i, 0], pca_2_result[label == i, 1], label='cluster' + str(i))
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')

    plt.legend()
    plt.show()

    return label


def process_output(original_data):
    # print(original_data)
    path = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'
    necropsy_data = pd.read_table(path)

    merged_data = (original_data.filter(items=['ID', 'K-mensLabels'])).merge(
        necropsy_data.filter(items=['ID', 'DrinkingCategory']),
        on=["ID"])
    print(merged_data)

    # texts = []
    # for i, txt in enumerate(id_array):
    #     texts.append(ax.text(pca1_scores[i], pca2_scores[i], id_array[i]))
    # ad.adjust_text(texts)


def main():
    # data = path_grabber()
    # scaled_data, original_data = data_processor(data)
    # pca_result = run_PCA(scaled_data)
    # kmeans_labels = run_kmeans(pca_result)
    # original_data['K-mensLabels'] = kmeans_labels
    # # print(original_data)
    # process_output(original_data)
    processor()


if __name__ == "__main__":
    main()
