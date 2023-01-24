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


def feature_selector(cor_data_path, target_path):
    features_cor = ['ID', 'corticalthickness', 'corticalvolume', 'crosssectionalvolume', 'marrowvolume', 'pmoi']
    cor_data = pd.read_csv(cor_data_path)
    cor_data = cor_data.filter(features_cor).dropna().reset_index(drop=True)

    features_tar = ['ID', 'DrinkingCategory']
    target_data = pd.read_table(target_path)
    target_data = target_data.filter(features_tar).query("DrinkingCategory in ['LD', 'VHD']").dropna().reset_index(
        drop=True)

    org_data = cor_data.merge(target_data.filter(items=['ID', 'DrinkingCategory']), on=["ID"])
    org_data = org_data.dropna().reset_index(drop=True)
    return org_data


def target_value(target_path):
    features_tar = ['ID', 'DrinkingCategory']
    target_data = pd.read_table(target_path)
    target_data = target_data.filter(features_tar).dropna().reset_index(drop=True)
    return target_data


def plot_correlation(data):
    plt.figure(figsize=(20, 20))
    corr = data.corr()
    heatmap = sns.heatmap(corr, center=0, mask=np.triu(np.ones_like(corr, dtype=bool)),
                          cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1, annot=True, square=True)
    heatmap.set_title('Fig 1 Features Heatmap', fontdict={'fontsize': 12}, pad=12)
    heatmap.figure.tight_layout()
    plt.show()


def normalize(data):
    scaler = preprocessing.StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data))
    return scaled_data


def run_PCA(data, components):
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(data)
    return pca_result


def plot_pca(data, components):
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(data)

    print('Cumulative variance explained by principal components: {:.2%}'.format(
        np.sum(pca.explained_variance_ratio_)))
    print(pca.explained_variance_ratio_)

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


def data_standerize(data):
    features = ['corticalthickness', 'corticalvolume', 'crosssectionalvolume', 'marrow volume', 'pmoi']
    std_data = data.filter(features).reset_index(drop=True)
    scaled_data = normalize(std_data)
    return scaled_data


def plot_kscore(data):
    scores = [0]
    for i in range(2, 11):
        fitx = KMeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(data)
        score = silhouette_score(data, fitx.labels_)
        scores.append(score)
    print("Optimized at", max(range(len(scores)), key=scores.__getitem__) + 1, "clusters")
    plt.figure(figsize=(11, 8.5))
    plt.plot(range(1, 11), np.array(scores), 'bx-')
    plt.xlabel('Number of clusters $k$')
    plt.ylabel('Average Silhouette')
    plt.title('The Silhouette Method showing the optimal $k$ (number of centroids)')
    plt.show()


def run_kmeans(pca_result, clusters):
    kmeans = KMeans(n_clusters=clusters)
    label = kmeans.fit_predict(pca_result)
    centroids = kmeans.cluster_centers_

    fig, ax = plt.subplots()
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(pca_result[label == i, 0], pca_result[label == i, 1], label='cluster' + str(i))
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, color='k', marker='*', label='centroid')

    plt.legend()
    plt.show()

    return label


def processor():
    cor_data_path = '/Users/nises/Desktop/RA/bone_cortical.csv'
    can_data_path = '/Users/nises/Desktop/RA/bone_cancellous.csv.csv'
    target_path = '/Users/nises/Desktop/RA/2021-10-05_necropsy_hormone_immune.tsv'

    data_process = feature_selector(cor_data_path, target_path)
    target_variables = target_value(target_path)

    standerized_data = data_standerize(data_process)
    pca_output = run_PCA(standerized_data, 2)
    kmeans_labels = run_kmeans(pca_output, 2)
    data_process['K-mensLabels'] = kmeans_labels
    print(data_process)
    print('asd')


def main():
    processor()


if __name__ == "__main__":
    main()
