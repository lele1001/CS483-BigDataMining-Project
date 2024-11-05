import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
import os

def read_data(file_name):
    return pd.read_csv(file_name)


def clean_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for file in os.listdir(directory):
            os.remove(f"{directory}/{file}")


def generate_pca_plot(data, features_2d, k):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=data['Cluster'])
    plt.title('Clusters of Health Profiles in 2D PCA Space')
    plt.xlabel(f'PCA Component 1: {features_2d[:, 0].var():.2f}')
    plt.ylabel(f'PCA Component 2: {features_2d[:, 1].var():.2f}')
    plt.legend(labels=[f'Cluster {i}' for i in range(k)], title='Cluster')
    plt.savefig("./graphs/cluster_visualization.png")
    plt.close()


def generate_plot(profiles, title, xlabel, ylabel, save_path, ncol=5, figsize=(14, 10)):
    profiles.plot(kind='bar', figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.legend(title='Legend', loc='upper center', ncol=ncol)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(profiles, title, path):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    with open(path, "w") as file:
        file.write(f"{title}:\n")
        file.write(profiles.to_string())


def traditional_clustering(data, features, features_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Add cluster labels to the data
    data['Cluster'] = clusters

    # Profile Each Cluster by calculating the average of each feature within each cluster
    cluster_profiles = data.groupby('Cluster').mean()
    generate_plot(cluster_profiles, "Cluster Profiles (Mean Feature Values)", "Health Indicators", "Average Value", "./graphs/cluster_profiles.png", 8)
    save_results(cluster_profiles, "Cluster Profiles (Mean Feature Values)", "./results/cluster_profiles.txt")

    # Analyze Cluster Composition in Terms of Diabetes Status
    diabetes_distribution = data.groupby(['Cluster', 'Diabetes_012']).size().unstack(fill_value=0)
    diabetes_distribution = diabetes_distribution.div(diabetes_distribution.sum(axis=1), axis=0)

    generate_plot(diabetes_distribution, "Cluster Composition by Diabetes Status", "Cluster", "Percentage", "./graphs/cluster_composition_by_diabetes_status.png")
    save_results(diabetes_distribution, "Cluster Composition by Diabetes Status", "./results/cluster_composition_by_diabetes_status.txt")

    # Visualize Clusters in 2D Space Using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    generate_pca_plot(data, features_2d, k)

    # Generate histograms for each feature
    for feature in features.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=feature, hue=data['Cluster'], multiple='dodge', kde=True, bins=20, palette="viridis", alpha=0.7)
        plt.title(f'Feature Distribution by Cluster: {feature}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(labels=[f'Cluster {i}' for i in range(k)], title='Cluster', loc='upper center')
        plt.tight_layout()
        plt.savefig(f"./graphs/{feature}_cluster_histogram.png")
        plt.close()

def stratified_kmeans(data_scaled):
    # Step 1: Segment data by diabetes status
    no_diabetes = data_scaled[data_scaled['Diabetes_012'] == 0]
    pre_diabetes = data_scaled[data_scaled['Diabetes_012'] == 1]
    diabetes = data_scaled[data_scaled['Diabetes_012'] == 2]

    # Step 2: Calculate feature means (or percentages if binary) within each subgroup
    cluster_profiles = pd.DataFrame({
        'No Diabetes (0)': no_diabetes.mean(),
        'Pre-Diabetes (1)': pre_diabetes.mean(),
        'Diabetes (2)': diabetes.mean()
    })

    # Step 3: Plot feature comparisons between the groups
    cluster_profiles.drop(index='Diabetes_012', inplace=True)  # Remove target column
    cluster_profiles = cluster_profiles.T

    generate_plot(cluster_profiles, "Feature Comparison by Diabetes Status Cluster", "Health Indicators", "Average (or Percentage) Value", "./graphs/feature_comparison_by_diabetes_status.png", 7)
    save_results(cluster_profiles, "Feature Comparison by Diabetes Status Cluster", "./results/feature_comparison_by_diabetes_status.txt")


def main():
    clean_directory("./graphs")
    clean_directory("./results")

    print("K-Means Clustering on CDC Diabetes Health Indicators Dataset")
    # Read data from labels.csv
    data = read_data("../data/labels.csv")

    # Separate the feature columns from the target column
    features = data.drop(columns=['Diabetes_012']) 
    target = data['Diabetes_012']

    # Standardize the feature data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    k = np.unique(data.iloc[:, 0]).shape[0]

    # Perform Traditional K-Means Clustering
    print("\nPerforming Traditional K-Means Clustering on the dataset")
    traditional_clustering(data, features, features_scaled, k)

    # Perform Stratified K-Means Clustering
    data_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    data_scaled['Diabetes_012'] = target

    print("\nPerforming Stratified K-Means Clustering on the dataset")
    stratified_kmeans(data_scaled)
    

if __name__ == "__main__":
    main()
