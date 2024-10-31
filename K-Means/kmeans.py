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


def generate_plots(data, features, features_2d, k):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=data['Cluster'], palette='viridis', alpha=0.6)
    plt.title('Clusters of Health Profiles in 2D PCA Space')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(labels=[f'Cluster {i}' for i in range(k)], title='Cluster')
    plt.savefig("./graphs/cluster_visualization.png")
    plt.close()

    # Save each feature's histogram in clusters for better understanding
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

def save_results(data, cluster_profiles, diabetes_distribution):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    with open("./results/cluster_profiles.txt", "w") as file:
        file.write("Cluster Profiles (Mean Feature Values):\n")
        file.write(cluster_profiles.to_string())

    with open("./results/diabetes_distribution.txt", "w") as file:
        file.write("Diabetes Distribution within Each Cluster:\n")
        file.write(diabetes_distribution.to_string())

    data.to_csv("./results/data_with_clusters.csv", index=False)


def traditional_clustering(data, features, features_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Add cluster labels to the data
    data['Cluster'] = clusters

    # Profile Each Cluster by calculating the average of each feature within each cluster
    cluster_profiles = data.groupby('Cluster').mean()
    print("\nCluster Profiles (Mean Feature Values):")
    print(cluster_profiles)

    # Analyze Cluster Composition in Terms of Diabetes Status
    diabetes_distribution = data.groupby(['Cluster', 'Diabetes_012']).size().unstack(fill_value=0)
    print("\nDiabetes Distribution within Each Cluster:")
    print(diabetes_distribution)

    # Visualize Clusters in 2D Space Using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    generate_plots(data, features, features_2d, k)

    # Save the results in a text file
    save_results(data, cluster_profiles, diabetes_distribution)


def stratified_kmeans(data_scaled, features, features_scaled, k):
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

    print("\nCluster Profiles by Diabetes Status (Mean Feature Values):")
    print(cluster_profiles)

    # Step 3: Plot feature comparisons between the groups
    cluster_profiles.drop(index='Diabetes_012', inplace=True)  # Remove target column
    cluster_profiles = cluster_profiles.T
    cluster_profiles.plot(kind='bar', figsize=(24, 10))

    plt.title('Feature Comparison by Diabetes Status Cluster')
    plt.xlabel('Health Indicators')
    plt.ylabel('Average (or Percentage) Value')

    # Place the legend horizontally at the bottom
    plt.legend(title='Diabetes Status', loc='upper center', ncol=5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./graphs/feature_comparison_by_diabetes_status.png")

def main():
    print("K-Means Clustering on CDC Diabetes Health Indicators Dataset")

    # Read data from labels.csv
    data = read_data("../data/labels.csv")

    # Clean the graphs folder
    clean_directory("./graphs")

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
    stratified_kmeans(data_scaled, features, features_scaled, k)
    

if __name__ == "__main__":
    main()
