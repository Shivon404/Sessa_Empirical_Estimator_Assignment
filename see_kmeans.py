# SEE.R converted to Python for specific data structure
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings('ignore')

# Function to calculate within-cluster variance (from your original SESSA code)
def within_cluster_variance(X, labels, centroids):
    variance = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            variance += np.sum(cdist(cluster_points, [centroid], 'sqeuclidean'))
    return variance / len(X)

# Enhanced SESSA function with adaptive alpha (from your original code)
def calculate_sessa(X, labels, centroids, alpha=0.1, cluster_count=None):
    """
    Calculate SESSA (Silhouette-Enhanced Self-Adjusting) score
    
    Parameters:
    -----------
    X : array-like
        Input data
    labels : array-like
        Cluster labels
    centroids : array-like
        Cluster centroids
    alpha : float
        Weight parameter for within-cluster variance
    cluster_count : int
        Number of clusters (used for adaptive alpha)
        
    Returns:
    --------
    sessa_score : float
        The calculated SESSA score
    silhouette : float
        Silhouette score component
    wcv : float
        Within-cluster variance component
    """
    # Calculate silhouette score (cohesion and separation metric)
    silhouette = silhouette_score(X, labels)
    
    # Calculate within-cluster variance (compactness metric)
    wcv = within_cluster_variance(X, labels, centroids)
    
    # Adaptive alpha: as cluster number increases, we penalize complexity more
    if cluster_count is not None:
        # Adjust alpha linearly with cluster count (higher clusters = higher alpha)
        adaptive_alpha = alpha * (1 + (cluster_count - 2) * 0.05)
    else:
        adaptive_alpha = alpha
    
    # Final SESSA score calculation
    sessa_score = silhouette - adaptive_alpha * wcv
    
    return sessa_score, silhouette, wcv

# Main SEE function adapted to work with your data structure
def See(data, features=['TIME', 'TOTAL_SPEND']):
    """
    Sequential Event Estimation function adapted for the specific data structure
    
    Parameters:
    -----------
    data : DataFrame
        Data with columns like LOCATION, TIME, TOTAL_SPEND, etc.
    features : list
        List of features to use for clustering
        
    Returns:
    --------
    DataFrame
        Enhanced data with cluster information and visualizations
    """
    print(f"Starting SEE analysis using features: {features}")
    
    # Ensure we have a copy to work with
    X = data[features].values
    
    # Check for any missing values
    if np.isnan(X).any():
        print("Warning: Data contains missing values. Handling them...")
        # Simple imputation by column means
        for col_idx in range(X.shape[1]):
            col_mean = np.nanmean(X[:, col_idx])
            X[np.isnan(X[:, col_idx]), col_idx] = col_mean
    
    # Calculate metrics for different values of k
    k_range = range(2, 11)
    sessa_scores = []
    silhouette_scores = []
    wcv_scores = []
    adaptive_alphas = []

    # Create dataframe to store all metrics
    metrics_df = pd.DataFrame(columns=['k', 'SESSA', 'Silhouette', 'WCV', 'Adaptive_Alpha'])

    for k in k_range:
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        # Calculate adaptive alpha
        adaptive_alpha = 0.1 * (1 + (k - 2) * 0.05)
        
        # Calculate SESSA with all components
        sessa_score, silhouette, wcv = calculate_sessa(X, labels, centroids, alpha=0.1, cluster_count=k)
        
        # Store scores
        sessa_scores.append(sessa_score)
        silhouette_scores.append(silhouette)
        wcv_scores.append(wcv)
        adaptive_alphas.append(adaptive_alpha)
        
        # Add to metrics dataframe
        metrics_df = metrics_df._append({
            'k': k,
            'SESSA': sessa_score,
            'Silhouette': silhouette,
            'WCV': wcv,
            'Adaptive_Alpha': adaptive_alpha
        }, ignore_index=True)

    # Find optimal k
    optimal_k = k_range[np.argmax(sessa_scores)]
    print(f"Optimal number of clusters based on SESSA: {optimal_k}")

    # Create visualizations
    fig = plt.figure(figsize=(16, 12))

    # 1. SESSA Score plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(k_range, sessa_scores, marker='o', linestyle='-', linewidth=2, markersize=10, color='#1f77b4')
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('SESSA Score')
    ax1.set_title('SESSA Score vs. Number of Clusters')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Component comparison plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(k_range, silhouette_scores, marker='s', linestyle='-', linewidth=2, markersize=8, 
             color='green', label='Silhouette Score')
    ax2.plot(k_range, wcv_scores, marker='^', linestyle='-', linewidth=2, markersize=8, 
             color='orange', label='Within-Cluster Variance')
    ax2.plot(k_range, adaptive_alphas, marker='*', linestyle='--', linewidth=2, markersize=8, 
             color='purple', label='Adaptive Alpha')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Score')
    ax2.set_title('Component Scores vs. Number of Clusters')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Final clustering with optimal k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=1234)
    labels_optimal = kmeans_optimal.fit_predict(X)
    centroids_optimal = kmeans_optimal.cluster_centers_

    # 3. Clustered data visualization
    ax3 = fig.add_subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))

    for i in range(optimal_k):
        cluster_points = X[labels_optimal == i]
        ax3.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], 
                   alpha=0.7, label=f'Cluster {i}')

    ax3.scatter(centroids_optimal[:, 0], centroids_optimal[:, 1], s=200, marker='X', 
               color='red', label='Centroids')
    ax3.set_xlabel(features[0])
    ax3.set_ylabel(features[1])
    ax3.set_title(f'Optimal Clustering (k={optimal_k})')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 4. Metrics table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    table_data = [
        ['k', 'SESSA', 'Silhouette', 'WCV', 'Alpha'],
    ]
    for _, row in metrics_df.iterrows():
        table_data.append([
            f"{row['k']:.0f}",
            f"{row['SESSA']:.4f}",
            f"{row['Silhouette']:.4f}",
            f"{row['WCV']:.4f}",
            f"{row['Adaptive_Alpha']:.4f}"
        ])

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Clustering Metrics Comparison')

    plt.tight_layout()
    plt.savefig('sessa_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create silhouette plot for optimal k
    silhouette_vals = silhouette_samples(X, labels_optimal)
    
    plt.figure(figsize=(10, 7))
    y_ticks = []
    y_lower, y_upper = 0, 0

    for i in range(optimal_k):
        cluster_silhouette_vals = silhouette_vals[labels_optimal == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        
        color = colors[i]
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                 height=1.0, edgecolor='none', color=color, alpha=0.7)
        
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(x=np.mean(silhouette_vals), color='red', linestyle='--')
    plt.yticks(y_ticks, [f'Cluster {i}' for i in range(optimal_k)])
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.title('Silhouette Analysis for Optimal Clustering')
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
    
    # Additional analysis: Cluster sizes and distribution
    cluster_sizes = np.bincount(labels_optimal)
    print("\nCluster size distribution:")
    for i in range(optimal_k):
        print(f"Cluster {i}: {cluster_sizes[i]} points ({cluster_sizes[i]/len(X)*100:.1f}%)")

    # Create a report dataframe with cluster statistics
    cluster_stats = pd.DataFrame(index=range(optimal_k))
    cluster_stats['Size'] = cluster_sizes
    cluster_stats['Percentage'] = (cluster_sizes / len(X) * 100).round(1)
    
    for f_idx, feature in enumerate(features):
        cluster_stats[f'Avg_{feature}'] = [X[labels_optimal == i, f_idx].mean() for i in range(optimal_k)]
        cluster_stats[f'Std_{feature}'] = [X[labels_optimal == i, f_idx].std() for i in range(optimal_k)]
        cluster_stats[f'Min_{feature}'] = [X[labels_optimal == i, f_idx].min() for i in range(optimal_k)]
        cluster_stats[f'Max_{feature}'] = [X[labels_optimal == i, f_idx].max() for i in range(optimal_k)]

    # Calculate average silhouette score per cluster
    cluster_stats['Avg_Silhouette'] = [silhouette_vals[labels_optimal == i].mean() for i in range(optimal_k)]

    print("\nCluster Statistics:")
    print(cluster_stats)

    # Add cluster labels to original data
    data_enhanced = data.copy()
    data_enhanced['SEE_Cluster'] = labels_optimal
    
    # Return the output as a dictionary
    return {
        'optimal_k': optimal_k,
        'kmeans_model': kmeans_optimal,
        'labels': labels_optimal,
        'centroids': centroids_optimal,
        'sessa_score': max(sessa_scores),
        'enhanced_data': data_enhanced,
        'cluster_stats': cluster_stats,
        'metrics': metrics_df
    }

# Main execution code
def run_see_analysis(data_file='data_csv.csv', features=['TIME', 'TOTAL_SPEND']):
    """
    Run the SEE analysis on the specified data file
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV data file
    features : list
        List of features to use for clustering
    """
    try:
        # Load the data
        print(f"Loading data from {data_file}...")
        data = pd.read_csv(data_file)
        print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Display column names to verify
        print("Available columns:", data.columns.tolist())
        
        # Verify that the required features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            print(f"Warning: The following features are missing from the data: {missing_features}")
            # Ask for alternative features
            valid_features = [f for f in features if f in data.columns]
            features = valid_features
            
        # Check data types and convert if necessary
        for feature in features:
            if data[feature].dtype == 'object':
                try:
                    data[feature] = pd.to_numeric(data[feature])
                    print(f"Converted {feature} to numeric type")
                except:
                    print(f"Warning: Could not convert {feature} to numeric, this may cause issues")
        
        # Run the SEE analysis
        result = See(data, features)
        
        # Print summary of results
        print("\nSummary of SEE Analysis:")
        print(f"Optimal number of clusters: {result['optimal_k']}")
        print(f"Best SESSA score: {result['sessa_score']:.4f}")
        
        # Display cluster statistics
        print("\nCluster Statistics:")
        print(result['cluster_stats'])
        
        # Save the enhanced data
        enhanced_file = 'enhanced_' + data_file
        result['enhanced_data'].to_csv(enhanced_file, index=False)
        print(f"\nEnhanced data saved to {enhanced_file}")
        
        print("\nAnalysis complete! Visualizations saved as 'sessa_analysis.png' and 'silhouette_analysis.png'")
        
        return result
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage in Jupyter notebook:
if __name__ == "__main__":
    # You would run this in your notebook:
    # result = run_see_analysis()
    
    # Optionally use different features:
    # result = run_see_analysis(features=['PC_HEALTHXP', 'PC_GDP'])
    
    # To analyze all possible feature pairs:
    # import itertools
    # all_features = ['TIME', 'PC_HEALTHXP', 'PC_GDP', 'USD_CAP', 'TOTAL_SPEND']
    # feature_pairs = list(itertools.combinations(all_features, 2))
    # for pair in feature_pairs:
    #     print(f"\nAnalyzing feature pair: {pair}")
    #     run_see_analysis(features=list(pair))
    
    print("SEE K-means module loaded successfully!")