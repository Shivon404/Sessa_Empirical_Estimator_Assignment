# Sessa Empirical Estimator (SEE) Implementation

## Project Overview
This project implements the Sessa Empirical Estimator for analyzing pharmaceutical prescription durations using different clustering algorithms.

## Authors
- Siobhan B. Leonor
- Heather M. Will

## Project Structure
- `assignment_main.ipynb`: Main implementation notebook
- `callable_functions.ipynb`: Reusable functions for SEE analysis
- `data_csv.csv`: Pharmaceutical Drug Spending dataset
- `clustered_data.csv`: Results from K-means clustering
- `data_with_dbscan_clusters.csv`: Results from DBSCAN clustering
- `data_with_gmm_clusters.csv`: Results from GMM clustering
- `dbscan_cluster_summary.csv`: Summary statistics for DBSCAN

## Features
1. Data preprocessing and simulation of prescription dates
2. Implementation of SEE using:
   - K-means clustering
   - DBSCAN clustering
   - Gaussian Mixture Models (GMM)
3. Comparative analysis of clustering methods
4. Statistical insights and visualizations

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib
