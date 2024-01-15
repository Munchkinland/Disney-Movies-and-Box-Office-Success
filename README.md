## Disney Movies Analysis
This project involves the analysis of Disney movies data, exploring various aspects such as voice actors, directors, revenues, and more. The analysis is carried out using the Python programming language and various libraries such as Pandas, Seaborn, and TPOT.

# Data Loading and Exploration
The project starts by loading multiple datasets related to Disney movies, including information about voice actors, directors, revenue streams, and total gross. Initial exploration involves displaying the first few rows of each dataset and summarizing their characteristics.

# Data Cleaning and Merging
Data consistency is ensured by normalizing movie titles and checking for common titles across different datasets. The datasets are then merged based on movie titles, resulting in a consolidated dataset that includes information about voice actors, directors, and total gross revenue.

# Data Preprocessing for Regression
The dataset is cleaned and prepared for regression modeling. This involves handling missing values, normalizing and transforming data, and eliminating duplicates. The cleaned dataset is saved as a CSV file for further analysis.

# Exploratory Data Analysis (EDA)
Exploratory Data Analysis is performed using various visualizations to understand the distribution of total and inflation-adjusted gross revenue, total gross revenue by genre, and the trend of total gross revenue over time. Special attention is given to the years 1990-2000 to observe revenue peaks.

# Univariate Analysis
Univariate analysis is conducted for each column, including counts of movies by genre and MPAA rating, distribution of total and inflation-adjusted gross revenue, and counts of top characters, voice actors, and directors.

# Multivariate Analysis
Multivariate analysis explores the relationships between variables, such as total revenue by genre and MPAA rating, inflation-adjusted revenue by genre and MPAA rating, the relationship between total and inflation-adjusted revenue, and top directors' impact on total revenue.

# Contingency Tables
Contingency tables are created to examine relationships between categorical variables. Tables are presented for genre and MPAA rating, character and voice actor, director and genre, and MPAA rating and release year.

# Correlation Matrix
A correlation matrix is generated to explore the relationships between numerical variables, providing insights into potential correlations between features.

# Dataset Backup for Regression
A backup of the dataset is created specifically for regression modeling. The dataset is exported to a CSV file for use in future regression analysis.

# Regression Modeling with TPOT
A regression model is built using TPOT (Tree-based Pipeline Optimization Tool). The dataset is preprocessed, including one-hot encoding for categorical variables, and then split into training and testing sets. TPOT is employed to automatically discover the best regression pipeline, and the resulting model is evaluated.

# Model Export
The best regression pipeline identified by TPOT is exported for future use. The pipeline is saved as both a Python script (best_pipeline_for_regression.py) and a joblib file (model_pipeline.pkl).

This README provides an overview of the entire Disney Movies Analysis project, from data loading and exploration to regression modeling and model export. Further details and insights can be found within the project scripts and notebooks.
