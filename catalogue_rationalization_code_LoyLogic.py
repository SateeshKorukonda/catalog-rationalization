#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Generating Data

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

def generate_catalog_data(n_products):

    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)

    # Define parameters
    categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports']
    start_date = datetime(2024, 11, 29)  # 6 months before May 29, 2025
    end_date = datetime(2025, 5, 29)

    # Generate product IDs
    product_ids = [f'P{str(i).zfill(4)}' for i in range(1, n_products + 1)]

    # Generate categories with realistic distribution (e.g., Clothing more common)
    category_weights = [0.25, 0.35, 0.20, 0.10, 0.10]  # Weighted probabilities
    category_data = random.choices(categories, weights=category_weights, k=n_products)

    # Generate product age (days since launch, 0 to 730 days ~ 2 years)
    product_age = np.random.exponential(scale=365, size=n_products).astype(int)
    product_age = np.clip(product_age, 0, 730)

    # Generate price and cost price based on category
    price_ranges = {
        'Electronics': (50, 1000),
        'Clothing': (10, 150),
        'Home': (20, 500),
        'Beauty': (5, 100),
        'Sports': (15, 300)
    }
    margin_ranges = {
        'Electronics': (0.1, 0.3),  
        'Clothing': (0.3, 0.6),
        'Home': (0.2, 0.5),
        'Beauty': (0.4, 0.7),  
        'Sports': (0.2, 0.5)
    }
    prices = []
    cost_prices = []
    for cat in category_data:
        price = random.uniform(price_ranges[cat][0], price_ranges[cat][1])
        margin = random.uniform(margin_ranges[cat][0], margin_ranges[cat][1])
        cost_price = price * (1 - margin)
        prices.append(round(price, 2))
        cost_prices.append(round(cost_price, 2))

    # Generate units sold with seasonal effects
    seasonal_factor = {
        'Electronics': [1.5, 1.5, 1.0, 0.8, 0.8, 0.9],  
        'Clothing': [1.8, 1.8, 0.9, 0.7, 0.7, 0.8],
        'Home': [1.2, 1.2, 1.0, 1.0, 1.0, 1.0],
        'Beauty': [1.6, 1.6, 0.8, 0.7, 0.7, 0.8],
        'Sports': [0.9, 0.9, 1.2, 1.2, 1.2, 1.0]
    }
    units_sold = []
    for i, cat in enumerate(category_data):
        base_sales = np.random.lognormal(mean=4, sigma=1.5)
        month_factor = seasonal_factor[cat][random.randint(0, 5)]  
        age_factor = 1.0 if product_age[i] > 90 else 0.5  # New products sell less
        units = base_sales * month_factor * age_factor
        units_sold.append(round(units, 1))
    units_sold = np.clip(units_sold, 0, 5000)
    # Outliers: viral products
    for i in random.sample(range(n_products), 15):
        units_sold[i] = random.randint(3500, 5000)

    # Generate revenue
    revenue = [units * price * random.uniform(0.95, 1.05) for units, price in zip(units_sold, prices)]
    revenue = [round(r, 2) for r in revenue]

    # Generate stock levels (correlated with units sold)
    stock_level = []
    for units, cat in zip(units_sold, category_data):
        base_stock = np.random.lognormal(mean=5, sigma=1.5)
        stock = base_stock * (0.5 if units > 1000 else 1.5 if units < 100 else 1.0)
        stock_level.append(int(np.clip(stock, 0, 10000)))
    # Outliers: stockouts and overstock
    for i in random.sample(range(n_products), 40):
        if random.random() < 0.4 and units_sold[i] > 1000:
            stock_level[i] = 0  # Stockout for high-demand products
        else:
            stock_level[i] = random.randint(6000, 10000)  # Overstock

    # Generate restock frequency (correlated with units sold and season)
    restock_frequency = []
    for i, units in enumerate(units_sold):
        base_freq = min(round(units / 100), 10)
        month_factor = seasonal_factor[category_data[i]][random.randint(0, 5)]
        freq = base_freq * month_factor
        restock_frequency.append(int(np.clip(np.random.normal(freq, 1), 0, 10)))
    # Outliers: no restocks for slow movers
    for i in random.sample(range(n_products), 60):
        if units_sold[i] < 50:
            restock_frequency[i] = 0

    # Generate days in inventory (inversely related to units sold)
    days_in_inventory = []
    for units, age in zip(units_sold, product_age):
        if units == 0:
            days = 180
        else:
            days = random.randint(0, min(180, age))
            days = days * (1.5 if units < 50 else 0.5 if units > 1000 else 1.0)
        days_in_inventory.append(int(days))
    # Outliers: old stock
    for i in random.sample(range(n_products), 50):
        days_in_inventory[i] = random.randint(150, 180)

    # Generate last sale date
    last_sale_date = []
    for units, days, age in zip(units_sold, days_in_inventory, product_age):
        if units == 0 or days >= min(180, age):
            last_sale_date.append(None)
        else:
            days_ago = random.randint(0, days)
            sale_date = end_date - timedelta(days=days_ago)
            last_sale_date.append(sale_date.strftime('%Y-%m-%d'))

    # Generate customer ratings (1–5, correlated with units sold)
    customer_rating = []
    for units in units_sold:
        if units == 0:
            rating = None
        else:
            base_rating = min(5, max(1, round(np.random.normal(3.5 + units / 1000, 0.5))))
            rating = base_rating if random.random() < 0.8 else None  # 20% missing
        customer_rating.append(rating)

    # Introduce missing values (tied to product age)
    for i in random.sample(range(n_products), 60):
        if product_age[i] < 30:  # New products
            units_sold[i] = np.nan
            revenue[i] = np.nan
            last_sale_date[i] = None
        elif units_sold[i] < 20:  # Low sellers
            last_sale_date[i] = None

    # Create DataFrame
    data = {
        'product_id': product_ids,
        'category': category_data,
        'product_age': product_age,
        'units_sold': units_sold,
        'revenue': revenue,
        'price': prices,
        'cost_price': cost_prices,
        'stock_level': stock_level,
        'restock_frequency': restock_frequency,
        'days_in_inventory': days_in_inventory,
        'last_sale_date': last_sale_date,
        'customer_rating': customer_rating
    }
    df = pd.DataFrame(data)

    # Add new columns as requested
    df['profit_margin'] = (df['price'] - df['cost_price']) / df['price']
    df['stock_to_sales_ratio'] = df['stock_level'] / (df['units_sold'] + 1)  # +1 to avoid division by zero
    df['sales_velocity'] = df['units_sold'] / (df['days_in_inventory'] + 1)  # +1 to avoid division by zero
    current_date = datetime(2025, 5, 30)  # Current date: May 30, 2025
    df['last_sale_date'] = pd.to_datetime(df['last_sale_date'], errors='coerce')
    df['days_since_last_sales'] = (current_date - df['last_sale_date']).dt.days
    df['days_since_last_sales'] = df['days_since_last_sales'].fillna(df['days_since_last_sales'].max())  # Fill missing with max

    return df

# Example usage
if __name__ == "__main__":
    df = generate_catalog_data(n_products=1000)



# 

# In[14]:


df


# # Pre-processing the data
# 

# In[15]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler

def handle_missing_values(df):
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Sales-related: assume 0 for no sales
    df_clean['units_sold'] = df_clean['units_sold'].fillna(0)
    df_clean['revenue'] = df_clean['revenue'].fillna(0)

    # Customer rating: fill with median to preserve distribution
    df_clean['customer_rating'] = df_clean['customer_rating'].fillna(df_clean['customer_rating'].median())

    # Numeric columns: fill with median for robustness
    for col in ['stock_level', 'restock_frequency', 'days_in_inventory', 
                'stock_to_sales_ratio', 'sales_velocity', 'days_since_last_sales']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Last sale date: if missing, assume no sales, set to start date (Nov 29, 2024)
    start_date = datetime(2024, 11, 29)
    df_clean['last_sale_date'] = df_clean['last_sale_date'].fillna(start_date)
    # Recalculate days_since_last_sales after filling
    current_date = datetime(2025, 5, 30)
    df_clean['days_since_last_sales'] = (current_date - df_clean['last_sale_date']).dt.days

    return df_clean

def handle_outliers(df):
    # Create a copy to avoid modifying the input
    df_clean = df.copy()

    # Numeric columns to check for outliers
    numeric_cols = ['units_sold', 'revenue', 'stock_level', 'restock_frequency', 
                    'days_in_inventory', 'profit_margin', 'stock_to_sales_ratio', 
                    'sales_velocity', 'days_since_last_sales']

    # IQR method: cap outliers at Q1 - 1.5*IQR and Q3 + 1.5*IQR
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    return df_clean

def encode_categorical(df):
    # Create a copy to avoid modifying the input
    df_clean = df.copy()

    # One-hot encode the 'category' column
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    category_encoded = encoder.fit_transform(df_clean[['category']])

    # Create column names for encoded features
    encoded_columns = [f"category_{cat}" for cat in encoder.categories_[0]]

    # Convert encoded data to DataFrame and join with original
    category_df = pd.DataFrame(category_encoded, columns=encoded_columns, index=df_clean.index)
    df_clean = pd.concat([df_clean, category_df], axis=1)

    # Drop original 'category' column to avoid redundancy
    df_clean = df_clean.drop(columns=['category'])

    return df_clean, encoder

def scale_features(df):
    # Create a copy to avoid modifying the input
    df_scaled = df.copy()

    # Numeric columns to scale
    numeric_cols = ['product_age', 'units_sold', 'revenue', 'price', 'cost_price', 
                    'stock_level', 'restock_frequency', 'days_in_inventory', 
                    'customer_rating', 'profit_margin', 'stock_to_sales_ratio', 
                    'sales_velocity', 'days_since_last_sales']

    # Apply StandardScaler (mean=0, std=1)
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # One-hot encoded columns (category_*) and product_id, last_sale_date not scaled
    return df_scaled, scaler


def preprocess_pipeline(n_products=1000, output_file='preprocessed_catalog_data_original.csv'):
    df = generate_catalog_data(n_products)
    df_preprocessed = df.copy()
    df_preprocessed = handle_missing_values(df_preprocessed)
    df_preprocessed = handle_outliers(df_preprocessed)
    df_preprocessed, encoder = encode_categorical(df_preprocessed)
    df_preprocessed, scaler = scale_features(df_preprocessed)
    df_preprocessed.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to '{output_file}'")
    print("\nSample of preprocessed data:")
    print(df_preprocessed.head())
    return df, df_preprocessed, encoder, scaler
# Run the pipeline
if __name__ == "__main__":
    df, df_preprocessed_original, encoder, scaler = preprocess_pipeline(n_products=1000)


# In[16]:


if __name__ == "__main__":
        df_preprocessed_original = pd.read_csv('preprocessed_catalog_data_original.csv')

df_preprocessed_original


# ### The data is cleaned and scaled now with added new features that will help us classify into the 4 categories.
# ### (The reason for these actions has been described in the Documentation of the Assignment (e.g : Why do we have to scale the data? What happens if we don't?))
# 
# ### Now, we will focus on Analysis this data to come up with a recommendation system based on the necessary/important features.
# 
# ### So, the next question is : Which features are the most important one?
# 
# ### Let's find out!

# In[17]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def create_proxy_labels(df):
    print("Creating proxy labels...")
    try:
        df_labeled = df.copy()
        low_sales = df['units_sold'].quantile(0.25)
        low_revenue = df['revenue'].quantile(0.25)
        high_days_inv = df['days_in_inventory'].quantile(0.75)
        high_days_since = df['days_since_last_sales'].quantile(0.75)
        high_sales = df['units_sold'].quantile(0.75)
        high_revenue = df['revenue'].quantile(0.75)
        high_velocity = df['sales_velocity'].quantile(0.75)
        high_stock_ratio = df['stock_to_sales_ratio'].quantile(0.75)
        low_velocity = df['sales_velocity'].quantile(0.25)
        conditions = [
            (df_labeled['units_sold'] < low_sales) & 
            (df_labeled['revenue'] < low_revenue) & 
            (df_labeled['days_in_inventory'] > high_days_inv) & 
            (df_labeled['days_since_last_sales'] > high_days_since),
            (df_labeled['units_sold'] > high_sales) & 
            (df_labeled['revenue'] > high_revenue) & 
            (df_labeled['sales_velocity'] > high_velocity),
            (df_labeled['stock_to_sales_ratio'] > high_stock_ratio) & 
            (df_labeled['sales_velocity'] < low_velocity),
        ]
        choices = ['Discontinue', 'Prioritize', 'Reprice']
        df_labeled['recommendation'] = np.select(conditions, choices, default='Promote')
        print("Proxy labels created successfully.")
        return df_labeled
    except Exception as e:
        print(f"Error in create_proxy_labels: {e}")
        raise

def analyze_feature_importance(df):
    print("Analyzing feature importance...")
    try:
        # Verify required columns
        required_cols = ['product_id', 'last_sale_date', 'recommendation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        X = df.drop(columns=['product_id', 'last_sale_date', 'recommendation'])
        y = df['recommendation']

        # Verify data integrity
        if X.empty or y.empty:
            raise ValueError("Feature matrix or target is empty")
        if X.isna().any().any() or y.isna().any():
            raise ValueError("NaN values detected in features or target")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, 
                                    min_samples_leaf=2, random_state=42)
        rf.fit(X_train, y_train)
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("\nFeature Importance (Top 10):")
        print(importance.head(10))
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        print(f"\nTrain Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        return importance, rf
    except Exception as e:
        print(f"Error in analyze_feature_importance: {e}")
        raise

def rf_analysis_pipeline(df_preprocessed):

    # Can remove this, this is just to check whether the columns are there or not.
    print("Starting Random Forest analysis pipeline...")
    try:
        # Verify df_preprocessed columns
        expected_cols = ['product_id', 'last_sale_date', 'product_age', 'units_sold', 'revenue', 
                         'price', 'cost_price', 'stock_level', 'restock_frequency', 
                         'days_in_inventory', 'customer_rating', 'profit_margin', 
                         'stock_to_sales_ratio', 'sales_velocity', 'days_since_last_sales',
                         'category_Electronics', 'category_Clothing', 'category_Home', 
                         'category_Beauty', 'category_Sports']
        missing_cols = [col for col in expected_cols if col not in df_preprocessed.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in df_preprocessed: {missing_cols}")

        df_rf = df_preprocessed.copy()
        df_rf = create_proxy_labels(df_rf)
        importance, model = analyze_feature_importance(df_rf)
        df_rf.to_csv('rf_labeled_catalog_data.csv', index=False)
        print("Labeled data saved to 'rf_labeled_catalog_data.csv'")
        print("\nSample of labeled data:")
        print(df_rf[['product_id', 'units_sold', 'revenue', 'stock_level', 'recommendation']].head())
        return df_rf, importance, model
    except Exception as e:
        print(f"Error in rf_analysis_pipeline: {e}")
        raise

# Example usage
if __name__ == "__main__":
        df_preprocessed_original = pd.read_csv('preprocessed_catalog_data_original.csv')

        # Run Random Forest analysis
        df_rf, importance, model = rf_analysis_pipeline(df_preprocessed_original)



# In[18]:


df_rf


# ## We used Random Forest. Now we will use K-Means Clustering to cluster the products into 4 categories.
# ## Let's see how it works.

# In[19]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

def apply_kmeans_clustering(df, n_clusters=4):
    print("Applying K-means clustering...")
    try:
        # Verify required columns
        expected_cols = ['product_id', 'last_sale_date', 'product_age', 'units_sold', 'revenue', 
                         'price', 'cost_price', 'stock_level', 'restock_frequency', 
                         'days_in_inventory', 'customer_rating', 'profit_margin', 
                         'stock_to_sales_ratio', 'sales_velocity', 'days_since_last_sales',
                         'category_Electronics', 'category_Clothing', 'category_Home', 
                         'category_Beauty', 'category_Sports']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Features for clustering
        X = df.drop(columns=['product_id', 'last_sale_date'])

        # Verify data integrity
        if X.empty:
            raise ValueError("Feature matrix is empty")
        if X.isna().any().any():
            raise ValueError("NaN values detected in features")

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)

        # Compute evaluation metrics
        silhouette = silhouette_score(X, df['cluster'])
        db_score = davies_bouldin_score(X, df['cluster'])
        ch_score = calinski_harabasz_score(X, df['cluster'])
        print(f"Silhouette Score: {silhouette:.3f} (range: -1 to 1, higher is better)")
        print(f"Davies-Bouldin Index: {db_score:.3f} (lower is better)")
        print(f"Calinski-Harabasz Index: {ch_score:.3f} (higher is better)")

        return df, kmeans, silhouette
    except Exception as e:
        print(f"Error in apply_kmeans_clustering: {e}")
        raise

def analyze_feature_importance(df):
    print("Analyzing feature importance...")
    try:
        X = df.drop(columns=['product_id', 'last_sale_date', 'cluster'])
        cluster_means = df.groupby('cluster')[X.columns].mean()
        between_cluster_variance = cluster_means.var()
        importance = pd.DataFrame({
            'Feature': between_cluster_variance.index,
            'Importance': between_cluster_variance.values / between_cluster_variance.sum()
        }).sort_values(by='Importance', ascending=False)
        print("\nFeature Importance (Top 10) from K-means Clustering:")
        print(importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(10))
        plt.title('Top 10 Feature Importance from K-means Clustering')
        plt.xlabel('Normalized Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_kmeans.png')
        plt.close()
        print("Feature importance plot saved to 'feature_importance_kmeans.png'")

        return importance, cluster_means
    except Exception as e:
        print(f"Error in analyze_feature_importance: {e}")
        raise

def map_clusters_to_recommendations(df, cluster_means):
    print("Mapping clusters to recommendations...")
    try:
        df_mapped = df.copy()
        labels = {}
        for cluster in cluster_means.index:
            mean_velocity = cluster_means.loc[cluster, 'sales_velocity']
            mean_sales = cluster_means.loc[cluster, 'units_sold']
            mean_stock_ratio = cluster_means.loc[cluster, 'stock_to_sales_ratio']
            mean_days_inv = cluster_means.loc[cluster, 'days_in_inventory']
            if mean_sales < df['units_sold'].quantile(0.25) and mean_days_inv > df['days_in_inventory'].quantile(0.75):
                labels[cluster] = 'Discontinue'
            elif mean_velocity > df['sales_velocity'].quantile(0.75) and mean_sales > df['units_sold'].quantile(0.75):
                labels[cluster] = 'Prioritize'
            elif mean_stock_ratio > df['stock_to_sales_ratio'].quantile(0.75) and mean_velocity < df['sales_velocity'].quantile(0.25):
                labels[cluster] = 'Reprice'
            else:
                labels[cluster] = 'Promote'
        df_mapped['recommendation'] = df_mapped['cluster'].map(labels)
        print("Cluster mapping completed.")
        return df_mapped
    except Exception as e:
        print(f"Error in map_clusters_to_recommendations: {e}")
        raise


def kmeans_analysis_pipeline(df_preprocessed, n_clusters=4):
    print(f"Starting K-means analysis pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # Copy df_preprocessed
        df_kmeans = df_preprocessed.copy()

        # Apply K-means clustering
        df_kmeans, kmeans, silhouette = apply_kmeans_clustering(df_kmeans, n_clusters)

        # Analyze feature importance
        importance, cluster_means = analyze_feature_importance(df_kmeans)

        # Map clusters to recommendations
        df_kmeans = map_clusters_to_recommendations(df_kmeans, cluster_means)

        # Save results
        df_kmeans.to_csv('kmeans_clustered_catalog_data.csv', index=False)
        print("Clustered data saved to 'kmeans_clustered_catalog_data.csv'")
        print("\nSample of clustered data:")
        print(df_kmeans[['product_id', 'units_sold', 'revenue', 'stock_level', 'cluster', 'recommendation']].head())
        print("\nCluster Distribution:")
        print(df_kmeans['recommendation'].value_counts())

        return df_kmeans, kmeans, importance, silhouette
    except Exception as e:
        print(f"Error in kmeans_analysis_pipeline: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Load df_preprocessed
        df_preprocessed_original = pd.read_csv('preprocessed_catalog_data_original.csv')

        # Run K-means analysis
        df_kmeans, kmeans, importance, silhouette = kmeans_analysis_pipeline(df_preprocessed_original)
    except Exception as e:
        print(f"Error in main execution: {e}")


# ## K-Means might not do well because it assumes the data are spread in a somewhat sperical manner. 

# ## We will use XGBoost (Reason given in the report)

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def create_proxy_labels(df):
    print("Creating proxy labels...")
    try:
        df_labeled = df.copy()
        low_sales = df['units_sold'].quantile(0.25)
        low_revenue = df['revenue'].quantile(0.25)
        high_days_inv = df['days_in_inventory'].quantile(0.75)
        high_days_since = df['days_since_last_sales'].quantile(0.75)
        high_sales = df['units_sold'].quantile(0.75)
        high_revenue = df['revenue'].quantile(0.75)
        high_velocity = df['sales_velocity'].quantile(0.75)
        high_stock_ratio = df['stock_to_sales_ratio'].quantile(0.75)
        low_velocity = df['sales_velocity'].quantile(0.25)
        conditions = [
            (df_labeled['units_sold'] < low_sales) & 
            (df_labeled['revenue'] < low_revenue) & 
            (df_labeled['days_in_inventory'] > high_days_inv) & 
            (df_labeled['days_since_last_sales'] > high_days_since),
            (df_labeled['units_sold'] > high_sales) & 
            (df_labeled['revenue'] > high_revenue) & 
            (df_labeled['sales_velocity'] > high_velocity),
            (df_labeled['stock_to_sales_ratio'] > high_stock_ratio) & 
            (df_labeled['sales_velocity'] < low_velocity),
        ]
        choices = ['Discontinue', 'Prioritize', 'Reprice']
        df_labeled['recommendation'] = np.select(conditions, choices, default='Promote')

        # Ensure all classes are represented
        label_map = {'Discontinue': 0, 'Prioritize': 1, 'Reprice': 2, 'Promote': 3}
        class_counts = df_labeled['recommendation'].value_counts()
        min_samples = 1  # Minimum samples per class
        for label in label_map.keys():
            if label not in class_counts or class_counts[label] < min_samples:
                # Assign the label to a random sample
                idx = df_labeled.sample(1).index
                df_labeled.loc[idx, 'recommendation'] = label

        print(f"Proxy labels created successfully. Total products: {len(df_labeled)}")
        return df_labeled
    except Exception as e:
        print(f"Error in create_proxy_labels: {e}")
        raise


def train_xgboost_classifier(df):
    print("Training XGBoost classifier...")
    try:
        # Verify required columns
        required_cols = ['product_id', 'last_sale_date', 'recommendation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        X = df.drop(columns=['product_id', 'last_sale_date', 'recommendation'])
        y = df['recommendation']

        # Verify data integrity
        if X.empty or y.empty:
            raise ValueError("Feature matrix or target is empty")
        if X.isna().any().any() or y.isna().any():
            raise ValueError("NaN values detected in features or target")
        if len(X) != 1000:
            print(f"Warning: Expected 1000 products, found {len(X)}")

        # Encode labels
        label_map = {'Discontinue': 0, 'Prioritize': 1, 'Reprice': 2, 'Promote': 3}
        y_encoded = y.map(label_map)

        # Verify label mapping
        if y_encoded.isna().any():
            raise ValueError("NaN values in encoded labels")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Verify test set size
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Train XGBoost with stronger regularization
        xgb_clf = xgb.XGBClassifier(
            n_estimators=50,            
            max_depth=3,                
            min_child_weight=10,        
            learning_rate=0.05,       
            reg_alpha=0.5,            
            reg_lambda=2.0,         
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred_train = xgb_clf.predict(X_train)
        y_pred_test = xgb_clf.predict(X_test)

        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        cv_scores = cross_val_score(xgb_clf, X, y_encoded, cv=5, scoring='accuracy')
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

        print(f"\nTrain Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"Precision (weighted): {precision:.3f}")
        print(f"Recall (weighted): {recall:.3f}")
        print(f"F1-Score (weighted): {f1:.3f}")

        # Feature importance
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': xgb_clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("\nFeature Importance (Top 10):")
        print(importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(10))
        plt.title('Top 10 Feature Importance from XGBoost')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance_xgb.png')
        print("Feature importance plot saved to 'feature_importance_xgb.png'")
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print("\nConfusion Matrix (Test Set):")
        cm_df = pd.DataFrame(cm, index=label_map.keys(), columns=label_map.keys())
        print(cm_df)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_map.keys(), yticklabels=label_map.keys())
        plt.title('Confusion Matrix for XGBoost Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_xgb.png')
        print("Confusion matrix plot saved to 'confusion_matrix_xgb.png'")
        plt.close()

        # Add predictions to df
        df['recommendation_pred'] = xgb_clf.predict(X)
        df['recommendation_pred'] = df['recommendation_pred'].map({v: k for k, v in label_map.items()})

        return df, xgb_clf, importance, cm
    except Exception as e:
        print(f"Error in train_xgboost_classifier: {e}")
        raise

def xgb_analysis_pipeline(df_preprocessed):
    print(f"Starting XGBoost analysis pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # Verify df_preprocessed columns
        expected_cols = ['product_id', 'last_sale_date', 'product_age', 'units_sold', 'revenue', 
                         'price', 'cost_price', 'stock_level', 'restock_frequency', 
                         'days_in_inventory', 'customer_rating', 'profit_margin', 
                         'stock_to_sales_ratio', 'sales_velocity', 'days_since_last_sales',
                         'category_Electronics', 'category_Clothing', 'category_Home', 
                         'category_Beauty', 'category_Sports']
        missing_cols = [col for col in expected_cols if col not in df_preprocessed.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in df_preprocessed: {missing_cols}")

        # Copy df_preprocessed
        df_xgb = df_preprocessed.copy()

        # Apply proxy labels
        df_xgb = create_proxy_labels(df_xgb)

        # Train XGBoost and evaluate
        df_xgb, xgb_clf, importance, cm = train_xgboost_classifier(df_xgb)

        # Save results
        df_xgb.to_csv('xgb_classified_catalog_data.csv', index=False)
        print("Classified data saved to 'xgb_classified_catalog_data.csv'")
        print("\nSample of classified data:")
        print(df_xgb[['product_id', 'units_sold', 'revenue', 'stock_level', 'recommendation', 'recommendation_pred']].head())
        print("\nRecommendation Distribution:")
        print(df_xgb['recommendation_pred'].value_counts())

        return df_xgb, xgb_clf, importance, cm
    except Exception as e:
        print(f"Error in xgb_analysis_pipeline: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Load df_preprocessed
        df_preprocessed_original = pd.read_csv('preprocessed_catalog_data_original.csv')

        # Run XGBoost analysis
        df_xgb, xgb_clf, importance, cm = xgb_analysis_pipeline(df_preprocessed_original)
    except Exception as e:
        print(f"Error in main execution: {e}")


# In[21]:


df_xgb


# ## Now we will work on Upgradability and Tuning our model (We will work on XGBoost since it is very efficient)

# ## Fine Tuning using Bayesian Optimization

# In[22]:


import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real

def tune_thresholds(df, xgb_clf, n_calls=50):
    print(f"Starting threshold tuning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # Features and true labels
        X = df.drop(columns=['product_id', 'last_sale_date', 'recommendation', 'recommendation_pred'])
        y = df['recommendation']
        label_map = {'Discontinue': 0, 'Prioritize': 1, 'Reprice': 2, 'Promote': 3}
        y_encoded = y.map(label_map)

        # Get probability predictions
        y_prob = xgb_clf.predict_proba(X)

        # Define custom scoring: increased weight for Discontinue for caution
        def custom_score(y_true, y_pred, weights={'Discontinue': 0.5, 'Promote': 0.3, 'Prioritize': 0.15, 'Reprice': 0.05}):
            precision = precision_score(y_true, y_pred, average=None, labels=list(label_map.values()), zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, labels=list(label_map.values()), zero_division=0)
            score = 0
            for i, label in enumerate(label_map.keys()):
                score += weights[label] * (0.6 * precision[i] + 0.4 * recall[i])
            return score

        # Objective function for Bayesian optimization (minimize negative score)
        def objective(thresholds):
            y_pred = np.argmax(np.where(y_prob >= thresholds, y_prob, 0), axis=1)
            return -custom_score(y_encoded, y_pred)  # Negative for minimization

        # Define search space: thresholds for all classes (0.1 to 0.9)
        space = [Real(0.1, 0.9, name=f'thresh_{label}') for label in label_map.keys()]

        # Run Bayesian optimization with more iterations
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=50,  # Increased for more precision
            n_random_starts=10,  # Initial random points
            random_state=42,
            verbose=False
        )

        # Extract best thresholds
        best_thresholds = {label: result.x[i] for i, label in enumerate(label_map.keys())}

        # Apply tuned thresholds
        y_prob_tuned = np.where(y_prob >= list(best_thresholds.values()), y_prob, 0)
        df['recommendation_tuned'] = np.argmax(y_prob_tuned, axis=1)
        df['recommendation_tuned'] = df['recommendation_tuned'].map({v: k for k, v in label_map.items()})

        # Evaluate
        precision = precision_score(y_encoded, df['recommendation_tuned'].map(label_map), 
                                   average='weighted', zero_division=0)
        recall = recall_score(y_encoded, df['recommendation_tuned'].map(label_map), 
                              average='weighted', zero_division=0)
        f1 = f1_score(y_encoded, df['recommendation_tuned'].map(label_map), 
                      average='weighted', zero_division=0)
        print(f"\n=== Tuning Results ===")
        print(f"Tuned Thresholds: {best_thresholds}")
        print(f"Precision (weighted): {precision:.3f}")
        print(f"Recall (weighted): {recall:.3f}")
        print(f"F1-Score (weighted): {f1:.3f}")
        print("\nTuned Recommendation Distribution:")
        print(df['recommendation_tuned'].value_counts())
        print("\nSample of Predictions (Original vs Tuned):")
        print(df[['product_id', 'recommendation', 'recommendation_pred', 'recommendation_tuned']].head())

        # Save results
        df.to_csv('xgb_tuned_catalog_data.csv', index=False)
        print("Tuned data saved to 'xgb_tuned_catalog_data.csv'")

        return df, best_thresholds
    except Exception as e:
        print(f"Error in tune_thresholds: {e}")
        raise

def tuned_xgb_pipeline(df_preprocessed):
    print(f"Starting tuned XGBoost pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # Apply proxy labels and train XGBoost
        df_xgb, xgb_clf, importance, cm = xgb_analysis_pipeline(df_preprocessed)
        print("\n=== XGBoost Training Complete ===")

        # Tune thresholds
        df_tuned, best_thresholds = tune_thresholds(df_xgb, xgb_clf)

        return df_tuned, xgb_clf, best_thresholds
    except Exception as e:
        print(f"Error in tuned_xgb_pipeline: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Load preprocessed data
        df_preprocessed_original = pd.read_csv('preprocessed_catalog_data_original.csv')

        # Run tuned pipeline
        df_tuned, xgb_clf, best_thresholds = tuned_xgb_pipeline(df_preprocessed_original)
    except Exception as e:
        print(f"Error in main execution: {e}")

