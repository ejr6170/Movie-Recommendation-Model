import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import zipfile
import urllib.request
import os

def download_dataset():
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = 'ml-100k.zip'
    extract_path = 'ml-100k'
    if not os.path.exists(extract_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_path)
        print(f"Extracted data to {extract_path}.")
    else:
        print(f"Data already exists at {extract_path}. Skipping download.")

# load rating & movie data
def load_data():
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)
    
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], 
                         names=['movie_id', 'title'], header=None)
    
    ratings = ratings.merge(movies[['movie_id', 'title']], on='movie_id')
    return ratings, movies

# build and train model using matrix_factorization as it is more customizable then other libraries 
def matrix_factorization(ratings, n_factors=50):
    # Pivot ratings to user-item matrix
    rating_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    

    user_means = rating_matrix.mean(axis=1)
    def fill_row(row):
        return row.fillna(row.mean())
    rating_matrix_filled = rating_matrix.apply(fill_row, axis=1)
 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for train_idx, test_idx in kf.split(rating_matrix_filled):
        train_mat = rating_matrix_filled.iloc[train_idx]
        test_mat = rating_matrix_filled.iloc[test_idx]
        
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        user_factors = svd.fit_transform(train_mat)
        item_factors = svd.components_.T
        predicted_ratings = np.dot(user_factors, item_factors.T)
        
        test_actual = test_mat[test_mat > 0].stack()
        test_pred_indices = test_actual.index
        test_pred = [predicted_ratings[i - 1, j - 1] for i, j in test_pred_indices 
                     if i - 1 < predicted_ratings.shape[0] and j - 1 < predicted_ratings.shape[1]]
        rmse_scores.append(np.sqrt(mean_squared_error(test_actual.values[:len(test_pred)], test_pred)))
    
    cv_rmse = np.mean(rmse_scores)
    print(f"Cross-validated RMSE: {cv_rmse:.2f} (+/- {np.std(rmse_scores):.2f})")
    
    #Final model complete with all data
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(rating_matrix_filled)
    item_factors = svd.components_.T
    predicted_ratings = np.dot(user_factors, item_factors.T)
    
    return predicted_ratings, rating_matrix, svd, rating_matrix_filled

    # recommendation function 
def recommend_movies(user_id, predicted_ratings, rating_matrix, movies, top_n=5):

    try:
        user_index = rating_matrix.index.get_loc(user_id)
    except KeyError:
        print(f"User ID {user_id} not found in rating matrix. Available users: {rating_matrix.index.tolist()}")
        return []
    
    user_pred_ratings = predicted_ratings[user_index]
    
    rated_movies = rating_matrix.columns[rating_matrix.loc[user_id] > 0]
    unrated_movies = [mid for mid in rating_matrix.columns if mid not in rated_movies]
    
    if not unrated_movies:
        print(f"No unrated movies found for user {user_id}. All movies may be rated.")
        return []
    
    try:
        top_recommendations = sorted(unrated_movies, key=lambda x: user_pred_ratings[x - 1], reverse=True)[:top_n]
    except IndexError as e:
        print(f"Indexing error: {e}. Check predicted_ratings shape {predicted_ratings.shape} vs unrated_movies {len(unrated_movies)}")
        return []
    
    recommended_titles = movies[movies['movie_id'].isin(top_recommendations)]['title'].tolist()
    return recommended_titles

# Main execution 
if __name__ == "__main__":
    download_dataset()
    ratings, movies = load_data()
    predicted_ratings, rating_matrix, svd, train_matrix = matrix_factorization(ratings, n_factors=50)
    user_id = 1
    recommendations = recommend_movies(user_id, predicted_ratings, rating_matrix, movies, top_n=5)
    print(f"Top 5 recommendations for user {user_id}:")
    if recommendations:
        for title in recommendations:
            print(f"- {title}")
    else:
        print("No recommendations generated due to data or indexing issues.")