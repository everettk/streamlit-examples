import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error as rmse
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import sys
import streamlit as st


# movie_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
# user_cols = ['user_id','age','gender','occupation','zip_code']
# users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
# movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')

rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('../data/ml-20m/ratings.csv', sep=',', names=rating_cols, skiprows=1, dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int}, encoding='latin-1')
ratings = ratings.drop(['timestamp'], axis=1)

st.title('Run at scale')
