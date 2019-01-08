import streamlit as st
import numpy as np
import pandas as pd

def read_users():
    user_cols = ['user_id','age','gender','occupation','zip_code']
    return pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')

def read_ratings():
    rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    return pd.read_csv('../data/ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')

def read_movies():
    movie_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
    return pd.read_csv('../data/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')
