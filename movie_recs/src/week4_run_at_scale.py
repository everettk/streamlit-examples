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
n_users, n_movies = len(ratings.user_id.unique()), len(ratings.item_id.unique())


st.write(f'Number of ratings: {len(ratings)}')

from sklearn.model_selection import train_test_split

@st.cache
def split():
    return train_test_split(ratings, test_size=0.2)
x_train, x_test = split()
y_true = x_test.rating
st.write('x_train:')
st.write(x_train)

st.write('x_test:')
st.write(x_test)

st.subheader('Keras with Adam Optimizer')


import keras
import math
from streamlit.Chart import Chart

class MyCallback(keras.callbacks.Callback):
    def __init__(self, x_test, num_epochs):
        self._x_test = x_test
        self._num_epochs = num_epochs
        self._sample_tests = x_test[0:10]

    def on_train_begin(self, logs=None):
        st.header('Progress')
        self._summary_chart = self._create_chart('area', 300)

        st.header('Percentage Complete')
        self._progress = st.info('No stats yet.')
        self._progress.progress(0)

        st.header('Current Epoch')
        self._epoch_header = st.empty()

        st.header('A Few Tests')
        self._sample_test_results = st.empty()
        self._sample_test_results.text(self._sample_tests)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._epoch_header.text(f'Epoch in progress: {epoch}')

    def on_batch_end(self, batch, logs=None):
        rows = pd.DataFrame([[logs['mean_squared_error']]],
            columns=['mean_squared_error'])
        if batch % 100 == 99:
            self._summary_chart.add_rows(rows)
        batch_percent_complete = logs['batch']*logs['size'] /\
         self.params['samples']

        overall_percent_complete = self._epoch / self._num_epochs +\
         (batch_percent_complete / self._num_epochs)
        self._progress.progress(math.ceil(overall_percent_complete * 100))

    def on_epoch_end(self, epoch, logs=None):
        t = self._sample_tests
        prediction = np.round(self.model.predict([t.user_id, t.item_id]),0)
        self._sample_tests[f'epoch {epoch}'] = prediction
        self._sample_test_results.text(self._sample_tests)
        #TODO: would be cool for this to be a visualization instead of text!

    def _create_chart(self, type='line', height=0):
        empty_data = pd.DataFrame(columns=['mean_squared_error'])
        epoch_chart = Chart(empty_data, f'{type}_chart', height=height)
        epoch_chart.y_axis(type='number', orientation='right',
            y_axis_id="mse_axis", allow_data_overflow="true")
        epoch_chart.cartesian_grid(stroke_dasharray='3 3')
        epoch_chart.legend()
        getattr(epoch_chart, type)(type='monotone', data_key='mean_squared_error',
            stroke='#82ca9d', fill='#82ca9d',
            dot="false", y_axis_id='mse_axis')
        return st.DeltaConnection.get_connection().get_delta_generator()._native_chart(epoch_chart)

def adam_predictions(x_train, x_test):
    n_latent_factors = 3
    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
    user_input = keras.layers.Input(shape=[1],name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

    prod = keras.layers.dot([movie_vec, user_vec], axes = 1)

    model = keras.Model([user_input, movie_input], prod)
    model.compile('adam', 'mean_squared_error', metrics=["accuracy", "mae", "mse"])

    num_epochs = 3
    # model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating) ,epochs=num_epochs, verbose=0)
    model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating),epochs=num_epochs, verbose=0, callbacks=[MyCallback(x_test, num_epochs)])
    return np.round(model.predict([x_test.user_id, x_test.item_id]),0)

adam_preds = adam_predictions(x_train, x_test)
st.write('**Keras Adam Predictions**')
st.write(adam_preds)
st.write('**MSE for Keras Adam Prediction**: %s' % mean_squared_error(y_true, adam_preds))
