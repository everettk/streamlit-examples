# The published output of this file currently lives here:
#

import keras
import math
import numpy as np
import pandas as pd
import streamlit as st
import os.path
import urllib.request
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from streamlit.Chart import Chart
from zipfile import ZipFile

interactive_mode = False

st.title('Run at scale')
st.write("""
In this part, we take the matrix-filling part of the Week 3 and run it on a
[larger MovieLens dataset](https://grouplens.org/datasets/movielens/20m/).
This dataset has 20 million ratings, which is **200X** larger than what we
worked with in Week 3!

Sadly, this larger dataset does not have demographic information about users -
so we don't get to play with our full recommendation system this time. But let's
watch our model train on this 20 million row dataset!
""")

st.info("""
Uncomment the next section to download the large dataset, unzip it, and read in
ratings.csv.
""")

st.header("Downloading & Reading the Large Dataset")
st.write("""
Here we download the 20M MovieLens dataset, save it to the `tmp/` directory,
and then we read in ratings.csv. We do two things to save ourselves time. First,
we only download the zip if we haven't already. Second, we `@st.cache` the
`read_ratings()` function.
""")
rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
movielens_20m_url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
tmp_zip_path = "/tmp/ml-20m.zip"

st.subheader("Downloading MovieLens 20M Dataset")
if not os.path.isfile(tmp_zip_path):
    st.write("%s doesn't exist, so let's download it." % tmp_zip_path)
    progress = st.info('No stats yet.')
    progress.progress(0)

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = math.ceil(downloaded / total_size * 100)
        progress.progress(percent)

    urllib.request.urlretrieve(movielens_20m_url, tmp_zip_path, show_progress)
    st.write("Successfully downloaded to %s" % tmp_zip_path)
else:
    st.write("%s exists, so we skip the download step." % tmp_zip_path)

@st.cache
def read_ratings(tmp_zip_path):
    z = ZipFile(tmp_zip_path)
    # ratings = pd.read_csv(z.open('ml-20m/ratings.csv'), sep=',', names=rating_cols, skiprows=1, dtype={"user_id": int, "item_id": int, "rating": float}, encoding='latin-1')
    ratings = pd.read_csv("~/workspace/old/streamlit-examples-orig/movie_recs/data/ml-20m/ratings.csv", sep=',', names=rating_cols, skiprows=1, dtype={"user_id": int, "item_id": int, "rating": float}, encoding='latin-1')

    ratings = ratings.drop(['timestamp'], axis=1)
    return ratings

ratings = read_ratings(tmp_zip_path)
n_users, n_movies = len(ratings.user_id.unique()), len(ratings.item_id.unique())


# # # -----------------------------------------------------------------------------





@st.cache
def split(ratings):
    return train_test_split(ratings, test_size=0.2)
x_train, x_test = split(ratings)
y_true = x_test.rating

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
        self._sample_test_results.dataframe(self._sample_tests)


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
        self._sample_test_results.dataframe(self._sample_tests)
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

    num_epochs = 10
    # model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating) ,epochs=num_epochs, verbose=0)
    model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating),epochs=num_epochs, verbose=0, callbacks=[MyCallback(x_test, num_epochs)])
    return np.round(model.predict([x_test.user_id, x_test.item_id]),0), model

adam_preds, model = adam_predictions(x_train, x_test)
st.write('**Keras Adam Predictions**')
st.write(adam_preds)
st.write('**MSE for Keras Adam Prediction**: %s' % mean_squared_error(y_true, adam_preds))

st.write("""
Congratulations! You've now finished the final part of this project!
""")
st.balloons()

if not interactive_mode:
    st.write("""
    *Viewing this online? You can check out the underlying code
    [here](https://github.com/streamlit/streamlit-examples/blob/master/movie_recs/src/week4_run_at_scale.py).*
    """)
