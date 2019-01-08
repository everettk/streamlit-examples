import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error as rmse
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import sys
import streamlit as st

rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
movie_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
user_cols = ['user_id','age','gender','occupation','zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')

st.title('Iterating on our recommendation system')

#1
st.write("""
In part 2, we put together a very basic recommendation system, that we now want
to improve. We identified that the main problem with our current approach is
that the ratings matrix we have is very sparse. Only 7% of the matrix is filled.
Can we fix this?
""")

st.header('Filling in the Ratings Matrix')
st.write("""
In order to make the matrix less sparse, we need to predict user ratings for
unseen movies. We try two different techniques for doing this - both of which
use low-dimensional representations of the ratings matrix to predict the missing
values.

1. The first technique uses SciPy’s singular value decomposition, and
is inspired by [Agnes Johannsdottir’s work](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html).
2. The second approach uses matrix factorization in Keras (with an Adam
optimizer). This technique is inspired by [Nipun Batra’s work](https://nipunbatra.github.io/blog/2017/recommend-keras.html),
and has outperformed other ML algorithms for predicting ratings.
""")

#2
rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')
ratings = ratings.drop(['timestamp'], axis=1)
n_users, n_movies = len(ratings.user_id.unique()), len(ratings.item_id.unique())

st.write(f'Number of ratings: {len(ratings)}')

st.subheader('Preparing the train & test data')
st.write("""
First we need to split our data into a training and testing. We also capture the
true ratings for test data. We will use this later when we want to measure the
error of our predictions.
""")


with st.echo():
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

st.subheader('SVD')
st.write("""
Now that we've prepared our data, let's try out SVD. To better understand SVD,
we encourage you to st.write() u, s, and vt, as well as play with different
values for k.
""")

with st.echo():
    from scipy.sparse.linalg import svds

    @st.cache
    def convert_to_matrix(r):
        matrix = np.zeros((n_users, n_movies))
        for line in r.itertuples():
            matrix[line[1]-1, line[2]-1] = line[3]
        return matrix

    @st.cache
    def svds_predictions(x_train):
        train_data_matrix = convert_to_matrix(x_train)
        u, s, vt = svds(train_data_matrix, k = 5)
        x_pred = np.dot(np.dot(u, np.diag(s)), vt)
        return x_test.apply(lambda row : np.round(x_pred[row['user_id']-1, row['item_id']-1], 0), axis=1)

    svds_preds = svds_predictions(x_train)
    st.write('*SVDS Predictions*')
    st.write(svds_preds)

st.write("""
Well, how did we do? Let's measure the mean squared error.
""")
with st.echo():
    from sklearn.metrics import mean_squared_error
    st.write('**MSE for SVD**: %s' % mean_squared_error(y_true, svds_preds))

st.subheader('Keras with Adam Optimizer')
st.write("""
Let's do the same thing, but this time with Keras!

This training process takes
a long time - so we pass some callbacks to monitor the progress of our model
training process. If you look closely at our MyCallback class, you'll see that
we are tracking the mean squared error through the training process, and also
looking at the rating predictions for a small set of movies at the end of each
epoch.
""")

#TODO: this really needs to be a tutorial where you check out the code &
# run it & watch stuff happen coz otherwise you completely miss the interactivity
# and the animations!
with st.echo():
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

    @st.cache
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
        return np.round(model.predict([x_test.user_id, x_test.item_id]),0), model

    adam_preds, model = adam_predictions(x_train, x_test)
    st.write('**Keras Adam Predictions**')
    st.write(adam_preds)
    st.write('**MSE for Keras Adam Prediction**: %s' % mean_squared_error(y_true, adam_preds))

st.write("""
We've determined that the Keras approach achieves a significantly lower error,
so let's use that to fill our matrix. We do so as follows.
""")

with st.echo():
    @st.cache
    def filled_matrix():
        n_movies = np.arange(1,1683,1)
        n_users = np.arange(1,944,1)
        user_movie_matrixA = np.repeat(n_users, len(n_movies))
        user_movie_matrixB = np.tile(n_movies, len(n_users))
        user_movie_matrix = np.array([user_movie_matrixA,user_movie_matrixB])

        st.write('Starting predict ... ')
        all_rating = model.predict([user_movie_matrixA[::],user_movie_matrixB[::]])
        st.write('Finished predict.')

        df_users = pd.DataFrame(user_movie_matrixA)
        df_movies = pd.DataFrame(user_movie_matrixB)
        df_ratings = pd.DataFrame(all_rating)

        df_all_rate = pd.concat([df_users,df_movies,df_ratings],axis=1)
        df_all_rate.columns = ['user_id', 'item_id','rating']
        return df_all_rate

    filled_matrix = filled_matrix()

st.write("Do we get here immediately?")

st.header('Recommendations from a filled matrix')
#TODO: clean up cosine similarity code & getting recs so that it's easy to switch out!
