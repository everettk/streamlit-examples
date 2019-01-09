import keras
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from streamlit.Chart import Chart

rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
movie_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
user_cols = ['user_id','age','gender','occupation','zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')
ratings = ratings.drop(['timestamp'], axis=1)
n_users, n_movies = len(ratings.user_id.unique()), len(ratings.item_id.unique())

st.title('Iterating on our recommendation system')
st.write("""
In `week2_rec_v0.py`, we put together a very basic recommendation system, that
we now want to improve. We identified that the main problem with our current
approach is that the ratings matrix we have is very sparse. Only 7% of the
matrix is filled. Can we fix this?

In order to make the matrix less sparse, we need to predict user ratings for
unseen movies. We try two different techniques for doing this - both of which
use low-dimensional representations of the ratings matrix to predict the missing
values. In both cases, we are basically learning an embedding for movies and
users, and then in order to predict a rating for a specific (user, movie) pair -
we take the dot product between the two.

1. The first technique uses SciPy’s singular value decomposition (SVD), and
is inspired by [Agnes Johannsdottir’s work](https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html).
2. The second approach uses matrix factorization in Keras (with an Adam
optimizer). This technique is inspired by [Nipun Batra’s work](https://nipunbatra.github.io/blog/2017/recommend-keras.html),
and has outperformed other ML algorithms for predicting ratings.

""")

st.header('The Tale of 2 Prediction Techniques')
st.subheader('Preparing the train & test data')
st.write("""
First we need to split our data into a training and testing. We also capture the
true ratings for test data. We will use this later when we want to measure the
error of our predictions.
""")

st.info("""
1. Uncomment the next section to see how we split the data into training and
testing datasets.
""")

# # -----------------------------------------------------------------------------

# x_train, x_test = train_test_split(ratings, test_size=0.2)
# y_true = x_test.rating
# st.write('x_train:')
# st.write(x_train)
#
# st.write('x_test:')
# st.write(x_test)
#
# st.write("""
# You'll notice that every time we rerun our code, `x_train` and `x_test` are
# recalculated (and thus different).
# """)
#
# st.info("""
# 2a.  To help us reason about our code later on,
# let's cache the result so that it stays static as we iterate on this code.
# Comment this section. Uncomment the next.
# """)

# # -----------------------------------------------------------------------------

st.write("""
In Streamlit, you can cache a function by adding `@st.cache` right above it.
This tells Streamlit to not recompute the result *unless* something changes (e.g.
the input or the function body). This is a great way to speed up your code and
by wrapping up the slow parts of your code in it!
""")
with st.echo():
    @st.cache
    def split():
        return train_test_split(ratings, test_size=0.2)
    x_train, x_test = split()
    y_true = x_test.rating

st.info("""
2. Uncomment the next section to learn about SVD.
""")

# # -----------------------------------------------------------------------------

st.subheader('SVD')
st.write("""
Now that we've split our data, let's try out SVD. First, we convert `ratings`
into a matrix (n_users x n_movies). SVD then factorizes this matrix into smaller
matrices (`u`, `s`, `vt`). Taking their dot product then produces a fully filled
out matrix of predicted ratings.

We can then use this matrix to make predictions about our specific test cases in
`x_test`.
""")

@st.cache
def convert_to_matrix(r):
    matrix = np.zeros((n_users, n_movies))
    for line in r.itertuples():
        matrix[line[1]-1, line[2]-1] = line[3]
    return matrix

@st.cache
def svds_predicted_matrix(x_train):
    train_data_matrix = convert_to_matrix(x_train)
    u, s, vt = svds(train_data_matrix, k = 5)
    return np.dot(np.dot(u, np.diag(s)), vt)

@st.cache
def svds_predictions(x_train, x_test):
    x_pred = svds_predicted_matrix(x_train)
    return x_test.apply(lambda row : np.round(x_pred[row['user_id']-1, row['item_id']-1], 0), axis=1)

svds_preds = svds_predictions(x_test, x_test)
st.write('*SVDS Predictions for x_test*')
st.write(svds_preds)

st.info("""
3. Uncomment the next section to see how we did?
""")
# # -----------------------------------------------------------------------------

st.write("Well, how did we do? Let's measure the mean squared error.")
st.write('**MSE for SVD**: %s' % mean_squared_error(y_true, svds_preds))
st.info("4. Uncomment the next section to see how we do this with Keras.")

# # -----------------------------------------------------------------------------

# st.subheader('Keras with Adam Optimizer')
# st.write("""
# Let's do the same thing, but this time using Keras, with the Adam Optimizer.
# (Warning this takes a while so you'll have to wait...)
# """)
#
# @st.cache
# def adam_predictions(x_train, x_test):
#     n_latent_factors = 3
#     movie_input = keras.layers.Input(shape=[1],name='Item')
#     movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
#     movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
#     user_input = keras.layers.Input(shape=[1],name='User')
#     user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input)
#     user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
#
#     prod = keras.layers.dot([movie_vec, user_vec], axes = 1)
#
#     model = keras.Model([user_input, movie_input], prod)
#     model.compile('adam', 'mean_squared_error', metrics=["accuracy", "mae", "mse"])
#
#     num_epochs = 10
#     model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating) ,epochs=num_epochs, verbose=0)
#     return np.round(model.predict([x_test.user_id, x_test.item_id]),0), model
#
# adam_preds, model = adam_predictions(x_train, x_test)
# st.write('*Keras Adam Predictions for x_test*')
# st.write(adam_preds)
# st.write('**MSE for Keras Adam Prediction**: %s' % mean_squared_error(y_true, adam_preds))
#
# st.write("""
# Awesome. This does much better than the SVD algorithm above. But, wow, it's so
# much slower! `@st.cache` helps, but what if we want to iterate on our model?
# Perhaps we want to understand how the model is doing epoch-by-epoch? Do we
# really need 10 epochs? Is it too many? Do we need more? Or maybe
# we want to try out different values for `n_latent_factors`?
# """)
#
# st.info("""
# To see how we can sprinkle a few lines of streamlit into the Keras callbacks to
# get more insight into the training process, comment this section and uncomment
# the next.
# """)

# # -----------------------------------------------------------------------------

st.subheader('Keras with Adam Optimizer')
st.write("""
Here we have the same code as before for training the model, but we define some
of our own callbacks. This allows us to track the mean squared error through the
training process, and also looking at the rating predictions for a small set of
movies at the end of each epoch.
""")


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

    num_epochs = 10
    # model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating) ,epochs=num_epochs, verbose=0)
    model.fit([x_train.user_id, x_train.item_id], x_train.rating, validation_data=([x_test.user_id, x_test.item_id], x_test.rating),epochs=num_epochs, verbose=0, callbacks=[MyCallback(x_test, num_epochs)])
    return np.round(model.predict([x_test.user_id, x_test.item_id]),0), model

adam_preds, model = adam_predictions(x_train, x_test)
st.write('**Keras Adam Predictions**')
st.write(adam_preds)
st.write('**MSE for Keras Adam Prediction**: %s' % mean_squared_error(y_true, adam_preds))

st.write("""
We've determined that the Keras approach achieves a significantly lower error,
so let's proceed with filling the matrix with this algorithm. Uncomment the next
section.
""")
# # -----------------------------------------------------------------------------

st.subheader('Filling the Matrix')
st.write("""
We now fill the matrix with predictions using our model.
""")

@st.cache
def filled_matrix():
    n_movies = np.arange(1,1683,1)
    n_users = np.arange(1,944,1)
    user_movie_matrixA = np.repeat(n_users, len(n_movies))
    user_movie_matrixB = np.tile(n_movies, len(n_users))
    user_movie_matrix = np.array([user_movie_matrixA,user_movie_matrixB])

    st.write('Starting matrix fill process ... ')
    all_rating = model.predict([user_movie_matrixA[::],user_movie_matrixB[::]])
    st.write('Finished.')
    #TODO: should we add a progress bar for this too?

    df_users = pd.DataFrame(user_movie_matrixA)
    df_movies = pd.DataFrame(user_movie_matrixB)
    df_ratings = pd.DataFrame(all_rating)

    df_all_rate = pd.concat([df_users,df_movies,df_ratings],axis=1)
    df_all_rate.columns = ['user_id', 'item_id','rating']
    return df_all_rate

filled_matrix = filled_matrix()

# # -----------------------------------------------------------------------------

st.subheader('Making Recommendations from Filled Matrix')

#
# st.write("Do we get here immediately?")
#
# st.header('Recommendations from a filled matrix')
# #TODO: clean up cosine similarity code & getting recs so that it's easy to switch out!
