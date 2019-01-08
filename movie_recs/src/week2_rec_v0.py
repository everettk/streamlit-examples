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

data_cols = ['user_id', 'item_id', 'rating', 'timestamp']
item_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
user_cols = ['user_id','age','gender','occupation','zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=data_cols, encoding='latin-1')

st.title('Recommendation System v0')
st.write("""
The recommended process for a machine learning project is similar to the process
of creating a car. For example, we start by building out a skateboard (the
minimum viable product) and then transition to a more complex model every
week (week 2: a scooter, week 3: a bike, etc) until ultimately we have a
car/motorcycle at the end of week 4. This will ensure that your success at
Insight because even if the final car is incomplete, you will still have a
working end-to-end deliverable.

In that spirit, in this section we build a very simple recommendation system,
using cosine similarity between users. We simply define a user, find
similar users (via cosine similarity), and recommend 5 movies based on the most
common liked movies among those similar users.
""")

st.subheader('User Input')
st.write("""
Since this is the MVP, we simply hardcode the user info (aka the input) as
follows:
""")

with st.echo():
    gender = 'F'
    occupation = 'scientist'
    age = 50
    location = 'CA'
    if location=='CA':
        location = '90000'

st.write("""
We also specify weights for each variable. Right now we are weighing each
variable equally. Later, we can adjust these weights to tune the
recommendations.
""")
with st.echo():
    weight_age = 0.25
    weight_gender = 0.25
    weight_job = 0.25
    weight_zip = 0.25

def nearest_5years(x, base=5):
    return int(base * round(float(x)/base))

def nearest_region(x1):
    x = x1[0]
    if x=='0' or x=='1':
        return 'Eastcoast'
    if x=='2' or x=='3':
        return 'South'
    if x=='4' or x=='5' or x=='6':
        return 'Midwest'
    if x=='7' or x=='8':
        return 'Frontier'
    if x=='9' :
        return 'Westcoast'
    else:
        return 'None'

st.subheader('Data Preprocessing')
st.write("""
Next, we convert all the categorical variables in the users dataframe into
dummy/indicator variables. This allows us to then weigh each variable
according to the input weights. You can check out the underlying code <HERE> to
see the definitions of nearest_5years and nearest_region).
""")
#TODO: add link above

with st.echo():
    def users_weighted(users, weight_gender, weight_job, weight_age, weight_zip):
        A = weight_gender * pd.get_dummies(users.gender)
        B = weight_job * pd.get_dummies(users.occupation)
        C = weight_age * pd.get_dummies(users['age'].apply(nearest_5years))
        D = weight_zip * pd.get_dummies(users['zip_code'].apply(nearest_region))
        return pd.concat([A,B,C,D], axis = 1)

    users_weighted = users_weighted(users, weight_gender, weight_job, weight_age, weight_zip)

st.write('Here is what the users dataframe looks like before conversion:')
st.write(users)
st.write('And here it is after the conversion:')
st.write(users_weighted)

st.subheader('Finding Similar Users')
st.write("""
Now that our data is prepared, we can look for similar users. First, let's
convert our user info so that it has the same variables as users_weighted.
""")

with st.echo():
    user_info = users_weighted.iloc[0].copy() #get an example user profile
    user_info.iloc[0:] = 0 #empty profile to fill with input user
    user_info[gender] = weight_gender
    user_info[nearest_5years(age)] = weight_age
    user_info[nearest_region(location)] = weight_zip
    user_info[occupation] = weight_job

st.write('user_info')
st.write(user_info)

st.write("""
Next, we look for the 3 most similar users.
""")

with st.echo():
    sim=[]
    for i in range(len(users_weighted)): #finds the
        sim.append(cosine_similarity([user_info], [users_weighted.iloc[i]]))
    item = np.argsort(sim, axis=0)[-3:]# 3 users with the highest similarity to input user
    item = item+1 #to get the correct indexing
    ratings_sort = ratings.sort_values('user_id', ascending=True)

st.write("""
Now, let's take a look at which movies these users like in common.
""")

with st.echo():
    #sort movie IDs/recommendations by user ID
    df_top_data = pd.DataFrame
    U1 = ratings_sort.loc[ratings_sort['user_id'] == item[0][0][0]]
    U2 = ratings_sort.loc[ratings_sort['user_id'] == item[1][0][0]]
    U3 = ratings_sort.loc[ratings_sort['user_id'] == item[2][0][0]]
    df_top_data = pd.concat([U1,U2,U3],axis=0)
    df_top_data = df_top_data.sort_values('user_id', ascending=True)
    df_top_data = df_top_data[df_top_data.rating >= 4] #must have 4-5 star rating
    top_movies_list = df_top_data['item_id'].value_counts().index.tolist()#.iloc[:5]
    st.write(df_top_data['item_id'].value_counts())

st.write("""
The above shows us for each movie ID, the number of similar users who rated it
4 or above. Looks like there are 6 such movies. We return 5 of them.
""")

with st.echo():
    top_movies_list = [x - 1 for x in top_movies_list] #correct indexing
    idx = top_movies_list[::]
    st.write(movies['movie_title'].loc[idx[0:5]])

st.write("""
Awesome! We now have an end-to-end movie recommendation system! Feel free to
check out the code <HERE> and play with it. Clearly, this is a fairly naive
approach but now we have a baseline from which we can improve.

One of the problems with this approach is due to the fact that the data we have
is very sparse. We do not know the rating for most (user, movie) pairs. In the
next part of this tutorial, we explore approaches where we address this problem
among others.
""")

#TODO: is it just the popular movies and how often does it not have enough common movies to give us recs!

#TODO: you learn that the data is very sparse: for administrator e.g. there are only 2 movies in common
# with rating above 4.
