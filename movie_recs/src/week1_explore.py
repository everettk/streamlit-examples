import streamlit as st
import numpy as np
import pandas as pd

st.title('Exploring Data with Streamlit')
st.write("""
When working with a new dataset, it's important to explore and understand it.
Here we walk you through how we do this with the MovieLens dataset.

We begin by importing a few libraries, and examining the Users table.
You may notice the @st.cache line above the get_users() function. This makes it
so that
""")

st.info("""
1. Uncomment the next section and save. This will update the reporton the side.
""")

with st.echo():
    import streamlit as st
    import numpy as np
    import pandas as pd

    @st.cache
    def get_users():
        user_cols = ['user_id','age','gender','occupation','zip_code']
        return pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')

    users = get_users()
    st.subheader('Raw Data: users')
    st.write(users)

#TODO: finish the sentence below
st.write("""
You may have noticed the @st.cache line

Now that we see the Users table. There are natural questions we will
want to ask, e.g.
- how many users are there?
- how old do the users tend to be?
- what is the gender distribution of the users?
- what is the occuptation distribution of the users?
- where do the users live?
""")

st.write("""
Let's start with the easiest question first.
""")

st.subheader('How many users are there?')
st.write('Number of users: %s' % len(users))

st.subheader('How old do the users tend to be?')
st.write("""
To answer this question, let's build a bar chart representing the distribution
of ages across the users. We can do this with a Vega Lite chart as follows.
""")

with st.echo():
    st.vega_lite_chart(users, {
        'mark': 'bar',
        'encoding': {
            'x': {'field': 'age', 'type': 'ordinal'},
            'y': {'aggregate': 'count', 'type': 'quantitative'},
        },
    })

st.write("""
We can now do the same thing with gender and occupation.
""")

st.subheader('What is the gender distribution of the users?')
st.vega_lite_chart(users, {
    'mark': 'bar',
    'encoding': {
        'x': {'field': 'gender', 'type': 'ordinal'},
        'y': {'aggregate': 'count', 'type': 'quantitative'},

    },
})

st.subheader('What is the occupation distribution of the users?')
st.vega_lite_chart(users, {
    'mark': 'bar',
    'encoding': {
        'x': {'field': 'occupation', 'type': 'ordinal'},
        'y': {'aggregate': 'count', 'type': 'quantitative'},

    },
})

st.subheader('Where do the users live?')
st.write("""
We could build a histogram of the zipcode distribution, but unless you *really*
know your zipcodes, it would be difficult to make sense of.

It would be much more informative to plot the users on a map. We can do this,
but first we need to convert the zipcodes into latitudes and longitudes.
""")

with st.echo():
    from uszipcode import SearchEngine
    search = SearchEngine(simple_zipcode=True)
    users['lat'] =  users['zip_code'].apply(lambda z : search.by_zipcode(z).lat )
    users['lon'] =  users['zip_code'].apply(lambda z : search.by_zipcode(z).lng )
    st.write(users)

# TODO: the points on the map are pretty far apart - so how do i have the map zoomed out enough but with the dots still visible?
# TODO: this could be a cool narrative point where we discover that they are far apart AND there aren't that many duplicated zipcodes
# TODO: I had to play with the getRadius & zoom levels to make it look right -- that was a cool iterative process that would have been nice if run-on-save worked 100%!

st.write("""
Now, we can draw this on a map! We do this as follows:
""")

with st.echo():
    st.deck_gl_chart(
        viewport={
            'latitude': 35,
            'longitude': -100,
            'zoom': 3
        },
        layers=[{
            'type': 'ScatterplotLayer',
            'data': users,
            'radiusScale': 10,
            'getRadius': 1000
        }]
    )

st.write("""
There are many other ways in which we can explore the Users table.

But for now, let's take a look at the other tables. We can follow a very
similar exploration process for the Movies and Ratings tables too. We don't
include explanations for these two tables in this report, but do check out
the underlying code here.
""")
#TODO: add <here> link above


st.header('Movies')
movie_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War' ,'Western']
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')

st.subheader('Raw Data: movies')
st.write(movies)

st.subheader('How many movies are there?')
st.write('Number of movies: %s' % len(movies))

st.subheader('How does the genre breakdown look?')
genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War' ,'Western']
movie_genres = pd.DataFrame(movies, columns=genre_cols)
#st.write(item_genres)

genre_counts = movie_genres.sum()
genre_counts = genre_counts.reset_index()
genre_counts = genre_counts.rename({'index':'genre', 0:'count'}, axis=1)
st.write(genre_counts)

st.subheader('Genre Distribution')
st.vega_lite_chart(genre_counts, {
    'mark': 'bar',
    'encoding': {
        'x': {'field': 'genre', 'type': 'ordinal'},
        'y': {'field': 'count', 'type': 'quantitative'},

    },
})


st.header('Ratings')
rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')

st.subheader('Raw Data: ratings')
st.write(ratings)

st.subheader('How many ratings are there?')
st.write('Number of ratings: %s' % len(ratings))

st.subheader('What is the distribution of ratings?')

st.vega_lite_chart(ratings, {
    'mark': 'bar',
    'encoding': {
        'x': {'field': 'rating', 'type': 'ordinal'},
        'y': {'aggregate': 'count', 'type': 'quantitative'},

    },
})

st.subheader('What is the distribution of ratings per user??')
# TODO: sometimes stops here - bug report!
#TODO: the history of this line
#ratings_per_user = df_data.groupby(['user_id']).agg(['count'])

ratings_per_user = pd.DataFrame(ratings.groupby(['user_id']).size())
ratings_per_user = ratings_per_user.rename({0:'ratings_by_user'}, axis=1)
st.write(ratings_per_user)
st.vega_lite_chart(ratings_per_user, {
    'mark': 'bar',
    'encoding': {
        'x': {'field': 'ratings_by_user', 'type': 'ordinal'},
        'y': {'aggregate': 'count', 'type': 'quantitative'},
    },
})
# TODO: clean up imports in each file
# TODO: make sure it works with python 2.7
# TODO: how do we show off the interactive, iterative nature of working with streamlit?
# TODO: link to the next part of the tutorial
