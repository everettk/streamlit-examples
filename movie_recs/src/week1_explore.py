import numpy as np
import pandas as pd
import streamlit as st

interactive_mode = True

st.title('Exploring Data with Streamlit')
st.write("""
When working with a new dataset, it's important to explore and understand it.
Here we walk you through how we do this with the MovieLens dataset.

To follow this guide step-by-step (*recommended*), in an interactive way,
simply follow the instructions in the blue boxes.

If you'd rather see the whole guide instantly: change the line at the top of
this file to: `interactive_mode = False`
and then uncomment *all* the lines below. You can uncomment multiple lines by
selecting all of them and then pressing `⌘ + /` (if you're on mac) or `ctrl + /`
(if you're on linux).
""")

if interactive_mode:
    st.info("""
    1. Uncomment the next section (i.e. up to the dashed line) and save your
    code. Streamlit runs on save. If this doesn't work, click on the report-pane
    and press `r` to rerun.
    """)

# st.write("""
# We begin by importing a few libraries, and examining the Users table.
# """)
#
# with st.echo():
#     import numpy as np
#     import pandas as pd
#     import streamlit as st
#
#     def get_users():
#         user_cols = ['user_id','age','gender','occupation','zip_code']
#         return pd.read_csv('../data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
#
#     users = get_users()
#     st.subheader('Raw Data: users')
#     st.write(users)
#
# st.write("""
# Now that we see the Users table. There are natural questions we will
# want to ask, e.g.
# - how many users are there?
# - how old do the users tend to be?
# - what is the gender distribution of the users?
# - what is the occuptation distribution of the users?
# - where do the users live?
# """)
#
# if interactive_mode:
#     st.info("""
#     2. Next up, we learn how to build vega lite charts with Streamlit. Uncomment
#     the next section see a few examples!
#     """)
#
# # # -----------------------------------------------------------------------------
#
# st.write("Let's start with the easiest question first.")
# st.subheader('How many users are there?')
# st.write('Number of users: %s' % len(users))
#
# st.subheader('How old do the users tend to be?')
# st.write("""
# To answer this question, let's build a bar chart representing the distribution
# of ages across the users. We can do this with a Vega Lite chart as follows.
# """)
#
# with st.echo():
#     st.vega_lite_chart(users, {
#         'mark': 'bar',
#         'encoding': {
#             'x': {'field': 'age', 'type': 'ordinal'},
#             'y': {'aggregate': 'count', 'type': 'quantitative'},
#         },
#     })
#
# st.write("""
# We can now do the same thing with gender and occupation.
# """)
#
# st.subheader('What is the gender distribution of the users?')
# st.vega_lite_chart(users, {
#     'mark': 'bar',
#     'encoding': {
#         'x': {'field': 'gender', 'type': 'ordinal'},
#         'y': {'aggregate': 'count', 'type': 'quantitative'},
#
#     },
# })
#
# st.subheader('What is the occupation distribution of the users?')
# st.vega_lite_chart(users, {
#     'mark': 'bar',
#     'encoding': {
#         'x': {'field': 'occupation', 'type': 'ordinal'},
#         'y': {'aggregate': 'count', 'type': 'quantitative'},
#
#     },
# })
#
# st.subheader('Where do the users live?')
# st.write("""
# To answer this question, we need to plot the users on a map. Streamlit has
# support for deck gl charts, but for that we need the latitude and longitude.
# We currently only have zipcodes. How do we convert?
# We found a [libary](https://pypi.org/project/uszipcode/) that can do this. Let's
# try it on a few examples.
# """)
#
# with st.echo():
#     from uszipcode import SearchEngine
#     search = SearchEngine(simple_zipcode=True)
#
# if interactive_mode:
#     st.info("""
#     3. Uncomment the next section to try the by_zipcode() function on a specific
#     zipcode. Replace 94612 with your own zipcode!
#     """)
#
# # # -----------------------------------------------------------------------------
# if interactive_mode:
#     st.write("Let's try our SearchEngine on a specific zipcode.")
#     st.text(search.by_zipcode(94612))
#
#     st.write("""
#     Cool! `by_zipcode` gives us a lot of information about the zipcode! We just
#     need two fields: `lat` and `lng`.
#     """)
#
#     st.info("""
#     3a. We don't need this in our final result, so comment out this section,
#     and uncomment the next to see how we can add the necessary columns to our users
#     dataframe.
#     """)
#
# # # -----------------------------------------------------------------------------
#
# users['lat'] =  users['zip_code'].apply(lambda z : search.by_zipcode(z).lat )
# users['lon'] =  users['zip_code'].apply(lambda z : search.by_zipcode(z).lng )
# st.write(users)
#
# if interactive_mode:
#     st.info("""
#     4. Great. Check out those beautiful lat & lon columns! Let's put this on a
#     map. Uncomment the next section.
#     """)
#
# # # -----------------------------------------------------------------------------
# if interactive_mode:
#     st.deck_gl_chart(
#         layers=[{
#             'type': 'ScatterplotLayer',
#             'data': users
#         }]
#     )
#
#     st.write("""
#     Ummm... where's our data? Are we zoomed too far out? Take a look at the
#     first user in the table above. His zip_code is 85711. Look this zipcode up
#     on Google Maps & see if you can zoom in to see any datapoints in that area.
#     (Hint: you should be able to see a few red dots if you zoom in enough).
#     """)
#
#     st.info("""
#     4a. Comment out this section and uncomment the next section.
#     """)
#
# # # -----------------------------------------------------------------------------
#
# if interactive_mode:
#     st.write("""
#     We've now centered the map around the latitude and longitude of the first
#     user. However, we still can't see the points on the map.
#     """)
#
#     st.deck_gl_chart(
#             viewport={
#                 'latitude': 32,
#                 'longitude': -110,
#                 'zoom': 3
#             },
#             layers=[{
#                 'type': 'ScatterplotLayer',
#                 'data': users
#             }]
#         )
#
#     st.info("""
#     4b. Let's draw our points with a larger radius (so that we can see them).
#     Comment out this section and uncomment the next.
#     """)
#
# # # -----------------------------------------------------------------------------
#
# st.deck_gl_chart(
#         viewport={
#             'latitude': 32,
#             'longitude': -110,
#             'zoom': 3
#         },
#         layers=[{
#             'type': 'ScatterplotLayer',
#             'data': users,
#             'radiusScale': 10,
#             'getRadius': 1000
#         }]
#     )
#
# if interactive_mode:
#     st.info("""
#     5. Almost done ... feel free to play with the viewport above to center the
#     map a little better. Once you're done, uncomment the next section.
#     """)
#
# # # -----------------------------------------------------------------------------
#
# if interactive_mode:
#     st.info("""
#     6. Uncomment the remainder of this file to see how we explore the other parts
#     of this dataset. Enjoy!
#     """)
#
# # # -----------------------------------------------------------------------------
#
# st.write("""
# There are many other ways in which we can explore the Users table.
#
# But for now, let's take a look at the other tables. We can follow a very
# similar exploration process for the Movies and Ratings tables too. We don't
# include explanations for these two tables in this report, but do check out
# the underlying code here.
# """)
#
# st.header('Movies')
# movie_cols = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War' ,'Western']
# movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')
#
# st.subheader('Raw Data: movies')
# st.write(movies)
#
# st.subheader('How many movies are there?')
# st.write('Number of movies: %s' % len(movies))
#
# st.subheader('How does the genre breakdown look?')
# genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War' ,'Western']
# movie_genres = pd.DataFrame(movies, columns=genre_cols)
#
# genre_counts = movie_genres.sum()
# genre_counts = genre_counts.reset_index()
# genre_counts = genre_counts.rename({'index':'genre', 0:'count'}, axis=1)
# st.write(genre_counts)
#
# st.subheader('Genre Distribution')
# st.vega_lite_chart(genre_counts, {
#     'mark': 'bar',
#     'encoding': {
#         'x': {'field': 'genre', 'type': 'ordinal'},
#         'y': {'field': 'count', 'type': 'quantitative'},
#
#     },
# })
#
# st.header('Ratings')
# rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
# ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')
#
# st.subheader('Raw Data: ratings')
# st.write(ratings)
#
# st.subheader('How many ratings are there?')
# st.write('Number of ratings: %s' % len(ratings))
#
# st.subheader('What is the distribution of ratings?')
#
# st.vega_lite_chart(ratings, {
#     'mark': 'bar',
#     'encoding': {
#         'x': {'field': 'rating', 'type': 'ordinal'},
#         'y': {'aggregate': 'count', 'type': 'quantitative'},
#
#     },
# })
#
# st.subheader('What is the distribution of ratings per user??')
# ratings_per_user = pd.DataFrame(ratings.groupby(['user_id']).size())
# ratings_per_user = ratings_per_user.rename({0:'ratings_by_user'}, axis=1)
# st.write(ratings_per_user)
# st.vega_lite_chart(ratings_per_user, {
#     'mark': 'bar',
#     'encoding': {
#         'x': {'field': 'ratings_by_user', 'type': 'ordinal'},
#         'y': {'aggregate': 'count', 'type': 'quantitative'},
#     },
# })
#
# st.info("Check out week2_rec_v0.py for the next part of this guide.")
#
# if not interactive_mode:
#     st.write("""
#     *If you're viewing this report online, you can check out the underlying code
#     [here](https://github.com/streamlit/streamlit-examples/blob/master/movie_recs/src/week1_explore.py).*
#     """)
