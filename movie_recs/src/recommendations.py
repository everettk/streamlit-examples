import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_recs(
    users,\
    ratings,\
    gender,\
    occupation,\
    age,\
    location,\
    weight_age = 0.25,\
    weight_gender = 0.25,\
    weight_job = 0.25,\
    weight_zip = 0.25):

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

    def users_weighted(users, weight_gender, weight_job, weight_age, weight_zip):
        A = weight_gender * pd.get_dummies(users.gender)
        B = weight_job * pd.get_dummies(users.occupation)
        C = weight_age * pd.get_dummies(users['age'].apply(nearest_5years))
        D = weight_zip * pd.get_dummies(users['zip_code'].apply(nearest_region))
        return pd.concat([A,B,C,D], axis = 1)


    users_weighted = users_weighted(
        users, \
        weight_gender, \
        weight_job, \
        weight_age, \
        weight_zip)

    user_info = users_weighted.iloc[0].copy() #get an example user profile
    user_info.iloc[0:] = 0 #empty profile to fill with input user
    user_info[gender] = weight_gender
    user_info[nearest_5years(age)] = weight_age
    user_info[nearest_region(location)] = weight_zip
    user_info[occupation] = weight_job

    sim=[]
    for i in range(len(users_weighted)): #finds the
        sim.append(cosine_similarity([user_info], [users_weighted.iloc[i]]))
    item = np.argsort(sim, axis=0)[-3:]# 3 users with the highest similarity to input user
    item = item+1 #to get the correct indexing
    ratings_sort = ratings.sort_values('user_id', ascending=True)

    df_top_data = pd.DataFrame
    U1 = ratings_sort.loc[ratings_sort['user_id'] == item[0][0][0]]
    U2 = ratings_sort.loc[ratings_sort['user_id'] == item[1][0][0]]
    U3 = ratings_sort.loc[ratings_sort['user_id'] == item[2][0][0]]
    df_top_data = pd.concat([U1,U2,U3],axis=0)
    df_top_data = df_top_data.sort_values('user_id', ascending=True)
    df_top_data = df_top_data[df_top_data.rating >= 4] #must have 4-5 star rating
    top_movies_list = df_top_data['item_id'].value_counts().index.tolist()#.iloc[:5]
    top_movies_list = [x - 1 for x in top_movies_list] #correct indexing
    return top_movies_list[::]
