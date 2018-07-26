

from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np

#df_movies = pd.read_csv('ml-20m/movies.csv')
df_ratings = pd.read_csv('ml-20m/ratings.csv')[:2000000]


def rating_splitter(df):
    
    df['liked'] = np.where(df['rating']>=4, 1, 0)
    df = df[df.liked == 1]
    df['movieId'] = df['movieId'].astype('str')
    gp_user_like = df.groupby(['liked', 'userId'])
    return gp_user_like
def getMovies(gp_user_like):
    return ([gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups])
pd.options.mode.chained_assignment = None
gp_user_like = rating_splitter(df_ratings)
splitted_movies = getMovies(gp_user_like)
print(len(splitted_movies))#20000 rows =>312 users
import gensim
assert gensim.models.word2vec.FAST_VERSION > -1


import random

for movie_list in splitted_movies:
    random.shuffle(movie_list)



from gensim.models import Word2Vec
import datetime
start = datetime.datetime.now()

model = Word2Vec(sentences = splitted_movies, # We will supply the pre-processed list of moive lists to this parameter
                 iter = 5, # epoch
                 min_count = 10, # a movie has to appear more than 10 times to be keeped
                 size = 200, # size of the hidden layer
                 workers = 4, # specify the number of threads to be used for training
                 sg = 1, # Defines the training algorithm. We will use skip-gram so 1 is chosen.
                 hs = 0, # Set to 0, as we are applying negative sampling.
                 negative = 5, # If > 0, negative sampling will be used. We will use a value of 5.
                 window = 9999999)

print("Time passed: " + str(datetime.datetime.now()-start))
model.save('item2vec_20180327.model')



