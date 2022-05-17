#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from nltk.stem import SnowballStemmer
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import KNNBaseline
from surprise.model_selection import train_test_split
import surprise
from surprise import Dataset
from surprise import Reader


# In[2]:


movies = pd.read_csv('/Users/Amin/Documents/GitHub/Review-Sentiment-Analysis-with-Recommendation-System/data/imdb_subset.csv')


# In[3]:


movies['Title'] = movies['Title'].str.strip()
movies[['year', 'genre','director','actors','description']] = movies[['year', 'genre','director','actors','description']].astype('string')
movies = movies.fillna('')


# In[4]:


movies['genre'] = movies['genre'].str.replace(","," ")
movies['genre'] = movies['genre'].str.lower()
movies['director'] = movies['director'].str.replace(","," ")
movies['director'] = movies['director'].str.lower()
movies['actors'] = movies['actors'].str.replace(","," ")
movies['actors'] = movies['actors'].str.lower()
movies['description'] = movies['description'].str.lower()
movies['description'] = movies['description'].str.replace(","," ")
stop = nltk.corpus.stopwords.words('english')
movies['description'] = movies['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
stemmer = SnowballStemmer('english')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
def stemmer_text(text):
    return ' '.join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])
movies['description'] = movies['description'].apply(stemmer_text)


# In[5]:


movies['soup'] = movies['year'] +' '+ movies ['director'] + ' '+ movies ['director'] +' '+ movies['genre'] + ' '+movies ['description'] +' '+ movies ['actors']


# In[6]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(movies['soup'])


# In[7]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)
movie_sim = pd.DataFrame(cosine_sim, index=movies.Title, columns=movies.Title)


# In[8]:


def content_recommender(movie_title):
    
    # Selecting the target movie similarity matrix
    cosine_similarity_series = movie_sim.loc[movie_title]
    
    # Sort these values highest to lowest and pick the first 30 movie.
    ordered_similarities = cosine_similarity_series.sort_values(ascending=False)[1:31]
    # 
    return (ordered_similarities.index.tolist())


# In[9]:


ratings = pd.read_csv('/Users/Amin/Documents/GitHub/Review-Sentiment-Analysis-with-Recommendation-System/data/rating_db.csv')


# In[10]:


reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User_ID', 'imdb_id', 'rating']], reader)


# In[11]:


model = KNNBaseline(k = 40, bsl_options = {'method':'sgd', 'reg': 0.1, 'learning_rate': 0.01})


# In[12]:


full_trainset = data.build_full_trainset()
model.fit(full_trainset)


# In[13]:


indices = movies[['Title','imdb_id']].set_index('Title')
movies_db = movies[['Title','director','year','genre']].set_index('Title')


# In[14]:


def recommender(user_id,movie_title):
    
    ''' This function used the two previous filter and create the final recommandation movie list.
    First it creates the list of 50 movie by using the content based filter.
    Secound, using the suprise model, predict the vote that target user would gives to those movie.
    Third sort the 50 movie based on that predicted vote
    Finally, it return the top 10 movies as a final recommandation
    
    '''
    
    est=[]
    # First use the content based filter to make a list of movie that are close to the target movie
    rec_movies = content_recommender(movie_title)
    # subset the movie metadata to have only movie that the content base filter suggested
    movies_list = movies_db.loc[rec_movies]
    # Get the movie_id of the suggested movie
    ids =  indices.loc[rec_movies]['imdb_id']
    # Using the surprise model and collaborative filtering to predict the vote that the target user gives to the suggested movie
    for b in ids:
        stm = model.predict(user_id,b).est
        est.append(stm)
    movies_list['est'] = est
    
    # Add the estimated vote from collaborative filter and add it to our suggested movie dataset, and sort the data based on that estimated vote
    movies_list = movies_list.sort_values('est', ascending=False)
    # Return the top 10 as our final movie recommandation
    return movies_list[0:10]

