#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


# In[2]:


review = pd.read_csv('/Users/Amin/Documents/GitHub/Review-Sentiment-Analysis-with-Recommendation-System/data/process_db.csv')


# In[3]:


def recommendation_movie(movie_title):

    # Delete any duplicate for each user. If user review a movie more than once, only keep the last review
    review_db = review.drop_duplicates(subset=['User_ID', 'movie'], keep='last')
    
    # Make a pivot table of the User and movie and set the rating as a score
    user_ratings_table = review_db.pivot(index='User_ID', columns='movie', values='rating')
    
    # Because not all the user review all the movie, the table need to be normalized and then fill the NA with 0
    avg_ratings = user_ratings_table.mean(axis=1)

    user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)

    user_ratings_table_normed = user_ratings_table_centered.fillna(0)

    # For movie-movie based recommender, the similarity matrix between movies has to be created, therefore the Table need to be transpose.
    movie_ratings_centered = user_ratings_table_normed.T

    # Generate the similarity matrix
    similarities = cosine_similarity(movie_ratings_centered)

    # Wrap the similarities in a DataFrame
    cosine_similarity_df = pd.DataFrame(similarities, index=movie_ratings_centered.index, columns=movie_ratings_centered.index)
    
    # Selecting the target movie similarity matrix
    cosine_similarity_series = cosine_similarity_df.loc[movie_title]

    # Sort these values highest to lowest and pick the first 30 movie.
    ordered_similarities = cosine_similarity_series.sort_values(ascending=False)[1:31]

    return(ordered_similarities.index.tolist())


# In[4]:


def recommendation_user(user_id):
    
    # Delete any duplicate for each user. If user review a movie more than once, only keep the last review
    review_db = review.drop_duplicates(subset=['User_ID', 'movie'], keep='last')
    
    # Make a pivot table of the User and movie and set the rating as a score
    user_ratings_table = review_db.pivot(index='User_ID', columns='movie', values='rating')
    
    # Because not all the user review all the movie, the table need to be normalized and then fill the NA with 0
    avg_ratings = user_ratings_table.mean(axis=1)

    user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)

    user_ratings_table_normed = user_ratings_table_centered.fillna(0)
    
    # Generate the similarity matrix
    similarities_user = cosine_similarity(user_ratings_table_normed)

    # Wrap the similarities in a DataFrame
    user_cosine_similarity_df = pd.DataFrame(similarities_user, index=user_ratings_table_normed.index, columns=user_ratings_table_normed.index)
    
    user_cosine_similarity_series = user_cosine_similarity_df.loc[user_id]

    # Sort these values highest to lowest
    similar_users = user_cosine_similarity_series.sort_values(ascending=False)[1:51]
    similar_movie_df = user_ratings_table_normed[user_ratings_table_normed.index.isin(similar_users.index)]


    item_score = {}
    for i in similar_movie_df.columns:
      # Get the ratings for movie i
      movie_rating = similar_movie_df[i]
      # Create a variable to store the score
      total = 0
      # Create a variable to store the number of scores
      count = 0
      # Loop through similar users
      for u in similar_users.index:
        # If the movie has rating
        if pd.isna(movie_rating[u]) == False:
          # Score is the sum of user similarity score multiply by the movie rating
          score = similar_users[u] * movie_rating[u]
          # Add the score to the total score for the movie so far
          total += score
          # Add 1 to the count
          count +=1
      # Get the average score for the item
      item_score[i] = total / count
    # Convert dictionary to pandas dataframe
    item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])

    # Sort the movies by score
    ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)
    # Select top m movies
    
    return(ranked_item_score.movie[0:31].tolist())


# In[23]:


def recommender(user_id,movie_title):
    
    final = list(set(recommendation_movie(movie_title)).intersection(set(recommendation_user(user_id))))
    
    if final :
        return final 
    
    return ((recommendation_movie(movie_title)) + list(recommendation_user(user_id)))[0:6]


# In[24]:


recommender(2343,'Scary Movie 3 ')


# In[ ]:




