import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#Importing the movies and ratings data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

#Cleaning the movies data and adding a year column
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#Splitting the genres into a list
movies_df['genres'] = movies_df.genres.str.split('|')
moviesWithGenres_df = movies_df.copy()

#Creating a dataframe where rows imply various movies and columns imply various genres
#Filling in 1 for a movie with the corresponding genres in the column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

#Cleaning the ratings data
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)

#Retrieving the movie ID for the movies taken in by the user
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

#Dot product to get weights which is the user's profile
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

#Get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

#Multiply the genres by the weights and then take the weighted average to find the recommended movies
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

#Sort the recommendation to bring the highly recommended movies to the top
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendedMovies = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(5).keys())]

#Print the top five recommendations
print("The top 5 recommended movies are:")
print(recommendedMovies['title'].head())
