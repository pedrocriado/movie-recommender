# Given that the original Books dataset does has a single CSV file with the ratings,
# we will need to manually split it into train and test sets.

import pandas as pd
import numpy as np

# Load the Ratings CSV file
df = pd.read_csv('Datasets/Movies/Ratings.csv')

# Remove ratings whose User ID or Book ID is not in the Users or Books CSV file
movies = pd.read_csv('Datasets/Movies/movies.csv')


df = df[df['movieId'].isin(movies['movieId'])] # Remove ratings for books not in Books.csv
#df = df[df['userid'].isin(movies['userid'])] # Remove ratings for users not in Users.csv

# Set random state for reproducibility
random_state = 0

# Split the data into train and test sets
train = df.sample(frac=0.8, random_state=random_state)
test = df.drop(train.index)

# Save the train and test sets to CSV files
train.to_csv('Datasets/Movies/Rating_training.csv', index=False)
test.to_csv('Datasets/Movies/Rating_test.csv', index=False)
