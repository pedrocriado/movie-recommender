import os
from os.path import exists
import csv
import json
import torch

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from config import getopt

import glob
import random

from collections import Counter

# Let's create our own dataset class from the Books Dataset CSV Files! :D
# Pytorch has a Dataset class that we can inherit from to create our own dataset class,
# and it is flexible enough to do so by overriding the __init__, __len__, and __getitem__ methods.
# By using this template, we can create our own dataset class for any dataset we want to use;
# we just need the data, which usually comes in the form of a single or multiple CSV files.
# The dataset we will be using is the Books Dataset, which can be found on Kaggle:
# https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

class Movies(Dataset):
    def __init__(self, split='train', opt=None):
        # Set random seed for reproducibility
        np.random.seed(0) 
        
        # Set the split (train or test)
        self.split = split 
        
        # Load the Ratings CSV file (User-ID | ISBN | Book-Rating)
        if split == 'train':
            self.ratings = pd.read_csv('Datasets/Movies/Rating_training.csv')
        if split == 'test':
            self.ratings = pd.read_csv('Datasets/Movies/Rating_test.csv')

        # Get data of all users and books

        self.all_movies = pd.read_csv('Datasets/Movies/movies.csv', dtype={'movieId': int, 'title': str,
                                                                        'genres': str})
        self.all_users = pd.read_csv('Datasets/Movies/Users.csv', dtype={'User-Id': int})
        # Note: Including the dtype parameter in the pd.read_csv function (^) is not necessary, 
        # but it is good practice to do so, as it will prevent pandas from having to infer
        # the data type of each column, which can be slow for large datasets.

        # Get (User ID, Book ID, User Rating) tuples
        self.user_ids = self.ratings['userId'].values
        self.movie_ids = self.ratings['movieId'].values
        self.ratings = self.ratings['rating'].values

        # Set general attributes
        self.num_users = len(self.all_users)
        self.num_movies = len(self.all_movies)

        # Since we our ids are random values, we need to map them to a range of
        # integers starting from 0 that we can later use to index into our embedding
        # in our matrix factorization model. We will use the LabelEncoder class from
        # scikit-learn to do this for us.
        self.user_id_encoder = LabelEncoder().fit(self.all_users['User-ID'].values)
        self.movie_id_encoder = LabelEncoder().fit(self.all_movies['movieId'].values)

        num_users = len(self.user_id_encoder.classes_)
        user_ids = self.user_id_encoder.classes_.tolist()

        num_movies = len(self.movie_id_encoder.classes_)
        movie_ids = self.movie_id_encoder.classes_.tolist()

        with open('label_encoder.json', 'w') as f:
            json.dump({'num_users': num_users, 'user_ids': user_ids, 'num_movies': num_movies, 'movie_ids': movie_ids},
                      f)

        self.index_user_ids = self.user_id_encoder.transform(self.user_ids)
        self.index_movie_ids = self.movie_id_encoder.transform(self.movie_ids)

        print("Loaded data, total number of ratings: ", len(self.ratings))

    def __getitem__(self, idx):
        # The __getitem__ method is used to get a single item from the dataset
        # given an index. It is used by the DataLoader class to create batches of data.
        # Let's think, what do we need to return for each item?

        # Given this arbitrary index we will return the user ID of the user who rated
        # the book, the book ID of the book that was rated, and the rating that the
        # user gave to the book. We will also return the encoded user ID and book ID
        # as well, which we will use to index into our embedding matrix in our model.

        # Get the user ID, book ID, and rating 
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]

        # Convert Rating to Torch Tensor (fancy array)
        rating = torch.tensor(rating, dtype=torch.float32)

        # Encode the user ID and book ID
        index_user_id = self.index_user_ids[idx]
        index_movie_id = self.index_movie_ids[idx]

        return index_user_id, index_movie_id, rating, user_id, movie_id

    def __len__(self):
        # The __len__ method is used to get the total number of items in the dataset.
        # (this is how the dataloader knows how many batches to create, and which indices
        # are legal to use when calling __getitem__!)
        return len(self.ratings)

import argparse
if __name__ == "__main__":

    # ---------------------- Testing our Books Dataset class --------------------- #
    parser = argparse.ArgumentParser()
    opt = getopt() # Get the options from the config file

    # Create the dataset object and pass it to the DataLoader class
    dataset = Movies(opt=opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, drop_last=False)

    # Iterate through the dataloader to get the first batch of data
    for i, (encoded_user_id, encoded_movie_id, rating, user_id, movie_id) in enumerate(dataloader):
        print("Batch: ", i)
        print("Encoded User ID: ", encoded_user_id)
        print("Encoded Movie ID: ", encoded_movie_id)
        print("Rating: ", rating)
        print("User ID: ", user_id)
        print("Book ID: ", movie_id)
        
        break # Break the loop after the first batch


