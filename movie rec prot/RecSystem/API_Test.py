from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json

import torch.nn.functional as F
import time

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=32):
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size


        # Create the embedding layers for our users and books
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        # Initialize the embeddings with a normal distribution
        self.user_embedding.weight.data.normal_(0, 0.1)
        self.movie_embedding.weight.data.normal_(0, 0.1)


    def forward(self, user_id, movie_id):
        # Get the embeddings for the user and book
        user_embedding = self.user_embedding(user_id)
        movie_embedding = self.movie_embedding(movie_id)

        # Compute the dot product between the embeddings
        dot_product = torch.sum(user_embedding * movie_embedding, dim=1)

        # Pass the dot product through a sigmoid function
        output = torch.sigmoid(dot_product)

        # Scale the output to be between 0 and 10
        rating = 10 * output

        return rating

    def predict_ratings(self, user_embedding, movie_ids):
        # Get the embeddings for the books
        movie_embeddings = self.movie_embedding(movie_ids)

        # Compute the dot product between the embeddings
        dot_product = torch.sum(user_embedding * movie_embeddings, dim=1)

        # Pass the dot product through a sigmoid function
        output = torch.sigmoid(dot_product)

        # Scale the output to be between 0 and 10
        ratings = 10 * output

        return ratings

def getIDLabelEncoder(path):
    """Given a path to a JSON file containing a sklearn LabelEncoder,
    load the LabelEncoder and return it. This LabelEncoder will be used to
    encode the Media IDs.

    Args:
        path (str): Path to the JSON file containing the LabelEncoder data.

    Returns:
        sklearn.preprocessing.LabelEncoder: LabelEncoder for the Media IDs.
    """
    with open(path, 'r') as f:
        encodings = json.load(f)
        id_encoder = LabelEncoder().fit(encodings['movie_ids'])

    return id_encoder


def get_n_random_ids(media_csv_path, n=10):
    """Given a CSV file containing the media(names of the books/movies/songs/videogames/etc and their IDs),
    return an list of n random IDs.

    Args:
        media_csv_path (str): Path to the CSV file containing the media.
        n (int, optional): Number of random IDs to return. Defaults to 10.

    Returns:
        list: List of n random IDs.
    """
    # Load the CSV file
    media = pd.read_csv(media_csv_path)

    # Get the IDs
    ids = media['movieId'].values

    # Get n random IDs
    random_ids = np.random.choice(ids, size=n, replace=False)

    return random_ids

def get_random_ratings(ids):
    """Given a list of Media IDs, return a list of random ratings for each ID
    in the form of a list of tuples (ID, rating).

    Args:
        ids (list): List of IDs.

    Returns:
        list: List of tuples (ID, rating).
    """
    # Get random ratings (0 - 5)
    ratings = np.random.randint(0, 6, size=len(ids))

    # Create a list of tuples (ID, rating)
    random_ratings = list(zip(ids, ratings))

    return random_ratings

def getNearestNeighborEmbedding(model, user_idx):
    """Given a trained model and a list of Media IDs and their ratings for a user,
    look through your dataset and find the user that has the most similar ratings
    to the user you are trying to predict for. Then, return the embedding for that user.

    Args:
        model (MatrixFactorization): Trained model.
        media_ratings (list): List of tuples (ID, rating).

    Returns:
        torch.Tensor (embedding_size): Embedding for the user that has the most similar
                                       ratings to the user you are trying to predict for.
    """
    # Get the embeddings for current user and then all other users.
    user_embeddings = model.user_embedding.weight[user_idx]
    other_users = torch.cat([model.user_embedding.weight[:user_idx], model.user_embedding.weight[user_idx + 1:]])

    # Compute the cosine similarities between the user's ratings and all other users
    # similarities = torch.nn.functional.cosine_similarity(user_embeddings, user_ratings, dim=1)
    similarities = torch.nn.functional.cosine_similarity(user_embeddings, other_users)

    # Find the index of the user with the highest similarity
    most_similar_user_index = torch.argmax(similarities)

    # Extract the embedding for the most similar user
    most_similar_user_embedding = other_users[most_similar_user_index]

    return most_similar_user_embedding

def getApproximateEmbedding(model, media_ratings, lr=0.01, num_iterations =10):
    """Given a trained model and a list of Media IDs and their ratings for a user,
    compute the approximate embedding for that user. You can do this by getting the
    embeddings for each media that the user has rated minimising the loss between
    the predicted rating and the actual rating using the predict_ratings function.

    Try starting with a random embedding for the user and then iteratively updating
    the embedding to minimise the loss. Feel free to experiment with other methods!
    At the end of the day, the goal is to get a good embedding for the user.

    Args:
        model (MatrixFactorization): Trained model.
        media_ratings (list): List of tuples (ID, rating).

    Returns:
        torch.Tensor (embedding_size): Approximate embedding for the user.
    """
    # Create a tensor of the user's ratings
    user_ratings = torch.tensor([rating for _, rating in media_ratings]).float()

    # Get the IDs of the media the user has rated
    media_ids = torch.tensor([media_id for media_id, _ in media_ratings])

    # Initialize the user embedding with random values
    user_embedding = torch.randn((model.embedding_size,), requires_grad=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Perform gradient descent to update the user embedding
    for i in range(num_iterations):
        # Compute the predicted ratings for the user
        predicted_ratings = model.predict_ratings(user_embedding, media_ids)

        # Compute the loss between the predicted and actual ratings
        loss = F.mse_loss(predicted_ratings, user_ratings)

        # Compute the gradients of the loss with respect to the user embedding
        loss.backward()

        # Update weights
        optimizer.step()

        # Reset gradients to 0
        optimizer.zero_grad()

    return user_embedding

def test_model(model_path, media_csv_path, label_encoder_path, number_users=100, number_predictions=10):
    """ Tests your trained model by generating random ratings for a random number of users
    and then trying to make predictions for those users.

    Args:
        model_path (str): Path to the trained model.
        media_csv_path (str): Path to the CSV file containing the media IDs and names.
        label_encoder_path (str): Path to the JSON file containing the LabelEncoder data.
        number_users (int, optional): Number of random users to generate ratings for. Defaults to 100.
        number_predictions (int, optional): Number of medias to make predictions for. Defaults to 10.
    """
    with open(label_encoder_path, 'r') as j:
        label_encoder = json.loads(j.read())
    # Load the model
    model = MatrixFactorization(num_users=label_encoder['num_users'], num_movies=label_encoder['num_movies'])

    # Load the model's state dictionary (trained weights)
    model.load_state_dict(torch.load(model_path))

    # Load the ID LabelEncoder
    id_encoder = getIDLabelEncoder(label_encoder_path)

    # Set the model to evaluation mode
    model.eval()

    time_start = time.time()
    # Loop through the number of users
    for i in range(number_users): 
        # Generate random IDs for the Media
        random_ids = get_n_random_ids(media_csv_path, n=number_predictions)

        # Encode the IDs
        ids_encoded = torch.tensor(id_encoder.transform(random_ids))

        # Generate random ratings
        random_ratings = get_random_ratings(ids_encoded)

        # Generate random IDs for medias that the user has not rated
        # but we want to make predictions for
        ids_to_predict = torch.tensor(id_encoder.transform(get_n_random_ids(media_csv_path, n=number_predictions)))

        # Get the nearest neighbor embedding
        nearest_neighbor_embedding = getNearestNeighborEmbedding(model, i)

        # Get the approximate embedding
        approximate_embedding = getApproximateEmbedding(model, random_ratings, number_predictions)

        # Make predictions for the user
        predictions_nn = model.predict_ratings(nearest_neighbor_embedding, ids_to_predict)
        predictions_aprox = model.predict_ratings(approximate_embedding, ids_to_predict)

        # Print the predictions
        print(f"==================== User {i} ====================")
        print(f"Nearest Neighbor Predictions: {predictions_nn}")
        print(f"Approximate Predictions: {predictions_aprox}")

    time_end = time.time()

    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    # Path to the trained model
    model_path = 'model.pt'
    # Path to the CSV file containing the media IDs and names
    media_csv_path = "Datasets/Movies/movies.csv"

    # Path to the JSON file containing the LabelEncoder data
    label_encoder_path = "label_encoder.json"

    # Number of random users to generate ratings for
    number_users = 100

    # Number of medias to make predictions for
    number_predictions = 50

    test_model(model_path=model_path, media_csv_path=media_csv_path, label_encoder_path=label_encoder_path, number_users=number_users, number_predictions=number_predictions)

