import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, num_features):
        super(MF, self).__init__()

        # Creates embedding modules with a shaped based on given number of users or items and the number of features
        self.user_embed = nn.Embedding(num_users, num_features)
        self.item_embed = nn.Embedding(num_items, num_features)

    def forward(self, u, v):
        # Given a list of userids and itemids, return their embeddings from the embedding module
        u = self.user_embed(u)
        v = self.item_embed(v)

        # Return the list of dot product of all users and movies
        return (u*v).sum(1)