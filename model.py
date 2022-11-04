import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, features):
        super(MF, self).__init__()

        # Creates embedding modules with a shaped based on given number of users or items and the number of features
        self.user_embed = nn.Embedding(num_users, features)
        self.item_embed = nn.Embedding(num_items, features)

    def forward(self, u, v):
        # Given a userid and a movieid, return their embeddings from the embedding module (using the id as an index to get the embedding vector)
        u = self.user_embed(u)
        v = self.item_embed(v)

        # Return the dot product of the two embedding vectors
        return torch.dot(u, v)