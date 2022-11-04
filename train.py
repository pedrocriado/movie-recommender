import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
from model import MF

def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    batch_size = 128
    lr = 0.001
    num_epochs = 10
    num_features = 100

    df = pd.read_csv("C:/UCF/Projects/Multus Medium/dataset/ml-25m/ratings.csv")
    user_ids = np.sort(df.userId.unique())
    item_ids = np.sort(df.movieId.unique())
    num_users = len(user_ids)
    num_items = len(item_ids)

    userid_to_idx = {o:i for i,o in enumerate(user_ids)}
    df["userId"] = df["userId"].apply(lambda x: userid_to_idx[x])

    itemid_to_idx = {o:i for i,o in enumerate(item_ids)}
    df["movieId"] = df["movieId"].apply(lambda x: itemid_to_idx[x])

    model = MF(num_users, num_items, num_features).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    users = torch.LongTensor(df.userId.values)
    items = torch.LongTensor(df.movieId.values)
    ratings = torch.FloatTensor(df.rating.values)

    dataset = TensorDataset(users, items, ratings)
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(num_epochs):
        for (user, item, targets) in train_dl:
            user = user.to(device)
            item = item.to(device)
            targets = targets.to(device)

            preds = model(user, item).to(device)
            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(loss.item())


if __name__ == "__main__":
    main()