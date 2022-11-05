import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MF

def prep_df(path):
    df = pd.read_csv(path)
    user_ids = np.sort(df.userId.unique())
    item_ids = np.sort(df.movieId.unique())
    num_users = len(user_ids)
    num_items = len(item_ids)

    userid_to_idx = {o:i for i,o in enumerate(user_ids)}
    df["userId"] = df["userId"].apply(lambda x: userid_to_idx[x])

    itemid_to_idx = {o:i for i,o in enumerate(item_ids)}
    df["movieId"] = df["movieId"].apply(lambda x: itemid_to_idx[x])

    return df, num_users, num_items

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    # lr = 0.001
    # lr = 0.0001
    lr = 0.00001    
    num_epochs = 2
    num_features = 100
    save_interval = 50000

    # Set dataset_path to where the file called ratings.csv is located,
    # Set model_load_path to where you want to load a model from (put None if there is none),
    # Set model_save_path to where you want the model to be saved.
    dataset_path = "C:/UCF/Projects/Multus Medium/dataset/ml-25m/ratings.csv"
    model_load_path = "C:/UCF/Projects/Multus Medium/models/model_2_2.pt"
    model_save_path = "C:/UCF/Projects/Multus Medium/models"

    df, num_users, num_items = prep_df(dataset_path)

    model = MF(num_users, num_items, num_features).to(device)
    if(model_load_path != None):
        model.load_state_dict(torch.load(model_load_path))

    users = torch.LongTensor(df.userId.values)
    items = torch.LongTensor(df.movieId.values)
    ratings = torch.FloatTensor(df.rating.values)

    dataset = TensorDataset(users, items, ratings)
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, num_epochs + 1):
        accuracy = []
        for i, (user, item, targets) in enumerate(train_dl):
            user = user.to(device)
            item = item.to(device)
            targets = targets.to(device)

            preds = model(user, item).to(device)
            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_acc = (torch.sum(abs(preds - targets)) / len(preds)).item()
            accuracy.append(batch_acc)

            print(str(epoch) + "\tBatch loss: " + str(round(loss.item(), 4)) + "\t\tBatch avg difference: " + str(round(batch_acc, 2)))

            if i % save_interval == 0 and i != 0:
                torch.save(model.state_dict(), f"{model_save_path}/model_{epoch}_{int(i/save_interval)}.pt")
        print("Average difference from predicted rating to actual rating: " + str(round(sum(accuracy) / len(accuracy))))
        torch.save(model.state_dict(), f"{model_save_path}/model_{epoch}.pt")


if __name__ == "__main__":
    main()