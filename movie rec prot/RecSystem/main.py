from sched import scheduler


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import os, numpy as np, argparse, time
    from tqdm import tqdm

    from config import getopt
    
    import dataloader
    
    import wandb

    import models
    from train_and_eval import train, evaluate



    opt = getopt()

    config = {
        'learning_rate' : opt.lr,
        'epochs' : opt.n_epochs,
        'batch_size' : opt.batch_size,
        'architecture' : opt.archname
        }

    # Set the Loss Function
    criterion = nn.MSELoss()

    # ------------------------------- Load Our Data ------------------------------ #
    train_dataset = dataloader.Movies(split=opt.trainset, opt=opt)
    val_dataset = dataloader.Movies(split=opt.testset, opt=opt)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

    # ------------------------------ Load Our Model ------------------------------ #


    model = models.MatrixFactorization(num_users=train_dataset.num_users,
                                           num_movies=train_dataset.num_movies,
                                           embedding_size=opt.embedding_size)
    model.load_state_dict(torch.load('model.pt'))
    model = model.to(opt.device)


    # ----------------------- Setting up Weight and Biases ----------------------- #
    if opt.wandb:
        w = wandb.init(project='Movie Recommender',
                       entity='movie-recommender-ai',
                       settings=wandb.Settings(start_method='thread'),
                       config=config)

        wandb.run.name = opt.description
    
    # --------------------------- Setting our Optimizer -------------------------- #
    optimizer = torch.optim.SGD(model.parameters(), nesterov=True, momentum=0.9, lr=opt.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

    # --------------------------- Training our Model ---------------------------- #
    _ = model.to(opt.device)
    wandb.watch(model, criterion, log="all")

    for epoch in range(opt.n_epochs):
        evaluate(val_dataloader=val_dataloader, model=model, criterion=criterion, epoch=epoch, opt=opt)
        if not opt.evaluate:
            _ = model.train()
            loss = train(train_dataloader=train_dataloader,
                         model=model, criterion=criterion,
                         optimizer=optimizer,
                         opt=opt, epoch=epoch,
                         val_dataloader=val_dataloader,
                         wandb_log=opt.wandb)

            scheduler.step()

    # --------------------------- Saving the Model  ------------------------------ #
    torch.save(model.state_dict(), 'model.pt')
