import torch
import pytorch_lightning as pl
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
# model
###model = nn.Sequential(
#    nn.Linear(28 * 28, 64),
#    nn.ReLu(),\
#    nn.Linear(64, 64),
#    nn.Linear(),
#    nn.Dropout(0.1), # if we're overfiting
#    nn.Linear(64, 10)
###)
# resedual connected model
class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logit = self.l3(do)
        return logit
model = ResNet()
# define my optimiser
params = model.parameters()

# Adam, SGD, RMSprop, Adagrad,and NAG are some commonly used optimisers
optimiser = optim.SGD(params, lr=1e-2)

# Define my loss
loss = nn.CrossEntropyLoss()

#data
train_data = datasets.MNIST('data', train=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)



# my training loops
nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train() # because of dropout
    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b,-1)
        # 1 forward
        l = model(x) # l: logits
        # 2 compute the objective function
        j = loss(l, y)

        # 3 cleaning the gradients
        model. zero_grad()
        # params.grad.zero_()

        # 4 accumulate the partial derivatives of j wrt params
        j.backward()
        # params.grad.add_(dj/dparams)

        # 5 step in the opposite direction of the gradient
        optimiser.step()
        # with torch,no_grad(): params = params - eta * params.grad
        losses.append(j.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
    print(f'Epoch {epoch +1}', end= ', ')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'train loss: {torch.tensor(accuracies).mean():.2f}')


    losses = list()
    accuracies = list()
    model.eval()
    for batch in val_loader:
        x, y = batch

        # x: b x 1 x 28 x28
        b = x.size(0)
        x = x.view(b, -1)
        # 1 forward
        with torch.no_grad():

            l = model(x)  # l: logits

        # 2 compute the objective function
        j = loss(l, y)

        losses.append(j.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
    print(f'Epoch {epoch + 1}', end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'valiadtion accuracies: {torch.tensor(accuracies).mean():.2f}')


