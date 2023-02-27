# Multus Medium (Multimedia Recommendation System in Pytorch by AI@UCF)
An implementation of a Book Recommendation System by using Matrix Factorization.
The set of files and their format is the  same that is commonly used in the
Computer Vision Lab, and the one that we will start using from now on to create
our project (it provides a organized way of training our models, which will be
useful to train our models in the cloud and merge all of them when we create our
Multimedia System).

To start the training process, just run `main.py`:
```
>>> python main.py
```

By default, Weights and Biases is used to log the training process. To disable
it, just set `wandb` to `False` in `config.py`. In case you want to use it, you
will need to create an account in [Weights and Biases](https://wandb.ai/site) and
type your username in the call to wandb.init() in `main.py`.
