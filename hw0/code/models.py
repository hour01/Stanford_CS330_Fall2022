"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.u1 = ScaledEmbedding(num_users, embedding_dim)
        self.q1 = ScaledEmbedding(num_items, embedding_dim)
        self.a1 = ZeroEmbedding(num_users, 1)
        self.b1 = ZeroEmbedding(num_items, 1)
        self.u1.reset_parameters()
        self.q1.reset_parameters()
        self.a1.reset_parameters()
        self.b1.reset_parameters()

        if not embedding_sharing:
            self.u2 = ScaledEmbedding(num_users, embedding_dim)
            self.q2 = ScaledEmbedding(num_items, embedding_dim)
            self.u2.reset_parameters()
            self.q2.reset_parameters()
        else:
            self.u2 = self.u1
            self.q2 = self.q1

        self.regression = nn.Sequential(
            nn.Linear(layer_sizes[0],layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1],layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1],1)
        )

        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        # [batch, 1]
        predictions = (torch.sum(self.u1(user_ids)*self.q1(item_ids),dim=1,keepdim=True)\
              + self.a1(user_ids) + self.b1(item_ids)).squeeze(1)
        
        # [batch, hidden]
        u, q = self.u2(user_ids), self.q2(item_ids)
        # in: [batch, 3*hidden]
        # score: [batch, 1]
        score = (self.regression(torch.concat([u,q,u*q],dim=1))).squeeze(1)

        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score