import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import datetime


class VAEncoder(nn.Module):
    def __init__(self, n_hidden, n_latent, activation):
        super(VAEncoder, self).__init__()
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.linear1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)  # l_enc1
        self.linear2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)  # l_enc2
        self.mu_layer = nn.Linear(in_features=self.n_hidden, out_features=self.n_latent)  # q_mu
        self.log_var_layer = nn.Linear(in_features=self.n_hidden, out_features=self.n_latent)  # q_log_sigma

        # Define activation function
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))

        # Gaussian distribution parameters
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        log_var = torch.clamp(log_var, min=-10, max=10)

        return mu, log_var


class GaussDecoder(nn.Module):
    def __init__(self, n_hidden, out_dim, activation, out_activation):
        super(GaussDecoder, self).__init__()
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        self.mu_layer = nn.Linear(in_features=self.n_hidden, out_features=self.out_dim)
        self.std_layer = nn.Linear(in_features=n_hidden, out_features=self.out_dim)

        # Activation functions
        self.activation = activation
        self.out_activation = out_activation

    def forward(self, z):
        z = self.activation(self.fc1(z))
        z = self.activation(self.fc2(z))
        mu = self.out_activation(self.mu_layer(z))
        std = F.softplus(self.std_layer(z))

        return mu, std


class BernoulliDecoder(nn.Module):
    def __init__(self, n_hidden, d, activation):
        super(BernoulliDecoder, self).__init__()

        self.hidden_dim = n_hidden
        self.out_dim     = d
        self.fc1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        self.logits_layer = nn.Linear(in_features=self.n_hidden, out_features=self.d)
        self.activation = activation

    def forward(self, z):

        z = self.activation(self.fc1(z))
        z = self.activation(self.fc2(z))
        logits = self.logits_layer(z)

        return logits


class TDecoder(nn.Module):
    def __init__(self, n_hidden, d, activation, out_activation):
        super(TDecoder, self).__init__()

        self.n_hidden = n_hidden
        self.out_dim = d
        self.fc1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)

        self.mu_layer = nn.Linear(in_features=self.n_hidden, out_features=self.out_dim)
        self.log_sigma_layer = nn.Linear(in_features=self.n_hidden, out_features=self.out_dim)
        self.df_layer = nn.Linear(in_features=self.n_hidden, out_features=self.out_dim)  # df layer

        # Activation functions
        self.activation = activation  # Hidden layer activation, e.g., nn.ReLU()
        self.out_activation = out_activation  # Output activation for mu

        # Initialize weights using orthogonal initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Apply orthogonal initialization to all layers
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.mu_layer.weight)
        nn.init.orthogonal_(self.log_sigma_layer.weight)
        nn.init.orthogonal_(self.df_layer.weight)

    def forward(self, z):
        z = self.activation(self.fc1(z))
        z = self.activation(self.fc2(z))

        # Output layer for mu with output activation
        mu = self.out_activation(self.mu_layer(z))

        # Output layer for log_sigma with value clipping
        log_sigma = self.log_sigma_layer(z)
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)

        # Output layer for degrees of freedom (df), no activation
        df = self.df_layer(z)

        return mu, log_sigma, df


class BernoulliDecoderMiss(nn.Module):
    def __init__(self, n_hidden, d, missing_process):
        super(BernoulliDecoderMiss, self).__init__()

        self.n_hidden = n_hidden
        self.d = d
        self.missing_process = missing_process  # Type of missing process

        # Define layers for 'linear' and 'nonlinear' cases
        if missing_process == 'linear' or missing_process == 'nonlinear':
            self.fc1 = nn.Linear(in_features=n_hidden, out_features=d)  # Linear case: 1 dense layer
            if missing_process == 'nonlinear':
                self.fc_hidden = nn.Linear(in_features=n_hidden, out_features=n_hidden)  # Nonlinear: hidden layer

        # Define trainable parameters for 'selfmasking' and 'selfmasking_known' cases
        if missing_process == 'selfmasking' or missing_process == 'selfmasking_known':
            self.W = nn.Parameter(torch.randn(1, 1, d))  # Initialize W parameter
            self.b = nn.Parameter(torch.randn(1, 1, d))  # Initialize b parameter

    def forward(self, z):
        if self.missing_process == 'selfmasking':
            logits = - self.W * (z - self.b)

        elif self.missing_process == 'selfmasking_known':
            W_softplus = F.softplus(self.W)
            logits = - W_softplus * (z - self.b)

        elif self.missing_process == 'linear':
            logits = self.fc1(z)

        elif self.missing_process == 'nonlinear':
            z = torch.tanh(self.fc_hidden(z))  # Nonlinear hidden layer
            logits = self.fc1(z)

        else:
            print("Use 'selfmasking', 'selfmasking_known', 'linear' or 'nonlinear' as 'missing_process'")
            logits = None

        return logits


def compute_classic_ELBO(q_z, log_p_x_given_z):
    """
    Computes the Evidence Lower Bound (ELBO) for a Variational Autoencoder (VAE).

    Parameters:
    - q_z (torch.distributions.Normal): The approximate posterior distribution q(z|x).
    - log_p_x_given_z (torch.Tensor): Log-likelihood of the data given z, log(p(x|z)).

    Returns:
    - elbo (torch.Tensor): The computed ELBO value.
    """
    # Manual KL divergence between q(z|x) and p(z) where p(z) is standard normal
    mu = q_z.loc  # Mean of q(z|x)
    log_var = torch.log(q_z.scale.pow(2))  # Log variance (log(sigma^2))

    KL = torch.sum(KL_loss(mu, log_var), axis = -1)

    # Mean log-likelihood across the sample dimension (log(p(x|z)))
    log_p_x_given_z_mean = log_p_x_given_z.mean(dim=-1)

    elbo = (log_p_x_given_z_mean - KL).mean()

    return elbo




class notMIWAE(nn.Module):
    def __init__(self, X, Xval,
                 n_latent=50, n_hidden=100, n_samples=1,
                 activation=nn.Tanh(),
                 out_dist='gauss',
                 out_activation=None,
                 learnable_imputation=False,
                 permutation_invariance=False,
                 embedding_size=20,
                 code_size=20,
                 missing_process='selfmask',
                 testing=False,
                 name='/tmp/notMIWAE'):

        super(notMIWAE, self).__init__()

        # ---- data
        self.Xorg = X.copy()
        self.Xval_org = Xval.copy()
        self.n, self.d = X.shape

        # ---- missing
        self.S = np.array(~np.isnan(X), dtype=np.float32)
        self.Sval = np.array(~np.isnan(Xval), dtype=np.float32)

        if np.sum(self.S) < self.d * self.n:
            self.X = self.Xorg.copy()
            self.X[np.isnan(self.X)] = 0
            self.Xval = self.Xval_org.copy()
            self.Xval[np.isnan(self.Xval)] = 0
        else:
            self.X = self.Xorg
            self.Xval = self.Xval_org

        # ---- settings
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.activation = activation
        self.out_dist = out_dist
        self.out_activation = out_activation
        self.embedding_size = embedding_size
        self.code_size = code_size
        self.missing_process = missing_process
        self.testing = testing
        self.batch_pointer = 0
        self.eps = np.finfo(float).eps

        # ---- input
        self.x_pl = None
        self.s_pl = None
        self.n_pl = None

        if learnable_imputation and not testing:
            self.imp = nn.Parameter(torch.randn(1, self.d))

        # ---- encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.d, self.n_hidden),
            self.activation,
            nn.Linear(self.n_hidden, self.n_latent * 2)
        )

        # ---- decoder
        if out_dist in ['gauss', 'normal', 'truncated_normal']:
            self.decoder = nn.Sequential(
                nn.Linear(self.n_latent, self.n_hidden),
                self.activation,
                nn.Linear(self.n_hidden, self.d * 2)
            )
        elif out_dist == 'bern':
            self.decoder = nn.Sequential(
                nn.Linear(self.n_latent, self.n_hidden),
                self.activation,
                nn.Linear(self.n_hidden, self.d)
            )
        elif out_dist in ['t', 't-distribution']:
            self.decoder = nn.Sequential(
                nn.Linear(self.n_latent, self.n_hidden),
                self.activation,
                nn.Linear(self.n_hidden, self.d * 3)
            )

        # ---- missing process
        self.missing_decoder = nn.Sequential(
            nn.Linear(self.d, self.n_hidden),
            self.activation,
            nn.Linear(self.n_hidden, self.d)
        )

        # ---- optimizer
        # self.optimizer = optim.Adam(self.parameters())

    def compute_MIWAE_ELBO(self,lpxz, lqzx, lpz):
        """

        :param lpxz:
        :param lqzx:
        :param lpz:
        :return:
        """
        # ---- importance weights
        l_w = lpxz + lpz - lqzx

        # ---- sum over samples using log-sum-exp trick
        log_sum_w = torch.logsumexp(l_w, dim=1)

        # ---- average over samples
        log_avg_weight = log_sum_w - torch.log(torch.tensor(n_pl, dtype=torch.float32))

        # ---- average over minibatch to get the average log likelihood
        return log_avg_weight.mean()




# Losses

def gauss_loss(x, s, mu, log_sig2):
    """ Gauss as p(x | z) """
    eps = torch.finfo(torch.float32).eps

    p_x_given_z = - 0.5 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * log_sig2 \
                  - 0.5 * (x - mu) ** 2 / (torch.exp(log_sig2) + eps)

    return torch.sum(p_x_given_z * s, dim=-1)  # sum over d-dimension

def bernoulli_loss(x, s, y):
    eps = torch.finfo(torch.float32).eps
    p_x_given_z = x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps)
    return torch.sum(s * p_x_given_z, dim=-1)  # sum over d-dimension


def bernoulli_loss_miss(x, y):
    eps = torch.finfo(torch.float32).eps
    p_x_given_z = x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps)
    return torch.sum(p_x_given_z, dim=-1)  # sum over d-dimension

def KL_loss(q_mu, q_log_sig2):
    KL = 1 + q_log_sig2 - q_mu ** 2 - torch.exp(q_log_sig2)
    return - 0.5 * torch.sum(KL, dim=1)