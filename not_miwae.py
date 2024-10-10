import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import torchrl


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
    def __init__(self, nb_features, n_latent=50, n_hidden=100, n_samples=1,
                 activation=nn.Tanh(), out_dist='gauss', out_activation=None,
                 learnable_imputation=False, missing_process='selfmask',
                 testing=False):
        super(notMIWAE, self).__init__()

        # Model settings
        self.nb_features = nb_features
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.activation = activation
        self.out_dist = out_dist
        self.out_activation = out_activation
        self.missing_process = missing_process
        self.testing = testing
        self.batch_pointer = 0
        self.eps = torch.finfo(torch.float32).eps

        # Encoder
        self.encoder = VAEncoder(self.n_hidden, self.n_latent, self.activation)

        # Decoder
        if out_dist in ['gauss', 'normal', 'truncated_normal']:
            self.decoder = GaussDecoder(self.n_hidden, self.nb_features, self.activation, self.out_activation)
        elif out_dist == 'bern':
            self.decoder = BernoulliDecoder(self.n_hidden, self.nb_features, self.activation)
        elif out_dist in ['t', 't-distribution']:
            self.decoder = TDecoder(self.n_hidden, self.nb_features, self.activation, self.out_activation)

        # Missing process
        self.missing_decoder = BernoulliDecoderMiss(self.n_hidden, self.nb_features,  missing_process)

        if learnable_imputation and not testing:
            self.imp = nn.Parameter(torch.randn(1, self.d))

    def forward(self, x, s):
        # Encoder
        q_mu, q_log_var = self.encoder(x)

        # Reparameterization trick
        q_z = dist.Normal(q_mu, torch.exp(0.5 * q_log_var))
        # Sample from the normal dist
        z = q_z.rsample((self.n_samples,))
        z = z.permute(1, 0, 2)

        # Decoder
        p_x_given_z = None
        if self.out_dist in ['gauss', 'normal', 'truncated_normal']:
            mu, std = self.decoder(z)
            if self.out_dist == 'truncated_normal':
                p_x_given_z = torchrl.modules.TruncatedNormal(loc=mu, scale=std, min=0.0, max=1.0)
            else:
                p_x_given_z = dist.Normal(mu, std)
        elif self.out_dist == 'bern':
            logits = self.decoder(z)
            p_x_given_z = dist.Bernoulli(logits=logits)
        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sigma, df = self.decoder(z)
            p_x_given_z = dist.StudentT(df=3 + F.softplus(df), loc=mu, scale=F.softplus(log_sigma) + 0.0001)
        else:
            raise ValueError("out_dist is not recognized.")

        # Compute log probabilities
        # TODO verify dims
        log_p_x_given_z = p_x_given_z.log_prob(x.unsqueeze(0).expand(self.n_samples, -1, -1))
        log_p_x_given_z = (log_p_x_given_z * s.unsqueeze(0)).sum(dim=-1)

        # Missing process
        l_out_mixed = p_x_given_z.sample() * (1 - s).unsqueeze(0) + x.unsqueeze(0) * s.unsqueeze(0)
        logits_miss = self.missing_decoder(l_out_mixed)
        # p(s|x)
        p_s_given_x = dist.Bernoulli(logits=logits_miss)
        # evaluate s in p(s|x)
        log_p_s_given_x = p_s_given_x.log_prob(s.unsqueeze(0)).sum(dim=-1)

        # --- evaluate the z-samples in q(z|x)
        q_z2 = dist.Normal(loc=q_mu.unsqueeze(1),
                      scale=torch.sqrt(torch.exp(q_log_var.unsqueeze(1))))
        log_q_z_given_x = torch.sum(q_z2.log_prob(z), dim=-1)

        # ---- evaluate the z-samples in the prior
        prior = dist.Normal(loc=0.0, scale=1.0)
        log_p_z = torch.sum(prior.log_prob(z), dim=-1)

        return log_p_x_given_z, log_p_s_given_x, log_q_z_given_x, log_p_z


def get_MIWAE(n_samples, lpxz, lqzx, lpz):
    l_w = lpxz + lpz - lqzx
    log_sum_w = torch.logsumexp(l_w, dim=0)
    log_avg_weight = log_sum_w - torch.log(torch.tensor(n_samples, dtype=torch.float32))
    return log_avg_weight.mean()

def get_notMIWAE(n_samples, lpxz, lpmz, lqzx, lpz):
    l_w = lpxz + lpmz + lpz - lqzx
    log_sum_w = torch.logsumexp(l_w, dim=0)
    log_avg_weight = log_sum_w - torch.log(torch.tensor(n_samples, dtype=torch.float32))
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


