import torch.nn as nn
import torch
import torch.distributions as dist
import torch.nn.functional as F

from code.code_UCI_experiments.not_miwae import VAEncoder, GaussDecoder, BernoulliDecoder, TDecoder


class Miwae(nn.Module):

    def __init__(self, n_input_features, n_latent=10, n_hidden=128,
                 activation=nn.Tanh(), out_dist='gauss', out_activation=nn.functional.relu,
                 learnable_imputation=False,
                 testing=False):
        super(Miwae, self).__init__()

        # Model settings
        self.n_input_features   = n_input_features
        self.n_latent           = n_latent
        self.n_hidden           = n_hidden
        self.activation         = activation
        self.out_dist           = out_dist
        self.out_activation     = out_activation
        self.testing            = testing
        self.eps                = torch.finfo(torch.float32).eps

        # Encoder
        self.encoder            = VAEncoder(self.n_input_features, self.n_hidden, self.n_latent, self.activation)

        # Decoder
        if out_dist in ['gauss', 'normal', 'truncated_normal']:
            self.decoder = GaussDecoder(self.n_latent, self.n_hidden, self.n_input_features, self.activation, self.out_activation)
        elif out_dist == 'bern':
            self.decoder = BernoulliDecoder(self.n_latent, self.n_hidden, self.n_input_features, self.activation)
        elif out_dist in ['t', 't-distribution']:
            self.decoder = TDecoder(self.n_latent, self.n_hidden, self.n_input_features, self.activation, self.out_activation)

        if learnable_imputation and not testing:
            self.imp = nn.Parameter(torch.randn(1, self.d))

    def forward(self, x, s, n_samples_importance_sampling):
        # Encoder
        q_mu, q_log_var                 = self.encoder(x)
        n_samples_importance_sampling   = x.shape[0]
        # Reparameterization trick
        q_z     = dist.Normal(q_mu, torch.exp(0.5 * q_log_var))
        # Sample from the normal dist
        z       = q_z.rsample((n_samples_importance_sampling,))
        z       = z.permute(1, 0, 2)

        # Decoder
        if self.out_dist in ['gauss', 'normal']:
            mu, std = self.decoder(z)
            p_x_given_z = dist.Normal(mu, std)
        elif self.out_dist == 'bern':
            logits      = self.decoder(z)
            mu          = logits
            p_x_given_z = dist.Bernoulli(logits=logits)
        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sigma, df   = self.decoder(z)
            p_x_given_z         = dist.StudentT(df=3 + F.softplus(df), loc=mu, scale=F.softplus(log_sigma) + 0.0001)
        else:
            raise ValueError("out_dist is not recognized.")

        # Compute log probabilities
        log_p_x_given_z = torch.sum(
            torch.unsqueeze(s, dim=1) * p_x_given_z.log_prob(torch.unsqueeze(x, dim=1)), dim=-1
        )

        # --- evaluate the z-samples in q(z|x)
        q_z2            = dist.Normal(loc=q_mu.unsqueeze(1),
                                      scale=torch.sqrt(torch.exp(q_log_var.unsqueeze(1))))
        log_q_z_given_x = torch.sum(q_z2.log_prob(z), dim=-1)

        # ---- evaluate the z-samples in the prior
        prior           = dist.Normal(loc=0.0, scale=1.0)
        log_p_z         = torch.sum(prior.log_prob(z), dim=-1)

        return mu, log_p_x_given_z, log_q_z_given_x, log_p_z


def get_MIWAE(n_samples_importance_sampling, lpxz, lqzx, lpz):
    l_w             = lpxz + lpz - lqzx
    log_sum_w       = torch.logsumexp(l_w, dim=0)
    log_avg_weight  = log_sum_w - torch.log(torch.tensor(n_samples_importance_sampling, dtype=torch.float32))
    return log_avg_weight.mean()
