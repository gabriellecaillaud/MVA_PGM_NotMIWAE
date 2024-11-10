import datetime

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as dist
import torchvision
from torch.utils.data import Subset
from torchvision import transforms

from data_imputation import compute_imputation_rmse_conv_not_miwae
from not_miwae import get_notMIWAE
from not_miwae_cifar import ZeroBlueTransform, ZeroPixelWhereBlueTransform
from utils import seed_everything


class ConvVAEncoder(nn.Module):
    def __init__(
            self,
            in_channels=3,  # RGB images
            hidden_dims=[64, 128, 256],  # Adjusted for 32x32
            n_latent=128,  # Latent dimension
            activation=nn.ReLU()
    ):
        super(ConvVAEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.n_latent = n_latent

        # Build encoder architecture
        modules = []
        in_channels_conv = in_channels

        # Build convolutional layers with MaxPooling
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(
                    in_channels=in_channels_conv,
                    out_channels=h_dim,
                    kernel_size=3,  # Smaller kernel since we're using MaxPool
                    stride=1,  # No stride since we're using MaxPool
                    padding=1  # Same padding
                ),
                activation,
                nn.MaxPool2d(kernel_size=2, stride=2)  # Handle downsampling
            ])
            in_channels_conv = h_dim

        self.encoder = nn.Sequential(*modules)

        # For 32x32 input with 3 MaxPool layers, final feature map will be 4x4
        self.flatten_size = hidden_dims[-1] * 4 * 4

        # Latent space layers
        self.fc_mu = nn.Linear(self.flatten_size, n_latent)
        self.fc_var = nn.Linear(self.flatten_size, n_latent)

    def forward(self, x):
        # x shape: [batch_size, 3, 32, 32]

        # Convolutional layers
        x = self.encoder(x)  # Final shape: [batch_size, 256, 4, 4]

        # Flatten
        x = torch.flatten(x, start_dim=1)  # Shape: [batch_size, 256*4*4]

        # Get latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        log_var = torch.clamp(log_var, min=-10, max=10)

        return mu, log_var

    def get_flatten_size(self):
        return self.flatten_size

    def get_final_channels(self):
        return self.hidden_dims[-1]


class ConvGaussDecoder(nn.Module):
    def __init__(
            self,
            n_latent=128,
            hidden_dims=[256, 128, 64],  # Reverse of encoder dims
            out_channels=3,  # RGB output
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid()  # For pixel values [0,1]
    ):
        super(ConvGaussDecoder, self).__init__()

        self.n_latent = n_latent
        self.hidden_dims = hidden_dims

        # Initial projection from latent space
        self.linear_projection = nn.Linear(n_latent, hidden_dims[0] * 4 * 4)
        self.activation = activation

        # Build decoder architecture
        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                activation
            ])

        self.decoder = nn.Sequential(*modules)

        # Output layers for mean and standard deviation
        self.final_conv_mu = nn.ConvTranspose2d(
            hidden_dims[-1],
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.final_conv_std = nn.ConvTranspose2d(
            hidden_dims[-1],
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.out_activation = out_activation

    def forward(self, z):
        # z shape: [batch_size, n_latent]

        # Project and reshape
        x = self.linear_projection(z)
        x = self.activation(x)
        x = x.view(-1, self.hidden_dims[0], 4, 4)

        # Pass through decoder layers
        x = self.decoder(x)

        # Generate mean and std
        mu = self.out_activation(self.final_conv_mu(x))  # Shape: [batch_size, 3, 32, 32]

        # Apply softplus to ensure positive std dev
        std = F.softplus(self.final_conv_std(x))  # Shape: [batch_size, 3, 32, 32]

        return mu, std + 1e-8

    def get_beta(self, beta_min=1e-8, beta_max=1.0):
        """Optional temperature parameter for sampling"""
        if self.training:
            return beta_max
        return beta_min

    def sample(self, mu, std):
        """Generate a sample from the decoded distribution"""
        eps = torch.randn_like(mu)
        return mu + std * eps * self.get_beta()


class BernoulliDecoderConvMiss(nn.Module):
    def __init__(self, n_hidden, missing_process):
        super(BernoulliDecoderConvMiss, self).__init__()
        self.n_hidden = n_hidden
        self.missing_process = missing_process

        # Calculate flattened dimension for a single image
        self.flat_dim = 3 * 32 * 32

        if missing_process == 'linear':
            # Using Conv2d instead of Linear for spatial structure preservation
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        elif missing_process == 'nonlinear':
            # Two convolutional layers with intermediate hidden channels
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=n_hidden, out_channels=3, kernel_size=3, padding=1)

        elif missing_process in ['selfmasking', 'selfmasking_known']:
            # Reshape W and b to match image dimensions
            self.W = nn.Parameter(torch.randn(1, 3, 32, 32))  # One weight per spatial location and channel
            self.b = nn.Parameter(torch.randn(1, 3, 32, 32))  # One bias per spatial location and channel

    def forward(self, z):
        # z shape: (N, 3, 32, 32)

        if self.missing_process == 'selfmasking':
            logits = -self.W * (z - self.b)

        elif self.missing_process == 'selfmasking_known':
            W_softplus = F.softplus(self.W)
            logits = -W_softplus * (z - self.b)

        elif self.missing_process == 'linear':
            logits = self.conv1(z)

        elif self.missing_process == 'nonlinear':
            h = torch.tanh(self.conv1(z))
            logits = self.conv2(h)

        else:
            print("Use 'selfmasking', 'selfmasking_known', 'linear' or 'nonlinear' as 'missing_process'")
            logits = None

        return logits  # Output shape: (N, 3, 32, 32)


class ConvNotMIWAE(nn.Module):
    def __init__(
            self,
            n_latent=128,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims=[64, 128, 256],
            missing_process='selfmasking',
            testing=False
    ):
        super(ConvNotMIWAE, self).__init__()

        # Model settings
        self.n_latent = n_latent
        self.activation = activation
        self.out_activation = out_activation
        self.testing = testing
        self.missing_process = missing_process
        self.eps = torch.finfo(torch.float32).eps

        # Encoder
        self.encoder = ConvVAEncoder(
            in_channels=3,
            hidden_dims=hidden_dims,
            n_latent=n_latent,
            activation=activation
        )

        # Decoder
        self.decoder = ConvGaussDecoder(
            n_latent=n_latent,
            hidden_dims=hidden_dims[::-1],  # Reverse encoder dims
            out_channels=3,
            activation=activation,
            out_activation=out_activation
        )

        # Missing process decoder
        self.missing_decoder = BernoulliDecoderConvMiss(
            n_hidden = 64, missing_process = self.missing_process
        )

    def forward(self, x, s, n_samples_importance_sampling):
        # x shape: [batch_size, 3, 32, 32]
        # s shape: [batch_size, 3, 32, 32] (binary mask)

        # Encoder
        q_mu, q_log_var = self.encoder(x)

        # Reparameterization trick
        q_z = dist.Normal(q_mu, torch.exp(0.5 * q_log_var))
        # Sample from the normal dist
        z = q_z.rsample((n_samples_importance_sampling,))
        z = z.permute(1, 0, 2)  # [batch_size, n_samples, n_latent]

        # Decoder
        mu, std = self.decoder(z.reshape(-1, self.n_latent))

        # Reshape outputs to include samples dimension
        batch_size = x.shape[0]
        mu = mu.reshape(batch_size, n_samples_importance_sampling, *x.shape[1:])
        std = std.reshape(batch_size, n_samples_importance_sampling, *x.shape[1:])

        # Create distribution
        p_x_given_z = dist.Normal(mu, std)

        # Compute log probabilities
        x_expanded = x.unsqueeze(1)
        s_expanded = s.unsqueeze(1)

        log_p_x_given_z = torch.sum(
            s_expanded * p_x_given_z.log_prob(x_expanded),
            dim=[-1, -2, -3]  # Sum over channels, height, width
        )

        # Missing process
        # Sample from p(x|z) for unobserved values
        samples = p_x_given_z.sample()
        # Mix observed and sampled values
        l_out_mixed = samples * (1 - s_expanded) + x_expanded * s_expanded

        # Compute missing probabilities
        logits_miss = self.missing_decoder(l_out_mixed.reshape(-1, *x.shape[1:]))
        logits_miss = logits_miss.reshape(batch_size, n_samples_importance_sampling, *x.shape[1:])

        # p(s|x)
        p_s_given_x = dist.Bernoulli(logits=logits_miss)
        # evaluate s in p(s|x)
        log_p_s_given_x = torch.sum(
            p_s_given_x.log_prob(s_expanded),
            dim=[-1, -2, -3]  # Sum over channels, height, width
        )

        # Evaluate the z-samples in q(z|x)
        q_z2 = dist.Normal(
            loc=q_mu.unsqueeze(1),
            scale=torch.sqrt(torch.exp(q_log_var.unsqueeze(1)))
        )
        log_q_z_given_x = torch.sum(q_z2.log_prob(z), dim=-1)

        # Evaluate the z-samples in the prior
        prior = dist.Normal(loc=0.0, scale=1.0)
        log_p_z = torch.sum(prior.log_prob(z), dim=-1)

        return mu, log_p_x_given_z, log_p_s_given_x, log_q_z_given_x, log_p_z


class ConvNotMIWAEWithReparametrizationTrick(nn.Module):
    def __init__(
            self,
            n_latent=128,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims=[64, 128, 256],
            missing_process='selfmasking',
            testing=False
    ):
        super(ConvNotMIWAEWithReparametrizationTrick, self).__init__()
        print("Conv not MIWAE with reparametrization trick (no .sample from a torch.dist)")
        # Model settings
        self.n_latent = n_latent
        self.activation = activation
        self.out_activation = out_activation
        self.testing = testing
        self.missing_process = missing_process
        self.eps = torch.finfo(torch.float32).eps

        # Encoder
        self.encoder = ConvVAEncoder(
            in_channels=3,
            hidden_dims=hidden_dims,
            n_latent=n_latent,
            activation=activation
        )

        # Decoder
        self.decoder = ConvGaussDecoder(
            n_latent=n_latent,
            hidden_dims=hidden_dims[::-1],  # Reverse encoder dims
            out_channels=3,
            activation=activation,
            out_activation=out_activation
        )

        # Missing process decoder
        self.missing_decoder = BernoulliDecoderConvMiss(
            n_hidden=64, missing_process=self.missing_process
        )

    def forward(self, x, s, n_samples_importance_sampling):
        # x shape: [batch_size, 3, 32, 32]
        # s shape: [batch_size, 3, 32, 32] (binary mask)

        # Encoder
        q_mu, q_log_var = self.encoder(x)

        # Reparameterization trick
        epsilon = torch.randn(n_samples_importance_sampling, *q_mu.shape, device=q_mu.device)
        # Calculate standard deviation from log variance
        std = torch.exp(0.5 * q_log_var)
        # Apply reparameterization trick: z = mu + std * epsilon
        z = q_mu.unsqueeze(0) + std.unsqueeze(0) * epsilon
        z = z.permute(1, 0, 2)  # [batch_size, n_samples, n_latent]

        # Decoder
        mu, std = self.decoder(z.reshape(-1, self.n_latent))

        # Reshape outputs to include samples dimension
        batch_size = x.shape[0]
        mu = mu.reshape(batch_size, n_samples_importance_sampling, *x.shape[1:])
        std = std.reshape(batch_size, n_samples_importance_sampling, *x.shape[1:])

        epsilon = torch.randn_like(mu)

        # Apply reparametrization trick
        samples = mu + std * epsilon

        # Compute log probabilities
        x_expanded = x.unsqueeze(1)
        s_expanded = s.unsqueeze(1)

        log_2pi = torch.tensor(2 * torch.pi, device=z.device)
        # Compute log probabilities manually for Normal distribution
        log_p_x_given_z = torch.sum(
            s_expanded * (-0.5 * torch.log(log_2pi) - torch.log(std) - 0.5 * ((x_expanded - mu) / std) ** 2),
            dim=[-1, -2, -3]  # Sum over channels, height, width
        )

        # Mix observed and sampled values using the reparametrized samples
        l_out_mixed = samples * (1 - s_expanded) + x_expanded * s_expanded

        # Compute missing probabilities
        logits_miss = self.missing_decoder(l_out_mixed.reshape(-1, *x.shape[1:]))
        logits_miss = logits_miss.reshape(batch_size, n_samples_importance_sampling, *x.shape[1:])

        # p(s|x)
        # Bernouilli distribution reparametrization trick
        log_p_s_given_x = torch.sum(
            -logits_miss * (1 - s_expanded) - torch.log(1 + torch.exp(-logits_miss)),
            dim=[-1, -2, -3]  # Sum over channels, height, width
        )

        std_q = torch.sqrt(torch.exp(q_log_var.unsqueeze(1)))
        log_q_z_given_x = torch.sum(
            -0.5 * torch.log(log_2pi)
            - torch.log(std_q)
            - 0.5 * ((z - q_mu.unsqueeze(1)) / std_q) ** 2,
            dim=-1
        )

        # Evaluate the z-samples in the prior (standard normal: μ=0, σ=1)
        log_p_z = torch.sum(
            -0.5 * torch.log(log_2pi)
            - 0.5 * (z ** 2),  # simplified since μ=0, σ=1
            dim=-1
        )

        return mu, log_p_x_given_z, log_p_s_given_x, log_q_z_given_x, log_p_z


def train_conv_notMIWAE_on_cifar10(model, train_loader, val_loader, optimizer, scheduler, num_epochs,
                              total_samples_x_train, device, date):
    model.to(device)
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_rmse = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, s, xtrue = data[0]
            x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
            optimizer.zero_grad()
            mu, lpxz, lpmz, lqzx, lpz = model(x, s, total_samples_x_train)
            loss = -get_notMIWAE(total_samples_x_train, lpxz, lpmz, lqzx, lpz)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            model.eval()
            # compute rmse on batch
            with torch.no_grad():
                batch_rmse = compute_imputation_rmse_conv_not_miwae(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                train_rmse += batch_rmse

        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)

        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for data in val_loader:
                x, s, xtrue = data[0]
                x, s, xtrue = x.to(device), s.to(device), xtrue.to(device)
                mu, lpxz, lpmz, lqzx, lpz = model(x, s, total_samples_x_train)
                loss = -get_notMIWAE(total_samples_x_train, lpxz, lpmz, lqzx, lpz)
                val_loss += loss.item()
                batch_rmse = compute_imputation_rmse_conv_not_miwae(mu, lpxz, lpmz, lqzx, lpz, xtrue, s)
                val_rmse += batch_rmse
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"temp/not_miwae_{date}_best_val_loss.pt")

        print(
            f'Epoch {(epoch + 1):4.0f}, Train Loss: {train_loss:8.4f} , Train rmse: {train_rmse:7.4f} , Val Loss: {val_loss:8.4f} , Val RMSE: {val_rmse:7.4f}  last value of lr: {scheduler.get_last_lr()[-1]:.4f}')



if __name__ == "__main__":
    calib_config = [
        {'model': 'not_miwae', 'lr': 3e-3, 'epochs': 100, 'pct_start': 0.1, 'final_div_factor': 1e4, 'batch_size': 64,
         'n_hidden': 512, 'n_latent': 128, 'missing_process': 'nonlinear', 'weight_decay': 0, 'betas': (0.9, 0.999),
         'random_seed': 0, 'out_dist': 'gauss', 'dataset_size' : None, 'transform': 'ZeroBlueTransform', 'hidden_dims' : [64,128,256]},
        ][-1]


    if calib_config['transform'] == 'ZeroBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroBlueTransform(do_flatten=False)
        ])
    elif calib_config['transform'] == 'ZeroPixelWhereBlueTransform':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            ZeroPixelWhereBlueTransform(do_flatten=False)
        ])
    else:
        raise KeyError('Transforms is not correctly defined.')
    batch_size = calib_config['batch_size']

    if calib_config['dataset_size'] is not None:
        # set download to True if this is the first time you are running this file
        train_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                download=False, transform=transform), torch.arange(calib_config['dataset_size']))

        test_set = Subset(torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                       download=False, transform=transform), torch.arange(calib_config['dataset_size']))
    else:
        train_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                download=False, transform=transform)

        test_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                                       download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(date)



    seed_everything(calib_config['random_seed'])

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = ConvNotMIWAE(n_latent=calib_config['n_latent'],
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            hidden_dims= calib_config['hidden_dims'],
            missing_process=calib_config['missing_process'])

    model.to(device)
    print(f"Number of parameters in the model: {sum (p.numel() if p.requires_grad else 0 for p in model.parameters()) }")
    optimizer = torch.optim.Adam(model.parameters(), lr=calib_config['lr'], weight_decay=calib_config['weight_decay'], betas=calib_config['betas'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = calib_config['lr'],
                                                    epochs = calib_config['epochs'],
                                                    steps_per_epoch= len(train_loader),
                                                    pct_start= calib_config['pct_start'],
                                                    # final_div_factor=calib_config['final_div_factor']
                                                    )
    print(f"calib_config:{calib_config}")
    train_conv_notMIWAE_on_cifar10(model=model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, scheduler=scheduler, num_epochs = calib_config['epochs'], total_samples_x_train= 10, device=device, date=date )



