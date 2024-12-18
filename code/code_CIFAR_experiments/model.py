import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as dist


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


class BernoulliDecoderLinearMiss(nn.Module):
    def __init__(self, n_hidden, missing_process):
        super(BernoulliDecoderLinearMiss, self).__init__()
        self.n_hidden = n_hidden
        self.missing_process = missing_process
        print(f"Using BernoulliDecoderLinearMiss with missing process{self.missing_process}")
        # Calculate flattened dimension for a single image
        self.flat_dim = 3 * 32 * 32

        if missing_process == 'linear':
            # Single linear layer
            self.fc1 = nn.Linear(in_features=self.flat_dim, out_features=self.flat_dim)

        elif missing_process == 'nonlinear':
            print(f"{self.n_hidden=}")
            # Two linear layers with intermediate hidden dimension
            self.fc1 = nn.Linear(in_features=self.flat_dim, out_features=n_hidden)
            self.fc2 = nn.Linear(in_features=n_hidden, out_features=self.flat_dim)

        elif missing_process in ['selfmasking', 'selfmasking_known']:
            # Reshape W and b to match flattened dimensions
            self.W = nn.Parameter(torch.randn(1, self.flat_dim))
            self.b = nn.Parameter(torch.randn(1, self.flat_dim))

        elif missing_process == "conv":
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        elif missing_process == "2conv":
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=n_hidden, out_channels=3, kernel_size=3, padding=1)

    def forward(self, z):
        # z shape: (N, 3, 32, 32)
        batch_size = z.shape[0]

        # Flatten the input
        z_flat = z.view(batch_size, -1)  # Shape: (N, 3*32*32)

        if self.missing_process == 'selfmasking':
            logits = -self.W * (z_flat - self.b)

        elif self.missing_process == 'selfmasking_known':
            w_softplus = F.softplus(self.W)
            logits = -w_softplus * (z_flat - self.b)

        elif self.missing_process == 'linear':
            logits = self.fc1(z_flat)

        elif self.missing_process == 'nonlinear':
            h = torch.tanh(self.fc1(z_flat))
            logits = self.fc2(h)

        elif self.missing_process == "conv":

            logits = self.conv1(z)

        elif self.missing_process == '2conv':
            h = torch.tanh(self.conv1(z))
            logits = self.conv2(h)

        else:
            print("Use 'selfmasking', 'selfmasking_known', 'linear', 'nonlinear', 'conv'  or '2conv' as 'missing_process'")
            logits = None

        # Reshape back to image dimensions
        logits = logits.view(batch_size, 3, 32, 32)

        return logits


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
        self.missing_decoder = BernoulliDecoderLinearMiss(
            n_hidden=512, missing_process=self.missing_process
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
