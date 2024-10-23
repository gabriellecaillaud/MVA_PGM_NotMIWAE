import torch
import torch.nn.functional as F


def softmax(x):
    return F.softmax(x, dim=1)


def imp(model, xz, s, L):
    """
    Compute the imputation using the model.

    Parameters:
    -----------
    model: PyTorch model, not-MIWAE
    xz   : torch.Tensor, input with missing values (1, num_features)
    s    : torch.Tensor, binary mask for missing data (1, num_features)
    L    : int, number of latent samples

    Returns:
    --------
    l_out : torch.Tensor, latent output
    wl    : torch.Tensor, softmax-weighted importance weights
    xm    : torch.Tensor, imputed missing values
    xmix  : torch.Tensor, input + imputed values
    """
    with torch.no_grad():
        log_p_x_given_z, log_p_s_given_x, log_q_z_given_x, log_p_z = model(xz, s, L)

        mu = torch.exp(log_q_z_given_x.loc) # TODO check if it's really torch.exp
        # Compute the importance weights
        wl = softmax(log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x)

        # Compute the missing data imputation
        xm = torch.sum((mu.T * wl.T).T, dim=0)
        xmix = xz + xm * (1 - s)

    return mu, wl, xm, xmix


def not_imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Compute the imputation error of missing data using the not-MIWAE model.

    Parameters:
    -----------
    model: PyTorch model, not-MIWAE
    Xorg : torch.Tensor, original data with no missing values (num_samples, num_features)
    Xz   : torch.Tensor, data with missing values (num_samples, num_features)
    X    : torch.Tensor, complete data (used for dimensions) (num_samples, num_features)
    S    : torch.Tensor, binary mask for missing data (num_samples, num_features)
    L    : int, number of latent samples

    Returns:
    --------
    rmse: float, Root Mean Square Error of imputation
    XM  : torch.Tensor, imputed missing values (num_samples, num_features)
    """
    N = X.shape[0]
    D = X.shape[1]

    XM = torch.zeros_like(Xorg)  # To store imputed missing values

    for i in range(N):
        # Get one data sample and its mask
        xz = Xz[i, :].unsqueeze(0)  # Shape (1, num_features)
        s = S[i, :].unsqueeze(0)  # Shape (1, num_features)

        # Impute the missing values using the model
        l_out_mu, wl, xm, xmix = imp(model, xz, s, L)

        # Store the imputed value in the matrix
        XM[i, :] = xm

        # Print progress
        if i % 100 == 0:
            print(f'{i} / {N}')

    # Compute RMSE on the missing data
    rmse = torch.sqrt(torch.sum((Xorg - XM) ** 2 * (1 - S)) / torch.sum(1 - S))

    return rmse.item(), XM


def compute_rmse(mu, log_p_x_given_z, log_p_s_given_x, log_q_z_given_x, log_p_z, Xtrue, S):
    mu = torch.exp(mu)  # TODO check if it's really torch.exp
    # Compute the importance weights
    wl = softmax(log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x)

    # Compute the missing data imputation
    Xm = torch.sum((mu.T * wl.T).T, dim=0)

    # TODO check dim of xm
    assert Xm.shape == Xtrue.shape
    rmse = torch.sqrt(torch.sum((Xtrue - Xm) ** 2 * (1 - S)) / torch.sum(1 - S))
    return rmse

