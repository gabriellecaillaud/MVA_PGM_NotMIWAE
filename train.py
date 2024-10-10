import torch
import numpy as np
import pandas as pd

def introduce_mising(X):
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz


def train_notMIWAE(model, train_loader, val_loader, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, s) in enumerate(train_loader):
            x, s = x.to(device), s.to(device)
            optimizer.zero_grad()
            lpxz, lpmz, lqzx, lpz = model(x, s)
            loss = -model.get_notMIWAE(x.shape[-1], lpxz, lpmz, lqzx, lpz)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, s in val_loader:
                x, s = x.to(device), s.to(device)
                lpxz, lpmz, lqzx, lpz = model(x, s)
                loss = -model.get_notMIWAE(lpxz, lpmz, lqzx, lpz)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


if __name__ == "__main__":

    # white wine
    from mva_project.MVA_PGM_NotMIWAE.not_miwae import notMIWAE, get_notMIWAE

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
    # ---- drop the classification attribute
    data = data[:, :-1]
    # ----

    N, D = data.shape

    dl = D - 1

    # ---- standardize data
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)

    # ---- random permutation
    p = np.random.permutation(N)
    data = data[p, :]

    Xtrain = data.copy()
    Xval_org = data.copy()

    # ---- introduce missing process
    Xnan, Xz = introduce_mising(Xtrain)
    S = np.array(~np.isnan(Xnan), dtype=np.float)
    Xval, Xvalz = introduce_mising(Xval_org)

    # Data preparation
    Xorg = torch.tensor(Xnan, dtype=torch.float32)
    Xval_org = torch.tensor(Xval, dtype=torch.float32)
    n, d = Xnan.shape

    # Missing data mask
    S = torch.tensor(~np.isnan(Xnan), dtype=torch.float32)
    Sval = torch.tensor(~np.isnan(Xval), dtype=torch.float32)

    if torch.sum(S) < d * n:
        X = Xorg.clone()
        X[torch.isnan(X)] = 0
        Xval = Xval_org.clone()
        Xval[torch.isnan(Xval)] = 0
    else:
        X = Xorg
        Xval = Xval_org



    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(X, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process=mprocess,
                        name=name)

