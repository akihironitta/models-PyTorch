import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder"""
    
    def __init__(self, in_size=784, hidden_size=400, out_size=20, activation=nn.ReLU(), binary=True):
        super(VAE, self).__init__()
        self.name = "VAE"
        self.activate = activation
        self.binary = binary
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size)
        )

        self.encoder_mean = nn.Sequential(
            self.encoder,
            nn.Linear(hidden_size, out_size)
        )
        
        self.encoder_log_var = nn.Sequential(
            self.encoder,
            nn.Linear(hidden_size, out_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(out_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, in_size)
        )

    def rsample(self, z_mean, z_log_var):
        """
        Sample Z = {z_1, ..., z_N} from Gaussian distribution N(z_mean, z_var)
        using the reparameterization trick.
        """
        N, M = z_mean.size()
        if self.training:
            eps = torch.randn(N, M)
            z = z_mean + torch.exp(z_log_var) * eps
            return z
        else:
            return z_mean

    def encode(self, x):
        """
        Encode input variable x
        into mean and log variance of latent variable z
        """
        # FIXME:
        # Completely splitting z_mean and z_log_var
        # may be inefficient.
        z_mean = self.encoder_mean(x)
        z_log_var = self.encoder_log_var(x)
        return z_mean, z_log_var

    def decode(self, z):
        """
        Decode latent variable z
        into mean (and log variance) of input variable x
        """
        x_mean = self.decoder(z)
        return x_mean

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.rsample(z_mean, z_log_var)
        x_mean =self.decode(z)
        if self.binary:
            x_mean = torch.sigmoid(x_mean)
        return x_mean, z_mean, z_log_var, z

def vae_loss(x, x_mean, z_mean, z_log_var):
    """
    VAE loss, a.k.a. negative Evidence Lower Bound (ELBO).
    """
    z_std = torch.exp(z_log_var / 2.)
    # reconstruction loss --- BCE in binary case
    # FIXME: Not implemented yet for non-binary variable case.
    bce = nn.functional.binary_cross_entropy(x_mean, x, reduction="sum")
    # KL divergence
    kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    # nagative ELBO
    loss = bce + kl
    return loss, bce, kl

    
def main():
    import torch
    from utils import load_dataloader, parse_parameters, save_model, load_model
    try:
        import colored_traceback.always
    except:
        pass
    
    args = parse_parameters()
    print(args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    epochs = args.epochs
    activation = torch.nn.ReLU()
    in_size = args.in_size
    out_size = args.out_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    dataset_name = args.dataset
    data_dir = "/tmp/data/"
    save_state_dir = "./checkpoints/"
    learning_rate = args.learning_rate
    log_interval = args.log_interval
    loss_func = vae_loss
    
    dl_train, dl_test = load_dataloader(dataset_name, batch_size, data_dir)
    
    model = VAE(in_size, hidden_size, out_size, activation, True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # === TRAIN ===
        total_loss = 0.
        total_bce = 0.
        total_kl = 0.
        model.train()
        for batch_idx, (x, y) in enumerate(dl_train):
            x = x.to(device).view(-1, in_size)
            x_mean, z_mean, z_log_var, z = model(x)
            loss, bce, kl = loss_func(x, x_mean, z_mean, z_log_var)
            loss /= len(x)
            bce /= len(x)
            kl /= len(x)
            total_loss += loss.item()
            total_bce += bce.item()
            total_kl += kl.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                msg = f"Batch [{batch_idx:4}/{len(dl_train):4}]: Batch Loss {loss.item():2.4f} KL {kl.item():2.4f} BCE {bce.item():2.4f}"
                print(msg)

        # === VALIDATION ===
        total_val_loss = 0.
        total_val_bce = 0.
        total_val_kl = 0.
        model.eval()
        for i, (x, y) in enumerate(dl_test):
            x = x.to(device).view(-1, in_size)
            x_mean, z_mean, z_log_var, z = model(x)
            loss, bce, kl = loss_func(x, x_mean, z_mean, z_log_var)
            loss /= len(x)
            bce /= len(x)
            kl /= len(x)
            total_val_loss += loss.item()
            total_val_bce += bce.item()
            total_val_kl += kl.item()

        msg = f"Epoch [{epoch+1:4}/{epochs:4}]: Train Loss {total_loss/len(dl_train):2.4f} (KL {total_kl/len(dl_train):2.4f} BCE {total_bce/len(dl_train):2.4f}) / Validation Loss {total_val_loss/len(dl_test):2.4f} (KL {total_val_kl/len(dl_test):2.4f} BCE {total_val_bce/len(dl_test):2.4f})"
        print(msg)
        
        save_model(model, epoch, dataset_name, save_state_dir)
        

if __name__ == "__main__":
    main()
