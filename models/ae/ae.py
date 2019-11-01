import torch
import torch.nn as nn


class AE(nn.Module):
    """Autoencoder"""
    
    def __init__(self, in_size=784, hidden_size=400, out_size=20, activation=nn.ReLU(), binary=True):
        super(AE, self).__init__()
        self.name = "AE"
        self.activate = activation
        self.binary = binary
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, out_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, in_size)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)
        if self.binary:
            x_reconst = torch.sigmoid(x_reconst)
        return x_reconst, z


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
    loss_func = nn.BCELoss()
    
    dl_train, dl_test = load_dataloader(dataset_name, batch_size, data_dir)
    print(len(dl_train), len(dl_test))
    
    model = AE(in_size, hidden_size, out_size, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # === TRAIN ===
        total_loss = 0.
        model.train()
        for batch_idx, (x, y) in enumerate(dl_train):
            x = x.to(device).view(-1, in_size)
            x_reconst, z = model(x)
            loss = loss_func(x_reconst, x) / len(x)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                msg = f"Batch [{batch_idx:4}/{len(dl_train):4}]: Batch Loss {loss.item():2.4f}"
                print(msg)

        # === VALIDATION ===
        total_val_loss = 0.
        model.eval()
        for i, (x, y) in enumerate(dl_test):
            x = x.to(device).view(-1, in_size)
            x_reconst, z = model(x)
            val_loss = loss_func(x_reconst, x) / len(x)
            total_val_loss += val_loss.item()

        msg = f"Epoch [{epoch+1:4}/{epochs:4}]: Train Loss {total_loss:2.4f} / Validation Loss {total_val_loss:2.4f}"
        print(msg)
        
        save_model(model, epoch, dataset_name, save_state_dir)
        

if __name__ == "__main__":
    main()
