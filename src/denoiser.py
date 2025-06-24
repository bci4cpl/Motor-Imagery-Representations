from models import convolution_AE
from properties import hyper_params as params
from properties import result_params
from utils import EEGDataSet_signal_by_day
from torch.utils.data import DataLoader
from torch import device
import torch
import os
import numpy as np
import matplotlib.pyplot as plt



class Denoiser():
    def __init__(self, model_adjustments,mode, test_dataset):
        self.model = None
        self.model_adjustments = model_adjustments
        self.mode = mode
        # self.logger = TensorBoardLogger('../tb_logs', name='EEG_Logger')
        # device settings
        self.proccessor = params['device']
        self.device = device(self.proccessor)
        self.accelerator = self.proccessor if self.proccessor == 'cpu' else 'gpu'
        self.devices = 1 if self.proccessor == 'cpu' else -1


        # Initialize the model with parameters extracted from test_dataset
        self.initialize_model(test_dataset)

    def initialize_model(self, dataset):
        # Extract necessary parameters from the dataset
        n_days_labels = dataset.n_days_labels
        n_channels = dataset.n_channels
        n_task_labels = dataset.n_task_labels

        self.model = convolution_AE(n_channels, n_days_labels, n_task_labels, self.model_adjustments,
                                    params['ae_lrn_rt'], filters_n=params['cnvl_filters'], mode=self.mode)
        self.model.to(self.device)

    # Training using pytorch lightingg
    # def fit(self, train_dataset):
    #     n_days_labels = train_dataset.n_days_labels
    #     n_task_labels = train_dataset.n_task_labels

    #     signal_data_loader = DataLoader(dataset=train_dataset, batch_size=params['btch_sz'], shuffle=True,
    #                                     num_workers=0)
    #     self.model = convolution_AE(train_dataset.n_channels, n_days_labels, n_task_labels, self.model_adjustments,
    #                                 params['ae_lrn_rt'], filters_n=params['cnvl_filters'], mode=self.mode)
    #     self.model.to(self.device)

    #     trainer_2 = pl.Trainer(max_epochs=params['n_epochs'], logger=self.logger, accelerator=self.accelerator,
    #                            devices=self.devices)
    #     trainer_2.fit(self.model, train_dataloaders=signal_data_loader)


    def train_and_save(self, dataset, n_epochs=50, save_every=5, save_dir="AE/"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Re-init the model cleanly
        n_days_labels = dataset.n_days_labels
        n_task_labels = dataset.n_task_labels

        self.model = convolution_AE(dataset.n_channels, n_days_labels, n_task_labels, self.model_adjustments,
                                    params['ae_lrn_rt'], filters_n=params['cnvl_filters'], mode=self.mode)
        self.model.to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['ae_lrn_rt'])
        loss_fn = torch.nn.MSELoss()

        data_loader = DataLoader(dataset=dataset, batch_size=params['btch_sz'], shuffle=True, num_workers=0)

        train_losses = []

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            for batch in data_loader:
                x, _, _ = batch
                x = x.to(self.device)

                optimizer.zero_grad()
                encoded = self.model.encode(x)
                reconstructed = self.model.decoder(encoded)
                loss = loss_fn(reconstructed, x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(data_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch}/{n_epochs} - Train Loss: {avg_loss:.4f}")

            if epoch % save_every == 0:
                save_path = os.path.join(save_dir, f"ae_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model checkpoint to {save_path}")

        # Plot training loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, n_epochs + 1), train_losses, marker='o')
        plt.title("Training Reconstruction Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def denoise(self, noisy_dataset):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            noisy_signal, y_label, days_label = noisy_dataset.getAllItems()
            denoised_signal = self.model(noisy_signal).detach().numpy()
        return denoised_signal,y_label,days_label,noisy_signal

    def load_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model is not initialized. Please initialize the model before loading weights.")
        checkpoint = torch.load(weights_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint)


        # Extracting the default iniziliatezed random weights
        model_dict = self.model.state_dict()

        # Filter out unnecessary keys
        #
        filtered_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                elif k == 'classiffier_days.1.weight':
                    num_days = np.random.uniform(-1, 1, size=(model_dict[k].shape[0]))
                    filtered_dict[k] = v[num_days, :]
                elif k == 'classiffier_days.1.bias':
                    filtered_dict[k] = v[num_days]
                else:
                    print(
                        f"Skipping {k} due to shape mismatch: checkpoint shape {v.shape}, model shape {model_dict[k].shape}")

        # Load the state dictionary with matched keys
        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)