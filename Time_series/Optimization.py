import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x).to(device)
        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()



        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        # self.model_path = f'models/lstm_{datetime.now().strftime("%Y-%m-%d%H:%M:%S")}.h5'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

    def save_model(self, name_of_path):
        torch.save(self.model.state_dict(), name_of_path+".h5")

    def evaluate(self, loader, batch_size=64, n_features=1, test=False):
        with torch.no_grad():
            predictions = []
            if test:
             values = []
            # print(loader)

            for x_test, y_test in loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()

                yhat = self.model(x_test)

                predictions.append(yhat.cpu().detach().numpy())
                if test:
                    values.append(y_test.cpu().detach().numpy())
            # print(yhat, "Y hat")
        if test:
            return predictions, values
        else:
            return predictions

    def forecast_with_lag_features(self, test_loader, batch_size=1, n_features=1, n_steps=100):
        test_loader_iter = iter(test_loader)
        predictions = []

        *_, (X, y) = test_loader_iter

        y = y.cpu().detach().numpy()
        X = X.view([batch_size, -1, n_features]).to(device)
        X = torch.roll(X, shifts=1, dims=2)
        X[..., -1, 0] = y.item(0)

        with torch.no_grad():
            self.model.eval()
            for _ in range(n_steps):
                X = X.view([batch_size, -1, n_features]).to(device)
                yhat = self.model(X)
                yhat = yhat.cpu().detach().numpy()
                X = torch.roll(X, shifts=1, dims=2)
                X[..., -1, 0] = yhat.item(0)
                predictions.append(yhat.item(0))

        return predictions

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()