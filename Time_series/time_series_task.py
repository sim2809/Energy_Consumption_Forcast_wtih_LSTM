from datetime import datetime
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.offline import iplot
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from my_LSTM_2 import my_LSTM_2
from Optimization import Optimization
import holidays
from valin_RNN import vanil_RNN
from my_LSTM import my_LSTM
import numpy as np
#### defining device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pred_plotting import pred_plotting
from time_series_preprocessing import time_series_preprocessor

class train():
    def __init__(self, dataset_path, model_name):
        self.us_holidays = holidays.US()
        self.model_name = model_name
        self.dataset_path = dataset_path

        ### stters
        self.upload_data(self.dataset_path)
        self.set_configs()
        self.set_model()
        self.set_loss_optim()




    def upload_data(self, df_path):
        """
        -   here we upload the dataset and make the dataset's id the Datetime column by set_index function

        -  we also make the index to the type of datetime by pandas to_datetime function
            https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

        - we also check wather the data is monotonicly growing or not by is_monotonic pandas function


        :param df_path: specify the path of the dataset
        :return: this is a void function
        """
        df = pd.read_csv(df_path)
        df = df[:100000]
        df = df.set_index(['Datetime'])
        df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic:
            df = df.sort_index()

        df = df.rename(columns={'PJME_MW': 'value'})
        self.preprocessor =  time_series_preprocessor(df ,"train")

    def set_preprocessed_data(self):
        self.data_loaders = self.preprocessor.get_data_loaders()

    def plot_dataset(self, title="dataset"):
        data = []
        value = go.Scatter(
            x=self.df.index,
            y=self.df.value,
            mode="lines",
            name="values",
            marker=dict(),
            text=self.df.index,
            line=dict(color="rgba(0,0,0, 0.3)"),
        )
        data.append(value)

        layout = dict(
            title=title,
            xaxis=dict(title="Date", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )

        fig = dict(data=data, layout=layout)
        iplot(fig)

    def get_model(self):
        """
        we have two options of models that we can use for such time series datasets
        RNN -> this is bad at learning long term dependences
        LSTM
        :return:
        """
        models = {
            "rnn": vanil_RNN,
            "lstm": my_LSTM,
            "lstm_2" : my_LSTM_2
        }

        final_model = models.get(self.model_name.lower())(**self.model_params)
        return final_model

    def set_configs(self):

        self.input_dim = 101
        # print(self.input_dim)
        output_dim = 1
        hidden_dim = 64
        layer_dim = 3
        dropout = 0.2

        self.batch_size = 64
        self.n_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6

        self.model_params = {'input_dim': self.input_dim,
                        'hidden_dim': hidden_dim,
                        'layer_dim': layer_dim,
                        'output_dim': output_dim,
                        'dropout_prob': dropout}


    def set_loss_optim(self):
        """
        loss function -> mean squared error
        optimizer -> Adam, with regularization
        :return:
        """
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def set_model(self):
        """
        getteing model and moveing it to the device
        :return:
        """
        self.model = self.get_model().to(device)

    def train(self):
        self.set_preprocessed_data()

        self.opt = Optimization(model=self.model, loss_fn=self.loss_fn, optimizer=self.optimizer)
        print(self.data_loaders)
        self.opt.train(self.data_loaders["train_loader"], self.data_loaders["val_loader"], batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.input_dim)
        self.opt.plot_losses()

    def save_model(self):
        self.opt.save_model("cheack_points/"+self.model_name )

    def evaluate(self):
        return self.opt.evaluate(self.data_loaders["test_loader_one"], batch_size=1, n_features=self.input_dim, test=True)


if __name__ == "__main__":
    task = train("PJME_hourly.csv", "lstm_2")
    task.train()
    task.save_model()
