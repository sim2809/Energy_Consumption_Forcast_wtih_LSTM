import numpy as np
import torch

from my_LSTM_2 import my_LSTM_2
from my_LSTM import my_LSTM
from pred_plotting import pred_plotting
from valin_RNN import vanil_RNN
from Optimization import Optimization
import pandas as pd
from time_series_preprocessing import time_series_preprocessor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class test_and_predict():

    def __init__(self, model_name, data_path, test_or_predict="test"):
        self.test_or_predict = test_or_predict
        self.model_name = model_name
        self.data_path = data_path
        self.model = None

        self.upload_data()
        self.upload_model()
        self.upload_weights()
        # print(self.data_loaders, "Aaaa")



    def upload_data(self):
        """
        here we set the dataloaders gotten from the preprocessor,
        preprocessor will also add holiday column
        :return: void
        """
        df = pd.read_csv(self.data_path)
        df = df.set_index(['Datetime'])
        df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic:
            df = df.sort_index()

        df = df.rename(columns={'PJME_MW': 'value'})
        self.df = df[100000:]
        self.preprocessor =  time_series_preprocessor(self.df , "test")
        self.data_loaders = self.preprocessor.get_data_loaders()

    def upload_model(self):
        models = {
            "rnn": vanil_RNN,
            "lstm": my_LSTM,
            "lstm_2": my_LSTM_2
        }

        model_params = self.get_configs()

        if self.model_name in models.keys():
            self.model = models.get(self.model_name.lower())(**model_params)
        else:
            raise Exception("Sorry, we do not have such a model")

    def upload_weights(self):
        if self.model is None:
            raise Exception("please first upload the model")

        self.model.load_state_dict(torch.load("cheack_points/"+self.model_name + ".h5"))


    def test(self):
        # print("You can not test on dataset without specifying targets")

        opt = Optimization(self.model, None, None)
        predictions, values = opt.evaluate(loader=self.data_loaders["test_loader"], batch_size=64 ,n_features=101, test=True)
        return predictions, values


    def predict(self):
        opt = Optimization(self.model, None, None)
        predictions = opt.forecast_with_lag_features(test_loader= self.data_loaders["test_loader"], batch_size=64, n_features=101, n_steps=1000)
        return predictions
    def get_configs(self):

        ### note that we do not need dropout in testing and predictions
        # dropout = 0 is the initial value for this parameter defined in the models' class

        model_params = {'input_dim': 101,
                        'hidden_dim': 64,
                        'layer_dim': 3,
                        'output_dim': 1}

        self.batch_size = 64
        return model_params

    def work(self):
        if self.test_or_predict == "test":
            return self.test()
        elif self.test_or_predict == "predict":

            return self.format_forecasts(self.predict())

    def format_forecasts(self, forcasts):

        start_date, freq = self.get_datetime_index(self.preprocessor.y_test)
        index = pd.date_range(start=start_date, freq=freq, periods=1000)

        preds = np.array(forcasts)
        # preds = np.concatenate(forcasts, axis=0).ravel()
        df_forecast = pd.DataFrame(data={"prediction": preds}, index=index)

        df_result = df_forecast.sort_index()
        df_result = self.preprocessor.inverse_transform(df_result, [["prediction"]])

        return df_result

    def get_datetime_index(self, df):
        return (
            pd.to_datetime(df.index[-1])
            + (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[-2])),
            pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[-2]),
        )



if __name__ == "__main__":
    #### for testing your model follow this secuence of commands
    test = test_and_predict("lstm_2", "PJME_hourly.csv", test_or_predict="test")
    predictions, values = test.work()

    df_result = test.preprocessor.format_predictions(predictions,  values, test.df)

    plot = pred_plotting()
    plot.plot_predictions(df_result)


    #### for predicting future values call this scuence of commands
    test = test_and_predict("lstm_2", "PJME_hourly.csv", test_or_predict="predict")

    df_result = test.preprocessor.format_predictions(predictions, values, test.preprocessor.X_test)

    plot = pred_plotting()

    df_forecast = test.work()

    plot.plot_dataset_with_forecast(test.df, df_forecast, title="forcast for 1000 samples")