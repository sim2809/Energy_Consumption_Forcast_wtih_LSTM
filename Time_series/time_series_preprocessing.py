import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import pandas as pd
import holidays

class time_series_preprocessor:
    def __init__(self, df, test_train_pred):
        self.test_train_pred = test_train_pred
        self.scaler = MinMaxScaler()
        self.df = df

    def feature_label_split(self, df_n , target_col):
        """
        :param target_col: specify the target of the dataset
        :return: X- feature and y - lable
        """
        y = df_n[[target_col]]
        X = df_n.drop(columns=[target_col])


        return X, y


    def generate_time_lags(self, n_lags):
        """
        -   here we copy the oreginal data and make it with lagges to give to LSTM later
        :param n_lags: the number of lags that should be generated
        :return: df with lags
        """
        df_n = self.df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n["value"].shift(n)
        df_n = df_n.iloc[n_lags:]
        df_n = self.add_holiday_col(df_n)
        return df_n


    def train_val_test_split(self, df_n, target_col, test_ratio):
        if self.test_train_pred == "train":
            X, y = self.feature_label_split(df_n, target_col)
            val_ratio = test_ratio / (1 - test_ratio)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, shuffle=False)
            return X_train, X_val, y_train, y_val

    def fit_scaler(self):
        self.df = self.scaler.fit_transform(self.df)

    def getting_scaled_data(self):
        """
        here we bring the dataset into the range of 0 to 1
        this will make the network learn better (question)
        :return:
        """

        df_generated = self.generate_time_lags(100)
        if self.test_train_pred == "train":
            # X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(df_generated, 'value', 0.2)
            X_train, X_val, y_train, y_val = self.train_val_test_split(df_generated, "value", 0.2)
        elif self.test_train_pred == "test":
            X_test, y_test = self.feature_label_split(df_generated, "value")



        if self.test_train_pred == "train":
            X_train.loc[:, X_train.columns != "is_holiday"] = self.scaler.fit_transform(X_train.loc[:, X_train.columns != "is_holiday"])
            X_val.loc[:, X_val.columns != "is_holiday"] = self.scaler.transform(X_val.loc[:, X_val.columns != "is_holiday"])
            # X_test.loc[:, X_test.columns != "is_holiday"] = self.scaler.transform(X_test.loc[:, X_test.columns != "is_holiday"])

            y_train.loc[:, y_train.columns != "is_holiday"] = self.scaler.fit_transform(y_train.loc[:, y_train.columns != "is_holiday"])
            y_val.loc[:, y_val.columns != "is_holiday"] = self.scaler.transform(y_val.loc[:, y_val.columns != "is_holiday"])
            # y_test.loc[:, y_test.columns != "is_holiday"] = self.scaler.transform(y_test.loc[:, y_test.columns != "is_holiday"])
        #
        if self.test_train_pred == "test":
            X_test.loc[:, X_test.columns != "is_holiday"] = self.scaler.fit_transform(X_test.loc[:, X_test.columns != "is_holiday"])
            y_test.loc[:, y_test.columns != "is_holiday"] = self.scaler.fit_transform(y_test.loc[:, y_test.columns != "is_holiday"])
            self.X_test = X_test
            self.y_test = y_test


        # print(X_train)

        if self.test_train_pred == "test":
            train_test_val_data = {
                "X_test_arr": X_test,
            }
            train_test_val_data.update({"y_test_arr": y_test})


        if self.test_train_pred == "train":
            train_test_val_data = {
                "X_train_arr" : X_train,
                "X_val_arr"   : X_val,

                "y_train_arr": y_train,
                # "y_test_arr" : y_test,
                "y_val_arr"  : y_val
            }


        return train_test_val_data



    def get_data_loaders(self):
        train_test_val_data = self.getting_scaled_data()
        batch_size = 64


        if self.test_train_pred == "train":
            train_features = torch.Tensor(train_test_val_data["X_train_arr"].values).to(device)
            train_targets = torch.Tensor(train_test_val_data["y_train_arr"].values).to(device)
            val_features = torch.Tensor(train_test_val_data["X_val_arr"].values).to(device)
            val_targets = torch.Tensor(train_test_val_data["y_val_arr"].values).to(device)

            train = TensorDataset(train_features, train_targets)
            val = TensorDataset(val_features, val_targets)

            train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)

            data_loaders = {"train_loader": train_loader,"val_loader": val_loader}

        if self.test_train_pred == "test":
            test_features = torch.Tensor(train_test_val_data["X_test_arr"].values).to(device)
            test_targets = torch.Tensor(train_test_val_data["y_test_arr"].values).to(device)

            test = TensorDataset(test_features, test_targets)

            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
            # test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
            data_loaders = {
                "test_loader" : test_loader,
            }

        return data_loaders

    def format_predictions(self, predictions, values, df_test):
        vals = np.concatenate(values, axis=0).ravel()

        preds = np.concatenate(predictions, axis=0).ravel()

        df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        df_result = self.inverse_transform(df_result, [["value", "prediction"]])
        return df_result

    def inverse_transform(self, df, columns):
        """
        as we have scaled out data to the range from 0 to 1 we need to scale it back in order to have actual(real life) values of the prediction
        we do it by calling inverse transform function of the scaler
        :param df:
        :param columns:
        :return:
        """
        for col in columns:
            df[col] = self.scaler.inverse_transform(df[col])
        return df

    def is_holiday(self, date):

        return 1 if (date in holidays.US()) else 0

    def add_holiday_col(self, df):
        return df.assign(is_holiday=df.index.to_series().apply(self.is_holiday))