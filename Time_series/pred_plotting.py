from plotly.offline import iplot
import plotly.graph_objs as go

class pred_plotting():

    def plot_predictions(self, df_result):
        data = []

        value = go.Scatter(
            x=df_result.index,
            y=df_result.value,
            mode="lines",
            name="values",
            marker=dict(),
            text=df_result.index,
            line=dict(color="rgba(0,0,0, 0.3)"),
        )
        data.append(value)


        prediction = go.Scatter(
            x=df_result.index,
            y=df_result.prediction,
            mode="lines",
            line={"dash": "dot"},
            name='predictions',
            marker=dict(),
            text=df_result.index,
            opacity=0.8,
        )
        data.append(prediction)

        layout = dict(
            title="Predictions vs Actual Values for the dataset",
            xaxis=dict(title="Time", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )

        fig = dict(data=data, layout=layout)
        iplot(fig)

    # # Set notebook mode to work in offline
    # pyo.init_notebook_mode()

    def plot_dataset_with_forecast(self, df, df_forecast, title):
        data = []
        df = df[45267:]
        value = go.Scatter(
            x=df.index,
            y=df.value,
            mode="lines",
            name="values",
            marker=dict(),
            text=df.index,
            line=dict(color="rgba(0,0,0, 0.3)"),
        )
        data.append(value)

        forecast = go.Scatter(
            x=df_forecast.index,
            y=df_forecast.prediction,
            mode="lines",
            name="forecasted values",
            marker=dict(),
            text=df.index,
            line=dict(color="rgba(10,100,10, 0.3)"),
        )
        data.append(forecast)

        layout = dict(
            title=title,
            xaxis=dict(title="Date", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )

        fig = dict(data=data, layout=layout)
        iplot(fig)