# use facebook prophet to forecast time series data
#
import pandas as pd
import prophet
import plotly.express as px
from datetime import datetime

import numpy as np
import json

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(DateTimeEncoder, self).default(obj)

def predict_sales(periods=365, category=None, state=None):
    """Predicts the sales for the next n periods."""
    # load the dataset
    data = pd.read_csv('dataset/geo_data.csv')

    df = pd.DataFrame()
    df['ds'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-' + data['day'].astype(str))
    df['y'] = data['sales']

    # sort the dataframe
    df = df.sort_values(by='ds')

    model = prophet.Prophet(changepoint_prior_scale=0.5)

    if category and state:
        df = data[(data['category'] == category) & (data['state'] == state)]
    if state:
        df = df[df['state'] == state]
    if category:
        df = df[df['category'] == category]

    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future, vectorized=False)
    fig = px.line(forecast, x='ds', y='yhat')
    fig.add_scatter(x=df['ds'], y=df['y'], mode='lines', name='Sales')

    response = json.dumps(fig.to_plotly_json(), cls=DateTimeEncoder)
    return response


    