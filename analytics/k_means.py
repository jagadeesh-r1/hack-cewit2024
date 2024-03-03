import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import plotly.express as px
from constants import segments


dataset = pd.read_csv('dataset/prod.csv')
dataset['order_date'] = pd.to_datetime(dataset['order_date'])
dataset['ship_date'] = pd.to_datetime(dataset['ship_date'])

# print(dataset.columns)
end_date = max(dataset['order_date']) + dt.timedelta(days=1)



def create_rfm(user_selection, dataset):
    """Creates and returns the RFM dataframe based on the user selection column."""
    dataset['customer_id'] = dataset['customer_id'].astype('category')
    dataset['state'] = dataset['state'].astype('category')
    dataset['city'] = dataset['city'].astype('category')
    dataset['segment'] = dataset['segment'].astype('category')
    dataset['ship_mode'] = dataset['ship_mode'].astype('category')
    dataset['category'] = dataset['category'].astype('category')
    dataset['sub_category'] = dataset['sub_category'].astype('category')
    
    if user_selection == 'customer_id':
        dataset_rfm = dataset.groupby(user_selection).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            state=('state', 'first'),
            city=('city', 'first'),
            segment=('segment', 'first'),
            ship_mode=('ship_mode', 'first'),
            category=('category', 'first'),
            sub_category=('sub_category', 'first'),
        ).reset_index()
    elif user_selection == 'state':
        dataset_rfm = dataset.groupby(user_selection).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            customer_id=('customer_id', 'first'),
            city=('city', 'first'),
            segment=('segment', 'first'),
            ship_mode=('ship_mode', 'first'),
            category=('category', 'first'),
            sub_category=('sub_category', 'first'),
        ).reset_index()
    elif user_selection == 'city':
        dataset_rfm = dataset.groupby(user_selection).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            customer_id=('customer_id', 'first'),
            state=('state', 'first'),
            segment=('segment', 'first'),
            ship_mode=('ship_mode', 'first'),
            category=('category', 'first'),
            sub_category=('sub_category', 'first'),
        ).reset_index()
    elif user_selection == 'segment':
        dataset_rfm = dataset.groupby(user_selection).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            customer_id=('customer_id', 'first'),
            state=('state', 'first'),
            city=('city', 'first'),
            ship_mode=('ship_mode', 'first'),
            category=('category', 'first'),
            sub_category=('sub_category', 'first'),
        ).reset_index()
    elif user_selection == 'ship_mode':
        dataset_rfm = dataset.groupby(user_selection, observed=True).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            customer_id=('customer_id', 'first'),
            state=('state', 'first'),
            city=('city', 'first'),
            segment=('segment', 'first'),
            category=('category', 'first'),
            sub_category=('sub_category', 'first'),
        ).reset_index()
    elif user_selection == 'category':
        dataset_rfm = dataset.groupby(user_selection).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            customer_id=('customer_id', 'first'),
            state=('state', 'first'),
            city=('city', 'first'),
            segment=('segment', 'first'),
            ship_mode=('ship_mode', 'first'),
            sub_category=('sub_category', 'first'),
        ).reset_index()
    elif user_selection == 'sub_category':
        dataset_rfm = dataset.groupby(user_selection).agg(
            recency=('order_date', lambda x: (end_date - x.max()).days),
            frequency=('order_id', 'count'),
            monetary=('sales', 'sum'),
            customer_id=('customer_id', 'first'),
            state=('state', 'first'),
            city=('city', 'first'),
            segment=('segment', 'first'),
            ship_mode=('ship_mode', 'first'),
            category=('category', 'first'),
        ).reset_index()
    # print(dataset_rfm)
    return dataset_rfm

def create_df_rfm_graphs(user_selection, dataset):
    """Creates and returns the dataframe for the RFM graphs using ploty."""
    histogram_data = {}
    df_rfm = create_rfm('customer_id', dataset)
    fig = px.histogram(df_rfm, x="recency", nbins=50, title=f"Recency Distribution based on {user_selection}", labels={'recency': 'Recency (days)'})
    histogram_data['recency'] = fig.to_plotly_json()['data']

    fig = px.histogram(df_rfm, x="frequency", nbins=50, title=f"Frequency Distribution based on {user_selection}")
    histogram_data['frequency'] = fig.to_plotly_json()['data']

    fig = px.histogram(df_rfm, x="monetary", nbins=50, title=f"Monetary Distribution based on {user_selection}")
    histogram_data['monetary'] = fig.to_plotly_json()['data']
    return histogram_data




def preprocess(dataframe):
    """Preprocess data for KMeans clustering by normalizing and scaling the data."""
    # Convert categorical columns to numerical
    for col in dataframe.columns:
        if dataframe[col].dtype.name == 'category':
            dataframe[col] = dataframe[col].cat.codes
    
    df_log = np.log1p(dataframe)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    
    return norm

def find_best_k(dataframe, increment=0, decrement=0):
    """Find the best k value for KMeans clustering using the elbow method."""
    
    norm = preprocess(dataframe)
    
    sse = {}
    # _k = min of 21 or norm.shape[0]+1
    _k = min(21, norm.shape[0]+1)
    for k in range(1, _k):
        # print("Current k: ", k)
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(norm)
        sse[k] = kmeans.inertia_

        kn = KneeLocator(
                 x=list(sse.keys()), 
                 y=list(sse.values()), 
                 curve='convex', 
                 direction='decreasing'
                 )
    if kn.knee is not None:
        k = kn.knee + increment - decrement
    else:
        # handle the case where no knee was found
        k = 2
    return k

def run_kmeans(df, increment=0, decrement=0):
    """Run KMeans clustering, including the preprocessing of the data
    and the automatic selection of the optimum k. 
    """
    
    norm = preprocess(df)
    # print(norm)
    k = find_best_k(df, increment, decrement)
    kmeans = KMeans(n_clusters=k, 
                    random_state=1)
    kmeans.fit(norm)
    return df.assign(cluster=kmeans.labels_)
