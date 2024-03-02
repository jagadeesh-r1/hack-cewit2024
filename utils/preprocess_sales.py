import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import plotly.express as px


sns.set(rc={'figure.figsize':(15, 6)})

df = pd.read_csv('dataset/clean_sales.csv')

df = df.rename(columns={'Order ID':'order_id','Order Date':'order_date','Ship Date':'ship_date', 'Country':'country', 'City':'city', 'State':'state', 'Region' : 'region', 'Ship Mode':'ship_mode', 'Segment':'segment', 
                       'Postal Code':'postal_code','Customer ID':'customer_id', 'Product ID':'product_id', 'Category':'category', 'Sub-Category':'sub_category', 'Product Name':'product_name', 'Sales':'sales', 'Quantity':'quantity', 'Discount':'discount', 'Profit':'profit', 'Returned':'returned'})

# print(df.columns)
# print(df.describe())

# convert categorical data to numerical data
df['order_date'] = pd.to_datetime(df['order_date'])
df['ship_date'] = pd.to_datetime(df['ship_date'])

# df['order_date'] = df['order_date'].map(dt.datetime.toordinal)
# df['ship_date'] = df['ship_date'].map(dt.datetime.toordinal)

df['returned'] = df['returned'].map({True: 1, False: 0})

df['quantity'] = df['quantity'].astype(int)

# change state to numerical data but keep the original state names
df['state'] = df['state'].astype(str)
df['state_name'] = df['state']

df['state'] = df['state'].astype('category')
df['state'] = df['state'].cat.codes

# change city to numerical data but keep the original city names
df['city'] = df['city'].astype(str)
df['city_name'] = df['city']

df['city'] = df['city'].astype('category')
df['city'] = df['city'].cat.codes

#change category to numerical data but keep the original category names
df['category'] = df['category'].astype(str)
df['category_name'] = df['category']

df['category'] = df['category'].astype('category')
df['category'] = df['category'].cat.codes

#change sub_category to numerical data but keep the original sub_category names
df['sub_category'] = df['sub_category'].astype(str)
df['sub_category_name'] = df['sub_category']

df['sub_category'] = df['sub_category'].astype('category')
df['sub_category'] = df['sub_category'].cat.codes

# change ship_mode to numerical data but keep the original ship_mode names
df['ship_mode'] = df['ship_mode'].astype(str)
df['ship_mode_name'] = df['ship_mode']

df['ship_mode'] = df['ship_mode'].astype('category')
df['ship_mode'] = df['ship_mode'].cat.codes

# change segment to numerical data but keep the original segment names
df['segment'] = df['segment'].astype(str)
df['segment_name'] = df['segment']

df['segment'] = df['segment'].astype('category')
df['segment'] = df['segment'].cat.codes

# change customer_id to numerical data but keep the original customer_id names
df['customer_id'] = df['customer_id'].astype(str)
df['customer_id'] = df['customer_id']

df['customer_id'] = df['customer_id'].astype('category')
df['customer_id_code'] = df['customer_id'].cat.codes


df.to_csv('dataset/prod.csv', index=False)
# print(df.head())
# print(df.describe())
# print(df.columns)

end_date = max(df['order_date']) + dt.timedelta(days=1)

df_rfm = df.groupby('customer_id').agg(
    recency=('order_date', lambda x: (end_date - x.max()).days),
    frequency=('order_id', 'count'),
    monetary=('sales', 'sum'),
    # first_purchase=('order_date', 'min'),
    # last_purchase=('order_date', 'max'),
    state=('state', 'first'),
    city=('city', 'first'),
    segment=('segment', 'first'),
    ship_mode=('ship_mode', 'first'),
    category=('category', 'first'),
    sub_category=('sub_category', 'first'),
    customer_id_code=('customer_id_code', 'first')
).reset_index()

print(df_rfm)

# print(df_rfm.describe())

# form histograms for recency, frequency, and monetary with labels and titles and show the histograms
# plt.hist(df_rfm['recency'], bins=50)
# plt.xlabel('Recency')
# plt.ylabel('Frequency')
# plt.title('Recency Histogram')
# plt.show()

# plt.hist(df_rfm['frequency'], bins=50)
# plt.xlabel('Frequency')
# plt.ylabel('Frequency')
# plt.title('Frequency Histogram')
# plt.show()

# plt.hist(df_rfm['monetary'], bins=50)
# plt.xlabel('Monetary')
# plt.ylabel('Frequency')
# plt.title('Monetary Histogram')

# plt.show()

# show the histogram
# plt.show()
def preprocess(df):
    """Preprocess data for KMeans clustering"""
    # Convert categorical columns to numerical
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.codes
    
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    
    return norm

norm = preprocess(df_rfm)

def elbow_plot(df):
    """Create elbow plot from normalized data"""
    
    norm = preprocess(df)
    
    sse = {}
    
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(norm)
        sse[k] = kmeans.inertia_
    
    # plt.title('Elbow plot for K selection')
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # sns.pointplot(x=list(sse.keys()),
    #              y=list(sse.values()))
    # plt.show()
    
elbow_plot(df_rfm)

def find_k(df, increment=0, decrement=0):
    """Find the optimum k clusters"""
    
    norm = preprocess(df)
    sse = {}
    
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(norm)
        sse[k] = kmeans.inertia_
    
    kn = KneeLocator(
                 x=list(sse.keys()), 
                 y=list(sse.values()), 
                 curve='convex', 
                 direction='decreasing'
                 )
    k = kn.knee + increment - decrement
    return k

k = find_k(df_rfm)

print(k, "clusters ***********************")


def run_kmeans(df, increment=0, decrement=0):
    """Run KMeans clustering, including the preprocessing of the data
    and the automatic selection of the optimum k. 
    """
    
    norm = preprocess(df)
    k = find_k(df, increment, decrement)
    kmeans = KMeans(n_clusters=k, 
                    random_state=1)
    kmeans.fit(norm)
    return df.assign(cluster=kmeans.labels_)

clusters = run_kmeans(df_rfm)

clusters.groupby('cluster').agg(
    recency=('recency','mean'),
    frequency=('frequency','mean'),
    monetary=('monetary','mean'),
    cluster_size=('customer_id','count')
).round(1).sort_values(by='recency')

print(clusters.head())


clusters_decrement = run_kmeans(df_rfm, decrement=1)
clusters_decrement.groupby('cluster').agg(
    recency=('recency','mean'),
    frequency=('frequency','mean'),
    monetary=('monetary','mean'),
    cluster_size=('customer_id','count')
).round(1).sort_values(by='monetary')

segments = {3:'bronze', 0:'silver',1:'gold',2:'platinum'}
clusters_decrement['cluster_rank'] = clusters_decrement['cluster'].map(segments)
clusters_decrement.head()

print(clusters_decrement.head(1000))
print(clusters_decrement.describe())

clusters_decrement.segment.value_counts()

# plot clusters
def plot_clusters(df):
    """Plot clusters along with legend and labels."""
    fig = px.scatter(df, x='recency', y='frequency', color='cluster_rank', opacity=0.7, size_max=10, width=800, height=800, title='Clusters')
    fig.show()
    # plt.scatter(df['segment'], df['frequency'], c=df['cluster'], cmap='viridis')
    # plt.title('Clusters')
    # plt.xlabel('Segment')
    # plt.ylabel('Frequency')
    # plt.show()    

plot_clusters(clusters_decrement)
