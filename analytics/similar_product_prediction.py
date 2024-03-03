import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import ast


product_data = pd.read_csv('dataset/clean_products.csv')

# make str into list
product_data['top_similar_products'] = product_data['top_similar_products'].apply(lambda x: ast.literal_eval(x))
product_data['top_similar_products'] = product_data['top_similar_products'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# print(product_data['top_similar_products'])

# only consider 100 rows for now
# product_data = product_data.head(100)

# create a dictionary to map sku to product name
sku_to_name = product_data.set_index('sku')['Product Name'].to_dict()

# print(sku_to_name)





# def return_similar_products(sku):
#     """Returns a list of similar products based on the input sku."""
#     try:

