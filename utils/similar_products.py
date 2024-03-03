import pandas as pd 
from datetime import datetime
import json


data = pd.read_csv('dataset/clean_products.csv')

# print(data.head())

product_data = {}

for ind in data.index:
    if product_data.get(data['Category'][ind]):
        if product_data[data['Category'][ind]].get(data['Sub-Category'][ind]):
            product_data[data['Category'][ind]][data['Sub-Category'][ind]]['skus'].append(data['sku'][ind])
            product_data[data['Category'][ind]][data['Sub-Category'][ind]]['product_names'].append(data['Product Name'][ind])
        else:
            product_data[data['Category'][ind]][data['Sub-Category'][ind]] = {'skus': [data['sku'][ind]], 'product_names': [data['Product Name'][ind]]}
    else:
        product_data[data['Category'][ind]] = {data['Sub-Category'][ind]: {'skus': [data['sku'][ind]], 'product_names': [data['Product Name'][ind]]}}

print(product_data)

with open('dataset/similar_product_data.json', 'w') as outfile:
    json.dump(product_data, outfile, indent=4)