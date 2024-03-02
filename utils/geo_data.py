import pandas as pd 
from datetime import datetime
import json


data = pd.read_csv('dataset/storesales.csv')

# print(data.head())

geo_data = {}

for ind in data.index:
    try:
        temp_month = datetime.strptime(data['Order Date'][ind], '%m/%d/%y').month
    except ValueError:
        temp_month = datetime.strptime(data['Order Date'][ind], '%Y-%m-%d').month
    if geo_data.get(data['State'][ind]):
        if geo_data[data['State'][ind]].get(data['City'][ind]):
            if geo_data[data['State'][ind]][data['City'][ind]].get(temp_month):
                if geo_data[data['State'][ind]][data['City'][ind]][temp_month].get(data['Category'][ind]):
                    geo_data[data['State'][ind]][data['City'][ind]][temp_month][data['Category'][ind]] += 1
                else:
                    geo_data[data['State'][ind]][data['City'][ind]][temp_month][data['Category'][ind]] = 1
            else:
                geo_data[data['State'][ind]][data['City'][ind]][temp_month] = {data['Category'][ind]: 1}
        else:
            geo_data[data['State'][ind]][data['City'][ind]] = {temp_month: {data['Category'][ind]: 1}}
    else:
        geo_data[data['State'][ind]] = {data['City'][ind]: {temp_month: {data['Category'][ind]: 1}}}

# print(geo_data)
        
with open('dataset/geo_data.json', 'w') as outfile:
    json.dump(geo_data, outfile, indent=4)