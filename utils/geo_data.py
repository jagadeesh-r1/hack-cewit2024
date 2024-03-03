import pandas as pd 
from datetime import datetime
import json


data = pd.read_csv('dataset/storesales.csv')

# print(data.head())

geo_data = {}

for ind in data.index:
    try:
        temp_date = datetime.strptime(data['Order Date'][ind], '%m/%d/%y')
    except ValueError:
        temp_date = datetime.strptime(data['Order Date'][ind], '%Y-%m-%d')
    temp_month = temp_date.month
    temp_year = temp_date.year
    temp_day = temp_date.day
    if geo_data.get(data['State'][ind]):
        if geo_data[data['State'][ind]].get(data['City'][ind]):
            if geo_data[data['State'][ind]][data['City'][ind]].get(temp_year):
                if geo_data[data['State'][ind]][data['City'][ind]][temp_year].get(temp_month):
                    if geo_data[data['State'][ind]][data['City'][ind]][temp_year][temp_month].get(temp_day):
                        if geo_data[data['State'][ind]][data['City'][ind]][temp_year][temp_month][temp_day].get(data['Category'][ind]):
                            geo_data[data['State'][ind]][data['City'][ind]][temp_year][temp_month][temp_day][data['Category'][ind]] += 1
                        else:
                            geo_data[data['State'][ind]][data['City'][ind]][temp_year][temp_month][temp_day][data['Category'][ind]] = 1
                    else:
                        geo_data[data['State'][ind]][data['City'][ind]][temp_year][temp_month][temp_day] = {data['Category'][ind]: 1}
                else:
                    geo_data[data['State'][ind]][data['City'][ind]][temp_year][temp_month] = {temp_day: {data['Category'][ind]: 1}}
            else:
                geo_data[data['State'][ind]][data['City'][ind]][temp_year] = {temp_month: {temp_day: {data['Category'][ind]: 1}}}
        else:
            geo_data[data['State'][ind]][data['City'][ind]] = {temp_year: {temp_month: {temp_day: {data['Category'][ind]: 1}}}}
    else:
        geo_data[data['State'][ind]] = {data['City'][ind]: {temp_year: {temp_month: {temp_day: {data['Category'][ind]: 1}}}}}

# print(geo_data)
        
with open('dataset/geo_data.json', 'w') as outfile:
    json.dump(geo_data, outfile, indent=4)