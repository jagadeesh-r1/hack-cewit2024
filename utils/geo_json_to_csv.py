# load the geo_data and convert it to csv
import json
import pandas as pd
import csv
from datetime import datetime

with open('dataset/geo_data.json') as f:
    data = json.load(f)

# print(data)

# create a list of dictionaries
rows = []
for state in data:
    for city in data[state]:
        for year in data[state][city]:
            for month in data[state][city][year]:
                for day in data[state][city][year][month]:
                    for category in data[state][city][year][month][day]:
                        rows.append({
                            'state': state,
                            'city': city,
                            'year': year,
                            'month': month,
                            'day': day,
                            'category': category,
                            'sales': data[state][city][year][month][day][category]
                        })

# print(rows)
                    
# write to csv
with open('dataset/geo_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['state', 'city', 'year', 'month','day', 'category', 'sales'])
    writer.writeheader()
    writer.writerows(rows)