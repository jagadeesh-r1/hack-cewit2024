import pandas as pd

sales_data = pd.read_csv('dataset/storesales.csv')

print(sales_data.head())
print(sales_data.columns)

sales_data = sales_data.drop(['similar_products_vectors', 'top_similar_products', 'similar_products', 'Row ID', 'Customer Name'], axis=1)

sales_data.to_csv('dataset/clean_sales.csv', index=False)