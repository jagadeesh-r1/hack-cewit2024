import pandas as pd

products = pd.read_csv('dataset/products.csv')

print(products.head())
print(products.columns)

products = products.drop(['similar_products', 'similar_products_vectors'], axis=1)

print(products.head())

products.to_csv('dataset/clean_products.csv', index=False)