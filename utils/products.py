# load data and separate the products into a separate dataframe

import pandas as pd
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec


glove_input_file = 'dataset/glove.6B.100d.txt'
word2vec_output_file = 'dataset/glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

data = pd.read_csv('dataset/storesales.csv')

# print(data.head())

products = data[['Category', 'Sub-Category', 'Product Name', 'similar_products', 'similar_products_vectors']]

print(products.head())

# assign an sku to each product like sku0001, sku0002, etc
products['sku'] = "sku" + (products.index + 1).astype(str).str.zfill(4)
print(products.tail())

# create column with top 5 similar products using cosine similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_similar_products(product_vector, products):
    similarity = cosine_similarity(product_vector, products['similar_products_vectors'].tolist())
    # print(similarity)
    similarity = similarity[0]
    # print(similarity)
    products['similarity'] = similarity
    products = products.sort_values(by='similarity', ascending=False)
    return products.head(6)

# get the top 5 similar products for each product
for index, row in products.iterrows():
    product_vector = row['similar_products_vectors'].tolist()
    products.loc[index, 'top_similar_products'] = get_top_similar_products(product_vector, products)

print(products.head())