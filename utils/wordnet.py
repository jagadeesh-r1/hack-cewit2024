import pandas as pd
import numpy as np

data = pd.read_csv('dataset/storesales.csv')


data['similar_products'] = data['Category'] + ' ' + data['Sub-Category'] + ' ' + data['Product Name']
data['top_similar_products'] = data['Category']

#load the glove model and query similar_products to get the vectors
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'dataset/glove.6B.100d.txt'
word2vec_output_file = 'dataset/glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


# lemmatize and stem words
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#remove stopwords and punctuation
import string
from nltk.corpus import stopwords


def lemmatize_and_punctuations_words(data):
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    words = data.split()
    words = [word.lower() for word in words]
    words = [word.translate(table) for word in words]
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  
    return ' '.join(words)


# all the words in the model

def get_vector(data):
    n_words = 0
    feature_vec = np.zeros((100,), dtype="float32")
    for word in data.split():
        try:
            feature_vec = np.add(feature_vec, model.get_vector(word))
            n_words += 1
        except:
            pass
        # result = model.get_vector(word.lower())
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    # print(feature_vec)
    return feature_vec

data['similar_products_vectors'] = data['similar_products'].apply(lambda x: get_vector(lemmatize_and_punctuations_words(x)))

# print(data['similar_products_vectors'])
products = data[['Category', 'Sub-Category', 'Product Name', 'similar_products', 'similar_products_vectors', 'top_similar_products']]
# remove duplicates
products = products.drop_duplicates(subset='similar_products', keep='first')

# assign an sku to each product like sku0001, sku0002, etc
products['sku'] = "sku" + (products.index + 1).astype(str).str.zfill(4)
print(products.tail())

# create column with top 5 similar products using cosine similarity
import numpy as np
from scipy import spatial



def get_top_similar_products(product_vector, products_df):
    """return sku of top 5 similar products"""
    # initialize similarity column with 0
    products_df['similarity'] = 0
    # print(products_df.shape)

    similarity = []
    for index, row in products_df.iterrows():
        similarity.append(1 - spatial.distance.cosine(product_vector, row['similar_products_vectors']))

    similaritydf = pd.DataFrame(similarity, columns=['similarity'])
    # print(similaritydf.shape)
    for ind in products_df.index:
        try:
            products_df.loc[ind, 'similarity'] = similaritydf['similarity'][ind]
        except:
            # print(ind)
            break

    products_df = products_df.sort_values(by='similarity', ascending=False)
    # return sku of top 5 similar products as list
    return products_df.head(6)['sku'].tolist()

    # return products_df.head(6)

products_deepcopy = products.copy()

# get the top 5 similar products for each product
for index in products.index:
    print(products['similar_products'][index])
    product_vector = products['similar_products_vectors'][index]
    similar_skus = get_top_similar_products(product_vector, products_deepcopy)
    print(similar_skus)
    products['top_similar_products'][index] = similar_skus[1:]


print(products.head())

products.to_csv('dataset/products.csv', index=False)
data.to_csv('dataset/storesales.csv', index=False)


