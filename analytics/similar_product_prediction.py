import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import ast
import numpy as np
import json

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

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


def return_similar_products(product_name):
    """Returns a list of similar products based on the input product_name."""
    try:
        # product_data = pd.read_csv('dataset/clean_products.csv')
        # product_data['top_similar_products'] = product_data['top_similar_products'].apply(lambda x: ast.literal_eval(x))
        # product_data['top_similar_products'] = product_data['top_similar_products'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # sku_to_name = product_data.set_index('sku')['Product Name'].to_dict()
        sku = product_data[product_data['Product Name'] == product_name]['sku'].values[0]
        print(sku)

        similar_skus = product_data[product_data['sku'] == sku]['top_similar_products'].values[0]
        similar_products = []
        for sku in similar_skus:
            similar_products.append(sku_to_name[sku])

        # create a network graph to visualize the similar products
        G = nx.Graph()
        G.add_node(sku_to_name[sku])
        for product in similar_products:
            G.add_node(product)
            G.add_edge(sku_to_name[sku], product)

        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'{adjacencies[0]}: {len(adjacencies[1])}')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Network graph of similar products',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        # fig.show()
        return json.dumps(fig.to_plotly_json(), default=default)
        # return similar_products
    except Exception as e:
        return str(e)
    

