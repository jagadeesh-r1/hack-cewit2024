
from flask import Flask, render_template, request, redirect, url_for
import json
import numpy as np
import plotly.express as px
from analytics.k_means import create_rfm, run_kmeans, dataset, segments

app = Flask(__name__)

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

@app.route('/create_rfm', methods=['POST'])
def entrypoint():
    try:
        user_selection = request.json['user_selection']
        print('here')
        user_selection = 'customer_id'
        df_rfm = create_rfm(user_selection, dataset)
        print(df_rfm)
        clusters = run_kmeans(df_rfm)
        # print(clusters)
        clusters['cluster_rank'] = clusters['cluster'].map(segments)
        print(clusters)
        fig = px.scatter(clusters, x='recency', y='frequency', color='cluster_rank', opacity=0.7, size_max=10, width=800, height=800, title='Clusters')
        response = json.dumps(fig.to_plotly_json(), default=default)
        # print(response)

        return response
    except Exception as e:
        print(e)
        return str(e)

entrypoint()

if __name__ == '__main__':
    app.run(debug=True)