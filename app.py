import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from kmeans import KMeans

# Initialize the Dash app
app = dash.Dash(__name__)

# Ensure the app runs on port 3000
server = app.server
app.server.port = 3000
app.server.host = '0.0.0.0'

# App layout
app.layout = html.Div([
  html.H1("KMeans Clustering Visualization"),

  html.Div([
    html.Div([
      html.Label("Initialization Method:"),
      dcc.Dropdown(
        id='init-method-dropdown',
        options=[
          {'label': 'Random', 'value': 'random'},
          {'label': 'Farthest First', 'value': 'farthest'},
          {'label': 'KMeans++', 'value': 'kmeans++'},
          {'label': 'Manual', 'value': 'manual'}
        ],
        value='random',
        clearable=False
        ),
      ], className='control-element'),

      html.Div([
        html.Label("Number of Clusters: "),
        dcc.Input(
          id='num-clusters-input',
          type='number',
          value=4,  # Default value
          min=1,    # Minimum number of clusters
          step=1,   # Increment step
        ),
      ], className='control-element'),

  ], className='controls'),

  html.Div([
    html.Button('Generate new dataset', id='generate-dataset-button', n_clicks=0),
    html.Button('Reset algorithm', id='reset-button', n_clicks=0),
  ], className='button-group'),

  html.Div([
    html.Button('Step through KMeans', id='step-button', n_clicks=0),
    html.Button('Run to convergence', id='run-button', n_clicks=0),
  ], className='button-group'),

  dcc.Graph(
    id='cluster-graph',
    style={'height': '600px'},
    className='graph',
    clickData=None,
  ),

  # Store data in hidden divs
  dcc.Store(id='dataset'),
  dcc.Store(id='kmeans-state'),
  dcc.Store(id='current-step'),
  dcc.Store(id='manual-centroids', data=[]),
])

# Callback to generate new dataset
@app.callback(
  Output('dataset', 'data'),
  Input('generate-dataset-button', 'n_clicks'),
  State('num-clusters-input', 'value')
)
def generate_dataset(n_clicks, n_clusters):
  if n_clusters is None or n_clusters < 1:
    n_clusters = 1
  # Generate a new random dataset
  X = []
  for _ in range(n_clusters):
    center = np.random.uniform(-10, 10, 2)
    points = center + np.random.randn(50, 2)
    X.append(points)
  X = np.vstack(X)
  # Convert to list for JSON serialization
  return X.tolist()

# Combined callback for KMeans operations
@app.callback(
  Output('kmeans-state', 'data'),
  Output('current-step', 'data'),
  Output('manual-centroids', 'data'),
  Input('dataset', 'data'),
  Input('init-method-dropdown', 'value'),
  Input('num-clusters-input', 'value'),
  Input('reset-button', 'n_clicks'),
  Input('step-button', 'n_clicks'),
  Input('run-button', 'n_clicks'),
  Input('cluster-graph', 'clickData'),
  State('kmeans-state', 'data'),
  State('current-step', 'data'),
  State('manual-centroids', 'data')
)
def update_kmeans(dataset, init_method, n_clusters, reset_n_clicks, step_n_clicks, run_n_clicks, clickData, kmeans_state, current_step, manual_centroids):
  ctx = dash.callback_context
  if not ctx.triggered:
    return dash.no_update, dash.no_update, dash.no_update
  trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

  if dataset is None:
    return dash.no_update, dash.no_update, dash.no_update
  X = np.array(dataset)

  if n_clusters is None or n_clusters < 1:
    n_clusters = 1

  if trigger_id == 'dataset':
    kmeans_state = {
      'centroids': [],
      'labels': [-1]*len(X),
      'iteration': 0,
      'n_clusters': n_clusters
    }
    current_step = 0
    manual_centroids = []
    return kmeans_state, current_step, manual_centroids

  elif trigger_id == 'init-method-dropdown' or trigger_id == 'reset-button':
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
    if init_method == 'manual':
      if manual_centroids and len(manual_centroids) == n_clusters:
        kmeans.set_centroids(np.array(manual_centroids))
        kmeans.labels = np.full(len(X), -1)
      else:
        # Centroids are not set yet; avoid calling fit
        kmeans.centroids = None
        kmeans.labels = np.full(len(X), -1)
    else:
      kmeans.initialize_centroids(X)
      kmeans.labels = np.full(len(X), -1)
    kmeans_state = {
      'centroids': kmeans.centroids.tolist() if kmeans.centroids is not None else [],
      'labels': kmeans.labels.tolist(),
      'iteration': 0,
      'n_clusters': n_clusters
    }
    current_step = 0
    if init_method != 'manual':
      manual_centroids = []
    return kmeans_state, current_step, manual_centroids

  elif trigger_id == 'cluster-graph' and init_method == 'manual':
    # Handle manual centroid selection
    if clickData is None:
      return dash.no_update, dash.no_update, dash.no_update
    point = clickData['points'][0]
    new_centroid = [point['x'], point['y']]
    manual_centroids.append(new_centroid)
    # Limit the number of centroids to the number of clusters
    if len(manual_centroids) > n_clusters:
      manual_centroids = manual_centroids[-n_clusters:]
    # Update centroids in kmeans_state
    kmeans_state['centroids'] = manual_centroids
    centroids = np.array(manual_centroids)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    kmeans_state['labels'] = labels.tolist()
    kmeans_state['n_clusters'] = n_clusters
    return kmeans_state, current_step, manual_centroids

  elif trigger_id == 'step-button':
    # Perform one iteration
    centroids = np.array(kmeans_state['centroids'])
    labels = np.array(kmeans_state['labels'])
    # Assignment Step
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    # Update Step
    new_centroids = np.array([
      X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
      for i in range(len(centroids))
    ])
    current_step += 1
    # Update kmeans_state
    kmeans_state['centroids'] = new_centroids.tolist()
    kmeans_state['labels'] = labels.tolist()
    kmeans_state['iteration'] = current_step
    return kmeans_state, current_step, manual_centroids

  elif trigger_id == 'run-button':
    # Run until convergence
    centroids = np.array(kmeans_state['centroids'])
    labels = np.array(kmeans_state['labels'])
    max_iter = 100  # To prevent infinite loops
    tol = 1e-4
    for _ in range(max_iter):
      # Assignment Step
      distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
      labels = np.argmin(distances, axis=1)
      # Update Step
      new_centroids = np.array([
        X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
        for i in range(len(centroids))
      ])
      current_step += 1
      # Check for convergence
      if np.allclose(centroids, new_centroids, atol=tol):
        break
      centroids = new_centroids
    # Update kmeans_state
    kmeans_state['centroids'] = centroids.tolist()
    kmeans_state['labels'] = labels.tolist()
    kmeans_state['iteration'] = current_step
    return kmeans_state, current_step, manual_centroids

  else:
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('cluster-graph', 'figure'),
    Input('dataset', 'data'),
    Input('kmeans-state', 'data'),
)
def update_graph(dataset, kmeans_state):
  if dataset is None or kmeans_state is None:
    return dash.no_update

  X = np.array(dataset)
  centroids = np.array(kmeans_state['centroids'])
  labels = np.array(kmeans_state['labels'])
  n_clusters = kmeans_state.get('n_clusters', 1)

  # Create scatter plots for each cluster
  data = []
  if np.any(labels >= 0):
    unique_labels = np.unique(labels[labels >= 0])
    for i in unique_labels:
      cluster_points = X[labels == i]
      data.append(go.Scatter(
        x=cluster_points[:, 0],
        y=cluster_points[:, 1],
        mode='markers',
        name=f'Cluster {int(i)+1}'
      ))
  else:
    # No labels assigned yet
    data.append(go.Scatter(
      x=X[:, 0],
      y=X[:, 1],
      mode='markers',
      name='Data Points'
    ))

  # Plot centroids if available and properly shaped
  if centroids.size > 0:
    centroids = np.atleast_2d(centroids)  # Ensure centroids are 2D
    data.append(go.Scatter(
      x=centroids[:, 0],
      y=centroids[:, 1],
      mode='markers',
      marker=dict(symbol='x', size=12, color='black'),
      name='Centroids'
    ))

  figure = go.Figure(data=data)
  figure.update_layout(
    title='KMeans Clustering Visualization',
    clickmode='event+select'
  )
  return figure


if __name__ == '__main__':
  app.run_server(debug=True, host='0.0.0.0', port=3000)
