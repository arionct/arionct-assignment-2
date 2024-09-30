import numpy as np

class KMeans:
  def __init__(self, n_clusters=3, init_method='random', max_iter=300, tol=1e-10):
    self.n_clusters = n_clusters
    self.init_method = init_method
    self.max_iter = max_iter
    self.tol = tol
    self.centroids = None
    self.labels = None

  def _initialize_random(self, X):
    indices = np.random.choice(len(X), self.n_clusters, replace=False)
    return X[indices]

  def _initialize_farthest(self, X):
    centroids = []
    # Randomly choose the first centroid
    centroids.append(X[np.random.choice(len(X))])
    for _ in range(1, self.n_clusters):
      # Compute distances from the data points to the nearest centroid
      dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
      # Choose the data point with the maximum distance as the next centroid
      next_centroid = X[np.argmax(dist_sq)]
      centroids.append(next_centroid)
    return np.array(centroids)

  def _initialize_kmeanspp(self, X):
    centroids = []
    # Randomly choose the first centroid
    centroids.append(X[np.random.choice(len(X))])
    for _ in range(1, self.n_clusters):
      # Compute squared distances to the nearest centroid
      dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
      # Compute probabilities proportional to the squared distances
      probs = dist_sq / dist_sq.sum()
      # Select the next centroid based on the computed probabilities
      cumulative_probs = probs.cumsum()
      r = np.random.rand()
      for idx, prob in enumerate(cumulative_probs):
        if r < prob:
          centroids.append(X[idx])
          break
    return np.array(centroids)

  def set_centroids(self, centroids):
    self.centroids = np.array(centroids)

  def initialize_centroids(self, X):
    # Initialize centroids
    if self.init_method == 'random':
      self.centroids = self._initialize_random(X)
    elif self.init_method == 'farthest':
      self.centroids = self._initialize_farthest(X)
    elif self.init_method == 'kmeans++':
      self.centroids = self._initialize_kmeanspp(X)
    elif self.init_method == 'manual':
      if self.centroids is None:
        raise ValueError("Centroids must be set manually before calling fit.")
    else:
      raise ValueError(f"Unknown initialization method '{self.init_method}'")

  def fit(self, X):
    self.initialize_centroids(X)
    for iteration in range(self.max_iter):
      # Assignment Step
      distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
      labels = np.argmin(distances, axis=1)
      
      # Update Step
      new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i] for i in range(self.n_clusters)])
      
      # Convergence Check
      if np.allclose(self.centroids, new_centroids, atol=self.tol):
        print(f"Converged at iteration {iteration}")
        break
      self.centroids = new_centroids
    
    self.labels = labels

def generate_dataset(num_points=300, num_clusters=4, cluster_std=0.60, random_state=0):
  from sklearn.datasets import make_blobs
  X, y_true = make_blobs(n_samples=num_points, centers=num_clusters, cluster_std=cluster_std, random_state=random_state)
  return X

if __name__ == '__main__':
  import matplotlib.pyplot as plt

  # Generate sample data
  X = generate_dataset()

  # List of initialization methods to test
  init_methods = ['random', 'farthest', 'kmeans++']

  for method in init_methods:
    # Instantiate KMeans with the current initialization method
    kmeans = KMeans(n_clusters=4, init_method=method)
    kmeans.fit(X)

    # Plot the clustered data
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, s=30, cmap='viridis', label='Data Points')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.5, label='Centroids')
    plt.title(f'KMeans Clustering with {method} Initialization')
    plt.legend()
    plt.show()
