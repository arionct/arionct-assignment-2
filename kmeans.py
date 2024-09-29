import numpy as np

class KMeans:
  def __init__(self, n_clusters=3, init_method='random', max_iter=300, tol=1e-4):
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

