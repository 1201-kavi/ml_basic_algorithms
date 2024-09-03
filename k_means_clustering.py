from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a random dataset with 3 clusters
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Create and train the model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict the clusters
y_pred = kmeans.predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()
