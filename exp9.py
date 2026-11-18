import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate Dataset
X, y = make_blobs(
    n_samples=200,
    centers=3,
    random_state=42
)

# Create K-Means Model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train Model
model.fit(X)

# Cluster Labels
labels = model.labels_

# Cluster Centers
centers = model.cluster_centers_

print("K-Means Clustering")
print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Visualization
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap='viridis'
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
