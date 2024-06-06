import numpy as np
import cv2
from skimage.util import random_noise
import time
import networkx as nx
from cupyx.scipy.sparse.linalg import eigsh
import cupy as cp
import matplotlib.pyplot as plt
from scipy import sparse


# Load and preprocess the image
path = "C:/Users/Edgar/Desktop/Uni/Seminar/Figures/images/"
image_path = path + 'bear4.png'

original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_gaussian_noise = random_noise(original_img, mode='gaussian', mean=0, var=0.005)

normalized_img = img_gaussian_noise.astype(np.float32)
normalized_img = (normalized_img - np.min(normalized_img)) / (np.max(normalized_img) - np.min(normalized_img))


rows, cols = normalized_img.shape

# Construct the graph
G = nx.grid_2d_graph(rows, cols)

sigma_d = 1.0  # Spatial Gaussian standard deviation
sigma_r = 0.1  # Intensity Gaussian standard deviation

# Define the weight function based on the given formula
def weight_function(node1, node2):
    if node1 == node2:
        return 1.0
    spatial_distance = np.linalg.norm(np.array(node1) - np.array(node2))
    spatial_gaussian = np.exp(-spatial_distance**2 / (2 * sigma_d**2))
    intensity_diff = normalized_img[node1[0], node1[1]] - normalized_img[node2[0], node2[1]]
    intensity_gaussian = np.exp(-intensity_diff**2 / (2 * sigma_r**2))
    return spatial_gaussian * intensity_gaussian

for (u, v) in G.edges():
    G.edges[u, v]['weight'] = weight_function(u, v)

# Compute the degree matrix D
degrees = list(dict(G.degree(weight='weight')).values())
D = sparse.diags(degrees, format='csr')

# Compute the weight matrix W
W = nx.adjacency_matrix(G, weight='weight')

# Compute the combinatorial Laplacian matrix L
L_combinatorial = D - W
print(f"Combinatorial Laplacian matrix: {L_combinatorial}")

# Compute the identity matrix I
I = sparse.identity(L_combinatorial.shape[0], format='csr')
print(f"Identity matrix: {I}")

# Reshape the input signal to match the Laplacian matrix dimensions
x_in = normalized_img.flatten().reshape((-1, 1))

# Compute (I - L_combinatorial) * x_in
result_combinatorial = (I - L_combinatorial) @ x_in

# Reshape the result back to the image dimensions
filtered_image_signal_combinatorial = result_combinatorial.reshape((rows, cols))


# Display the original and filtered images
plt.figure(figsize=(16, 12))

plt.subplot(1, 3, 1)
plt.title("Original Image", fontsize=16)
plt.imshow(original_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gaussian Noise", fontsize=16)
plt.imshow(img_gaussian_noise, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Filtered Image Signal", fontsize=16)
plt.imshow(filtered_image_signal_combinatorial, cmap='gray')
plt.axis('off')

plt.show()
