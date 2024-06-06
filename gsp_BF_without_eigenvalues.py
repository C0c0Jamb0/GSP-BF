import numpy as np
import cv2
from skimage.util import random_noise
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
from config import path, save_path
from scipy.sparse.linalg import matrix_power

# Load and preprocess the image
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

# Compute the identity matrix I
I = sparse.identity(L_combinatorial.shape[0], format='csr')

# Reshape the input signal to match the Laplacian matrix dimensions
x_in = normalized_img.flatten().reshape((-1, 1))

# Compute the random walk Laplacian matrix L_random_walk
L_random_walk = sparse.linalg.inv(D) @ L_combinatorial

# Compute (I - L_random_walk) * x_in
result_random_walk = (I - L_random_walk) @ x_in

# Reshape the result back to the image dimensions
filtered_image_signal_random_walk = result_random_walk.reshape((rows, cols))

# Iteratively apply the filter for a certain number of iterations
test = (I - L_random_walk)
iterations = 10

result_iterative = sparse.linalg.matrix_power(test, iterations) @ x_in

# Reshape the result back to the image dimensions
filtered_image_signal_iterative = result_iterative.reshape((rows, cols))

# Display the original and filtered images
x1, x2, y1, y2 = 110, 190, 50, 110
plt.figure(figsize=(16, 12))

plt.subplot(2, 4, 1)
plt.title("Original Image", fontsize=16)
plt.imshow(original_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("Gaussian Noise", fontsize=16)
plt.imshow(img_gaussian_noise, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Filtered Image Signal (Random Walk Laplacian)", fontsize=16)
plt.imshow(filtered_image_signal_random_walk, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Filtered Image Signal Iterative", fontsize=16)
plt.imshow(filtered_image_signal_iterative, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(original_img[y1:y2, x1:x2], cmap='gray')
plt.title("", fontsize=16)
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(img_gaussian_noise[y1:y2, x1:x2], cmap='gray')
plt.title("", fontsize=16)
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(filtered_image_signal_random_walk[y1:y2, x1:x2], cmap='gray')
plt.title("", fontsize=16)
plt.axis('off')

plt.subplot(2, 4, 8)
plt.title("Filtered Image Signal Iterative", fontsize=16)
plt.imshow(filtered_image_signal_iterative[y1:y2, x1:x2], cmap='gray')
plt.axis('off')

plt.savefig(save_path + "new_bf.svg") #save in .svg

plt.show()
