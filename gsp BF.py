import numpy as np
import cv2
from skimage.util import random_noise

import networkx as nx
from scipy.sparse.linalg import eigsh

# Path to the image
path = "C:/Users/Edgar/Desktop/Uni/Seminar/Figures/images/"
image_path = path + 'bear3.jpg'

# Load the image in grayscale
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img_gaussian_noise = random_noise(original_img, mode='gaussian', mean=0, var=0.005)

# Normalize the image so that 0 is black and 1 is white
normalized_img = img_gaussian_noise.astype(np.float32)  # Convert to float32 for normalization
normalized_img = (normalized_img - np.min(normalized_img)) / (np.max(normalized_img) - np.min(normalized_img))

# Get the number of rows and columns
rows, cols = normalized_img.shape

# Print the number of rows and columns
print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")

# Create a grid graph using NetworkX
G = nx.grid_2d_graph(rows, cols)

# Define sigma values for spatial and intensity components
sigma_d = 1.0  # Spatial Gaussian standard deviation
sigma_r = 0.1  # Intensity Gaussian standard deviation

# Define the weight function based on the given formula
def weight_function(node1, node2):
    # Check if the nodes are the same
    if node1 == node2:
        return 1.0

    # Spatial Gaussian
    spatial_distance = np.linalg.norm(np.array(node1) - np.array(node2))
    spatial_gaussian = np.exp(-spatial_distance**2 / (2 * sigma_d**2))

    # Intensity Gaussian
    intensity_diff = normalized_img[node1[0], node1[1]] - normalized_img[node2[0], node2[1]]
    intensity_gaussian = np.exp(-intensity_diff**2 / (2 * sigma_r**2))

    # Combine the two components
    weight = spatial_gaussian * intensity_gaussian

    return weight

# Add weights to the edges
for (u, v) in G.edges():
    G.edges[u, v]['weight'] = weight_function(u, v)

# Compute the weighted graph Laplacian
L = nx.normalized_laplacian_matrix(G, weight='weight')

# Compute a smaller number of eigenvalues and eigenvectors using a sparse eigenvalue solver
num_eigenvalues = 50  # Number of eigenvalues to compute
eigenvalues, eigenvectors = eigsh(L, k=num_eigenvalues, which='SM')

# Print the eigenvalues
print("Eigenvalues:", eigenvalues)

# Reshape the normalized image into a vector
image_vector = normalized_img.flatten()

# Compute the Graph Fourier Transform (GFT)
gft = eigenvectors.T @ image_vector

# Apply the filter in the spectral domain
h_BF = 1 - eigenvalues
filtered_gft = h_BF * gft

# Compute the inverse GFT to get the filtered image back in the spatial domain
filtered_image_vector = eigenvectors @ filtered_gft
filtered_image = filtered_image_vector.reshape((rows, cols))

# Display the original and filtered images using OpenCV (optional)
cv2.imshow("Original Image", original_img)
cv2.imshow("Noise Image", normalized_img)
cv2.imshow("Filtered Image", (filtered_image * 255).astype(np.uint8))  # Scale the filtered image for display
cv2.waitKey(0)
cv2.destroyAllWindows()