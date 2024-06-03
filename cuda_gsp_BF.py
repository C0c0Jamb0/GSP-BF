import numpy as np
import cv2
from skimage.util import random_noise
import time
import networkx as nx
from cupyx.scipy.sparse.linalg import eigsh
import cupy as cp
import matplotlib.pyplot as plt

path = "..."
image_path = path + '...'

original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img_gaussian_noise = random_noise(original_img, mode='gaussian', mean=0, var=0.005)

# Normalize the image so that 0 is black and 1 is white
normalized_img = img_gaussian_noise.astype(np.float32)
normalized_img = (normalized_img - np.min(normalized_img)) / (np.max(normalized_img) - np.min(normalized_img))

rows, cols = normalized_img.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")

G = nx.grid_2d_graph(rows, cols)

sigma_d = 1.0  # Spatial Gaussian standard deviation
sigma_r = 0.1  # Intensity Gaussian standard deviation

# Define the weight function based on the given formula
def weight_function(node1, node2):
    if node1 == node2:
        return 1.0

    # Spatial Gaussian
    spatial_distance = np.linalg.norm(np.array(node1) - np.array(node2))
    spatial_gaussian = np.exp(-spatial_distance**2 / (2 * sigma_d**2))

    # Intensity Gaussian
    intensity_diff = normalized_img[node1[0], node1[1]] - normalized_img[node2[0], node2[1]]
    intensity_gaussian = np.exp(-intensity_diff**2 / (2 * sigma_r**2))

    weight = spatial_gaussian * intensity_gaussian

    return weight

for (u, v) in G.edges():
    G.edges[u, v]['weight'] = weight_function(u, v)

L = nx.normalized_laplacian_matrix(G, weight='weight')

L_cupy = cp.sparse.csr_matrix(L)


# Compute a smaller number of eigenvalues and eigenvectors using a sparse eigenvalue solver
num_eigenvalues = 10 

start_time = time.time()
eigenvalues, eigenvectors = eigsh(L_cupy, k=num_eigenvalues, which='SA')

end_time = time.time()

print(num_eigenvalues)
print("Eigenvalues:", eigenvalues)

elapsed_time = end_time - start_time
print(f"Time taken to compute eigenvalues: {elapsed_time:.2f} seconds")

# Reshape the normalized image into a vector
image_vector = normalized_img.flatten()

image_vector_cupy = cp.asarray(image_vector)

# Compute the Graph Fourier Transform (GFT)
gft = eigenvectors.T @ image_vector_cupy

# Apply the bilateral filter in the spectral domain
h_BF = 1 - eigenvalues
filtered_gft = h_BF * gft

# Compute the inverse GFT to get the filtered image back in the spatial domain
filtered_image_vector = eigenvectors @ filtered_gft
filtered_image = filtered_image_vector.reshape((rows, cols)).get()

filtered_image = (filtered_image * 255).astype(np.uint8)

# Display the original and filtered images
x1, x2, y1, y2 = 110, 190, 50, 110
plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
plt.title("Original Image", fontsize=16)
plt.imshow(original_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Gaussian Noise", fontsize=16)
plt.imshow(img_gaussian_noise, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Bilateral Filter", fontsize=16)    
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(filtered_image[y1:y2, x1:x2], cmap='gray')
plt.title("Zoomed Bilateral Filter", fontsize=16)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(original_img[y1:y2, x1:x2], cmap='gray')
plt.title("Zoomed Bilateral Filter", fontsize=16)
plt.axis('off')

save_path = "..."
plt.savefig(save_path + "...") #save in .svg
plt.show()