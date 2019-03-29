import random

import cv2
import numpy as np
from numpy.linalg import eig, norm, inv
from matplotlib import gridspec
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import OrthogonalMatchingPursuit
from ksvd import ApproximateKSVD
import sparselandtools

def create_square_patches(img, num_patches, square_length):
    patches = []

    for _ in range(num_patches):
        start_row = random.randint(0, len(img) - square_length)
        start_col = random.randint(0, len(img[0]) - square_length)
        new_patch = img[start_row : start_row + square_length, start_col : start_col + square_length]
        patches.append(new_patch)

    return np.array(patches)

def create_correlation_matrix(patches):
    correlation_matrix = np.zeros(len(patches[0]))
    for patch in patches:
        outer_product = np.outer(patch, patch)
        correlation_matrix = np.add(correlation_matrix, outer_product)

    return correlation_matrix / len(patches)

def energy_DL(X, W, Z, U, lam=1):
    L = W.shape[0]
    N = X.shape[0]
    s = 0
    for i in range(N):
        s += pow(norm(X[i] - Z[i] @ W), 2)
        for k in range(L):
            s += lam * (pow(Z[i, k], 2) + pow(U[i, k], 2)) / (2 * U[i, k])
    return s

def ksvd(num_atoms):
    return ApproximateKSVD(num_atoms)

def normalize(patches):
    s = np.zeros((len(patches[0]), 1))
    for patch in patches:
        s += patch
    mean = s / len(patches)
    for patch in patches:
        patch -= mean
    return patches

def plot_eigenvectors(eigenvectors64, title):
    fig = plt.figure(figsize=(4, 10)) 
    columns = 8
    rows = 8

    gs = gridspec.GridSpec(rows, columns, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
         wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 

    count = 0
    for i in range(rows):
        for j in range(columns):
            img = eigenvectors64[count]
            ax = plt.subplot(gs[i,j])
            for tic in ax.xaxis.get_major_ticks():  
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False  
            ax.imshow(img, cmap='gray')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            count += 1 
    plt.axis('off')
    plt.title(title)
    plt.show()

def update_U(Z):
    (N, L) = Z.shape
    U = np.zeros((N, L))
    for i in range(N):
        for k in range(L):
            U[i, k] = EPSILON if (Z[i, k] <= EPSILON) else Z[i, k]
    return U 

def update_Z(X, W, U, lam=1):
    Z = np.zeros((N, L))
    for i in range(N):
        Ui = np.zeros((L,L))
        for k in range(L):
            Ui[k,k] = 1 / (2 * U[i, k])
        Z[i] = inv(W @ W.T + lam * Ui) @ W @ X[i].T
    return Z

# Set global parameters
EPSILON = pow(10,-9)
NUM_ATOMS = 350
NUM_PCA_COMPONENTS = 64
NUM_ITERATIONS = 70
THRESHOLD = 500

#~~~~~~~~~~~~~~~~~~~~~~ Principal Component Analysis (PCA) ~~~~~~~~~~~~~~~~~~~~~

# 1 Read in image
filename = 'clockwork-angels.jpg'
img = mpimg.imread(filename)

# 2 Get red layer from the image tensor
img_r = img[0:, 0:, 0]
img_r = img_r.astype('double')

# 3 Create 1000 random patches within the image
num_patches = 1000
patch_square_size = 16
patches = create_square_patches(img_r, num_patches, patch_square_size)
flattened_patches = [patch.reshape(256,1) for patch in patches]
norm_flattened_patches = np.array(normalize(flattened_patches))

# 4 Create the correlation matrix
correlation_matrix = create_correlation_matrix(norm_flattened_patches)

# 5 Compute and sort in decreasing order the eigenvectors of the correlation matrix
(eigenvalues, eigenvectors) = np.linalg.eig(correlation_matrix)
eigen_tuples = zip(eigenvalues, eigenvectors)
eigen_tuples = [(tup[0], tup[1].reshape(16,16)) for tup in eigen_tuples] # Reshape the 256x1 eigenvectors as 16x16 eigenvectors   
eigen_tuples64 = eigen_tuples[:64]
eigenvalues64 = [tup[0] for tup in eigen_tuples64]

# 6 Plot variation around PCs
variation1 = eigenvalues[0] / (num_patches - 1)
variation2 = eigenvalues[1] / (num_patches - 1)
total_variation = variation1 + variation2
percent1 = variation1 / total_variation
percent2 = variation2 / total_variation

# 7 Show Accountability of Variation in Data (PC1 & PC2)
plt.bar([0, 1], [variation1 / total_variation * 100, variation2 / total_variation * 100], tick_label=['PC1', 'PC2'])
plt.ylabel("%")
plt.title("Accountability of Variation in Data (PC1 & PC2)")
plt.show()

# 8 Display top 64 eigenvectors as 16x16 images in an 8x8 table
eigenvectors64 = [tup[1] for tup in eigen_tuples64]
plot_eigenvectors(eigenvectors64, "Top 64 Eigenvectors")

#~~~~~~~~~~~~~~~~~~~~~~ Dictionary Learning (DL) ~~~~~~~~~~~~~~~~~~~~~

# 1 Learn the dictionary
aksvd = ksvd(NUM_ATOMS)
print("Using KSVD to compute dictionary...")
W = aksvd.fit(norm_flattened_patches.reshape(1000,256)).components_

# 2 Plot 64 atoms
atoms64 = [atom.reshape(16,16) for atom in W[:64]]
plot_eigenvectors(atoms64, "First 64 Atoms")

# 3 LASSO
N = num_patches
L = W.shape[0]
U = np.ones((N,L))
X = norm_flattened_patches.reshape(1000,256)

results = []
for r in range(NUM_ITERATIONS):
    Z = update_Z(X, W, U)
    U = update_U(Z)
    result = energy_DL(X, W, Z, U)
    results.append(result)
    print(f"Round {r} | Majorized Dictionary Learning Objective Function Value: {result}")
    if r > 1 and abs(results[-2] - results[-1]) < THRESHOLD:
        results.append(result)
        print("Convergence reached...")
        break

plt.plot(results[1:])
plt.yscale('log')
plt.xlabel("Alternating Algorithm Iteration")
plt.ylabel("Objective Function Value")
plt.title("Objective Function Value vs. Alternating Algorithm Iteration")
plt.show()