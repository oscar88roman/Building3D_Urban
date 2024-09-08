import numpy as np
import os 

from pathlib import Path
import pandas as pd

import laspy 
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy
from scipy import linalg
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.cluster import DBSCAN

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import trimesh

import geomapi.utils as ut
from geomapi.nodes import PointCloudNode
from geomapi.utils import geometryutils as gmu

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.ticker as ticker

# DL Packages 
import torch
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from pytorch3d.ops import knn_points

import context
# import utils 
import utils_ChallengeBuilding3D as ut_CB3D
from collections import defaultdict
import time




def xyz_to_las(folder):
    
    las_files = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder):

        if filename.endswith('.xyz'):

            xyz_file = os.path.join(folder, filename)
            las_file = os.path.join(folder, filename.replace('.xyz', '.las'))

            # Read XYZ data
            data = np.loadtxt(xyz_file)
            x, y, z = data[:, 0], data[:, 1], data[:, 2]

            # Create a new LAS file
            header = laspy.LasHeader(point_format=1, version="1.2")
            las = laspy.LasData(header)

            # Assign the data to LAS fields
            las.x = x
            las.y = y
            las.z = z

            # Write the data to LAS file
            las.write(las_file)

            # Append the path of the created LAS file to the list
            las_files.append(las_file)

    return las_files


# Function to read a .las file and extract 3D points
def extract_points(file_path):
    with laspy.open(file_path) as file:
        las_data = file.read()
        x = las_data.x
        y = las_data.y
        z = las_data.z
        return np.vstack((x, y, z)).T  # Stack and transpose to get an Nx3 array of points
    

def process_las_files(folder):
    
    points_dict = {}
    folder = Path(folder)

    for file_path in folder.glob('*.las'):
        points_3d = ut_CB3D.extract_points(file_path)
        points_3d = np.round(points_3d, 3)

        file_name = file_path.name
        normals = compute_normals(points_3d)
        normals = np.round(normals, 3)

        points_normals = np.hstack((points_3d, normals))

        points_dict[file_name] = {
            'name': file_name,
            'points_3d': points_3d,
            'points_normals': points_normals
        }

    return points_dict


def compute_covariance_matrix(points):
    return np.cov(points.T)



def center_data(points):
   
    mean = np.mean(points, axis=0)
    shifted_points = points - mean

    return mean, shifted_points


def recenter_data(shifted_points, mean):

    points = shifted_points + mean
    return points

def perform_svd(cov_matrix):
        
        try:
            U, S, Vt = np.linalg.svd(cov_matrix, full_matrices = True, compute_uv = True, hermitian = False)
            
            return U, S, Vt
        
        except np.linalg.LinAlgError as e:
           
            print(f"SVD did not converge: {e}")
            
            return None, None, None
        

def normalize_neighborhood(shifted_points, S, mean):
    
    S = np.array(S)
    
    # Compute scale factors as the square root of the singular values
    scale_factors = np.sqrt(S)
    
    # Avoid division by zero by setting a threshold
    scale_factors[scale_factors < 1e-10] = 1.0  
    
    # Reshape scale_factors to (1, 3) so it can be broadcasted across all points
    scale_factors = scale_factors.reshape(1, -1)
    
    # Normalize the shifted points by dividing each coordinate by the corresponding scale factor
    normalized_points = shifted_points / scale_factors
       
    return normalized_points
        

    
# def normalize_neighborhood_2(shifted_points, S, mean):

#     S = np.array(S)
#     scale_factors = np.sqrt(S)
#     # Avoid division by zero
#     scale_factors[scale_factors < 1e-10] = 1.0  
#     # Ensure normalization applies to the correct dimension
#     normalized_points = shifted_points / scale_factors
#     normalized_points = normalized_points + mean

#     return normalized_points


def extract_eigenvector_and_neighbors(normalized_points, Vt):
 
    n = Vt.T[:, 2]  # Eigenvector corresponding to smallest singular value
    print('n', n)
    distances = np.linalg.norm(normalized_points, axis = 1)
    print('Distances', distances)
    k = normalized_points.shape[0] // 2

    # if normalized_points.shape[0] < 3:
    #     k = normalized_points.shape[0]

    # else:
    #     k = normalized_points.shape[0] // 2
    
    # print('k', k)

    half_k = int(k // 2)

    print('half k', half_k)

    closest_indices = np.argsort(distances)[:half_k]
    closest_points = normalized_points[closest_indices]
 
    return n, closest_points


def partition_and_compute_covariance(closest_points, n):

    dot_products = np.dot(closest_points, n)

    N_i_upper = closest_points[dot_products >= 0]
    N_i_lower = closest_points[dot_products < 0]

    # Check if either partition is empty
    if N_i_upper.size == 0 or N_i_lower.size == 0:
        
        print("Empty partition detected.")
        
        return None, None, None, None, None, None

    # Compute covariance matrices for both partitions
    K_i_upper = np.cov(N_i_upper, rowvar=False)
    K_i_lower = np.cov(N_i_lower, rowvar=False)

    # Perform SVD on the covariance matrices
    _, Sigma_i_upper, _ = np.linalg.svd(K_i_upper)
    _, Sigma_i_lower, _ = np.linalg.svd(K_i_lower)

    # Compute the centers of mass for both partitions
    N_i_upper_center = np.mean(N_i_upper, axis=0)
    N_i_lower_center = np.mean(N_i_lower, axis=0)

    return N_i_upper, N_i_lower, Sigma_i_upper, Sigma_i_lower, N_i_upper_center, N_i_lower_center



def calculate_distances(points, mean, n):

    s_perpendicular = np.dot(points - mean, n)
    s_tangential = np.linalg.norm((points - mean) - s_perpendicular[:, np.newaxis] * n, axis=1)

    return s_perpendicular, s_tangential


####__________PYTORCH GEOMETRIC_____________________________________________

def compute_normals(points, k = 10):

    # Initialize Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    
    # Get the indices of k-nearest neighbors for each point
    distances, indices = nbrs.kneighbors(points)
    
    normals = np.zeros(points.shape)
    
    for i, idx in enumerate(indices):
        # Extract the local neighborhood
        neighbors = points[idx]
        
        # Compute the covariance matrix of the local neighborhood
        cov_matrix = np.cov(neighbors - neighbors.mean(axis = 0), rowvar = False)
        
        # Perform SVD to find the normal
        _, _, Vt = np.linalg.svd(cov_matrix)
        
        # The normal is the last row of Vt (corresponding to smallest eigenvalue)
        normal = Vt[-1]
        
        # Assign the normal to the current point
        normals[i] = normal
    
    return normals



def load_obj(file_obj_path):
    
    vertices = []
    edges = []

    try:
        with open(file_obj_path, 'r') as file:
            for line in file:
                line = line.strip().split()
                if not line:
                    continue

                if line[0] == 'v':
                    vertex = [float(coord) for coord in line[1:]]
                    vertices.append(vertex)
                    
                elif line[0] == 'l':
                    edge = [int(index) - 1 for index in line[1:]]
                    edges.append(edge)

        if not vertices:
            print(f"Warning: No vertices found in {file_obj_path}")
            return None, None

        if not edges:
            print(f"Warning: No edges found in {file_obj_path}")
            return None, None

        vertices = np.array(vertices)
        edges = np.array(edges)

        return vertices, edges

    except Exception as e:
        print(f"Error loading .obj file: {e}")
        return None, None

def compute_svd_features_obj(points):
    cov_matrix = np.cov(points, rowvar=False)
    _, singular_values, _ = np.linalg.svd(cov_matrix)
    return singular_values

def create_graph_data(vertices, edges):
    x = torch.tensor(vertices, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)  # Shape should be (2, num_edges)
    data = Data(x=x, edge_index=edge_index)
    return data

class EdgeDetectionGNN(nn.Module):
    def __init__(self):
        super(EdgeDetectionGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=3, out_channels=64)
        self.conv2 = pyg_nn.GCNConv(in_channels=64, out_channels=128)
        self.pool = pyg_nn.global_mean_pool  # Use mean pooling
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Binary classification: edge or not

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Pooling
        x = self.pool(x, batch)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x

def train_model(model, data_list, epochs=10, lr=0.01):
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # For binary classification
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for data in loader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data.x, data.edge_index, data.batch)
            
            # Assuming `data.y` is available and contains edge labels
            target = data.y
            if target is not None:
                loss = criterion(output.squeeze(), target.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                print("Warning: Target is not available or incompatible.")
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader)}')









##________________________________________________________


def compute_bbox_features(points_3d):

    pi2 = np.pi / 2

    # Get the convex hull for the points
    hull = ConvexHull(points_3d)
    hull_points = points_3d[hull.vertices]

    if hull_points.shape[1] == 3:
        # Compute PCA for oriented bounding box
        pca = PCA(n_components=3)
        pca.fit(hull_points)
        principal_axes = pca.components_
        centered_points = hull_points - pca.mean_
        rotated_points = np.dot(centered_points, principal_axes.T)

        # Find the min and max of the rotated points
        min_vals = np.min(rotated_points, axis=0)
        max_vals = np.max(rotated_points, axis=0)

        # Compute the bounding box vertices in the rotated space
        bbox_vertices = np.array([
            [min_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], max_vals[1], max_vals[2]],
            [min_vals[0], max_vals[1], max_vals[2]]
        ])

        # Convert bbox vertices back to original space
        bbox_vertices = np.dot(bbox_vertices, principal_axes) + pca.mean_

    else:
        raise ValueError("Points must be 3D")

    # Compute the center
    center_x = float(np.mean(points_3d[:, 0]))
    center_y = float(np.mean(points_3d[:, 1]))
    center_z = float(np.mean(points_3d[:, 2]))

    center = np.array([center_x, center_y, center_z])

    return center, bbox_vertices



def plot_bounding_box(bbox_vertices):

    edges = [
        [bbox_vertices[0], bbox_vertices[1]],
        [bbox_vertices[1], bbox_vertices[2]],
        [bbox_vertices[2], bbox_vertices[3]],
        [bbox_vertices[3], bbox_vertices[0]],
        [bbox_vertices[4], bbox_vertices[5]],
        [bbox_vertices[5], bbox_vertices[6]],
        [bbox_vertices[6], bbox_vertices[7]],
        [bbox_vertices[7], bbox_vertices[4]],
        [bbox_vertices[0], bbox_vertices[4]],
        [bbox_vertices[1], bbox_vertices[5]],
        [bbox_vertices[2], bbox_vertices[6]],
        [bbox_vertices[3], bbox_vertices[7]]
    ]
    return edges


# _________________________
def compute_hull_polygons(points_3d):
    # Ensure points are numpy array
    points_3d = np.array(float(points_3d))
    # Compute the convex hull
    hull = ConvexHull(points_3d)
    hull_polygons = []

    # Extract vertices of each face of the convex hull
    for simplex in hull.simplices:
        # Ensure the polygon is closed by repeating the first vertex
        simplex_closed = np.append(simplex, simplex[0])
        polygon = points_3d[simplex_closed]
        hull_polygons.append(polygon)
    
    return hull_polygons, hull



def filter_edges(hull, points_3d, frequency_threshold=1):

    edge_count = defaultdict(int)
    
    # Count frequency of each edge
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % len(simplex)]]))
            edge_count[edge] += 1
    
    # Filter edges by frequency threshold
    filtered_edges = [edge for edge, count in edge_count.items() if count >= frequency_threshold]
    
    # Construct the minimal wireframe model
    wireframe_edges = []
    for edge in filtered_edges:
        wireframe_edges.append(points_3d[list(edge)])
    
    return wireframe_edges







# base_folder = Path(os.getcwd())
# folder = base_folder.parents[0] / 'data'

# # Iterate over .las files in the directory
# for file_path in folder.glob('*.las'):
    
#     # Extract points and process them
#     points_3d = extract_points(file_path)  # Replace with actual function to extract points
#     points_3d = np.round(points_3d, 3)
    
#     # Compute convex hull polygons
#     hull_polygons, hull = compute_hull_polygons(points_3d)

#     # Filter the edges to get the wireframe
#     wireframe_edges = filter_edges(hull, points_3d, frequency_threshold=2)  # Adjust the threshold as needed

#     # # Plot the results
#     # plot_wireframe(points_3d, wireframe_edges, hull_polygons, file_path.name)










        





def extract_eigenvector_and_neighbors(normalized_points, Vt):
 
    n = Vt.T[:, 2]  # Eigenvector corresponding to smallest singular value
    distances = np.linalg.norm(normalized_points, axis=1)
    k = normalized_points.shape[0] // 2

    half_k = int(k // 2)

    closest_indices = np.argsort(distances)[:half_k]
    closest_points = normalized_points[closest_indices]

    return n, closest_points


def partition_and_compute_covariance(closest_points, n):

    dot_products = np.dot(closest_points, n)
    N_i_upper = closest_points[dot_products >= 0]
    if N_i_upper == 0 or N_i_upper <=0:
        return 1
    
    N_i_lower = closest_points[dot_products < 0]
    if N_i_lower == 0 or N_i_lower >= 0:
        return 1

    print (f'N_i_upper {N_i_upper} Ni_lower {N_i_lower}' )

    if N_i_upper.size == 0 or N_i_lower.size == 0:
        print("Empty partition detected.")
        return None, None, None, None, None, None

    K_i_upper = np.cov(N_i_upper, rowvar=False)
    K_i_lower = np.cov(N_i_lower, rowvar=False)

    _, Sigma_i_upper, _ = np.linalg.svd(K_i_upper)
    _, Sigma_i_lower, _ = np.linalg.svd(K_i_lower)

    N_i_upper_center = np.mean(N_i_upper, axis=0)
    N_i_lower_center = np.mean(N_i_lower, axis=0)

    return N_i_upper, N_i_lower, Sigma_i_upper, Sigma_i_lower, N_i_upper_center, N_i_lower_center



def calculate_distances(points, mean, n):

    s_perpendicular = np.dot(points - mean, n)
    s_tangential = np.linalg.norm((points - mean) - s_perpendicular[:, np.newaxis] * n, axis=1)

    return s_perpendicular, s_tangential








def svd_analysis(points):

    mean = np.mean (points)
    centered_points = points - mean
    cov_matrix = compute_covariance_matrix(centered_points)

    U, S, Vt = perform_svd(cov_matrix)
    
    if U is None:
        return None

    normalized_points = normalize_neighborhood(centered_points, S)
    normalized_points = normalized_points + mean  # Restore original scale
    
    n, closest_points = extract_eigenvector_and_neighbors(normalized_points, Vt, mean)
    
    N_i_upper, N_i_lower, Sigma_i_upper, Sigma_i_lower, N_i_upper_center, N_i_lower_center = partition_and_compute_covariance(closest_points, n)
    
    s_perpendicular, s_tangential = calculate_distances(points, mean, n)
    
    features_svd_analysis = {
        "cov_matrix": cov_matrix,
        "U": U,
        "S": S,
        "Vt": Vt,
        "singular_values": S,
        "mean": mean,
        "normalized_points": normalized_points,
        "n": n,
        "N_i_upper": N_i_upper,
        "N_i_lower": N_i_lower,
        "N_i_upper_center": N_i_upper_center,
        "N_i_lower_center": N_i_lower_center,
        "Sigma_i_upper": Sigma_i_upper,
        "Sigma_i_lower": Sigma_i_lower,
        "distances_between_centers": [np.dot(N_i_upper_center - N_i_lower_center, n),
                                      np.linalg.norm(N_i_upper_center - N_i_lower_center - np.dot(N_i_upper_center - N_i_lower_center, n) * n)],
        "point_distances": [[s_perpendicular[i], s_tangential[i]] for i in range(len(points))]
    }

    return features_svd_analysis



def print_svg_features(results):
    print("Results Summary:\n")
    
    # Print Means
    print("Means (Center of Mass) for Each File:")
    for file_name, mean in results["means"].items():
        print(f"File: {file_name}, Mean: {mean}")
    print()

    # Print Covariance Matrices
    print("Covariance Matrices for Each File:")
    for file_name, cov_matrix in results["cov_matrices"].items():
        print(f"File: {file_name}, Covariance Matrix:\n{cov_matrix}")
    print()

    # Print Singular Values
    print("Singular Values for Each File:")
    for file_name, singular_values in results["S"].items():
        print(f"File: {file_name}, Singular Values: {singular_values}")
    print()

    # Print Singular Vectors
    print("Singular Vectors (U) for Each File:")
    for file_name, U in results["U"].items():
        print(f"File: {file_name}, Singular Vectors (U):\n{U}")
    print()

    # Print Eigenvector
    print("Eigenvector (n) for Each File:")
    for file_name, n in results["n"].items():
        print(f"File: {file_name}, Eigenvector: {n}")
    print()

    # Print Partition and Covariance Data
    print("Partition and Covariance Data:")
    for file_name in results["N_i_upper"].keys():
        print(f"File: {file_name}")
        print(f"  N_i_upper: {results['N_i_upper'][file_name]}")
        print(f"  N_i_lower: {results['N_i_lower'][file_name]}")
        print(f"  Sigma_i_upper: {results['Sigma_i_upper'][file_name]}")
        print(f"  Sigma_i_lower: {results['Sigma_i_lower'][file_name]}")
        print(f"  N_i_upper_center: {results['N_i_upper_center'][file_name]}")
        print(f"  N_i_lower_center: {results['N_i_lower_center'][file_name]}")
        print()

    # Print Distances Between Centers
    print("Distances Between Centers:")
    for file_name, distances in results["distances_between_centers"].items():
        print(f"File: {file_name}, Distances: {distances}")
    print()

    # Print Distances from Points to the Plane
    print("Distances from Points to the Plane:")
    for file_name in results["s_perpendicular"].keys():
        print(f"File: {file_name}")
        print(f"  Perpendicular Distance: {results['s_perpendicular'][file_name]}")
        print(f"  Tangential Distance: {results['s_tangential'][file_name]}")
        print()














def center_data(points_3d):

        # Compute the mean of the points along each dimension (x, y, z)
        mean = np.mean(points_3d, axis=0)

        # Subtract the mean from each point to center the data
        centered_points = points_3d - mean

        return mean, centered_points


# for file_path in folder_path.glob('*.las') or folder_path.glob('*.xyz'):


# PLOT HULL POLYGONS
# def plot_hull_polygons(points_3d, hull_polygons, file_name):

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection = '3d')

#     # Plot the points
#     ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color = 'r', marker = 'o', s = 0.25, label = 'Point cloud')

#     # Plot the hull polygons
#     for polygon in hull_polygons:
#         poly = Poly3DCollection([polygon], alpha = 0.25, linewidths = 0.25, edgecolors = 'g')
#         ax.add_collection3d(poly)

#     # Set labels and title
#     ax.set_xlabel('x', fontsize = 8)
#     ax.set_ylabel('y', fontsize = 8)
#     ax.set_zlabel('z', fontsize = 8)
#     ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
#     plt.title(f'POINT CLOUD, CONVEX HULL AND BBOX FOR {file_name}', fontsize = 8)
#     ax.legend(fontsize=8, loc='best')

#     ax.tick_params(axis = 'both', which = 'major', labelsize = 8)  # Adjust `labelsize` as needed
#     ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)  # For minor ticks if needed

#     # Set axis format to remove scientific notation
#     ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
#     ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
#     ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))

#     plt.show()


# Extract Neighbors
# def extract_neighborhoods(points, k_values):
#     neighborhood_largest_scale = {}
#     nbrs = NearestNeighbors(n_neighbors=max(k_values) + 1).fit(points)
#     for k in k_values:
#         neighborhood_largest_scale[k] = []
#         for point in points:
#             distances, indices = nbrs.kneighbors([point], n_neighbors=k + 1)
#             neighborhood_points = points[indices[0][1:]]
#             neighborhood_largest_scale[k].append(neighborhood_points)
#     return neighborhood_largest_scale

# # Feature Preparation
# def prepare_features(x_hat_i_k, ci_k_perpendicular, ci_k_parallel):
#     ci_k = np.array([ci_k_perpendicular, ci_k_parallel])
#     return np.concatenate([x_hat_i_k, ci_k])

# def normalize_and_extract_features(neighborhood):
#     centroid = np.mean(neighborhood, axis=0)
#     normalized_points = neighborhood - centroid
#     return np.mean(normalized_points, axis=0)

# def extract_features(points_3d, neighborhood_largest_scale, neighborhood_current_scale):
#     N_i_k0 = np.mean(neighborhood_largest_scale, axis=0)
#     N_i_k = np.mean(neighborhood_current_scale, axis=0)
#     n_i_k0 = np.mean(neighborhood_largest_scale, axis=0)
#     N_bar = np.mean(neighborhood_current_scale, axis=0)
#     ci_k_perpendicular, ci_k_parallel = np.random.rand(), np.random.rand()
#     x_hat_i_k = normalize_and_extract_features(neighborhood_current_scale)
#     return prepare_features(x_hat_i_k, ci_k_perpendicular, ci_k_parallel)