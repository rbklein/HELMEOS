import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

#points = np.random.rand(30,3)

data = np.load('snapshot_data.npy')[0,:,:]
print(data.shape)

r = 3

U,_,_ = np.linalg.svd(data, full_matrices=False)
U = U[:,:r]

sample = np.random.randn(10000,r)
sample = 1 / np.linalg.norm(sample, axis = 1)[:,None] * sample
points = sample

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

Usample = U @ sample.T
positive_columns = np.all(Usample > 0, axis = 0)
print(positive_columns)
indices = np.where(positive_columns)[0]
points = sample[positive_columns, :]

'''
Ur = U @ points.T

positive_columns = np.all(Ur > 0, axis=0)
indices = np.where(positive_columns)[0]
'''


hull = ConvexHull(points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
ax.scatter(points[hull.vertices,0], points[hull.vertices, 1], points[hull.vertices, 2], marker = '+')

num_points = len(hull.vertices)
_, dim = points.shape

rays = []

for i in range(num_points):
    a_eq = points[hull.vertices[i],:]

    ineq_vertices = np.delete(hull.vertices, i)

    A_ineq = points[ineq_vertices, :]

    b = -0.0001 * np.ones(num_points - 1)

    c = np.zeros(dim)

    #hello robin, this is you, you can use slack variables for strict inequalities nikolaj says
    res = linprog(c, A_ub = A_ineq, b_ub = b, A_eq = a_eq[None,:], b_eq = 0, bounds = (None, None))

    #print(res.x, A_ineq @ res.x, a_eq @ res.x)

    if res.status == 0:
        print('success')
        rays.append(hull.vertices[i])

    print(i)

plt.scatter(points[rays,0], points[rays,1], points[rays,2], marker='x')
#plt.scatter(0,0, marker = '*')
for i in rays:
    plt.plot([0,points[i,0]], [0,points[i,1]], [0,points[i,2]], color = 'black')

plt.show()







'''
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def generate_point_cloud():
    # Example point cloud in 3D
    points = np.random.rand(30, 3)  # 30 random points in 3D space
    return points

def find_convex_hull(points):
    hull = ConvexHull(points)
    return hull

def find_cone_rays(hull_points):
    num_points, num_dims = hull_points.shape
    rays = []
    
    for i in range(num_points):
        c = np.zeros(num_points)
        c[i] = -1  # Objective to maximize the i-th coordinate
        
        A = np.vstack([np.ones(num_points), np.eye(num_points)])
        b = np.zeros(num_points + 1)
        b[0] = 1  # The sum of coefficients should be 1 (convex combination)
        
        bounds = [(0, None) for _ in range(num_points)]
        
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
        
        if res.success:
            rays.append(res.x @ hull_points)
    
    return np.array(rays)

def plot_hull(points, hull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')
    
    # Plot the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Cycle back to the starting vertex
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')
    
    plt.show()

def main():
    points = generate_point_cloud()
    hull = find_convex_hull(points)
    hull_points = points[hull.vertices]
    
    plot_hull(points, hull)

    rays = find_cone_rays(hull_points)
    
    print("Rays of the cone:")
    print(rays)

if __name__ == "__main__":
    main()
'''

'''
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def generate_point_cloud():
    # Example point cloud in 3D
    points = np.random.rand(30, 3)  # 30 random points in 3D space
    return points

def find_convex_hull(points):
    hull = ConvexHull(points)
    return hull

def plot_point_cloud_and_hull(points, hull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')
    
    # Plot the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Cycle back to the starting vertex
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')
    
    plt.show()

# Main function
def main():
    points = generate_point_cloud()
    hull = find_convex_hull(points)
    plot_point_cloud_and_hull(points, hull)
    
    # Output the vertices of the convex hull
    print("Vertices of the convex hull:")
    print(points[hull.vertices])

if __name__ == "__main__":
    main()
'''


'''
from config_discretization import *

import matplotlib.pyplot as plt

X = jnp.load("snapshot_data.npy")
X = X[0,:,:]

plt.imshow(X)
plt.show()

print(X.shape)

r = 3

U,s,V = jnp.linalg.svd(X, full_matrices = False)
U = U[:,:r]

import numpy as np
U = np.array(U)

from scipy.optimize import linprog
from numpy.linalg import matrix_rank
import numpy as np

def intersects_positive_orthant(basis): #basis is a d x N matrix where the ith row is v_i
  (d, N) = basis.shape
  M = basis.T

  if matrix_rank(basis) == N: #if basis spans R^N
    return True

  A = -M #negative sign for >=
  b = np.zeros(N)
  for i in range(N):
    c = -M[i,:] #negative sign to maximize

    opt = linprog(c, A, b)
    
    if opt.status == 2: #infeasible
      return False
    elif opt.status == 3: #unbounded
      continue
    elif opt.fun <= 0: #optimal value is not positive
      return False
  
  return True
    

if __name__ == '__main__':
    print(intersects_positive_orthant(U)) 

'''
'''
def generate_random_points_on_sphere(num_points):
    # Generate random azimuthal angles (theta) uniformly between 0 and 2*pi
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    
    # Generate random polar angles (phi) using the inverse transform method
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))
    
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Stack the coordinates into a (num_points, 3) array
    points = np.vstack((x, y, z)).T
    return points

num_points = 5000
points = generate_random_points_on_sphere(num_points)

Ur = U @ points.T

positive_columns = np.all(Ur > 0, axis=0)
indices = np.where(positive_columns)[0]
print(indices)
indices_tuple = tuple(indices)

# Plot the points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates
points_b = np.delete(points, indices_tuple, axis=0)

x = points_b[:, 0]
y = points_b[:, 1]
z = points_b[:, 2]

x_g = points[indices_tuple, 0]
y_g = points[indices_tuple, 1]
z_g = points[indices_tuple, 2]

# Create a 3D scatter plot
ax.scatter(x, y, z, c='blue', marker='o')
ax.scatter(x_g, y_g, z_g, c='red', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud on a Sphere')

# Show the plot
plt.show()

'''


