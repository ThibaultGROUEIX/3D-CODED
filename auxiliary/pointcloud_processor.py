# import pymesh
import numpy as np
import torch

def center(points):
    #center pointcloud
    #points N_points, 3
    points= points.copy()
    centroid = np.mean(points[:,0:3], axis = 0, keepdims = True)
    points[:,0:3] = points[:,0:3] - centroid
    return points

def normalize_unitL2ball_pointcloud(points):
    #normalize  to unit ball pointcloud
    #points N_points, 3
    points[:,0:3] = points[:,0:3] / np.sqrt(np.max(np.sum(points[:,0:3]**2, 1)))
    return points

def normalize_by_channel(points):
    #normalize  to unit ball pointcloud
    #points N_points, 3
    points[:,0] = points[:,0] / np.max(points[:,0])
    points[:,1] = points[:,1] / np.max(points[:,1])
    points[:,2] = points[:,2] / np.max(points[:,2])
    return points

def flip_points(points, axis):
    #center pointcloud
    #points N_points, 3
    points = points.copy()
    points[:,axis] = -points[:,axis]
    return points

def get_3D_rot_matrix(axis, angle):
    if axis == 0:
        return np.array([[1,0,0],[0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    if axis == 1:
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [- np.sin(angle), 0,  np.cos(angle)]])
    if axis == 2:
        return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(angle), np.cos(angle), 0], [1,0,0]])


def rotate_points(points, axis, angle):
    #center pointcloud
    #points N_points, 3
    rot_matrix = get_3D_rot_matrix(axis, angle)
    if isinstance(points, torch.Tensor):
        rot_matrix = torch.from_numpy(rot_matrix).cuda().transpose(0,1).contiguous().float()
        points = torch.matmul(points, rot_matrix)
    elif isinstance(points, np.ndarray):
        points = points.copy()
        points = points.dot(np.transpose(rot_matrix))
    else:
        print("Pierre-Alain was right.")
    return points



def center_pointcloud_torch(points):
    # input : torch Tensor N_pts, D_dim
    # ouput : torch Tensor N_pts, D_dim
    # Center first 3 dimensions of input
    centroid = torch.mean(points[:,0:3], dim = 0, keepdim = True)
    points[:,0:3] = points[:,0:3] - centroid
    return points

def normalize_unitL2ball_pointcloud_torch(points):
    # input : torch Tensor N_pts, D_dim
    # ouput : torch Tensor N_pts, D_dim
    # normalize first 3 dimensions of input to unit ball
    points[:,0:3] = points[:,0:3] / np.sqrt(torch.max(torch.sum(points[:,0:3]**2, 1)))
    return points

def UnitBall(points):
    # center and normalize to unit ball point cloud.
    points = center_pointcloud_torch(points)
    points = normalize_unitL2ball_pointcloud_torch(points)
    return points

def BoundingBox(points):
    # center the bounding box and uniformly scale the bounding box to diameter 1.
    points, _, bounding_box = center_bounding_box(points)
    diameter = torch.max(bounding_box)
    points = points/diameter
    return points

def BoundingBox_2(points):
    # center the bounding box and scale the bounding box to have edges of size 1
    points, _, bounding_box = center_bounding_box(points)
    points = points/bounding_box
    return points

def identity(points):
    print("identity")
    return points

def anisotropic_scaling(points):
    # input : points : N_point, 3
    scale = torch.rand(1,3)/2.0 + 0.75 #uniform sampling 0.75, 1.25
    return scale * points # Element_wize multiplication with broadcasting

def uniform_rotation_axis_matrix(axis=0, range_rot=360):
    # input : Numpy Tensor N_pts, 3
    # ouput : Numpy Tensor N_pts, 3
    # ouput : rot matrix Numpy Tensor 3, 3
    # Uniform random rotation around axis
    scale_factor = 360.0 / range_rot
    theta = np.random.uniform(- np.pi/scale_factor, np.pi/scale_factor)
    rot_matrix = get_3D_rot_matrix(axis, theta)
    return torch.from_numpy(np.transpose(rot_matrix)).float()

def uniform_rotation_axis(points, axis=0, normals=False, range_rot=360):
    # input : Numpy Tensor N_pts, 3
    # ouput : Numpy Tensor N_pts, 3
    # ouput : rot matrix Numpy Tensor 3, 3
    # Uniform random rotation around axis
    rot_matrix = uniform_rotation_axis_matrix(axis, range_rot)

    if isinstance(points, torch.Tensor):
        points[:,:3] = torch.mm(points[:,:3], rot_matrix)
        if normals:
            points[:,3:6] = torch.mm(points[:,3:6], rot_matrix)
        return points, rot_matrix
    elif isinstance(points, np.ndarray):
        points = points.copy()
        points[:,:3] = points[:,:3].dot(rot_matrix.numpy())
        if normals:
            points[:,3:6] = points[:,3:6].dot(rot_matrix.numpy())
        return points, rot_matrix
    else:
        print("Pierre-Alain was right.")


def uniform_rotation_sphere(points, normals=False):
    # input : Tensor N_pts, 3
    # ouput : Tensor N_pts, 3
    # ouput : rot matrix Numpy Tensor 3, 3
    # Uniform random rotation on the sphere
    x = torch.Tensor(2)
    x.uniform_()
    p = torch.Tensor([[np.cos(np.pi  * 2 * x[0] )* np.sqrt(x[1]), (np.random.binomial(1, 0.5, 1)[0]*2 -1) * np.sqrt(1-x[1]), np.sin(np.pi  * 2 * x[0]) * np.sqrt(x[1])]])
    z = torch.Tensor([[0,1,0]])
    v = (p-z)/(p-z).norm()
    H = torch.eye(3) - 2*torch.matmul( v.transpose(1,0), v)
    rot_matrix = - H

    if isinstance(points, torch.Tensor):
        points[:,:3] = torch.mm(points[:,:3], rot_matrix)
        if normals:
            points[:,3:6] = torch.mm(points[:,3:6], rot_matrix)
        return points, rot_matrix

    elif isinstance(points, np.ndarray):
        points[:,:3] = points[:,:3].dot(rot_matrix.numpy())
        if normals:
            points[:,3:6] = points[:,3:6].dot(rot_matrix.numpy())
        return points, rot_matrix
    else:
        print("Pierre-Alain was right.")

def add_random_translation(points, scale = 0.03):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # Uniform random translation on first 3 dimensions
    a = torch.FloatTensor(3)
    points[:,0:3] = points[:,0:3] + (a.uniform_(-1,1) * scale).unsqueeze(0).expand(-1, 3)
    return points

def get_vertex_normalised_area(mesh):
    #input : pymesh mesh
    #output : Numpy array #vertex summing to 1
    num_vertices = mesh.vertices.shape[0]
    print("num_vertices", num_vertices)
    a = mesh.vertices[mesh.faces[:, 0]]
    b = mesh.vertices[mesh.faces[:, 1]]
    c = mesh.vertices[mesh.faces[:, 2]]
    cross = np.cross(a - b, a - c)
    area = np.sqrt(np.sum(cross ** 2, axis=1))
    prop = np.zeros((num_vertices))
    prop[mesh.faces[:, 0]] = prop[mesh.faces[:, 0]] + area
    prop[mesh.faces[:, 1]] = prop[mesh.faces[:, 1]] + area
    prop[mesh.faces[:, 2]] = prop[mesh.faces[:, 2]] + area
    return prop / np.sum(prop)

def center_bounding_box(points):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # Center bounding box of first 3 dimensions
    if isinstance(points, torch.Tensor):
        points = points.squeeze()
        transpo = False
        if points.size(0)==3:
            transpo = True
            points=points.transpose(1,0).contiguous()
        min_vals = torch.min(points, 0)[0]
        max_vals = torch.max(points, 0)[0]
        points = points - (min_vals + max_vals) / 2
        if transpo:
            points=points.transpose(1,0).contiguous()
        return points, (min_vals + max_vals) / 2, (max_vals - min_vals)/2
    elif isinstance(points, np.ndarray):
        min_vals = np.min(points, 0)
        max_vals = np.max(points, 0)
        points = points - (min_vals + max_vals) / 2
        return points, (min_vals + max_vals) / 2, (max_vals - min_vals)/2
    else:
        print(type(points))
        print("Pierre-Alain was right.")


class RotateCenterPointCloud(object):
    def __init__(self, points):
        self.points = points.clone()

    def rotate_theta(self, theta):
        rot_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0, np.cos(theta)]])
        self.rot_matrix = torch.from_numpy(rot_matrix).float().cuda()

        inv_rot_matrix = np.array(
            [[np.cos(-theta), 0, np.sin(-theta)], [0, 1, 0], [- np.sin(-theta), 0, np.cos(-theta)]])
        inv_rot_matrix = torch.from_numpy(inv_rot_matrix).float().cuda()
        self.inv_rot_matrix = inv_rot_matrix

    def rotate_phi(self, phi):
        self.rot_matrix = torch.matmul(torch.from_numpy(np.array(
            [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1], ])).float().cuda(),
                                  self.rot_matrix)
        self.inv_rot_matrix = torch.matmul(self.inv_rot_matrix, torch.from_numpy(np.array(
            [[np.cos(-phi), np.sin(-phi), 0], [-np.sin(-phi), np.cos(-phi), 0], [0, 0, 1], ])).float().cuda(),
                                  )

    def apply_rotation(self, points):
        self.rotated_points = torch.matmul(self.rot_matrix, points)

    def center(self, points):
        points, mean_bb, size_bb = center_bounding_box(points)
        # point = points - mean_bb Already done in center_bounding_box
        self.mean_bb = mean_bb
        self.centered_points = points.unsqueeze(0)

    def rotate_center(self, phi, theta):
        self.rotate_theta(theta)
        self.rotate_phi(phi)
        self.apply_rotation(self.points.clone())
        self.center(self.rotated_points)


    def back(self, points):
        points = points + self.mean_bb
        return torch.matmul(points, self.inv_rot_matrix.transpose(1, 0))




def translate_positive(points,  translation_vector=False):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # translate pointcloud to positive quadrant
    if isinstance(points, torch.Tensor):
        min_vals = torch.min(points, 0)[0]
        max_vals = torch.max(points, 0)[0]
        points = points - min_vals
    elif isinstance(points, np.ndarray):
        min_vals = np.min(points, 0)
        max_vals = np.max(points, 0)
        points = points - min_vals
    else:
        print("Pierre-Alain was right.")

    if translation_vector:
        return points, min_vals
    else:
        return points

#Done for eurographics 19
def Barycentric(p, a, b, c):
    # input : p, a, b, c are numpy arrays of size 3
    # output barycentric coordinates point p in triangle (a,b,c)

    # output barycentric coordinates
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))

    denom = float(d00 * d11 - d01 * d01)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return (u, v, w)

#Done for eurographics 19
def Barycentric_batched(p, a, b, c):
    # input : p, a, b, c are numpy arrays of size N_points x 3
    # output barycentric coordinates point p in triangle (a,b,c)
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.sum( np.multiply(v0, v0), 1)
    d01 = np.sum( np.multiply(v0, v1), 1)
    d11 = np.sum( np.multiply(v1, v1), 1)
    d20 = np.sum( np.multiply(v2, v0), 1)
    d21 = np.sum( np.multiply(v2, v1), 1)

    denom = np.multiply(d00 , d11) - np.multiply(d01 , d01)

    v = (np.multiply(d11, d20) - np.multiply(d01, d21)) / denom
    w = (np.multiply(d00, d21) - np.multiply(d01, d20)) / denom
    u =  - v - w + 1.0

    return (u, v, w)


class Jet_colormap():
    # TODO : not finished !
    def __init__(self):
        self.r, self.g, self.b = np.zeros(256), np.zeros(256), np.zeros(256)
        for i in range(256):
            n = 4. * i / 256
            self.g[i] = 255 * np.min((np.max((np.min( (n - 0.5 , - n + 3.5 )), 0)), 1));
            self.b[i] = 255 * np.min((np.max((np.min( (n + 0.5 , - n + 2.5 )), 0)), 1));

    def __call__(self, points):

        # upgrade this line
        colors = np.floor(normalize_unitL2ball_pointcloud(translate_positive(points)) * 255).astype(int)

        colors[:,0] = self.r[colors[:,0]]
        colors[:,1] = self.g[colors[:,1]]
        colors[:,2] = self.b[colors[:,2]]
        print(colors)
        return colors

    def colorize_mesh(self, path):
        mesh = pymesh.load_mesh(path)
        vertices = mesh.vertices
        colors = self.__call__(vertices)
        print(colors[0])
        mesh.add_attribute("vertex_red")
        mesh.add_attribute("vertex_green")
        mesh.add_attribute("vertex_blue")
        mesh.set_attribute("vertex_red" , colors[:,0])
        mesh.set_attribute("vertex_green", colors[:,1])
        mesh.set_attribute("vertex_blue", colors[:,2])
        pymesh.save_mesh(path[:-4] + "color.ply", mesh, "vertex_red", "vertex_green", "vertex_blue", ascii=True )
        return mesh

class Custom_colormap():
    def __init__(self):
        self.r, self.g, self.b = np.arange(256), np.arange(256), np.arange(256)

    def __call__(self, points):
        # upgrade this line
        colors = np.floor(normalize_by_channel(translate_positive(points)**2) * 255).astype(int)
        colors[:,0] = self.r[colors[:,0]]
        colors[:,1] = self.g[colors[:,1]]
        colors[:,2] = self.b[colors[:,2]]
        return colors

    def colorize_mesh(self, path):
        mesh = pymesh.load_mesh(path)
        vertices = mesh.vertices
        colors = self.__call__(vertices)
        mesh.add_attribute("vertex_red")
        mesh.add_attribute("vertex_green")
        mesh.add_attribute("vertex_blue")
        mesh.set_attribute("vertex_red" , colors[:,0])
        mesh.set_attribute("vertex_green", colors[:,1])
        mesh.set_attribute("vertex_blue", colors[:,2])
        pymesh.save_mesh(path[:-4] + "color.ply", mesh, "vertex_red", "vertex_green", "vertex_blue", ascii=True )
        return mesh

    def save_mesh_color(self, path, mesh, colors=None):
        if colors is None:
            pymesh.save_mesh(path, mesh,  ascii=True )
        else:
            mesh.add_attribute("vertex_red")
            mesh.add_attribute("vertex_green")
            mesh.add_attribute("vertex_blue")
            mesh.set_attribute("vertex_red" , colors[:,0])
            mesh.set_attribute("vertex_green", colors[:,1])
            mesh.set_attribute("vertex_blue", colors[:,2])
            pymesh.save_mesh(path, mesh, "vertex_red", "vertex_green", "vertex_blue", ascii=True )

if __name__ == '__main__':
    point = uniform_rotation_sphere(np.random.uniform(0,5,(10,3)) +1)
    point = torch.zeros((10,3)) +1
    point.uniform_()
    print(point)
    point = BoundingBox(point)
    print(point)
    point = BoundingBox_2(point)
    print(point)
    # a = Custom_colormap()
    # a.colorize_mesh("example_0.ply")