import mayavi.mlab as mlab
import numpy as np
import torch
import ipdb

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='sphere',
                          colormap='spectral', scale_factor=0.07, figure=fig)  # 'spectral', 'bone', 'copper',
        # 'gnuplot'
        mlab.colorbar(object=G, title="score\n", orientation='vertical')
        # mlab.colorbar(object=G, title="score\n", orientation='horizontal')
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere',
                          colormap='gnuplot', scale_factor=0.04, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.05)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig

def draw_scenes_gu(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    # if not isinstance(points, np.ndarray):
    #     points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()
    
    fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), engine=None, size=(600, 600))
    for i in range(len(points)):
        visualize_pts(points[i], fig=fig)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    # return fig
    mlab.show(1)
    input()


def draw_scenes_tracking(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()

    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points, show_intensity=False)
    if gt_boxes is not None:
        corners3d = np.expand_dims(gt_boxes.corners().transpose(), 0)
        fig = draw_corners3d(corners3d, fig=fig, color=(1, 0, 0), max_num=100)

    if ref_boxes is not None:
        ref_corners3d = np.expand_dims(ref_boxes.corners().transpose(), 0)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            cur_color = tuple(box_colormap[1 % len(box_colormap)])
            fig = draw_corners3d(ref_corners3d, fig=fig, color=cur_color, cls=ref_scores, max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def mayavi_show(xyz, box=None, score=None):
    pred_box_xyz = xyz.view(1, -1, 3).squeeze(0).detach().cpu().numpy()
    pred_box_xyz_4d = np.ones((pred_box_xyz.shape[0], 4))
    pred_box_xyz_4d[:, :3] = pred_box_xyz

    if score is not None:
        pred_score = score.view(1, -1).squeeze(0).detach().cpu().numpy()
        pred_score[pred_score == 0] = -1
        pred_box_xyz_4d[:, -1] = pred_score

    final_pts = np.concatenate([pred_box_xyz_4d, np.array([[0, 0, 0, 2.0]])], axis=0)

    fig = visualize_pts(final_pts, show_intensity=True)

    if box is not None:
        corners3d = np.expand_dims(box.corners().transpose(), 0)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    mlab.show(1)
    input()


def mayavi_show_np(xyz: np.ndarray, box=None, score=None):
    try:
        pred_box_xyz = xyz.reshape([-1, 3])
    except:
        pred_box_xyz = xyz.reshape([-1, 4])[:, 1:]
    pred_box_xyz_4d = np.ones((pred_box_xyz.shape[0], 4))
    pred_box_xyz_4d[:, :3] = pred_box_xyz
    if score is not None:
        pred_score = score.reshape([-1])[:]
        pred_score[pred_score == 0] = -1
        pred_box_xyz_4d[:, -1] = pred_score
    else:
        pred_box_xyz_4d[:, -1] = 1.0
    final_pts = pred_box_xyz_4d
    fig = visualize_pts(final_pts, show_intensity=True)

    if box is not None:
        corners3d = np.expand_dims(box.corners().transpose(), 0)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    mlab.show(1)
    input()


def draw_box_by_one_corners(corners, fig, color=(0, 0, 1), line_width=4,
                            cls=None, cls_corner=6, tube_radius=None):
    b = corners

    if cls is not None:
        mlab.text3d(b[cls_corner, 0], b[cls_corner, 1], b[cls_corner, 2], '%s' % cls,
                    scale=(0.3, 0.3, 0.3), color=color, figure=fig)

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                    tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                    tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                    tube_radius=tube_radius,
                    line_width=line_width, figure=fig)


def draw_line(fig, pc1, pc2, color=(0, 0.6, 0), line_width=1.0, tube_radius=None):
    assert pc1.shape == pc2.shape
    for i in range(pc1.shape[0]):
        if isinstance(line_width, float):
            mlab.plot3d([pc1[i, 0], pc2[i, 0]], [pc1[i, 1], pc2[i, 1]], [pc1[i, 2], pc2[i, 2]], color=color,
                        tube_radius=tube_radius, line_width=line_width, figure=fig)
        elif isinstance(line_width, np.ndarray or list):
            mlab.plot3d([pc1[i, 0], pc2[i, 0]], [pc1[i, 1], pc2[i, 1]], [pc1[i, 2], pc2[i, 2]], color=color,
                        tube_radius=tube_radius, line_width=np.round(line_width[i]), figure=fig)

#open3d
import open3d as o3d

#vis 2 pcs
def vis_cloud(a,b):
	# a:n*3的矩阵
	# b:n*3的矩阵
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a.reshape(-1,3))
    pt1.paint_uniform_color([1,0,0])

    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(b.reshape(-1,3))
    pt2.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pt1,pt2],window_name='cloud[0] and corr',width=800,height=600)

def vis_cloud3pc(a,b,c):
	# a:n*3的矩阵
	# b:n*3的矩阵
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a.reshape(-1,3))
    pt1.paint_uniform_color([1,0,0])

    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(b.reshape(-1,3))
    pt2.paint_uniform_color([0,1,0])

    pt3=o3d.geometry.PointCloud()
    pt3.points=o3d.utility.Vector3dVector(c.reshape(-1,3))
    pt3.paint_uniform_color([0,0,1])

    o3d.visualization.draw_geometries([pt1,pt2,pt3],window_name='cloud[0] and corr',width=800,height=600)

from typing import Optional, Tuple, Union

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def merge_cylinder_segments(self):
         vertices_list = [np.asarray(mesh.vertices) for mesh in self.cylinder_segments]
         triangles_list = [np.asarray(mesh.triangles) for mesh in self.cylinder_segments]
         triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
         triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]
        
         vertices = np.vstack(vertices_list)
         triangles = np.vstack([triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])
        
         merged_mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vertices), 
                                                 o3d.open3d.utility.Vector3iVector(triangles))
         color = self.colors if self.colors.ndim == 1 else self.colors[0]
         merged_mesh.paint_uniform_color(color)
         self.cylinder_segments = [merged_mesh]

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

# COLORS = [[.8, .1, .1], [.1, .1, .8], [.1, .8, .1],[.6, .19608, .8], [1, 1, 0], [1, 0.87059,0.67843]] #r,b,g,p,sky-blue
COLORS = [[.1, .1, .8], [.1, .8, .1],[.6, .19608, .8], [1, 1, 0], [1, 0.87059,0.67843]] #r,b,g,p,sky-blue
COLORS1 = [[.1, .1, .8], [.1, .8, .1],[.6, .19608, .8], [1, 1, 0], [1, 0.87059,0.67843]] #r,b,g,p,sky-blue
def visualize_mesh(
    point_cloud: Optional[Union[np.array, Tuple]]=None, 
    bboxes: Optional[Tuple[np.array]]=None
):
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5],
                      [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]])
    visualization_group = []
    
    if point_cloud is not None:
        if not isinstance(point_cloud, tuple):
            point_cloud = (point_cloud, )
    
        for idx_p, pc in enumerate(map(np.array, point_cloud)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            if pc.shape[-1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])
            else:
                pcd.colors = o3d.utility.Vector3dVector(
                    np.ones_like(pc[:, :3]) * COLORS[idx_p]
                )
            if pc.shape[-1] >= 9:
                pcd.normals = o3d.utility.Vector3dVector(pc[:, 6:9])
            visualization_group.append(pcd)

    if bboxes is not None:
        if not isinstance(bboxes, tuple):
            bboxes = (bboxes,)
        for idx, boxgroup in enumerate(map(np.array, bboxes)):
            corners = boxgroup.reshape(-1, 3)
            # corners = boxgroup
            edges = lines[None, ...] + (np.ones_like(lines[None]).repeat(boxgroup.shape[0], axis=0) * np.arange(0, len(corners), boxgroup.shape[0])[:, None, None])
            edges = edges.reshape(-1, 2)
            # bounding box corners and bounding box edges
            # box_corner = o3d.geometry.PointCloud()
            # box_corner.points = o3d.utility.Vector3dVector(corners)
            # box_corner.colors = o3d.utility.Vector3dVector(
            #     np.ones_like(corners) * COLORS[idx]
            # )
            # box_edge = o3d.geometry.LineSet()
            # box_edge.lines = o3d.utility.Vector2iVector(edges)
            # box_edge.colors = o3d.utility.Vector3dVector(
            #     np.ones((len(edges), 3)) * COLORS[idx]
            # )            
            # box_edge.points = o3d.utility.Vector3dVector(corners)

            line_mesh1 = LineMesh(corners, edges, COLORS1[idx], radius=0.01)
            line_mesh1_geoms = line_mesh1.cylinder_segments
            # store #
            visualization_group.extend([*line_mesh1_geoms])

    o3d.visualization.draw_geometries(visualization_group)
    return None