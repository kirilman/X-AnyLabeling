import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw
from skimage.measure import EllipseModel

from ..logger import logger


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "rotation":
        assert len(xy) == 4, "Shape of shape_type=rotation must have 4 points"
        draw.polygon(xy=xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def masks_to_bboxes(masks):
    if masks.ndim != 3:
        raise ValueError(f"masks.ndim must be 3, but it is {masks.ndim}")
    if masks.dtype != bool:
        raise ValueError(
            f"masks.dtype must be bool type, but it is {masks.dtype}"
        )
    bboxes = []
    for mask in masks:
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes


def rectangle_from_diagonal(diagonal_vertices):
    """
    Generate rectangle vertices from diagonal vertices.

    Parameters:
    - diagonal_vertices (list of lists):
        List containing two points representing the diagonal vertices.

    Returns:
    - list of lists:
        List containing four points representing the rectangle's four corners.
        [tl -> tr -> br -> bl]
    """
    x1, y1 = diagonal_vertices[0]
    x2, y2 = diagonal_vertices[1]

    # Creating the four-point representation
    rectangle_vertices = [
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2],  # Bottom-left
    ]

    return rectangle_vertices


def max_distance(x_coords, y_coords):
    r = (x_coords[:, np.newaxis] - x_coords) ** 2 + (
        y_coords[:, np.newaxis] - y_coords
    ) ** 2
    return np.sqrt(r.max())


def distance_point(p1, p2):
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return np.sqrt(x0**2 + y0**2)


def neighbour_point(point, x_coords, y_coords):
    d_min = 10**9
    neighbour = (0, 0)
    for x, y in zip(x_coords, y_coords):
        d = distance_point(point, (x, y))
        if d < d_min:
            d_min = d
            neighbour = (x, y)
    return neighbour

class Polygone:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
    @property    
    def center_x(self):
        return np.mean(self.x)
    @property    
    def center_y(self):
        return np.mean(self.y)
    @property    
    def distance(self):
        return np.sqrt(self.center_x**2 + self.center_y**2)
    def __lt__(self, other):
        return self.distance < other.distance
    

def ellipse_parameters(x, y):
    a_points = np.array([x, y]).T
    ell = EllipseModel()
    ell.estimate(a_points)
    return ell.params

def maxdistance_beetwen_points(xx, yy):
    p1, p2 = (xx[0], yy[0]), (xx[1], yy[1])
    max_ro = -(10**6)
    for k, (x1, y1, x2, y2) in enumerate(zip(xx[:-1], yy[:-1], xx[1:], yy[1:])):
        d = distance_point((x1, y1), (x2, y2))

        if d > max_ro:
            max_ro = d
            p1 = (x1, y1)
            p2 = (x2, y2)
            index = k
    return max_ro, p1, p2, index


def add_points_to_polygone(xx, yy, N):
    start_len = len(xx)
    for i in range(0, N):
        ro, p1, p2, index = maxdistance_beetwen_points(xx, yy)
        new_x = p1[0] + (p2[0] - p1[0]) / 2
        new_y = p1[1] + (p2[1] - p1[1]) / 2
        xx = np.hstack((xx[: index + 1], [new_x], xx[index + 1 :]))  #!
        yy = np.hstack((yy[: index + 1], [new_y], yy[index + 1 :]))  #!
    return xx, yy
