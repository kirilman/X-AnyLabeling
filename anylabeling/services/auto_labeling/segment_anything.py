import logging
import os
import traceback

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from anylabeling.views.labeling.utils.shape import (
    neighbour_point,
    ellipse_parameters,
    Polygone,
    distance_point,
)
from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult
from .sam_onnx import SegmentAnythingONNX
from .__base__.clip import ChineseClipONNX

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from asbestutills.utils.geometry import add_points_to_polygone
from shapely import Polygon as ShapelyPolygone
import polygone_nms


class SegmentAnything(Model):
    """Segmentation model using SegmentAnything"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)
        self.input_size = self.config["input_size"]
        self.max_width = self.config["max_width"]
        self.max_height = self.config["max_height"]

        # Get encoder and decoder model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of Segment Anything.",
                )
            )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not os.path.isfile(
            decoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of Segment Anything.",
                )
            )

        # Load models
        self.model = SegmentAnythingONNX(
            encoder_model_abs_path, decoder_model_abs_path
        )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

        # CLIP models
        self.clip_net = None
        clip_txt_model_path = self.config.get("txt_model_path", "")
        clip_img_model_path = self.config.get("img_model_path", "")
        if clip_txt_model_path and clip_img_model_path:
            if self.config["model_type"] == "cn_clip":
                model_arch = self.config["model_arch"]
                self.clip_net = ChineseClipONNX(
                    clip_txt_model_path,
                    clip_img_model_path,
                    model_arch,
                    device=__preferred_device__,
                )
            self.classes = self.config.get("classes", [])

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks, image=None):
        """
        Post process masks
        """
        # Find contours
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])

                # Create shape
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = "AUTOLABEL_OBJECT"
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode in ["rectangle", "rotation"]:
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue

                # Get min/max
                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            # Create shape
            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.add_point(QtCore.QPointF(x_min, y_max))
            shape.shape_type = (
                "rectangle" if self.output_mode == "rectangle" else "rotation"
            )
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            if self.clip_net is not None and self.classes:
                img = image[y_min:y_max, x_min:x_max]
                out = self.clip_net(img, self.classes)
                shape.cache_label = self.classes[int(np.argmax(out))]
            shape.label = "AUTOLABEL_OBJECT"
            shape.selected = False
            shapes.append(shape)

        return shapes

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        shapes = []
        cv_image = qt_img_to_rgb_cv_img(image, filename)
        try:
            # Use cached image embedding if possible
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    filename,
                    image_embedding,
                )
            if self.stop_inference:
                return AutoLabelingResult([], replace=False)
            masks = self.model.predict_masks(image_embedding, self.marks)
            if len(masks.shape) == 4:
                masks = masks[0][0]
            else:
                masks = masks[0]
            shapes = self.post_process(masks, cv_image)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit()

    def preload_worker(self, files):
        """
        Preload next files, run inference and cache results
        """
        files = files[: self.preloaded_size]
        for filename in files:
            if self.image_embedding_cache.find(filename):
                continue
            image = self.load_image_from_filename(filename)
            if image is None:
                continue
            if self.stop_inference:
                return
            cv_image = qt_img_to_rgb_cv_img(image)
            image_embedding = self.model.encode(cv_image)
            self.image_embedding_cache.put(
                filename,
                image_embedding,
            )

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        if (
            self.pre_inference_thread is None
            or not self.pre_inference_thread.isRunning()
        ):
            self.pre_inference_thread = QThread()
            self.pre_inference_worker = GenericWorker(
                self.preload_worker, next_files
            )
            self.pre_inference_worker.finished.connect(
                self.pre_inference_thread.quit
            )
            self.pre_inference_worker.moveToThread(self.pre_inference_thread)
            self.pre_inference_thread.started.connect(
                self.pre_inference_worker.run
            )
            self.pre_inference_thread.start()


class SegmentAnythingAll(Model):
    def __init__(self, config_path, on_message) -> None:
        super().__init__(config_path, on_message)
        # self.input_size = self.config["input_size"]
        self.input_size = (684, 1024)
        self.max_width = self.config["max_width"]
        self.max_height = self.config["max_height"]
        sam = sam_model_registry[self.config["model_type"]](
            checkpoint=self.config["model_path"]
        )
        sam.to("cuda")
        self.model = SamAutomaticMaskGenerator(
            model=sam,
            pred_iou_thresh=0.5,
            points_per_side=32,
            # pred_iou_thresh=0.9,
            stability_score_thresh=0.92,
            # crop_n_layers=1,
            crop_overlap_ratio=0.9,
            # crop_n_points_downscale_factor=2,
            # min_mask_region_area=20000,  # Requires open-cv to run post-processing
        )
        print("create model")

    def to4image(self, image):
        h, w, c = image.shape
        dy = int(h / 2)
        dx = int(w / 2)
        parts = {
            "0": {"image": image[:dy, :dx], "dx": 0, "dy": 0},
            "1": {"image": image[:dy, dx:], "dx": dx, "dy": 0},
            "2": {"image": image[dy:, :dx], "dx": 0, "dy": dy},
            "3": {"image": image[dy:, dx:], "dx": dx, "dy": dy},
        }
        return parts

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        cv_image = qt_img_to_rgb_cv_img(image, filename)

        original_size = cv_image.shape[:2]
        # Calculate a transformation matrix to convert to self.input_size
        # scale_x = self.input_size[1] / cv_image.shape[1]
        # scale_y = self.input_size[0] / cv_image.shape[0]
        # scale = min(scale_x, scale_y)

        shapes = []
        parhs = self.to4image(cv_image)
        for name, path in parhs.items():
            cur_image = path["image"]
            print(cur_image.shape)
            scale_x = self.input_size[0] / cur_image.shape[0]
            scale_y = self.input_size[1] / cur_image.shape[1]
            scale = min(scale_x, scale_y)

            transform_matrix = np.array(
                [
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1],
                ]
            )

            cur_image = cv2.warpAffine(
                cur_image,
                transform_matrix[:2],
                (self.input_size[1], self.input_size[0]),
                flags=cv2.INTER_LINEAR,
            )
            masks = self.model.generate(cur_image)
            cv2.imwrite("test.jpeg", cur_image)
            shape = self.post_process(masks, scale, path["dx"], path["dy"])
            filtered = self.filter_shapes(shape)
            shapes = shapes + filtered

        print("Polygons", len(shapes))
        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.stop_inference = True
        del self.model

    def filter_shapes(self, shapes):
        rs = np.array([s.maxsize for s in shapes])
        q_min = np.quantile(rs, 0.06)
        q_max = np.quantile(rs, 0.96)
        filtered = [
            s
            for s in shapes
            if q_min < s.maxsize < q_max and len(s.points) > 5
        ]

        if len(shapes) > 100:
            polygones = []
            for shape in shapes:
                xx = [p.x() for p in shape.points]
                yy = [p.y() for p in shape.points]
                polygones.append(Polygone(xx, yy))

            t = np.linspace(0, 2 * np.pi, 70)
            diff_vars = []
            for p in polygones:
                try:
                    if len(p.x) < 10:
                        continue
                    xc, yc, a, b, theta = ellipse_parameters(p.x, p.y)
                    x_ell = xc + a * np.cos(t)
                    y_ell = yc + b * np.sin(t)
                    dists = []
                    for xn, yn in zip(x_ell, y_ell):
                        neighbour = neighbour_point((xn, yn), p.x, p.y)
                        dists.append(distance_point((xn, yn), neighbour))
                    diff_vars.append(np.var(dists))
                except:
                    diff_vars.append(0)

            thresh = np.quantile(diff_vars, 0.92)

            max_points = max([2 * len(shape.points) for shape in shapes])
            ans_polygones = []
            for p in polygones:
                if len(p.x) > 6:
                    xx = p.x
                    yy = p.y
                    xx, yy = add_points_to_polygone(
                        xx, yy, max_points - len(xx)
                    )
                    new_p = Polygone(xx, yy)
                    ans_polygones.append(new_p)

            indexs = []
            for k, pol_1 in enumerate(ans_polygones):
                p1 = ShapelyPolygone(np.array([pol_1.x, pol_1.y]).T)
                try:
                    p1.intersection(p1)
                    indexs += [k]
                except:
                    pass

            polys = []
            for k, p in enumerate(ans_polygones):
                if k in indexs:
                    ans = np.zeros(len(p.x) * 2 + 2)
                    xx = p.x
                    yy = p.y
                    if (max(xx) - min(xx)) * (max(yy) - min(yy)) > 200:
                        for i, (x, y) in enumerate(zip(p.x, p.y)):
                            ans[2 * i] = x
                            ans[2 * i + 1] = y
                        ans[-1] = 0.99
                        ans[-2] = 1.0
                        polys.append(ans)

            polys = np.array(polys)
            indexs = polygone_nms.nms(
                polys,
                thresh=0.8,
            )

            filtered = [
                s for i, s in enumerate(shapes) if diff_vars[i] < thresh
            ]

        return filtered

    def post_process(self, masks, scale, dx, dy):
        areas = np.array([x["area"] for x in masks])
        area_tresh = np.quantile(areas, 0.025)
        print("порог ", area_tresh)
        masks = [x for x in masks if x["area"] > area_tresh]
        shapes = []
        for mask in masks:
            mask = mask["segmentation"]
            image_area = mask.shape[0] * mask.shape[1]
            mask[mask > 0.0] = 255
            mask[mask <= 0.0] = 0
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            approx_contours = []
            # Approximate contour
            epsilon = 0.0015 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            # Remove too big contours ( >90% of image size)
            if cv2.contourArea(approx) > image_area * 0.7:
                print(cv2.contourArea(approx), image_area)
                continue

            points = approx.reshape(-1, 2) / scale
            points[:, 0] += dx
            points[:, 1] += dy
            # Create shape
            shape = Shape(flags={})
            for point in points:
                point[0] = int(point[0])
                point[1] = int(point[1])
                shape.add_point(QtCore.QPointF(point[0], point[1]))
            shape.shape_type = "polygon"
            shape.closed = True
            shape.fill_color = "#0FA90A"
            shape.line_color = "#a90a0a"
            shape.line_width = 8
            shape.label = "stone"
            shape.selected = False
            shapes.append(shape)
        return shapes

    def merge_parths(self):
        pass
