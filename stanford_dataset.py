# %%
import os
import re
import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from psd_tools import PSDImage

# %%
def generate_annotation_line(path, points_5pt, points, scale, center_w, center_h):
    points_5pt_str = ",".join([f"{x:.6f}" for point in points_5pt for x in point])
    points_str = ",".join([f"{x:.6f}" for point in points for x in point])

    # Format the scale, center_w, and center_h as floats with appropriate precision
    scale_str = f"{float(scale):.6f}"
    center_w_str = f"{float(center_w):.6f}"
    center_h_str = f"{float(center_h):.6f}"

    # Create the annotation line
    annotation_line = f"{path}\t{points_5pt_str}\t{points_str}\t{scale_str}\t{center_w_str}\t{center_h_str}"

    return annotation_line

# %%
class Data_Config:
    def __init__(self):
            self.data_definition = "stanford"
            
            self.train_tsv_file = 'stanford_dataset/annot/train.tsv'
            self.val_tsv_file = None
            self.raw_psd_dir = 'stanford_dataset/psd'
            self.train_pic_dir = 'stanford_dataset/image'
            self.val_pic_dir = None
            self.loader_type = 'alignment'
            self.batch_size = 16
            self.val_batch_size = 32
            self.train_num_workers = 1
            self.val_num_workers = 1
            self.width = 256
            self.height = 256
            self.channels = 3
            self.means = (127.5, 127.5, 127.5)
            self.scale = 0.00784313725490196
            self.classes_num = [28, 2, 28]
            self.crop_op = True
            self.aug_prob = 1.0
            self.label_num = 12
            self.edge_info = (
                (False, (9,10)), 
                (False, (11,12)),
            )
            self.flip_mapping = (
                [0, 1],[2, 3], 
                [5, 6],[7, 8],
                [9 ,10],[11,12],
                [14,15],[17,18],
                [21,22],[24,25],   
                )
            self.encoder_type = 'default'

            # Val
            self.norm_type = 'default'
            self.nme_left_index = 9
            self.nme_right_index = 10
data_config = Data_Config()

# %%
psd_files = [file for file in os.listdir(data_config.raw_psd_dir) if file.endswith('.psd')]
annotation_lines = []
# %%
# psd_file = psd_files[1]
for psd_file in psd_files:
    psd = PSDImage.open(os.path.join(data_config.raw_psd_dir, psd_file))
    image = psd[0].numpy()
    # %%
    def get_ctr(bbox):
        return (bbox[0] + bbox[2]) / 2 , (bbox[1] + bbox[3]) / 2

    # %%
    layers = []
    for layer in psd:
        if layer.kind == 'shape':
            layers.append(layer)
    layers = np.array(layers)

    irises = layers[:2]
    cardinal = layers[[2,7,8]]
    accessory = layers[[3,4,5,6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]
    dots = layers[2:]

    # %%
    def get_iris_points(iris):
        ctr = get_ctr(iris.bbox)
        d = iris.size[0]
        iris_dots = np.zeros((2,2))
        iris_dots[:,1] = ctr[1]
        iris_dots[0,0] = ctr[0] - d / 2
        iris_dots[1,0] = ctr[0] + d / 2 
        return iris_dots

    # %%
    points = np.zeros((28,2))
    iris_left_points = get_iris_points(irises[0])
    iris_right_points = get_iris_points(irises[1])
    points[0] = iris_left_points[0]
    points[1] = iris_right_points[1]
    points[2] = iris_left_points[1]
    points[3] = iris_right_points[0]
    points[4:] = np.array([get_ctr(layer.bbox) for layer in layers[2:]])

    points_5pt = np.zeros((5,2))
    points_5pt[0] = get_ctr(irises[0].bbox)
    points_5pt[1] = get_ctr(irises[1].bbox)
    points_5pt[2] = points[13]
    points_5pt[3] = points[24]
    points_5pt[4] = points[25]


    # %%
    predictor_path = '../shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    image = (image * 255).astype(np.uint8)
    dets = detector(image, 1)
    for detection in dets:
        face = sp(image,detection)
        shape = []
        for i in range(68):
            x = face.part(i).x
            y = face.part(i).y
            shape.append((x, y))
        shape = np.array(shape)
        # image_draw = draw_pts(image_draw, shape)
        x1, x2 = shape[:, 0].min(), shape[:, 0].max()
        y1, y2 = shape[:, 1].min(), shape[:, 1].max()
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2

        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        print(f"scale: {scale}, center_w: {center_w}, center_h: {center_h}")

    # %%
    plt.figure(figsize=(10,10),dpi=200)
    h,w,c = image.shape
    plt.imshow(image)
    # plt.scatter(points[:,0], points[:,1], c='r', s=3)
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, point in enumerate(points):
        plt.text(point[0], point[1], str(i), fontsize=5,c='r')
    plt.scatter(points_5pt[:,0], points_5pt[:,1], c='b', s=50,alpha=0.5, marker='^')
    plt.scatter(center_w, center_h, c='g', s=50, alpha=0.5, marker='x')
    # Plot the bounding box from dets
    for detection in dets:
        x, y, w, h = detection.left(), detection.top(), detection.width(), detection.height()
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()
    # plt.scatter(cardinal[:,0], cardinal[:,1], c='b', s=1)
    # plt.scatter(accessory[:,0], accessory[:,1], c='g', s=1)


    # %%
    mrn,dsc = re.match(r'MRN(\d+)_DSC_?(\d+)', psd_file).groups()
    image_name = f"MRN{mrn}_DSC{dsc}.jpg"
    image_path = os.path.join(data_config.train_pic_dir, image_name)
    annotation_line = generate_annotation_line(image_name, points_5pt, points, scale, center_w, center_h)
    # cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    annotation_lines.append(annotation_line)
    # %%
    # Plot the Cardinal Line
    # x0, y0 = cardinal[0]
    # x1, y1 = cardinal[1]
    # x2, y2 = cardinal[2]
    # k1 = (y2 - y1) / (x2 - x1)
    # b1 = y1 - k1 * x1
    # k2 = -1 / k1
    # b2 = y0 - k2 * x0
    # x_intersect = (b2 - b1) / (k1 - k2)
    # y_intersect = k1 * x_intersect + b1

    # y_range = np.linspace(0, h, 400)
    # x_range = (y_range - b2) / k2

    # plt.plot(x_range, y_range, 'r-', linewidth=0.3)
    # plt.plot([x1,x2], [y1,y2], 'r-', linewidth=0.3)
# %%
with open(data_config.train_tsv_file, 'w') as f:
    f.write("\n".join(annotation_lines))
# %%
# Dataloader