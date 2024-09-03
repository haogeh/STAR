# %%
from psd_tools import PSDImage
import os
import numpy as np
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt


# %%
class Data_Config:
    def __init__(self):
            self.data_definition = "300W"
            self.train_tsv_file = 'pub_annot/300W/train.tsv'
            self.val_tsv_file = 'pub_annot/300W/test.tsv'
            self.train_pic_dir = 'stanford_dataset'
            self.val_pic_dir = 'pub_dataset/300W'
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
            self.classes_num = [68, 9, 68]
            self.crop_op = True
            self.aug_prob = 1.0
            self.label_num = 12
            self.edge_info = (
                (False, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),  # FaceContour
                (False, (17, 18, 19, 20, 21)),  # RightEyebrow
                (False, (22, 23, 24, 25, 26)),  # LeftEyebrow
                (False, (27, 28, 29, 30)),  # NoseLine
                (False, (31, 32, 33, 34, 35)),  # Nose
                (True, (36, 37, 38, 39, 40, 41)),  # RightEye
                (True, (42, 43, 44, 45, 46, 47)),  # LeftEye
                (True, (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),  # OuterLip
                (True, (60, 61, 62, 63, 64, 65, 66, 67)),  # InnerLip
            )
            self.flip_mapping = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [31, 35], [32, 34], [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46], [48, 54], [49, 53], [50, 52], [61, 63], [60, 64], [67, 65], [58, 56], [59, 55])
            self.encoder_type = 'default'

            # Val
            self.norm_type = 'default'
            self.nme_left_index = 36
            self.nme_right_index = 45
data_config = Data_Config()

# %%
psd_files = [file for file in os.listdir(data_config.train_pic_dir) if file.endswith('.psd')]

# %%
psd_file = psd_files[0]
psd = PSDImage.open(os.path.join(data_config.train_pic_dir, psd_file))

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
ctrs = []
for dot in layers:
    ctr = get_ctr(dot.bbox)
    ctrs.append(ctr)
ctrs = np.array(ctrs)

# %%
iris = irises[0]
ctr = ctrs[0]
d = iris.size[0]
iris_dots = np.zeros((2,2))
iris_dots[:,1] = ctr[1]
iris_dots[0,0] = ctr[0] - d / 2
iris_dots[1,0] = ctr[0] + d / 2 

plt.figure(figsize=(10,10),dpi=200)
h,w,c = psd[0].numpy().shape
plt.imshow(psd[0].numpy())
plt.scatter(ctr[0], ctr[1], c='r', s=3)
plt.scatter(iris_dots[:,0], iris_dots[:,1], c='g', s=3)

# %%
plt.figure(figsize=(10,10),dpi=200)
h,w,c = psd[0].numpy().shape
plt.imshow(psd[0].numpy())
plt.scatter(ctrs[:,0], ctrs[:,1], c='r', s=3)

# plt.scatter(cardinal[:,0], cardinal[:,1], c='b', s=1)
# plt.scatter(accessory[:,0], accessory[:,1], c='g', s=1)

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

