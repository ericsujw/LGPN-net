import os
import glob

import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import scipy
from scipy.misc import imread
from scipy import signal
from skimage import util, filters
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize as skt_rs

from .util import create_mask
import cv2
import time
import logging
from .misc.panorama import draw_boundary_from_cor_id

FLOOR_ID = 2
WALL_ID = 1
CEILING_ID = 22


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, layout_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)
        self.layout_data = self.load_flist(layout_flist)

        self.input_size = config.INPUT_SIZE
        self._width = self.input_size[1]
        self._height = self.input_size[0]
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        
        max_mask_area: float = 1.0
        min_mask_area: float = 0.001  # good value, dont change
        self.rng = random.Random(self.config.SEED)
        self._classes4masking = {3, 4, 5, 6, 7, 10, 11, 12, 14, 15, 17, 19, 24, 25, 29, 30, 32, 33, 34, 36}  # 40
        self._min_mask_area, self._max_mask_area = min_mask_area, max_mask_area
        self._object_mask_only: bool = False
        self._dilate_convex_mask : bool = True
        self._random_mask_side_size_percentage = 0.3
        self._random_mask_side_deviation_percentage = 0.1

        # in test mode, there's a one-to-one relationship between mask and image masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        # except Exception as e:
        #     print("failed!!!: "+str(e))
        #     pass
        except:
            print('loading error: ' + self.data[index])
            
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        
        ### load empty room panorama
        if self.training:
        # load image
        # D:\Data\IndoorScene\structure3D\Structured3D\scene_00000\2D_rendering\485142\panorama\empty
            path = os.path.join('/CGVLAB3/datasets/Structured3D', self.data[index], 'empty/rgb_rawlight.png')
            empty = imread(path, mode='RGB')
        else:
            # path = os.path.join('../evaluation/raw', 'raw_' + str(format(index+1, '05d')) + '.png')
            path = self.data[index]
            empty = imread(path, mode='RGB')
        
        ### load full room panorama
        if self.training:
        # load image
        # D:\Data\IndoorScene\structure3D\Structured3D\scene_00000\2D_rendering\485142\panorama\empty
            path = os.path.join('/CGVLAB3/datasets/Structured3D', self.data[index], 'full/rgb_rawlight.png')
            full = imread(path, mode='RGB')
        else:
            # path = os.path.join('../evaluation/raw', 'raw_' + str(format(index+1, '05d')) + '.png')
            path = self.data[index]
            full = imread(path, mode='RGB')
            
        # gray to rgb
        #if len(img.shape) < 3:
        empty = gray2rgb(empty)
        full = gray2rgb(full)
            
        # resize/crop if needed
        if size != 0:
            empty = self.resize(empty, size[0], size[1])
            full = self.resize(full, size[0], size[1])
        
        ### load empty/full room semantic segmentation
        imgh, imgw = empty.shape[0:2]
        if self.training:
            empty_semantic_map = imread(os.path.join('/CGVLAB3/datasets/Structured3D', self.data[index], 'empty/semantic.png'), mode='P')
            full_semantic_map = imread(os.path.join('/CGVLAB3/datasets/Structured3D', self.data[index], 'full/semantic.png'), mode='P')
        else:
            empty_semantic_map = imread(self.data[index].replace('full_gt', 'empty_semantic'), mode='P')
            full_semantic_map = imread(self.data[index].replace('full_gt', 'full_semantic'), mode='P')
            
        empty_semantic_map = cv2.resize(empty_semantic_map.astype(np.uint8), (imgw, imgh), cv2.INTER_NEAREST)
        full_semantic_map = cv2.resize(full_semantic_map.astype(np.uint8), (imgw, imgh), cv2.INTER_NEAREST)
            
        foreground = 255 * ((full_semantic_map != CEILING_ID).astype(np.uint8) * (full_semantic_map != FLOOR_ID).astype(np.uint8) * (full_semantic_map != WALL_ID).astype(np.uint8))
        img = np.where(foreground.astype(np.bool)[...,None], full, empty)
        
        


        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index, empty_semantic_map, full_semantic_map)

        # load edge
        # edge = self.load_edge(img_gray, index, mask)

        # load layout
        layout = img_gray

        # load layout ont-hot
        # layout_ont_hot, plane_ont_hot = self.load_layout_instance(img, index)
        #
        # pad = 100
        # c, h, w = plane_ont_hot.size()
        # padding = torch.zeros(pad-c, h, w)
        # plane_ont_hot = torch.cat((plane_ont_hot, padding), 0)


        # return self.to_tensor(img), self.to_tensor(edge), self.to_tensor(mask), self.to_tensor(layout)
        return self.to_tensor(empty), self.to_tensor(mask), self.to_tensor(layout), self.to_tensor(empty)
        # return self.to_tensor(img), self.to_tensor(mask), self.to_tensor(layout)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float32)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma).astype(np.float32)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            # edge = imread(self.edge_data[index])
            path = os.path.join('../../dataset/Structured3D_extra/only_lines/', self.data[index], 'empty/lines_bi.png')
            edge = imread(path, mode='RGB')
            edge = rgb2gray(edge)
            edge = skt_rs(edge, (self.size[1], self.size[0]))
            # edge /= (edge.max() / 255.0)
            
            return edge.astype(np.float32)

    def load_mask(self, img, index, empty_semantic_map=None, full_semantic_map=None):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            # mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # refer to PanoDR mask generator
        if mask_type == 7:
            objects_empty, objects_full = self._extract_scene_objects(empty_semantic_map, full_semantic_map)
            candidate_objects_for_removal = self._select_candidates(objects_empty, objects_full)

            if len(candidate_objects_for_removal) == 0:
                if self._object_mask_only:
                    return None
                else:
                    mask = self._produce_random_mask()
            else:
                mask = self._compute_mask(full_semantic_map, list(candidate_objects_for_removal))
                if mask is None:
                    if self._object_mask_only:
                        return None
                    else:
                        mask = self._produce_random_mask()
                else:
                    if self._dilate_convex_mask:
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.dilate(mask, kernel, iterations=1)
                        
            return mask
            
    def _produce_random_mask(self) -> np.ndarray:
        min_size = min(self._width, self._height)
        width_size = int(min_size * (
                    self._random_mask_side_size_percentage + self.rng.random() * self._random_mask_side_deviation_percentage))
        height_size = int(min_size * (
                    self._random_mask_side_size_percentage + self.rng.random() * self._random_mask_side_deviation_percentage))

        pos_x = self.rng.randint(0, self._width - width_size - 1)
        pos_y = self.rng.randint(0, self._height - height_size - 1)

        mask = np.zeros((self._height, self._width), dtype=np.uint8)
        mask[pos_y: pos_y + height_size, pos_x: pos_x + width_size] = 255
        return mask

    def _compute_mask(self, semantic_map: np.ndarray, candidate_objects_for_removal: list) -> np.ndarray:
        image_area = semantic_map.size
        while len(candidate_objects_for_removal):
            chosen_id = self.rng.choice(candidate_objects_for_removal)
            object_mask = (semantic_map == chosen_id).astype(np.uint8) * 255
            # print(object_mask.dtype, object_mask.shape)
            # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
            contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            # Find the convex hull object for each contour


            boundary_objects = []
            for i in range(len(contours)):
                for p in contours[i]:
                    if (0 in p) or ((self._width - 1) in p):
                        boundary_objects.append(i)

            if len(boundary_objects) > 1:
                h = np.zeros_like(object_mask)
                for i in range(len(boundary_objects)):
                    hull = cv2.convexHull(contours[boundary_objects[i]])
                    h = cv2.fillConvexPoly(h, hull, (1))
                area = np.sum(h)
                if (area > self._min_mask_area * image_area) and (area < self._max_mask_area * image_area):
                    return h * 255
                else:
                    candidate_objects_for_removal.remove(chosen_id)
                    continue
            elif len(contours) >= 1:
                max_area, max_id = -1, -1
                for i in range(len(contours)):
                    hull = cv2.convexHull(contours[i])
                    h = np.zeros_like(object_mask)
                    h = cv2.fillConvexPoly(h, hull, (1))
                    area = np.sum(h)
                    if (area > max_area) and (area > self._min_mask_area * image_area) and (
                            area < self._max_mask_area * image_area):
                        max_area, max_id = area, i
                        mask = h
                if max_id != -1:
                    return mask * 255
                else:  # suitable object not found
                    candidate_objects_for_removal.remove(chosen_id)
                    continue
            return None
        return None

    def _extract_scene_objects(self, empty_semantic_map, full_semantic_map):
        objects_in_empty = np.unique(empty_semantic_map.flatten())
        objects_in_full = np.unique(full_semantic_map.flatten())
        return set(objects_in_empty), set(objects_in_full)

    def _select_candidates(self, objects_in_empty, objects_in_full):
        return (objects_in_full - objects_in_empty).intersection(self._classes4masking)
        # return (objects_in_full).intersection(self._classes4masking)



    # load GT layout, not for training or inference, only for visualization
    def load_layout_binary(self, img, index):
        imgh, imgw = img.shape[0:2]
        # edge = imread(self.edge_data[index])
        if self.training:
            path = os.path.join('../../dataset/Structured3D_extra/only_layout/',
                            self.data[index], 'empty/layout.png')
        else:
            path = self.layout_data[index]

        layout_rgb = imread(path, mode='RGB')

        layout_gray = rgb2gray(layout_rgb)
        layout_gray[layout_gray > 0.5] = 1.
        layout_gray[layout_gray < 0.5] = 0.

        layout_gray = self.resize(layout_gray, imgh, imgw)

        return layout_gray.astype(np.float32)

    # The old method for GT layout input, not used here
    def load_layout_instance(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training:
            cor_id_path = os.path.join('/CGVLAB3/datasets/Structured3D', self.data[index], 'layout.txt')
        else:
            cor_id_path = self.data[index].replace('full_gt', 'layout_txt').replace('png', 'txt')
        cor_id = np.loadtxt(cor_id_path)
        cor_id[:, 0] //= 1024//imgw
        cor_id[:, 1] //= 512 // imgh
        layout_viz, _, _ = draw_boundary_from_cor_id(cor_id, [imgh, imgw])
        layout_t = torch.from_numpy(layout_viz)
        layout_seg = self.Layout2Semantic(layout_t)

        raw_id = np.unique(cor_id[:,0])
        raw_id = np.insert(raw_id, 0, 0, axis=0)
        raw_id = np.insert(raw_id, len(raw_id), imgw, axis=0)

        layout_one_hot, plane_one_hot = self.one_hot(layout_seg.unsqueeze(0), 3, raw_id)

        return layout_one_hot, plane_one_hot


    def Layout2Semantic(self, layout_t):
        top_bottom = layout_t.cumsum(dim=0)>0
        bottom_up = 2 * (torch.flipud(torch.flipud(layout_t).cumsum(dim=0) > 0))
        semantic_mask = top_bottom + bottom_up
        return semantic_mask-1

    def one_hot(self, labels, C, raw_id):
        '''
            Converts an integer label torch.autograd.Variable to a one-hot Variable.

            Parameters
            ----------
            labels : torch.autograd.Variable of torch.cuda.LongTensor
                N x 1 x H x W, where N is batch size.
                Each value is an integer representing correct classification.
            C : integer.
                number of classes in labels.

            Returns
            -------
            target : torch.autograd.Variable of torch.cuda.FloatTensor
                N x C x H x W, where C is class number. One-hot encoded.
            '''
        # one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        one_hot = torch.FloatTensor(C, labels.size(1), labels.size(2)).zero_()
        target = one_hot.scatter_(0, labels.long(), 1)

        wall_segs = self.seg_by_cor(target, raw_id)

        # print(wall_segs.size(), target[0].unsqueeze(0).size())
        target_planes = torch.cat((target[0].unsqueeze(0), target[1].unsqueeze(0), wall_segs), dim=0)
        # target = Variable(target)

        return target, target_planes

    def seg_by_cor(self, target, raw_id):
        clip_idx = []
        for i in range(len(raw_id) - 1):
            tmp = i
            if raw_id[i + 1] > raw_id[tmp]:
                clip_idx.append([raw_id[tmp], raw_id[i + 1]])
            else:
                while raw_id[i + 1] < raw_id[tmp]:
                    i = i + 1
                clip_idx.append([raw_id[tmp], raw_id[i + 1]])

        wall_segs = self.wall_seg_by_clip_index(target, 2, clip_idx)

        return wall_segs

    def wall_seg_by_clip_index(self, target, wall_ch_id, clip_idx):
        wall_segs_list = []
        for i in range(len(clip_idx)):
            new_wall_seg = torch.FloatTensor(1, target.size(1), target.size(2)).zero_()
            new_wall_seg_1 = torch.FloatTensor(1, target.size(1), target.size(2)).zero_()
            new_wall_seg[:, :, int(clip_idx[i][0]):int(clip_idx[i][1])] = 1
            new_wall_seg = new_wall_seg + target[wall_ch_id]
            # new_wall_seg[new_wall_seg != 2] = 0
            new_wall_seg_1[new_wall_seg == 2] = 1
            wall_segs_list.append(new_wall_seg_1)

        wall_segs = torch.stack(wall_segs_list, dim=1).squeeze(0)
        wall_segs[0] = wall_segs[0] + wall_segs[-1]

        return wall_segs[:-1, :, :]





    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=False):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, (width, height))

        return img


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

