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

from shapely.geometry import LineString, Polygon
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import maximum_filter
from .misc import panostretch, post_proc

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

        self.structure3D_path = '/CGVLAB3/datasets/Structured3D'
        self.np_rng = np.random.default_rng(self.config.SEED)

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
        #     item = []
        except:
            print('loading error: ' + self.data[index])
            
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]

        if self.training:
            return '_'.join(name.split('/')) + '.png'
        else:
            return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        
        ### load empty room panorama
        if self.training:
            path = os.path.join(self.structure3D_path, self.data[index], 'empty/rgb_rawlight.png')
            empty = imread(path, mode='RGB')
        else:
            path = self.data[index]
            empty = imread(path, mode='RGB')
        
        ### load full room panorama
        if self.training:
            path = os.path.join(self.structure3D_path, self.data[index], 'full/rgb_rawlight.png')
            full = imread(path, mode='RGB')
        else:
            path = self.data[index]
            full = imread(path, mode='RGB')
            
        # resize/crop if needed
        if size != 0:
            empty = self.resize(empty, size[0], size[1])
            full = self.resize(full, size[0], size[1])
        
        if self.config.MASK == 8 or self.config.MASK == 7 or self.config.FURNISHED == 1:
            ### load empty/full room semantic segmentation
            imgh, imgw = empty.shape[0:2]
            if self.training:
                empty_semantic_map = imread(os.path.join(self.structure3D_path, self.data[index], 'empty/semantic.png'), mode='P')
                full_semantic_map = imread(os.path.join(self.structure3D_path, self.data[index], 'full/semantic.png'), mode='P')
            else:
                empty_semantic_map = imread(self.data[index].replace('full_gt', 'empty_semantic'), mode='P')
                full_semantic_map = imread(self.data[index].replace('full_gt', 'full_semantic'), mode='P')
                
            empty_semantic_map = cv2.resize(empty_semantic_map.astype(np.uint8), (imgw, imgh), cv2.INTER_NEAREST)
            full_semantic_map = cv2.resize(full_semantic_map.astype(np.uint8), (imgw, imgh), cv2.INTER_NEAREST)
        
        elif self.config.MASK == 9:
            empty_semantic_map = imread(os.path.join(self.structure3D_path, self.data[index], 'empty/semantic.png'), mode='P')
            imgh, imgw = empty_semantic_map.shape[0:2]

            mask = np.zeros((imgh, imgw), bool)
            object_mask = ( (empty_semantic_map ==  8) | 
                            (empty_semantic_map ==  9) | 
                            (empty_semantic_map == 13) | 
                            (empty_semantic_map == 16) | 
                            (empty_semantic_map == 38) |
                            (empty_semantic_map == 40) ).astype(np.uint8) * 255
            
            # kernel = np.ones((5, 5), np.uint8)
            # object_mask = cv2.dilate(object_mask, kernel, iterations=3)

            contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                h = np.zeros_like(object_mask)
                h = cv2.fillConvexPoly(h, hull, (1)).astype(bool)
                mask |= h
            
            mask = mask.astype(np.uint8) * 255

            imgh, imgw = empty.shape[0:2]
            empty_semantic_map = cv2.resize(mask, (imgw, imgh), cv2.INTER_NEAREST)
            full_semantic_map = empty_semantic_map
        
        else:
            empty_semantic_map = None
            full_semantic_map = None
        
        if self.config.FURNISHED == 1:
            foreground = 255 * ((full_semantic_map != CEILING_ID).astype(np.uint8) * (full_semantic_map != FLOOR_ID).astype(np.uint8) * (full_semantic_map != WALL_ID).astype(np.uint8))
            img = np.where(foreground.astype(np.bool)[...,None], full, empty)
        else:
            img = empty

        # load mask
        mask = self.load_mask(img, index, empty_semantic_map, full_semantic_map)

        # load edge
        # edge = self.load_edge(img_gray, index, mask)

        # NOTE: Disable horizonNet postprocessing
        return self.to_tensor(img), self.to_tensor(mask), torch.FloatTensor(), self.to_tensor(empty)

        # load layout
        if self.training:
            cor_id_path = os.path.join(self.structure3D_path, self.data[index], 'layout.txt')
            
            # post_proc
            h, w, h_o, w_o = 512, 1024, empty.shape[0], empty.shape[1]
            
            # bon, y_cor = self.load_layout_horizon_net(index)
            # y_bon_ = torch.unsqueeze(bon, 0).numpy()
            # y_cor_ = torch.sigmoid(torch.unsqueeze(y_cor, 0)).numpy()

            # Read ground truth corners
            with open(cor_id_path) as f:
                cor = np.array([line.strip().split() for line in f if line.strip()], np.int64)

                # Corner with minimum x should at the beginning
                cor_ = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            # cor_ = cors_from_horizon_net(y_bon_, y_cor_)
            bon_ = torch.FloatTensor(cor_2_1d(cor_, H=h, W=w))

            y_bon = torch.nn.functional.interpolate(torch.unsqueeze(bon_, 0), size=w_o, mode='nearest')
            y_bon = torch.clamp(((y_bon / np.pi + 0.5) * h_o).round().type(torch.int64), 0, h_o-1)
            
            layout_p = torch.zeros(1, h_o, w_o)
            layout_p[0, y_bon[0, 0], torch.arange(w_o)] = 1
            layout_p[0, y_bon[0, 1], torch.arange(w_o)] = 1

            occlusion = find_occlusion(cor_[::2].copy()).repeat(2)
            cor_ = cor_[~occlusion]

            cor_[:, 0] //= w // w_o
            cor_[:, 1] //= h // h_o
            for ix in range(len(cor_[0::2])):
                layout_p[0, torch.arange(cor_[ix*2, 1], cor_[ix*2+1, 1]+1), cor_[ix*2, 0]] = 1
            
            layout_seg = self.Layout2Semantic(layout_p[0])

            raw_id = cor_[0::2, 0]
            raw_id = np.insert(raw_id, 0, 0, axis=0)
            raw_id = np.insert(raw_id, len(raw_id), w_o, axis=0)

            layout_one_hot, plane_one_hot = self.one_hot(layout_seg.unsqueeze(0), 3, raw_id)

            pad = 100
            c, h, w = plane_one_hot.size()
            padding = torch.zeros(pad-c, h, w)
            plane_one_hot = torch.cat((plane_one_hot, padding), 0)

        else:
            layout_one_hot, plane_one_hot, layout_p = torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()

        return self.to_tensor(img), self.to_tensor(mask), torch.cat((layout_one_hot, plane_one_hot, layout_p)), self.to_tensor(empty)

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
            # if self.training:
            #     self._min_mask_area, self._max_mask_area = 0.06, 0.8
            # else:
            #     self._min_mask_area, self._max_mask_area = 0.01, 0.8

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
                        mask = cv2.dilate(mask, kernel, iterations=1, borderValue=255)

        if mask_type == 8:
            mask = np.zeros((imgh, imgw), np.uint8)
            r = self.np_rng.normal(loc=0.5, scale=0.05)
            r = np.clip(r, 0.3, 0.7)
            mh, mw = int(imgh * r), int(imgw * r)

            objects_empty, objects_full = self._extract_scene_objects(empty_semantic_map, full_semantic_map)
            candidate_objects_for_removal = list(self._select_candidates(objects_empty, objects_full))

            if len(candidate_objects_for_removal) == 0:
                rand_y, rand_x = self.rng.randint(int(mh / 4), imgh - int(mh / 4)), self.rng.randint(int(mw / 4), imgw - int(mw / 4))
                mask = self.create_mask_center(rand_y, rand_x, mh, mw, imgh, imgw)
            else:
                chosen_id = self.rng.choice(candidate_objects_for_removal)
                object_mask = full_semantic_map == chosen_id
                
                ys, xs = np.where(object_mask)
                mean_y, mean_x = np.average(ys), np.average(xs)
                
                ys_r, _ = np.where(np.roll(object_mask, imgh // 2, axis=0))
                _, xs_r = np.where(np.roll(object_mask, imgw // 2, axis=1))
                mean_y_r, mean_x_r = np.average(ys_r), np.average(xs_r)

                # yx, ry, rx, ryrx
                masks = []
                masks.append(self.create_mask_center(mean_y, mean_x, mh, mw, imgh, imgw))
                masks.append(self.create_mask_center(mean_y_r, mean_x, mh, mw, imgh, imgw))
                masks.append(self.create_mask_center(mean_y, mean_x_r, mh, mw, imgh, imgw))
                masks.append(self.create_mask_center(mean_y_r, mean_x_r, mh, mw, imgh, imgw))

                coverage = [np.sum(m * object_mask) for m in masks]
                mask = masks[np.argmax(coverage)]

        if mask_type == 9:
            mask = np.zeros((imgh, imgw), np.uint8)
            r = self.np_rng.normal(loc=0.5, scale=0.05)
            r = np.clip(r, 0.3, 0.7)
            mh, mw = int(imgh * r), int(imgw * r)

            object_mask = empty_semantic_map == 0

            grid_size = (2, 2)
            grid_h, grid_w = int(imgh / grid_size[0]), int(imgw / grid_size[1])

            masks = []
            for y in range(grid_size[0]):
                for x in range(grid_size[1]):
                    ys, xs = np.where(object_mask[y * grid_h:y * grid_h + grid_h, x * grid_w:x * grid_w + grid_w])
                    if len(ys) != 0:
                        rand_i = self.rng.randint(0, len(ys))
                        masks.append(self.create_mask_center(ys[rand_i], xs[rand_i], mh, mw, imgh, imgw))

            if len(masks) > 0:
                coverage = [np.sum(m * object_mask) for m in masks]
                mask = masks[np.argmax(coverage)]
            else:
                rand_y, rand_x = self.rng.randint(int(mh / 4), imgh - int(mh / 4)), self.rng.randint(int(mw / 4), imgw - int(mw / 4))
                mask = self.create_mask_center(rand_y, rand_x, mh, mw, imgh, imgw)

            # name = self.load_name(index)
            # cv2.imwrite(f'checkpoints/avoid_open_FULL/masks/{name}.png', np.dstack((np.zeros_like(mask), mask, (1 - object_mask.astype(np.uint8))*255)))
            # mask = empty_semantic_map

        return mask

    def create_mask_center(self, y, x, mh, mw, imgh, imgw):
        mask = np.zeros((imgh, imgw), np.uint8)
        py, px = int(y - mh / 2), int(x - mw / 2)

        pxs = []
        if px < 0:
            pxs.append((0, px + mw))
            pxs.append((px + imgw, imgw))
        elif px + mw >= imgw:
            pxs.append((px, imgw))
            pxs.append((0, mw + px - imgw))
        else:
            pxs.append((px, px + mw))
        
        pys = []
        if py < 0:
            pys.append((0, mh))
            # pys.append((0, py + mh))
            # pys.append((py + imgh, imgh))
        elif py + mh >= imgh:
            pys.append((imgh - mh, imgh))
            # pys.append((py, imgh))
            # pys.append((0, mh + py - imgh))
        else:
            pys.append((py, py + mh))
        
        for y0, y1 in pys:
            for x0, x1 in pxs:
                mask[y0:y1, x0:x1] = 255
        
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
            cor_id_path = os.path.join(self.structure3D_path, self.data[index], 'layout.txt')
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
    
    def load_layout_horizon_net(self, index):
        """
        Copy from https://github.com/sunset1995/HorizonNet/blob/master/dataset.py
        """
        if self.training:
            cor_id_path = os.path.join(self.structure3D_path, self.data[index], 'layout.txt')
        else:
            return torch.FloatTensor(), torch.FloatTensor()

        H, W = 512, 1024

        # Read ground truth corners
        with open(cor_id_path) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            # Detect occlusion
            occlusion = find_occlusion(cor[::2].copy()).repeat(2)
            assert (np.abs(cor[0::2, 0] - cor[1::2, 0]) > W/100).sum() == 0, path
            assert (cor[0::2, 1] > cor[1::2, 1]).sum() == 0, path
        
        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, H, W)

        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]
        dist_o = cdist(corx.reshape(-1, 1),
                       np.arange(W).reshape(-1, 1),
                       p=1)
        dist_r = cdist(corx.reshape(-1, 1),
                       np.arange(W).reshape(-1, 1) + W,
                       p=1)
        dist_l = cdist(corx.reshape(-1, 1),
                       np.arange(W).reshape(-1, 1) - W,
                       p=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        nearest_dist = dist.min(0)
        y_cor = (0.96 ** nearest_dist).reshape(1, -1)

        # Convert all data to tensor
        bon = torch.FloatTensor(bon.copy())
        y_cor = torch.FloatTensor(y_cor.copy())

        return bon, y_cor

def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
    y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)
    y_cor = y_cor.numpy()

    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy() * 0.5).astype(np.uint8)
    y1 = np.round(y_bon[0]).astype(int)
    y2 = np.round(y_bon[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

    return np.concatenate([gt_cor, img_pad, img_bon], 0)

def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]

def cors_from_horizon_net(y_bon_, y_cor_, H=512, W=1024):
    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0, 0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    r = int(round(W * 0.05 / 2))
    xs_ = find_N_peaks(y_cor_, r=r)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=False)

    # Check valid (for fear self-intersection)
    xy2d = np.zeros((len(xy_cor), 2), np.float32)
    for i in range(len(xy_cor)):
        xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
        xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
    if not Polygon(xy2d).is_valid:
        print(
            'Fail to generate valid general layout!! '
            'Generate cuboid as fallback.')
        xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
        cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    # cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
    cor_ = np.zeros((cor.shape[0]*2, cor.shape[1]))
    cor_[0::2] = cor
    cor_[1::2] = np.hstack([cor[:, 0][:, None], post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    return cor_.round().astype(np.int64)