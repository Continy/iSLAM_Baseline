import cv2
import pypose as pp
import numpy as np
import pandas
import torch
import yaml
import os

from os import listdir
from os.path import isdir, isfile
from torch.utils.data import Dataset
from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim

from yacs.config import CfgNode as CN

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
import copy


def build_cfg(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = CN(yaml.safe_load(f))
    return cfg


def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5) / fx
    hh = (hh.astype(np.float32) - oy + 0.5) / fy
    intrinsicLayer = np.stack((ww, hh)).transpose(1, 2, 0)
    return intrinsicLayer


class TartanAirV2StereoLoader:

    def __init__(self, datadir):
        self.baseline = 0.25
        self.datatype = 'tartanair'
        imgfolder = datadir + '/image_lcam_front'
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder + '/' + ff) for ff in files
                         if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles), dtype=np.float32) * 0.1
        self.rgb_ts = np.array([i for i in range(len(self.rgbfiles))],
                               dtype=np.float64) * 0.1

        if isdir(datadir + '/image_rcam_front'):
            imgfolder = datadir + '/image_rcam_front'
            files = listdir(imgfolder)
            self.rgbfiles_right = [
                (imgfolder + '/' + ff) for ff in files
                if (ff.endswith('.png') or ff.endswith('.jpg'))
            ]
            self.rgbfiles_right.sort()
        else:
            raise ValueError('No right camera images found')

        self.intrinsic_pre = np.array([320.0, 320.0, 320.0, 320.0],
                                      dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 320.0],
                                        dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0, 0, 0, 0,
                                       1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        if isfile(datadir + '/pose_lcam_front.txt'):
            posefile = datadir + '/pose_lcam_front.txt'
            self.poses = np.loadtxt(posefile).astype(np.float32)
        else:
            raise ValueError('No pose file found')

        self.cropcenter = (640, 640)


class TartanAirV1StereoLoader:

    def __init__(self, datadir):
        self.baseline = 0.25
        self.datatype = 'tartanair'
        imgfolder = datadir + '/image_left'
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder + '/' + ff) for ff in files
                         if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles), dtype=np.float32) * 0.1
        self.rgb_ts = np.array([i for i in range(len(self.rgbfiles))],
                               dtype=np.float64) * 0.1

        if isdir(datadir + '/image_right'):
            imgfolder = datadir + '/image_right'
            files = listdir(imgfolder)
            self.rgbfiles_right = [
                (imgfolder + '/' + ff) for ff in files
                if (ff.endswith('.png') or ff.endswith('.jpg'))
            ]
            self.rgbfiles_right.sort()
        else:
            raise ValueError('No right camera images found')

        self.intrinsic_pre = np.array([320.0, 320.0, 320.0, 240.0],
                                      dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 240.0],
                                        dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0, 0, 0, 0,
                                       1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        if isfile(datadir + '/pose_left.txt'):
            posefile = datadir + '/pose_left.txt'
            self.poses = np.loadtxt(posefile).astype(np.float32)
        else:
            raise ValueError('No pose file found')

        self.cropcenter = (448, 640)


class StereoDataset(Dataset):

    def __init__(self, loader):
        self.loader = loader
        self.imgl = loader.rgbfiles
        self.imgr = loader.rgbfiles_right
        self.transform = Compose([
            CropCenter(loader.cropcenter, fix_ratio=True),
            DownscaleFlow(),
            Normalize(mean=MEAN, std=STD, keep_old=True),
            ToTensor()
        ])
        self.poses = loader.poses
        self.intrinsic_pre = loader.intrinsic_pre
        self.right2left_pose = loader.right2left_pose

        self.extrinsic = self.right2left_pose.clone().numpy()

        self.intrinsic_calib = self.intrinsic_pre.copy()
        self.datatype = loader.datatype

    def __len__(self):
        return len(self.imgl) - 1

    def __getitem__(self, idx):
        imgl = cv2.imread(self.imgl[idx], cv2.IMREAD_COLOR)
        imgl_nxt = cv2.imread(self.imgl[idx + 1], cv2.IMREAD_COLOR)
        imgr = cv2.imread(self.imgr[idx], cv2.IMREAD_COLOR)
        self.intrinsic = [
            make_intrinsics_layer(imgl.shape[1], imgl.shape[0],
                                  self.intrinsic_pre[0], self.intrinsic_pre[1],
                                  self.intrinsic_pre[2], self.intrinsic_pre[3])
        ]

        # imgl = imgl.reshape(1, imgl.shape[0], imgl.shape[1], imgl.shape[2])
        # imgr = imgr.reshape(1, imgr.shape[0], imgr.shape[1], imgr.shape[2])
        # imgl_nxt = imgl_nxt.reshape(1, imgl_nxt.shape[0], imgl_nxt.shape[1],
        #                             imgl_nxt.shape[2])

        sample = {
            'img1': imgl_nxt,
            'intrinsic': self.intrinsic,
            'intrinsic_calib': self.intrinsic_calib,
            'baseline': self.loader.baseline,
            'img0': imgl,
            'img0_r': imgr,
            'datatype': self.datatype,
        }
        sample = copy.deepcopy(sample)
        sample = self.transform(sample)
        return sample
