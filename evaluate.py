from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
from Datasets.transformation import motion2pose_pypose
from Datasets.TrajFolderDataset import TrajFolderDataset
from Datasets.StereoDataset import build_cfg
from pathlib import Path
from TartanVO import TartanVO
import torch
from torch.utils.data import DataLoader
import pypose as pp
import numpy as np
import os
from tqdm import tqdm

EDN2NED = pp.from_matrix(
    torch.tensor([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., 0., 1.]],
                 dtype=torch.float32), pp.SE3_type).to('cuda')


class ColoredTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         colour="yellow",
                         ascii="░▒█",
                         dynamic_ncols=True,
                         **kwargs)

    def close(self, *args, **kwargs):
        if self.n < self.total:
            self.colour = "red"
            self.desc = "❌ Error"
        else:
            self.colour = "#35aca4"
            self.desc = "✅ Finish"
        super().close(*args, **kwargs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/local_tartanv1.yaml')
    parser.add_argument('--savepath', type=str, default='./Results')
    parser.add_argument('--data_type', type=str, default='v1')

    cfg = parser.parse_args()
    args = build_cfg(cfg.config)
    args.update(vars(cfg))
    if args.data_type == 'v1':
        cropsize = (448, 640)
    elif args.data_type == 'v2':
        cropsize = (640, 640)
    elif args.data_type == 'euroc':
        cropsize = (448, 640)
    else:
        raise ValueError('Unknown dataset type')

    root = args.ROOT

    for subset in args.DATASETS:
        motion_list = []
        torch.cuda.empty_cache()
        path = str(Path(root) / subset)
        print('Loading dataset:', path)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = Compose([
            CropCenter((448, 640), fix_ratio=True),
            DownscaleFlow(),
            Normalize(mean=mean, std=std, keep_old=True),
            ToTensor(),
            SqueezeBatchDim()
        ])

        dataset = TrajFolderDataset(datadir=path,
                                    datatype=args.data_type,
                                    transform=transform,
                                    start_frame=0,
                                    end_frame=-1)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=0,
                                shuffle=False,
                                drop_last=True)

        tartanvo = TartanVO(vo_model_name=args.VO_MODEL,
                            pose_model_name=None,
                            correct_scale=False,
                            fix_parts=('flow', 'stereo'))
        tartanvo.eval()
        for index, sample in ColoredTqdm(enumerate(dataloader),
                                         total=len(dataloader),
                                         desc=subset):
            with torch.no_grad():
                res = tartanvo(sample)
            motion = res['motion']
            motion_list.append(motion)

        motion_list = torch.cat(motion_list, dim=0)
        poses = motion2pose_pypose(motion_list)
        poses = EDN2NED @ poses @ EDN2NED.Inv()
        poses_np = poses.detach().cpu().numpy()

        os.makedirs(Path(args.savepath) / 'txt', exist_ok=True)
        os.makedirs(Path(args.savepath) / 'npy', exist_ok=True)
        subset = subset.replace('/', '_')
        np.save(Path(args.savepath) / 'npy' / f'{subset}.npy', poses_np)
        np.savetxt(Path(args.savepath) / 'txt' / f'{subset}.txt', poses_np)
        print(f'{path} done')
