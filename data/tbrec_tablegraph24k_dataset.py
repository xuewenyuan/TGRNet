import os
import torch
import pickle
from data.base_dataset import BaseDataset
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TbRecTableGraph24KDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--alpha', type=float, default=3, help='the adjustment factor alpha')
        parser.add_argument('--num_cols', type=int, default=22, help='the number of columns for classification')
        parser.add_argument('--num_rows', type=int, default=38, help='the number of rows for classification')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.data_list = open(os.path.join(opt.dataroot, opt.phase+'.txt'), 'r', encoding='UTF-8').readlines()
        self.dataroot = opt.dataroot
        self.re_h = opt.load_height
        self.re_w = opt.load_width

    def __getitem__(self, index): 
        """
        """
        table_pkl, node_csv, _, target_csv = self.data_list[index].strip().split()
        table_pkl = os.path.join(self.dataroot, 'gt', os.path.split(table_pkl)[1])
        node_csv = os.path.join(self.dataroot, 'graph_node', os.path.split(node_csv)[1])
        target_csv = os.path.join(self.dataroot, 'graph_target', os.path.split(target_csv)[1])

        table_anno = pickle.load(open(table_pkl, 'rb'))
        table_name = os.path.split(table_pkl)[1].replace('.pkl','')

        img_path = os.path.join(self.dataroot, 'image', os.path.split(table_anno['image_path'])[1])
        tb_img  = Image.open(img_path).convert("RGB")
        tb_w, tb_h = tb_img.size
        img_transformation, row_label_transformation, col_label_transformation, rc_label_transformation, im_scale, pdl, pdt= self.get_transform(tb_w, tb_h)
        tb_img = img_transformation(tb_img)

        seg_label_pkl = table_pkl.replace('gt','seg_label')
        seg_label = pickle.load(open(seg_label_pkl, 'rb'))
        seg_row_label = Image.fromarray(np.expand_dims(seg_label['row_label'],1).astype('uint8'))
        seg_col_label = Image.fromarray(np.expand_dims(seg_label['col_label'],0).astype('uint8'))
        seg_row_label = row_label_transformation(seg_row_label)
        seg_col_label = col_label_transformation(seg_col_label)
        seg_row_label = torch.from_numpy(np.array(seg_row_label)).to(torch.int64)
        seg_col_label = torch.from_numpy(np.array(seg_col_label)).to(torch.int64)

        seg_rc_label = Image.fromarray(seg_label['seg_label'].astype('uint8'))
        seg_rc_label = rc_label_transformation(seg_rc_label)
        seg_rc_label = torch.from_numpy(np.array(seg_rc_label)).to(torch.int64)

        nodes = pd.read_csv(node_csv).values
        cell_boxes = nodes[:,[3,4,7,8]].astype(np.float64)
        cell_boxes = cell_boxes*im_scale+np.array([pdl,pdt]*2)
        cell_boxes = torch.from_numpy(cell_boxes).to(torch.float32)

        targets = pd.read_csv(target_csv).values
        cls_row_label = torch.from_numpy(targets[:,[2,3]]).to(torch.int64)
        cls_col_label = torch.from_numpy(targets[:,[4,5]]).to(torch.int64)

        return table_name, tb_img, cell_boxes, seg_row_label, seg_col_label, seg_rc_label, cls_row_label, cls_col_label, im_scale, pdl, pdt

    def __len__(self):
        return len(self.data_list)

    def get_transform(self, tb_w, tb_h):
        img_transform = []
        seg_row_label_transform = []
        seg_col_label_transform = []
        seg_rc_label_transform = []

        min_size = min(self.re_h, self.re_w)
        max_size = max(self.re_h, self.re_w)

        im_min_size = min(tb_w, tb_h)
        im_max_size = max(tb_w, tb_h)
        im_scale = float(min_size) / float(im_min_size)
        if int(im_scale*im_max_size) > max_size:
            im_scale = float(max_size) / float(im_max_size)
        rew = int(tb_w * im_scale)
        reh = int(tb_h * im_scale)
        pdl, pdt = ((max_size-rew)//2, (max_size-reh)//2) 
        pdr, pdd = (max_size-rew-pdl, max_size-reh-pdt)

        img_transform.append(transforms.Resize((reh,rew), Image.BICUBIC))
        img_transform.append(transforms.Pad((pdl, pdt, pdr, pdd), padding_mode='edge'))
        img_transform.append(transforms.ToTensor())
        img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        seg_row_label_transform.append(transforms.Resize((reh,1), Image.NEAREST))
        seg_row_label_transform.append(transforms.Pad((0, pdt, 0, pdd)))
        seg_col_label_transform.append(transforms.Resize((1,rew), Image.NEAREST))
        seg_col_label_transform.append(transforms.Pad((pdl, 0, pdr, 0)))
        seg_rc_label_transform.append(transforms.Resize((reh,rew), Image.NEAREST))
        seg_rc_label_transform.append(transforms.Pad((pdl, pdt, pdr, pdd)))

        return transforms.Compose(img_transform), transforms.Compose(seg_row_label_transform), \
                transforms.Compose(seg_col_label_transform), transforms.Compose(seg_rc_label_transform), im_scale, pdl, pdt

    def collate_fn(self, batch):
        names, images, cell_boxes, seg_row_label, seg_col_label, seg_rc_label, cls_row_label, cls_col_label, im_scales, pdls, pdts = list(zip(*batch))
        batched_imgs = torch.stack(images,0)
        batched_seg_row_label = torch.stack(seg_row_label,0)
        batched_seg_col_label = torch.stack(seg_col_label,0)
        batched_seg_rc_label = torch.stack(seg_rc_label,0)
        batched_cls_row_label = torch.cat(cls_row_label,0)
        batched_cls_col_label = torch.cat(cls_col_label,0)
        batched_im_scales = torch.as_tensor(list(im_scales), dtype=torch.float32)
        batched_pdls = torch.as_tensor(list(pdls), dtype=torch.int32)
        batched_pdts = torch.as_tensor(list(pdts), dtype=torch.int32)
        return {'img_names': names, 'tb_imgs': batched_imgs, 'cell_boxes': cell_boxes, 'seg_row_label': batched_seg_row_label, \
                'seg_col_label': batched_seg_col_label, 'seg_rc_label': batched_seg_rc_label, \
                'cls_row_label': batched_cls_row_label, 'cls_col_label': batched_cls_col_label, \
                'im_scales': batched_im_scales, 'pdls': batched_pdls, 'pdts': batched_pdts}