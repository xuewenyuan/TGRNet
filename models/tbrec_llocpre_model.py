import torch
import itertools
from .base_model import BaseModel
from . import networks2 as networks
import sys, cv2, os, pickle
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from util import util

class TbrecLlocPreModel(BaseModel):
    """
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--rm_layers', type=str, default='', help='remove layers when load the pretrained model')
        return parser

    def __init__(self, opt):

        self.opt = opt

        BaseModel.__init__(self, self.opt)

        
        self.loss_names = ['cls_row', 'cls_col']

        self.model_names = ['Backbone','CellLlocPre']

        init_mode = False
        if self.isTrain:
            init_mode = not self.opt.continue_train

        self.netBackbone    = networks.resnet_fpn_backbone('resnet50', init_mode, self.opt.distributed, self.opt.gpu)
        #self.netCellBboxSeg = networks.cell_seg_head(self.opt.distributed, self.opt.gpu)
        self.netCellLlocPre = networks.cell_loc_head(self.opt.num_rows, self.opt.num_cols, self.opt.load_height, \
                                                    self.opt.load_width, self.opt.alpha, self.device, self.opt.distributed, self.opt.gpu)

        if self.isTrain:
            #self.criterionBbox = torch.nn.CrossEntropyLoss(weight = torch.FloatTensor([0.1,0.1,1.0])).to(self.device)
            #self.criterion_Lloc_cols = torch.nn.CrossEntropyLoss().to(self.device)
            #self.criterion_Lloc_rows = torch.nn.CrossEntropyLoss().to(self.device)
            gamma_dict = pickle.load(open(os.path.join(self.opt.dataroot,'gamma.pkl'), 'rb'))
            self.criterion_Lloc_cols = networks.OrdinalRegressionLoss(self.opt.num_cols, gamma_dict['col_gamma'])
            self.criterion_Lloc_rows = networks.OrdinalRegressionLoss(self.opt.num_rows, gamma_dict['row_gamma'])
            self.optimizer = torch.optim.Adam(itertools.chain(self.netBackbone.parameters(),\
                            self.netCellLlocPre.parameters()), lr=self.opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer])

    def set_input(self, input):
        self.table_names = input['img_names']
        self.table_images = input['tb_imgs'].to(self.device)
        self.cell_gt_boxes = list(input['cell_boxes'])
        self.cell_pred_boxes = list(input['pred_boxes'])
        #self.seg_row_label = input['seg_row_label'].to(self.device)
        #self.seg_col_label = input['seg_col_label'].to(self.device)
        #self.seg_rc_label  = input['seg_rc_label'].to(self.device)
        self.cls_row_label = input['cls_row_label'].to(self.device)
        self.cls_col_label = input['cls_col_label'].to(self.device)
        self.im_scales = input['im_scales']
        self.pdls = input['pdls']
        self.pdts = input['pdts']

    def forward(self):
        inter_feat = self.netBackbone(self.table_images)
        #self.seg_row_score, self.seg_col_score, self.seg_rc_score, self.cell_pred_boxes = self.netCellBboxSeg(inter_feat)
        #self.seg_row_score, self.seg_col_score, self.cell_pred_boxes = self.netCellBboxSeg(inter_feat)
        if self.isTrain:
            self.cls_row_score, self.cls_col_score, self.cls_inds = self.netCellLlocPre(inter_feat, self.cell_pred_boxes, \
                                                                    self.im_scales, self.pdls, self.pdts, self.cell_gt_boxes)
        else:
            self.cls_row_score, self.cls_col_score, _ = self.netCellLlocPre(inter_feat, self.cell_pred_boxes, \
                                                                    self.im_scales, self.pdls, self.pdts, None)
        

    def backward(self):
        #loss_seg_row = self.criterionBbox(self.seg_row_score, self.seg_row_label)
        #loss_seg_col = self.criterionBbox(self.seg_col_score, self.seg_col_label)
        #loss_seg_rc = self.criterionBbox(self.seg_rc_score, self.seg_rc_label)
        loss_cls_row = self.criterion_Lloc_rows(self.cls_row_score, self.cls_row_label)
        loss_cls_col = self.criterion_Lloc_cols(self.cls_col_score, self.cls_col_label)
        #losses = (loss_seg_row + loss_seg_col + loss_seg_rc) + (loss_cls_row + loss_cls_col)
        losses = (loss_cls_row + loss_cls_col)
        #losses = loss_seg_row + loss_seg_col + loss_seg_rc
        #loss_dict = {'seg_row': loss_seg_row, 'seg_col': loss_seg_col, 'seg_rc': loss_seg_rc, \
        #            'cls_row': loss_cls_row, 'cls_col': loss_cls_col}
        loss_dict = {'cls_row': loss_cls_row, 'cls_col': loss_cls_col}
        #loss_dict = {'seg_row': loss_seg_row, 'seg_col': loss_seg_col, 'seg_rc': loss_seg_rc}
        loss_dict_reduced = util.reduce_dict(loss_dict)
        #self.loss_seg_row = loss_dict_reduced['seg_row']
        #self.loss_seg_col = loss_dict_reduced['seg_col']
        #self.loss_seg_rc  = loss_dict_reduced['seg_rc']
        self.loss_cls_row = loss_dict_reduced['cls_row']
        self.loss_cls_col = loss_dict_reduced['cls_col']
        losses.backward()

    def optimize_parameters(self):
        if not self.isTrain:
            self.isTrain = True
        self.netBackbone.train()
        #self.netCellBboxSeg.train()
        self.netCellLlocPre.train()
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
    
    @torch.no_grad()
    def test(self):
        if self.isTrain:
            self.isTrain = False
        self.forward()
        #cls_pred_row = torch.argmax(self.cls_row_score,dim=1) # [num_cells, 2]
        #cls_pred_col = torch.argmax(self.cls_col_score,dim=1) # [num_cells, 2]
        cls_pred_row = torch.argmax(1-F.softmax(self.cls_row_score,dim=-1),dim=-1) # [num_cells, 2, rows_classes-1]
        cls_pred_col = torch.argmax(1-F.softmax(self.cls_col_score,dim=-1),dim=-1) # [num_cells, 2, cols_classes-1]
        cls_pred_row = torch.sum(cls_pred_row, dim=-1) #
        cls_pred_col = torch.sum(cls_pred_col, dim=-1) # 

        pred_lloc = torch.cat((cls_pred_row, cls_pred_col),dim=1).cpu().int().numpy() # [num_cells, 4]
        pred_box = [box.numpy() for box in self.cell_pred_boxes]
        #pred_box = [box.numpy() for box in self.cell_gt_boxes]
        preds = {}
        cell_count = 0
        for ind, tb_name in enumerate(self.table_names):
            num_cells = pred_box[ind].shape[0]
            preds[tb_name] = {'bbox': pred_box[ind], 'lloc': pred_lloc[cell_count:cell_count+num_cells,:]}
            #preds[tb_name] = {'lloc': pred_lloc[cell_count:cell_count+num_cells,:]}
            #preds[tb_name] = {'bbox': pred_box[ind]}
            cell_count += num_cells
        return preds
    '''
    @torch.no_grad()
    def test(self):
        if self.isTrain:
            self.isTrain = False
        self.forward()
        #cls_pred_row = torch.argmax(self.cls_row_score,dim=1) # [num_cells, 2]
        #cls_pred_col = torch.argmax(self.cls_col_score,dim=1) # [num_cells, 2]
        cls_pred_row = torch.argmax(1-F.softmax(self.cls_row_score,dim=-1),dim=-1) # [num_cells, 2, rows_classes-1]
        cls_pred_col = torch.argmax(1-F.softmax(self.cls_col_score,dim=-1),dim=-1) # [num_cells, 2, cols_classes-1]
        cls_pred_row = torch.sum(cls_pred_row, dim=-1) # [num_cells, 2]
        cls_pred_col = torch.sum(cls_pred_col, dim=-1) # [num_cells, 2]

        pred_lloc = torch.cat((cls_pred_row, cls_pred_col),dim=1).cpu().int().numpy()
        gt_lloc = torch.cat((self.cls_row_label,self.cls_col_label),dim=1).cpu().int().numpy()

        cls_rowSt = np.where(pred_lloc[:,0]==gt_lloc[:,0])[0].shape[0]
        cls_rowEd = np.where(pred_lloc[:,1]==gt_lloc[:,1])[0].shape[0]
        cls_colSt = np.where(pred_lloc[:,2]==gt_lloc[:,2])[0].shape[0]
        cls_colEd = np.where(pred_lloc[:,3]==gt_lloc[:,3])[0].shape[0]
        #print(pred_lloc.shape[0],gt_lloc.shape[0])
        cls_lloc  = np.where(np.sum(pred_lloc==gt_lloc,1)==4)[0].shape[0]

        cell_count = gt_lloc.shape[0]

        return (cls_rowSt, cls_rowEd, cls_colSt, cls_colEd, cls_lloc, cell_count)
    '''