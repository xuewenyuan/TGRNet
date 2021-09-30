import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import ops
from torchvision.ops import boxes as box_ops
import numpy as np
import cv2, os

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from torch.jit.annotations import Tuple, List, Dict, Optional
from collections import OrderedDict

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 and m.affine:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, use_distributed, gpu_id, no_init=False, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    """
    if use_distributed:
        assert(torch.cuda.is_available())
        net.to(torch.device('cuda'))
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_id])
    if not no_init:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def define_ResNet50(gpu_ids=[]):
    net = models.resnet50(pretrained=True)
    #net = nn.Sequential(*list(net.children())[:-2])
    net = ResNet50(net)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids) 
    return net

def resnet_fpn_backbone(backbone_name, pretrained, use_distributed, gpu_id, norm_layer=ops.misc.FrozenBatchNorm2d, trainable_layers=5):
    backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, 
            norm_layer=ops.misc.FrozenBatchNorm2d)
    
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    #return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256

    net = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    net = FeatureFusionForFPN(net)

    ## initalize the FeatureFusion layers
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    for submodule in net.children():
        if submodule.__class__.__name__ != "BackboneWithFPN":
            submodule.apply(init_func)
    
    return init_net(net, use_distributed, gpu_id, no_init=True)

def cell_seg_head(use_distributed, gpu_id):
    net = Cell_Bbox_Seg()
    return init_net(net, use_distributed, gpu_id)

def cell_loc_head(rows_classes, cols_classes, img_h, img_w, alpha, device, use_distributed, gpu_id):
    net = Cell_Lloc_Pre(rows_classes, cols_classes, img_h, img_w, alpha, device)
    return init_net(net, use_distributed, gpu_id)


##############################################################################
# Classes
##############################################################################

class OrdinalRegressionLoss(nn.Module):
    """
    """

    def __init__(self, num_class, gamma=None):
        """ 
        """
        super(OrdinalRegressionLoss, self).__init__()
        self.num_class = num_class
        self.gamma = torch.as_tensor(gamma, dtype=torch.float32)

    def _create_ordinal_label(self, gt):
        gamma_i = torch.ones(list(gt.shape)+[self.num_class-1])*self.gamma
        gamma_i = gamma_i.to(gt.device)
        gamma_i = torch.stack([gamma_i,gamma_i],-1)

        ord_c0 = torch.ones(list(gt.shape)+[self.num_class-1]).to(gt.device)
        mask = torch.zeros(list(gt.shape)+[self.num_class-1])+torch.linspace(0, self.num_class - 2, self.num_class - 1, requires_grad=False)
        mask = mask.contiguous().long().to(gt.device)
        mask = (mask >= gt.unsqueeze(len(gt.shape)))
        ord_c0[mask] = 0
        ord_c1 = 1-ord_c0
        ord_label = torch.stack([ord_c0,ord_c1],-1)
        return ord_label.long(), gamma_i

    def __call__(self, prediction, target):
        # original
        #ord_label = self._create_ordinal_label(target)
        #pred_score = F.log_softmax(prediction,dim=-1)
        #entropy = -pred_score * ord_label
        #entropy = entropy.view(-1,2,(self.num_class-1)*2)
        #loss = torch.sum(entropy, dim=-1).mean()
        # using nn.CrossEntropyLoss()
        #ord_label = self._create_ordinal_label(target)
        #criterion = nn.CrossEntropyLoss().to(ord_label.device)
        #loss = criterion(prediction, ord_label)
        # add focal
        ord_label, gamma_i = self._create_ordinal_label(target)
        pred_score = F.softmax(prediction,dim=-1)
        pred_logscore = F.log_softmax(prediction,dim=-1)
        entropy = -ord_label * torch.pow((1-pred_score), gamma_i) * pred_logscore
        entropy = entropy.view(-1,2,(self.num_class-1)*2)
        loss = torch.sum(entropy,dim=-1)
        return loss.mean()


class BackboneWithFPN(nn.Module):
    """
    copy from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py
    without extra_blocks=LastLevelMaxPool() in FeaturePyramidNetwork
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

class FeatureFusionForFPN(nn.Module):
    def __init__(self, backbone):
        super(FeatureFusionForFPN, self).__init__()
        
        self.fpn_backbone = backbone

        self.layer1_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.layer2_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
        
        self.layer3_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.layer4_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.smooth1 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.smooth2 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.smooth3 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return nn.functional.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H, W), mode='bilinear') + y
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        fpn_outputs = self.fpn_backbone(x)
        #print(fpn_outputs['0'].shape,fpn_outputs['1'].shape,fpn_outputs['2'].shape)
        # the output of a group of fpn feature: 
        # [('0', torch.Size([1, 256, 128, 128])), 
        #  ('1', torch.Size([1, 256, 64, 64])), 
        #  ('2', torch.Size([1, 256, 32, 32])), 
        #  ('3', torch.Size([1, 256, 16, 16]))]
        layer1 = self.layer1_bn_relu(fpn_outputs['0'])
        layer2 = self.layer2_bn_relu(fpn_outputs['1'])
        layer3 = self.layer3_bn_relu(fpn_outputs['2'])
        layer4 = self.layer4_bn_relu(fpn_outputs['3'])

        fusion4_3 = self.smooth1(self._upsample_add(layer4, layer3))
        fusion4_2 = self.smooth2(self._upsample_add(fusion4_3, layer2))
        fusion4_1 = self.smooth3(self._upsample_add(fusion4_2, layer1))

        fusion4_2 = self._upsample(fusion4_2, fusion4_1)
        fusion4_3 = self._upsample(fusion4_3, fusion4_1)
        layer4 = self._upsample(layer4, fusion4_1)
        #fusion4_3 = self._upsample(fusion4_3, fusion4_2)
        #layer4 = self._upsample(layer4, fusion4_2)

        inter_feat = torch.cat((fusion4_1, fusion4_2, fusion4_3, layer4), 1) # [N, 1024, H, W]
        inter_feat = self._upsample(inter_feat, x) # [N, 1024, x_h, x_w]
        #inter_feat = torch.cat((fusion4_2, fusion4_3, layer4), 1) # [N, 1024, H, W]
        #inter_feat = self._upsample(inter_feat, x) # [N, 1024, x_h, x_w]

        return inter_feat

class Cell_Bbox_Seg(nn.Module):
    def __init__(self, in_channels = 1024, num_classes=3):
        super(Cell_Bbox_Seg, self).__init__()

        self.decode_out = nn.Sequential(
                        nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
        
        self.row_out = nn.Sequential(
                        nn.Conv2d(256, 64, kernel_size=(3,1), stride=1, padding=(1,0)),
                        #nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=(3,1), stride=1, padding=(1,0)),
                        #nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
                        #nn.BatchNorm2d(num_classes),
                        nn.LeakyReLU(inplace=True)
                    )

        self.col_out = nn.Sequential(
                        nn.Conv2d(256, 64, kernel_size=(1,3), stride=1, padding=(0,1)),
                        #nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=(1,3), stride=1, padding=(0,1)),
                        #nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
                        #nn.BatchNorm2d(num_classes),
                        nn.LeakyReLU(inplace=True)
                    )
        
        self.twodim_out = nn.Sequential(
                        nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
                        #nn.BatchNorm2d(num_classes),
                        nn.LeakyReLU(inplace=True)
                    )
        
        self.fusion = nn.Sequential(
                        nn.Conv2d(num_classes*3, num_classes, kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(inplace=True)
                    )
        

    def postprocess(self, row_pred, col_pred, seg_pred, table_names=None):
        #pred_mat = torch.argmax(row_pred,dim=1) * torch.argmax(col_pred,dim=1)
        pred_mat = torch.argmax(seg_pred,dim=1)
        pred_mat = pred_mat.data.cpu().int().numpy()
        pred_mat[np.where(pred_mat>2)] = 2
        pred_mask = np.where(pred_mat == 1, 255, 0).astype('uint8')
        #self.vis_seg(pred_mask, table_names, '/data/xuewenyuan/dev/tablerec/results/delet_vis')
        N, H, W = pred_mask.shape
        batch_bboxes = []
        for ind in range(N):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pred_mask[ind] = cv2.morphologyEx(pred_mask[ind], cv2.MORPH_OPEN, kernel)
            contours = cv2.findContours(pred_mask[ind].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            bboxes = [[ct[:,:,0].min()-2, ct[:,:,1].min()-2, ct[:,:,0].max()+2, ct[:,:,1].max()+2] for ct in contours]
            bboxes = torch.as_tensor(bboxes).to(torch.float32)
            batch_bboxes.append(bboxes)
        return batch_bboxes

    def vis_seg(self, label_mat, table_names, vis_path):
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        batch_size = len(table_names)
        for ind in range(batch_size):
            vis_mat = np.zeros((label_mat[ind].shape[0],label_mat[ind].shape[1],3),dtype=np.int32)
            vis_mat[np.where(label_mat[ind] == 0)] = np.array([255,0,0],dtype=np.int32)
            vis_mat[np.where(label_mat[ind] == 1)] = np.array([0,255,0],dtype=np.int32)
            vis_mat[np.where(label_mat[ind] == 2)] = np.array([0,0,255],dtype=np.int32)
            cv2.imwrite(os.path.join(vis_path,table_names[ind]+'_pred.png'), vis_mat.astype('uint8'))

    def forward(self, input):
        
        decode_feat = self.decode_out(input)
        #decode_feat = nn.functional.interpolate(decode_feat, size=(src_img_shape[2], src_img_shape[3]), mode='bilinear', align_corners=False)

        seg_pred = self.twodim_out(decode_feat)

        row_pred = self.row_out(torch.mean(decode_feat, 3, True))
        col_pred = self.col_out(torch.mean(decode_feat, 2, True))

        row_expand = torch.repeat_interleave(row_pred, input.shape[3], dim = 3)
        col_expand = torch.repeat_interleave(col_pred, input.shape[2], dim = 2)

        seg_pred = self.fusion(torch.cat((seg_pred,row_expand,col_expand),1))

        #det_bboxes = self.postprocess(row_pred, col_pred, None)
        det_bboxes = self.postprocess(None, None, seg_pred)

        return row_pred, col_pred, seg_pred, det_bboxes

class Cell_Lloc_Pre(nn.Module):
    def __init__(self, rows_classes, cols_classes, img_h, img_w, alpha, device, 
                in_channels = 1024, cnn_emb_feat = 512, box_emb_feat = 256, gcn_out_feat = 512,
                cell_iou_thresh = 0.5, min_cells_percent = 1.0):
        super(Cell_Lloc_Pre, self).__init__()

        self.cell_iou_thresh = cell_iou_thresh
        self.min_cells_percent = min_cells_percent
        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.rows_classes = rows_classes
        self.cols_classes = cols_classes
        self.alpha = alpha

        self.decode_out = nn.Sequential(
                        nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(256,affine=False),
                        nn.ReLU(inplace=True)
                    )
        self.cnn_emb = nn.Sequential(
                        nn.Linear(256*2*2, cnn_emb_feat),
                        #nn.BatchNorm1d(cnn_emb_feat,affine=False),
                        nn.ReLU(inplace=True)
                    )
        self.box_emb = nn.Sequential(
                        nn.Linear(4, box_emb_feat),
                        #nn.BatchNorm1d(box_emb_feat,affine=False),
                        nn.ReLU(inplace=True)
                    )

        self.gconv_row = GCNConv(cnn_emb_feat+box_emb_feat, gcn_out_feat)
        self.gconv_col = GCNConv(cnn_emb_feat+box_emb_feat, gcn_out_feat)

        self.row_cls = nn.Sequential(
                        nn.Linear(gcn_out_feat, 2*(rows_classes-1)*2),
                        #nn.BatchNorm1d(2*(rows_classes-1)*2,affine=False),
                        nn.LeakyReLU(inplace=True)
                        )

        self.col_cls = nn.Sequential(
                        nn.Linear(gcn_out_feat, 2*(cols_classes-1)*2),
                        #nn.BatchNorm1d(2*(cols_classes-1)*2,affine=False),
                        nn.LeakyReLU(inplace=True)
                        )

    def get_box_feat(self, cell_boxes):
        # roi_bboxes: List(Tensor(x1,y1,x2,y2)) 
        # image_shapes: [N,C,H,W]
        boxes = torch.cat(cell_boxes, dim=0)
        box_w = boxes[:,2]-boxes[:,0]
        box_h = boxes[:,3]-boxes[:,1]
        ctr_x = (boxes[:,2]+boxes[:,0])/2
        ctr_y = (boxes[:,3]+boxes[:,1])/2
        #rel_x = torch.log(ctr_x/self.img_w)
        #rel_y = torch.log(ctr_y/self.img_h)
        #rel_w = torch.log(box_w/self.img_w)
        #rel_h = torch.log(box_h/self.img_h)
        rel_x = ctr_x/self.img_w
        rel_y = ctr_y/self.img_h
        rel_w = box_w/self.img_w
        rel_h = box_h/self.img_h
        boxes_feat = torch.stack((rel_x,rel_y,rel_w,rel_h),dim=1)
        return boxes_feat

    def edge_weight(self, edge_ind, cell_boxes, im_scale, pdl, pdt):
        assert cell_boxes.size(1) == 4
        assert edge_ind.size(1) == 2
        org_box = (cell_boxes - torch.stack((pdl,pdt)*2))/im_scale
        centr_x1 = (org_box[edge_ind[:,0],0] + org_box[edge_ind[:,0],2]) / 2
        centr_y1 = (org_box[edge_ind[:,0],1] + org_box[edge_ind[:,0],3]) / 2
        centr_x2 = (org_box[edge_ind[:,1],0] + org_box[edge_ind[:,1],2]) / 2
        centr_y2 = (org_box[edge_ind[:,1],1] + org_box[edge_ind[:,1],3]) / 2

        tb_w = org_box[:,[0,2]].max()
        tb_h = org_box[:,[1,3]].max()

        row_attr = torch.exp(-(torch.square((centr_y1-centr_y2)*self.alpha/tb_h)))
        col_attr = torch.exp(-(torch.square((centr_x1-centr_x2)*self.alpha/tb_w)))
        
        row_pres = torch.sort(row_attr,descending=True)[1][:8*cell_boxes.size(0)]
        col_pres = torch.sort(col_attr,descending=True)[1][:8*cell_boxes.size(0)]
        
        return row_attr[row_pres], col_attr[col_pres], edge_ind[row_pres], edge_ind[col_pres]

    def build_graph(self, cell_boxes, im_scales, pdls, pdts):
        #device = roi_bboxes[0].device
        num_images = len(cell_boxes)
        graphs = []
        for img_id in range(num_images):
            edge_ind  = []
            num_nodes = cell_boxes[img_id].shape[0]
            for n1 in range(num_nodes):
                for n2 in range(num_nodes):
                    if n1 == n2: continue
                    edge_ind.append([n1,n2])
            edge_ind = torch.as_tensor(edge_ind, dtype=torch.int64)
            #print(edge_ind.t())
            #edge_attr = self.edge_weight(edge_ind,cell_boxes[img_id], im_scales[img_id], pdls[img_id], pdts[img_id])
            #row_attr, col_attr = self.edge_weight(edge_ind,cell_boxes[img_id], im_scales[img_id], pdls[img_id], pdts[img_id])
            row_attr, col_attr, row_edge, col_edge = self.edge_weight(edge_ind,cell_boxes[img_id], im_scales[img_id], pdls[img_id], pdts[img_id])
            tb_graph = GraphData(edge_index=edge_ind.t(), num_nodes = num_nodes)
            tb_graph.row_attr = row_attr
            tb_graph.col_attr = col_attr
            tb_graph.row_edge = row_edge.t()
            tb_graph.col_edge = col_edge.t()
            graphs.append(tb_graph)
        graphs = GraphBatch.from_data_list(graphs).to(self.device)
        #print('graph')
        #print(graphs.edge_index, graphs.edge_attr)
        return graphs

    def filter_box(self, pred_boxes, gt_boxes):
        batch_size = len(gt_boxes)
        train_boxes = []
        train_inds = []
        count = 0
        for b_ind in range(batch_size):
            if pred_boxes[b_ind].size(0) != 0:
                match_quality_matrix = box_ops.box_iou(pred_boxes[b_ind], gt_boxes[b_ind])
                # find best pred candidate for each gt
                matched_val, matched_ind = match_quality_matrix.max(dim=0)
                rm_gts = torch.where(matched_val>self.cell_iou_thresh)[0]
            else:
                rm_gts = torch.Tensor([])
            res_ind = torch.as_tensor([ i for i in range(gt_boxes[b_ind].size(0)) if (i not in rm_gts)], dtype=torch.int32)
            #res_gt_boxes = gt_boxes[b_ind][res_ind]

            num_preserved = ((torch.rand((1,))+self.min_cells_percent*10)/10*gt_boxes[b_ind].shape[0]).to(torch.int32) # [0.9 ~ 1)
            num_preserved = max(num_preserved - rm_gts.shape[0], 0)
            preserved_ind = torch.randperm(len(res_ind))[:num_preserved]

            #pred_ind = matches[rm_gts]

            #select_boxes = torch.cat((res_gt_boxes[preserved_ind],pred_boxes[b_ind][pred_ind]), dim=0)
            #train_boxes.append(select_boxes)
            boxes = []
            for box_i in range(gt_boxes[b_ind].size(0)):
                
                if box_i in res_ind[preserved_ind]:
                    boxes.append(gt_boxes[b_ind][box_i])
                    train_inds.append(count+box_i)
                elif box_i in rm_gts:
                    pred_ind = matched_ind[box_i]
                    boxes.append(pred_boxes[b_ind][pred_ind])
                    train_inds.append(count+box_i)
                
            train_boxes.append(torch.stack(boxes,dim=0))
            count += gt_boxes[b_ind].size(0)

        return train_boxes, train_inds

    def forward(self, input, pred_cell_boxes, im_scales, pdls, pdts, gt_cell_boxes=None):
        train_inds = None
        if (gt_cell_boxes is not None) and (pred_cell_boxes is not None):
            cell_boxes, train_inds = self.filter_box(pred_cell_boxes, gt_cell_boxes)
        elif (pred_cell_boxes is None) and (gt_cell_boxes is not None): 
            cell_boxes = []
            for img_boxes in gt_cell_boxes:
                num_node = img_boxes.size(0)
                if num_node < 2:
                    cell_boxes.append(torch.cat((img_boxes, torch.as_tensor([[0,0,0,0]]*(2-num_node)).to(torch.float32).to(img_boxes.device)),0))
                else:
                    cell_boxes.append(img_boxes)
        elif (gt_cell_boxes is None) and (pred_cell_boxes is not None):
            cell_boxes = []
            for img_boxes in pred_cell_boxes:
                num_node = img_boxes.size(0)
                if num_node < 2:
                    cell_boxes.append(torch.cat((img_boxes, torch.as_tensor([[0,0,0,0]]*(2-num_node)).to(torch.float32).to(img_boxes.device)),0))
                else:
                    cell_boxes.append(img_boxes)
        
        box_feat = self.get_box_feat(cell_boxes)
        box_feat = self.box_emb(box_feat).to(self.device)

        decode_feat = self.decode_out(input)
        bbox_count = [i.shape[0] for i in cell_boxes]
        cnn_feat = ops.roi_align(decode_feat, cell_boxes, 2) #[num_node, 256, 2, 2]
        cnn_feat = self.cnn_emb(cnn_feat.view(cnn_feat.size(0), -1))

        graphs = self.build_graph(cell_boxes, im_scales, pdls, pdts)
        fusion_feat = torch.cat([box_feat, cnn_feat], dim=1)

        row_feat = self.gconv_row(fusion_feat, graphs.row_edge, graphs.row_attr)
        row_feat = F.relu(row_feat)
        col_feat = self.gconv_col(fusion_feat, graphs.col_edge, graphs.col_attr)
        col_feat = F.relu(col_feat)

        cls_row_score = self.row_cls(row_feat)
        cls_col_score = self.col_cls(col_feat)

        #cls_row_score = torch.reshape(cls_row_score, (cls_row_score.size(0), self.rows_classes, 2))
        #cls_col_score = torch.reshape(cls_col_score, (cls_col_score.size(0), self.cols_classes, 2))
        cls_row_score = torch.reshape(cls_row_score, (cls_row_score.size(0), 2, self.rows_classes-1, 2))
        cls_col_score = torch.reshape(cls_col_score, (cls_col_score.size(0), 2, self.cols_classes-1, 2))

        return cls_row_score, cls_col_score, train_inds 


        









