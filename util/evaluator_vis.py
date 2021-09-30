import numpy as np
import Polygon as plg
import os
import pickle
from PIL import Image
import cv2
from . import util

class IC15Evaluator(object):
    def __init__(self, opt, LTRB = True):
        self.data_list = open(os.path.join(opt.dataroot, opt.phase+'.txt'), 'r', encoding='UTF-8').readlines()
        #self.data_list = open(os.path.join('/data/xuewenyuan/dataset/TABLE2LATEX-450K/', 'test.txt'), 'r', encoding='UTF-8').readlines()
        self.re_h = opt.load_height
        self.re_w = opt.load_width
        if opt.phase == 'test' and opt.max_test_size != float("inf"):
            self.data_list = self.data_list[0:opt.max_test_size]
        if opt.phase == 'val' and opt.max_val_size != float("inf"):
            self.data_list = self.data_list[0:opt.max_val_size]
        self.gt_dict, self.scale_dict, self.gt_box_sum, self.masks = self.create_gt_dict(self.data_list)
        self.gtImg = len(self.gt_dict.keys())
        self.IOU_CONSTRAINT = [0.5]

    def reset(self):
        self.predImg = 0.0
        self.all_pred_name = []

        self.local_matchedSum     = [0.0] * len(self.IOU_CONSTRAINT)
        self.local_pred_box_sum   =  0.0
        self.local_pred_rowSt_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.local_pred_rowEd_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.local_pred_colSt_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.local_pred_colEd_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.local_pred_lloc_sum  = [0.0] * len(self.IOU_CONSTRAINT)

        self.all_matchedSum     = [0.0] * len(self.IOU_CONSTRAINT)
        self.all_pred_box_sum   =  0.0
        self.all_pred_rowSt_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.all_pred_rowEd_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.all_pred_colSt_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.all_pred_colEd_sum = [0.0] * len(self.IOU_CONSTRAINT)
        self.all_pred_lloc_sum  = [0.0] * len(self.IOU_CONSTRAINT)

    def summary(self, select_iou = None):
        metric = ''
        select_metric = 0.0
        #print(util.all_gather(self.predImg), self.gtImg)
        self.all_matchedSum =  np.array(util.all_gather(self.local_matchedSum)).sum(0)
        self.all_pred_box_sum  =  sum(util.all_gather(self.local_pred_box_sum))
        self.all_pred_rowSt_sum = np.array(util.all_gather(self.local_pred_rowSt_sum)).sum(0)
        self.all_pred_rowEd_sum = np.array(util.all_gather(self.local_pred_rowEd_sum)).sum(0)
        self.all_pred_colSt_sum = np.array(util.all_gather(self.local_pred_colSt_sum)).sum(0)
        self.all_pred_colEd_sum = np.array(util.all_gather(self.local_pred_colEd_sum)).sum(0)
        self.all_pred_lloc_sum = np.array(util.all_gather(self.local_pred_lloc_sum)).sum(0)
        for ind_iou in range(len(self.IOU_CONSTRAINT)):
            # bbox metric
            recall = self.all_matchedSum[ind_iou]/self.gt_box_sum if self.gt_box_sum != 0 else 0
            precision = self.all_matchedSum[ind_iou]/self.all_pred_box_sum if self.all_pred_box_sum != 0 else 0
            hmean = 0 if recall + precision==0 else 2*recall*precision/(recall+precision)
            # lloc metric
            acc_rowSt = self.all_pred_rowSt_sum[ind_iou]/self.all_matchedSum[ind_iou] if self.all_matchedSum[ind_iou] != 0 else 0
            acc_rowEd = self.all_pred_rowEd_sum[ind_iou]/self.all_matchedSum[ind_iou] if self.all_matchedSum[ind_iou] != 0 else 0
            acc_colSt = self.all_pred_colSt_sum[ind_iou]/self.all_matchedSum[ind_iou] if self.all_matchedSum[ind_iou] != 0 else 0
            acc_colEd = self.all_pred_colEd_sum[ind_iou]/self.all_matchedSum[ind_iou] if self.all_matchedSum[ind_iou] != 0 else 0
            acc_lloc  = self.all_pred_lloc_sum[ind_iou]/self.all_matchedSum[ind_iou] if self.all_matchedSum[ind_iou] != 0 else 0
            
            if select_iou == self.IOU_CONSTRAINT[ind_iou]:
                select_metric = (1+0.5*0.5)*(hmean * acc_lloc)/(0.5*0.5*hmean+acc_lloc) if hmean+acc_lloc != 0 else 0

            metric += 'IoU: {:.3f}, Sloc, Recall: {:.4f}, Precision: {:.4f}, Hmean(F1): {:.4f}\n'.format(self.IOU_CONSTRAINT[ind_iou], recall, precision, hmean)
            metric += 'IoU: {:.3f}, Lloc, Acc_all: {:.4f}, Acc_rowSt: {:.4f}, Acc_rowEd: {:.4f}, Acc_colSt: {:.4f}, Acc_colEd: {:.4f}\n'.format(\
                        self.IOU_CONSTRAINT[ind_iou], acc_lloc, acc_rowSt, acc_rowEd, acc_colSt, acc_colEd)
            metric += 'IoU: {:.3f}, F(beta=0.5): {:.3f}'.format(self.IOU_CONSTRAINT[ind_iou], select_metric)
            if ind_iou != len(self.IOU_CONSTRAINT)-1:
                metric += '\n'
            
        return metric, select_metric

    def update(self, preds):
        # pred: {'table_name':{'bbox':Array[L,4],'lloc':Array[L, 4]},...}
        for tb_name in preds.keys():
            if tb_name in self.all_pred_name:
                continue
            else:
                self.predImg += 1
                self.all_pred_name += util.all_gather([tb_name])

            pdl, pdt, scale_w, scale_h = self.scale_dict[tb_name]
            pred_bbox = [((bbox+([-pdl,-pdt])*2)*([scale_w, scale_h]*2)).astype(np.int32) for bbox in preds[tb_name]['bbox']]
            #im_scale, pdl, pdt = self.scale_dict[tb_name]
            #pred_bbox = [((bbox+([-pdl,-pdt])*2)/im_scale).astype(np.int32) for bbox in preds[tb_name]['bbox']]
            #pred_bbox = preds[tb_name]['bbox']

            len_pred = len(pred_bbox)
            len_gt   = len(self.gt_dict[tb_name]['bbox'])
            vis_flag = np.zeros(len_pred, dtype=np.uint8)
            self.local_pred_box_sum += len_pred
            iouMat = np.empty((len_gt,len_pred))
            gt_flag = np.zeros((len(self.IOU_CONSTRAINT),len_gt),np.int32)
            pred_flag = np.zeros((len(self.IOU_CONSTRAINT),len_pred),np.int32)
            for ind_gt in range(len_gt):
                for ind_pred in range(len_pred):
                    gt_plg = self.gt_dict[tb_name]['bbox'][ind_gt]
                    pred_plg = self.rectangle_to_polygon(pred_bbox[ind_pred])
                    iouMat[ind_gt,ind_pred] = self.calc_iou(pred_plg, gt_plg)
            
            for ind_gt in range(len_gt):
                for ind_pred in range(len_pred):
                    for ind_iou in range(len(self.IOU_CONSTRAINT)):
                        if gt_flag[ind_iou,ind_gt] == 0 and pred_flag[ind_iou,ind_pred] == 0 \
                            and iouMat[ind_gt,ind_pred] > self.IOU_CONSTRAINT[ind_iou]:
                            gt_flag[ind_iou, ind_gt] = 1
                            pred_flag[ind_iou, ind_pred] = 1
                            self.local_matchedSum[ind_iou] += 1
                            vis_flag[ind_pred] = 1
                            
                            if 'lloc' not in preds[tb_name].keys():
                                continue
                            if preds[tb_name]['lloc'][ind_pred, 0] == self.gt_dict[tb_name]['lloc'][ind_gt, 0]:
                                self.local_pred_rowSt_sum[ind_iou] += 1
                            if preds[tb_name]['lloc'][ind_pred, 1] == self.gt_dict[tb_name]['lloc'][ind_gt, 1]:
                                self.local_pred_rowEd_sum[ind_iou] += 1
                            if preds[tb_name]['lloc'][ind_pred, 2] == self.gt_dict[tb_name]['lloc'][ind_gt, 2]:
                                self.local_pred_colSt_sum[ind_iou] += 1
                            if preds[tb_name]['lloc'][ind_pred, 3] == self.gt_dict[tb_name]['lloc'][ind_gt, 3]:
                                self.local_pred_colEd_sum[ind_iou] += 1
                            if (preds[tb_name]['lloc'][ind_pred] == self.gt_dict[tb_name]['lloc'][ind_gt]).all():
                                self.local_pred_lloc_sum[ind_iou] += 1
                                vis_flag[ind_pred] = 2
            self.vis_mask(tb_name, pred_bbox, vis_flag)
                            
    def create_gt_dict(self, data_list):
        gt_dict = {} #'img_name': bboxes[]
        scale_dict = {}
        masks = {}
        gt_size = len(data_list)
        gt_box_sum = 0.0
        for ind in range(gt_size):
            table_pkl = data_list[ind].strip().split()[0]
            table_anno = pickle.load(open(table_pkl, 'rb'))
            table_name = os.path.split(table_pkl)[1].replace('.pkl','')
            table_img  = Image.open(table_anno['image_path']).convert("RGB")
            table_cv2 = cv2.imread(table_anno['image_path'])
            masks[table_name] = [table_cv2, np.zeros((table_cv2.shape), dtype=np.uint8), np.zeros((table_cv2.shape), dtype=np.uint8)]
            scale_dict[table_name] = self.calc_scale(table_img)
            #cells_bbox = [ self.rectangle_to_polygon(cell_i['bbox'], table_name) for cell_i in table_anno['cells_anno']]
            cells_bbox = []
            for cell_i in table_anno['cells_anno']:
                if 'ps' in cell_i:
                    cells_bbox.append(self.points_to_polygon(cell_i['ps']))
                else:
                    cells_bbox.append(self.rectangle_to_polygon(cell_i['bbox']))
            cells_lloc = np.array([ cell_i['lloc'] for cell_i in table_anno['cells_anno']], dtype=np.int32)
            gt_dict[table_name] = {'bbox': cells_bbox, 'lloc': cells_lloc}
            gt_box_sum += len(cells_bbox)
        return gt_dict, scale_dict, gt_box_sum, masks
        

    def rectangle_to_polygon(self, rect, table_name=None):
        # rect: [x1,y1,x2,y2]
        assert len(rect) == 4 or len(rect) == 8, 'The bbox should have four or eight points. {:.0f} points got'.format(len(rect))
        if len(rect) == 4:
            rect_8pts = np.array([rect[0],rect[1],rect[2],rect[1],rect[2],rect[3],rect[0],rect[3]],dtype=np.int32)
        else:
            rect_8pts = np.array(rect, dtype=np.int32)
        #if table_name == 'cTDaR_t00080_0':
        #    print(rect_8pts)
        rect_8pts = rect_8pts.reshape(4,2)
        return plg.Polygon(rect_8pts)

    def points_to_polygon(self, pts):
        return plg.Polygon(np.array(pts, dtype=np.int32))

    def calc_iou(self, pred, target):
        inter = pred & target
        inter_area = 0 if len(inter) == 0 else inter.area()
        union = pred.area() + target.area() - inter_area
        try:
            return inter_area / union
        except:
            return 0
    
    def calc_scale(self, table_image):
        tb_w, tb_h = table_image.size
        rew = self.re_w if tb_w>tb_h else int(self.re_h*tb_w/tb_h)
        reh = self.re_h if tb_h>tb_w else int(self.re_w*tb_h/tb_w)
        if rew>reh:
            pdl, pdt, pdr, pdd = (0, (self.re_h-reh)//2, 0, self.re_h-reh-(self.re_h-reh)//2) 
        else:
            pdl, pdt, pdr, pdd = ((self.re_w-rew)//2, 0, self.re_w-rew-(self.re_w-rew)//2, 0)
        return (pdl, pdt, tb_w*1.0/rew, tb_h*1.0/reh)
    
    '''
    def calc_scale(self, table_image):
        tb_w, tb_h = table_image.size

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
        return (im_scale, pdl, pdt)
    '''

    def vis_mask(self, table_name, pred_bbox, flags):
        for i, bi in enumerate(pred_bbox):
            if flags[i] == 0:
                self.masks[table_name][1] = cv2.rectangle(self.masks[table_name][1], (bi[0]+8,bi[1]+8), (bi[2]-8,bi[3]-8), color=(0,0,255), thickness=-1)
            elif flags[i] == 1:
                self.masks[table_name][1] = cv2.rectangle(self.masks[table_name][1], (bi[0]+8,bi[1]+8), (bi[2]-8,bi[3]-8), color=(0,255,0), thickness=-1)
            else:
                self.masks[table_name][1] = cv2.rectangle(self.masks[table_name][1], (bi[0]+8,bi[1]+8), (bi[2]-8,bi[3]-8), color=(0,255,0), thickness=-1)
                self.masks[table_name][1] = cv2.rectangle(self.masks[table_name][1], (bi[0]+8,bi[1]+8), (bi[2]-8,bi[3]-8), color=(255,0,0), thickness=4)
        alpha = 0.4
        beta = 0.7
        gamma = 0
        mask_img = cv2.addWeighted(self.masks[table_name][0], alpha, self.masks[table_name][1], beta, gamma)
        cv2.imwrite('./temp/'+table_name+'.png', mask_img)