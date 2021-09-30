
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
import torch
import pandas as pd
from tqdm import tqdm
from util import util
#from util.evaluator import IC15Evaluator
from util.evaluator_vis import IC15Evaluator



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    util.init_distributed_mode(opt)
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    evaluator = IC15Evaluator(opt)
    test_size = len(dataset)
    print('The number of test images = %d. Testset: %s' % (test_size, opt.dataroot))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    #save_dir = os.path.join(os.getcwd(), opt.results_dir, opt.name, opt.dataroot.split('/')[-1], '%s_%s' % (opt.phase, opt.epoch)) 
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    model.eval()
    evaluator.reset()
    eval_start_time = time.time()
    for data in tqdm(dataset):
        torch.cuda.synchronize()
        model.set_input(data)
        preds = model.test()
        evaluator.update(preds)
    eval_time = time.time() - eval_start_time
    res = '==>Evaluation time: {:.0f}, \n'.format(eval_time)
    metric, select_score = evaluator.summary(select_iou = 0.5)
    res += metric
    print(res)