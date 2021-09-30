import torch
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.evaluator import IC15Evaluator
from util import util

def eval(opt, dataset, model, evaluator = None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model.eval()
    evaluator.reset()
    eval_start_time = time.time()
    for i, data in enumerate(dataset):
        torch.cuda.synchronize()
        model.set_input(data)
        preds = model.test()
        evaluator.update(preds)
    eval_time = time.time() - eval_start_time
    res = '==>Evaluation time: {:.0f}, \n'.format(eval_time)
    metric, select_score = evaluator.summary(select_iou = 0.5)
    res += metric
    torch.set_num_threads(n_threads)
    return res, select_score

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    util.init_distributed_mode(opt)
    torch.manual_seed(10)
    if opt.device == 'cuda':
        torch.cuda.manual_seed(10)

    # train dataset
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d. Trainset: %s' % (train_size, opt.dataroot))

    # val dataset
    opt.phase    = 'val'
    val_dataset = create_dataset(opt)
    val_evaluator = IC15Evaluator(opt)
    val_size    = len(val_dataset)
    print('The number of test images = %d. Valset: %s' % (val_size, opt.dataroot))
    opt.phase = 'train'

    # test dataset
    #opt.phase    = 'test'
    #test_dataset = create_dataset(opt)
    #test_evaluator = IC15Evaluator(opt)
    #test_size    = len(test_dataset)
    #print('The number of test images = %d. Testset: %s' % (test_size, opt.dataroot))
    #opt.phase = 'train'

    model = create_model(opt)      
    model.setup(opt)               
    visualizer = Visualizer(opt)   
    total_iters = 0                
    best_score  = 0.0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): 
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0            

        if opt.distributed:
            train_dataset.set_epoch(epoch)      

        for i, data in enumerate(train_dataset): 
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size*opt.world_size
            epoch_iter += opt.batch_size*opt.world_size
            model.set_input(data)        
            model.optimize_parameters() 

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                val_res, metric_score = eval(opt, val_dataset, model, val_evaluator) 
                visualizer.print_current_val(epoch, epoch_iter, val_res)
                if metric_score > best_score:
                    best_score = metric_score
                    print('saving the best model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    model.save_networks('best')
                    #model.metric = best_acc
                best_res = 'current avg score: {:.4f}, best score: {:.6f}'.format(metric_score, best_score)
                visualizer.print_current_val(epoch, epoch_iter, best_res)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()     


    