import argparse
import os
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import json
from tqdm import tqdm
from data import HSTrainingData
from data import HSTestData
from network.WWD import *
from common import *
from metrics import compare_mpsnr
# loss
from loss import HLoss
# from loss import HyLapLoss
from metrics import quality_assessment
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import scipy.io as sio

 
# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = ''


def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 64")
    train_parser.add_argument("--epochs", type=int, default=800, help="epochs, default set to 20")
    train_parser.add_argument("--n_feats", type=int, default=180, help="n_feats, default set to 256")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="WWD", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument('--la1', type=float, default=0.3, help="")
    train_parser.add_argument('--la2', type=float, default=0.1, help="")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids (default: 7)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids (default: 7)")
    test_parser.add_argument("--dataset_name", type=str, default="Chikusei",help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--model_title", type=str, default="WWD",help="model_title, default set to model_title")
    test_parser.add_argument("--n_feats", type=int, default=180, help="n_feats, default set to 256")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")

    args = main_parser.parse_args()
    # print(args)
    print('===>GPU:',args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass


def train(args):
    traintime = str(time.ctime())
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    train_path = ''+args.dataset_name+'_x'+str(args.n_scale)+'/trains/'
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale)+'/'
    test_data_dir = ''+args.dataset_name+'_x'+str(args.n_scale)+'/'+args.dataset_name+'_test.mat'

    train_set = HSTrainingData(image_dir=train_path, augment=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True,pin_memory=True)
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,pin_memory=True)

    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    elif args.dataset_name=='Houston':
        colors = 48
    else:
        colors = 128

    print('===> Building model:{}'.format(args.model_title))
    net = WWD(inp_channels=colors, dim=args.n_feats, depths=[4,4,4,4], num_heads=[6,6,6,6],mlp_ratio=2, scale=args.n_scale)
    # print(net)
    model_title = args.dataset_name + "_" + args.model_title+'_x'+ str(args.n_scale)

    args.model_title = model_title

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0

    if resume:
        model_name = './checkpoints/' + model_title + "_ckpt_epoch_" + str(300) + ".pth"
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()
    print_network(net)
    # loss functions to choose
    h_loss = HLoss(args.la1,args.la2)


    print("===> Setting optimizer and logger")
    # add adam optimizer
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('runs/'+model_title+'_'+traintime)


    best_psnr = 0
    best_epoch = 0

    print('===> Start training')


    for e in range(start_epoch, args.epochs):
        psnr = []
        adjust_learning_rate(args.learning_rate, optimizer, e+1)
        epoch_meter.reset()
        net.train()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        
        for iteration, (x, lms, gt) in enumerate(tqdm(train_loader, leave=False)):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            psnr = []
            optimizer.zero_grad()
            y = net(x, lms)
            loss = h_loss(y, gt)
            epoch_meter.add(loss.item())
            loss.backward()
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} \tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), e + 1, iteration + 1, len(train_loader)-1, loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)
                
        print("Running testset")
        net.eval()
        with torch.no_grad():
            output = []
            test_number = 0
            for i, (ms, lms, gt) in enumerate(test_loader):
                ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
                y = net(ms, lms)
                y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
                y = y[:gt.shape[0], :gt.shape[1], :]
                psnr_value = compare_mpsnr(gt, y, data_range=1.)
                psnr.append(psnr_value)
                output.append(y)
                test_number += 1

        avg_psnr = sum(psnr) / test_number
        if avg_psnr >best_psnr:
            best_psnr = avg_psnr
            best_epoch = e+1
            save_checkpoint(args, net, e + 1, traintime)
            # print("alpha:{}  beta:{}  gamma:{}", net.module.alpha.item(),net.module.beta.item(),net.module.gamma.item())
        writer.add_scalar('scalar/test_psnr', avg_psnr, e + 1)

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f} PSNR:{:.3f} best_psnr:{:.3f} best_epoch:{}".format(
            time.ctime(), e+1, epoch_meter.value()[0], avg_psnr, best_psnr, best_epoch))
        # run validation set every epoch
        # eval_loss = validate(args, eval_loader, net, L1_loss)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        # writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 5 == 0:
            save_checkpoint(args, net, e+1, traintime)

    ## Save the testing results

    print('===> Start testing')
    model_name = './checkpoints/' + traintime +'/' + args.model_title + "_ckpt_epoch_" + str(best_epoch) + ".pth"
    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        net = WWD(inp_channels=colors, dim=args.n_feats, depths=[4, 4, 4, 4],
                       num_heads=[6, 6, 6, 6], mlp_ratio=2, scale=args.n_scale)
        net.to(device).eval()
        state_dict = torch.load(model_name)
        net.load_state_dict(state_dict['model'])

        output = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.\
                to(device)
            # y = model(ms)
            y = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:]
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number


    save_dir = result_path + model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    save_dir = result_path + model_title + '.mat'
    sio.savemat(save_dir, {'gt': output})
    print("Test finished, test results saved to .mat file at ", save_dir)
    print(indices)
    QIstr = model_title+'_'+str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 150 epochs"""
    lr = start_lr * (0.5 ** (epoch // 150))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = model(ms, lms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())

    # back to training mode
    model.train()
    return epoch_meter.value()[0]


def test(args):
    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    elif args.dataset_name=='Houston':
        colors = 48
    else:
        colors = 128
    test_data_dir = '' + args.dataset_name + '_x' + str(args.n_scale) + '/' + args.dataset_name + '_test.mat'
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale) + '/'
    model_title = args.model_title+'_x' + str(args.n_scale)
    model_name = './checkpoints/' + 'Thu Nov 14 14:30:23 2024/' + args.dataset_name +"_" + model_title   + "_ckpt_epoch_" + str(207) + ".pth"
    # model_name = './model/' +args.dataset_name +  model_title + ".pth"
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')

    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')

    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        net = WWD(inp_channels=colors, dim=args.n_feats, depths=[4, 4, 4, 4],
                       num_heads=[6, 6, 6, 6], mlp_ratio=2, scale=args.n_scale)
        net.to(device).eval()
        state_dict = torch.load(model_name)
        net.load_state_dict(state_dict['model'])

        output = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.\
                to(device)
            # y = model(ms)
            y = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:]
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    #save_dir = "./test.npy"
    save_dir = result_path + model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    save_dir = result_path + model_title + '.mat'
    sio.savemat(save_dir, {'gt': output})
    print("Test finished, test results saved to .mat file at ", save_dir)
    print(indices)
    QIstr = model_title+'_'+str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))

def save_checkpoint(args, model, epoch, traintime):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'+traintime+'/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)

    if torch.cuda.device_count() > 1:
        state = {"epoch": epoch, "model": model.module.state_dict()}
    else:
        state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


if __name__ == "__main__":
    #print(torch.cuda.current_device())
    main()
