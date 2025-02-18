import os
import sys
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.ContrastiveLoss import ContrastiveLoss
from utils.ContrastiveLoss import TensorBuffer
from models.vnet import VNet,Projector_v
from models.ResNet34 import Resnet34
from utils import ramps, losses
from dataloaders.LA_Data import LA_Heart_Dataset, RandomCrop, ToTensor, TwoStreamBatchSampler, RandomRotFlip

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='../dataset/LA/LA_Data', help='Name of Experiment')
parser.add_argument('--root_path', type=str, default='dataset/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str, default="C3S3", help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='4', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')

args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = "checkpoints/" + args.exp + "/"

# torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
T = 0.1

# 0: vnet 1:resnet
Good_student = 0

def sharpening(P):
    T = 0.1
    P_clone = P[labeled_bs:, :, :, :, :].clone().detach()
    P_clone1 = torch.pow(P_clone, 1 / T)
    P_clone2 = torch.sum(P_clone1, dim=1, keepdim=True)
    P_sharpen = torch.div(P_clone1, P_clone2)
    return P_sharpen


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
    return  result

if __name__ == "__main__":
    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='vnet'):
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
            model = torch.nn.DataParallel(model)
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
            model = torch.nn.DataParallel(model)
        return model

    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    db_train = LA_Heart_Dataset(base_dir=train_data_path,
                               split='train',
                               train_flod='train0.list',
                               common_transform=transforms.Compose([
                                   RandomCrop(patch_size),
                               ]),
                               sp_transform=transforms.Compose([
                                   ToTensor(),
                               ]))

    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr

    Contrastive_loss = ContrastiveLoss(sample_num=50).cuda()

    model_vnet.train()
    model_resnet.train()

    for epoch_num in tqdm(range(max_epoch),ncols = 70):
        time1=time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num, i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            s1_input, s1_label = volume_batch1.cuda(), volume_label1.cuda()
            s2_input, s2_label = volume_batch2.cuda(), volume_label2.cuda()

            v_out1 = model_vnet(s1_input)
            r_out1 = model_resnet(s1_input)
            v_out2 = model_vnet(s2_input)
            r_out2 = model_resnet(s2_input)

            v_outputs1 = v_out1['out']
            r_outputs1 = r_out1['out']
            v_outputs2 = v_out2['out']
            r_outputs2 = r_out2['out']

            v_loss_seg1 = F.cross_entropy(v_outputs1[:labeled_bs], s1_label[:labeled_bs])
            v_outputs1_soft = F.softmax(v_outputs1, dim=1)
            v_loss_seg_dice1 = losses.dice_loss(v_outputs1_soft[:labeled_bs, 1, :, :, :], s1_label[:labeled_bs] == 1)

            v_loss_seg2 = F.cross_entropy(v_outputs2[:labeled_bs], s2_label[:labeled_bs])
            v_outputs2_soft = F.softmax(v_outputs2, dim=1)
            v_loss_seg_dice2 = losses.dice_loss(v_outputs2_soft[:labeled_bs, 1, :, :, :], s2_label[:labeled_bs] == 1)


            r_loss_seg1 = F.cross_entropy(r_outputs1[:labeled_bs], s1_label[:labeled_bs])
            r_outputs1_soft = F.softmax(r_outputs1, dim=1)
            r_loss_seg_dice1 = losses.dice_loss(r_outputs1_soft[:labeled_bs, 1, :, :, :], s1_label[:labeled_bs] == 1)

            r_loss_seg2 = F.cross_entropy(r_outputs2[:labeled_bs], s2_label[:labeled_bs])
            r_outputs2_soft = F.softmax(r_outputs2, dim=1)
            r_loss_seg_dice2 = losses.dice_loss(r_outputs2_soft[:labeled_bs, 1, :, :, :], s2_label[:labeled_bs] == 1)


            if (0.2 * (v_loss_seg1+v_loss_seg2)+ 1 * (v_loss_seg_dice1 + v_loss_seg_dice2)) < (0.2 * (r_loss_seg1+r_loss_seg2)+ 1 * (r_loss_seg_dice1 + r_loss_seg_dice2)):
                Good_student = 0
            else:
                Good_student = 1

            v_supervised_loss = (v_loss_seg1 + v_loss_seg_dice1 + v_loss_seg2 + v_loss_seg_dice2)/2
            r_supervised_loss = (r_loss_seg1 + r_loss_seg_dice1 + r_loss_seg2 + r_loss_seg_dice2)/2

            v_cosine_loss = losses.cosine_similarity_loss(v_outputs1 , v_outputs2)
            r_cosine_loss = losses.cosine_similarity_loss(r_outputs1 , r_outputs2)

            v_outputs2_soft = F.softmax(v_outputs2, dim=1)
            r_outputs2_soft = F.softmax(r_outputs2, dim=1)

            v_outputs1_sharpen = sharpening(v_outputs1_soft)
            v_outputs2_sharpen = sharpening(v_outputs2_soft)
            r_outputs1_sharpen = sharpening(r_outputs1_soft)
            r_outputs2_sharpen = sharpening(r_outputs2_soft)

            if Good_student == 0:
                Plabel1 = v_outputs1_sharpen
                Plabel2 = v_outputs2_sharpen
            if Good_student == 1:
                Plabel1 = r_outputs1_sharpen
                Plabel2 = r_outputs2_sharpen

            consistency_weight = get_current_consistency_weight(iter_num//150)
            if Good_student == 0:

                r_consistency_dist_1_1 = consistency_criterion(r_outputs1_soft[labeled_bs:, :, :, :, :], Plabel1)
                r_consistency_dist_1_2 = consistency_criterion(r_outputs2_soft[labeled_bs:, :, :, :, :], Plabel2)



                b, c, w, h, d = r_consistency_dist_1_1.shape

                r_consistency_dist_1_1 = torch.sum(r_consistency_dist_1_1) / (b * c * w * h * d)
                r_consistency_dist_1_2 = torch.sum(r_consistency_dist_1_2) / (b * c * w * h * d)


                r_consistency_loss_1_1 = r_consistency_dist_1_1
                r_consistency_loss_1_2 = r_consistency_dist_1_2

                r_consistency_loss_total = r_consistency_loss_1_1 + r_consistency_loss_1_2

                v_loss = v_supervised_loss + v_cosine_loss
                r_loss = r_supervised_loss + r_cosine_loss + consistency_weight * r_consistency_loss_total
                writer.add_scalar('loss/r_consistency_loss_tatal',r_consistency_loss_total,iter_num)

            if Good_student == 1:

                v_consistency_dist_1_1 = consistency_criterion(v_outputs1_soft[labeled_bs:, :, :, :, :], Plabel1)
                v_consistency_dist_1_2 = consistency_criterion(v_outputs2_soft[labeled_bs:, :, :, :, :], Plabel2)

                b, c, w, h, d = v_consistency_dist_1_1.shape

                v_consistency_dist_1_1 = torch.sum(v_consistency_dist_1_1) / (b * c * w * h * d)
                v_consistency_dist_1_2 = torch.sum(v_consistency_dist_1_2) / (b * c * w * h * d)


                v_consistency_loss_1_1 = v_consistency_dist_1_1
                v_consistency_loss_1_2 = v_consistency_dist_1_2

                v_consistency_loss_total = v_consistency_loss_1_1 + v_consistency_loss_1_2

                v_loss = v_supervised_loss + v_cosine_loss + consistency_weight * v_consistency_loss_total
                r_loss = r_supervised_loss + r_cosine_loss
                writer.add_scalar('loss/v_consistency_loss_tatal',v_consistency_loss_total,iter_num)



            v_outputs1_pro = v_out1['projector']
            r_outputs1_pro = r_out1['projector']
            v_outputs2_pro = v_out2['projector']
            r_outputs2_pro = r_out2['projector']

            loss_contrastive1 = torch.utils.checkpoint.checkpoint(Contrastive_loss,v_outputs1_pro, r_outputs1_pro, v_outputs1, r_outputs1)
            loss_contrastive2 = torch.utils.checkpoint.checkpoint(Contrastive_loss,v_outputs2_pro, r_outputs2_pro, v_outputs2, r_outputs2)
            loss_contrastive3 = torch.utils.checkpoint.checkpoint(Contrastive_loss,v_outputs1_pro, r_outputs2_pro, v_outputs1, r_outputs2)
            loss_contrastive4 = torch.utils.checkpoint.checkpoint(Contrastive_loss,v_outputs2_pro, r_outputs1_pro, v_outputs2, r_outputs1)


            v_loss = v_loss + loss_contrastive1 + loss_contrastive2 + loss_contrastive3 + loss_contrastive4
            r_loss = r_loss + loss_contrastive1 + loss_contrastive2 + loss_contrastive3 + loss_contrastive4

            if (torch.any(torch.isnan(v_loss)) or torch.any(torch.isnan(r_loss)) ):
                print('nan find')
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()

            v_loss.backward(retain_graph=True)
            r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            writer.add_scalar('lr', lr_, iter_num)

            writer.add_scalar('loss/v_loss_seg1', v_loss_seg1, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice1', v_loss_seg_dice1, iter_num)
            writer.add_scalar('loss/v_loss_seg2', v_loss_seg2, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice2', v_loss_seg_dice2, iter_num)
            writer.add_scalar('loss/v_cosine_loss', v_cosine_loss, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/loss_contrastive1', loss_contrastive1, iter_num)
            writer.add_scalar('loss/loss_contrastive2', loss_contrastive2, iter_num)
            writer.add_scalar('loss/loss_contrastive3', loss_contrastive3, iter_num)
            writer.add_scalar('loss/loss_contrastive4', loss_contrastive4, iter_num)
            writer.add_scalar('loss/', v_loss, iter_num)

            writer.add_scalar('loss/r_loss_seg1', r_loss_seg1, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice1', r_loss_seg_dice1, iter_num)
            writer.add_scalar('loss/r_loss_seg2', r_loss_seg2, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice2', r_loss_seg_dice2, iter_num)
            writer.add_scalar('loss/r_cosine_loss', r_cosine_loss, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)

            writer.add_scalar('loss/r_loss', r_loss, iter_num)
            writer.add_scalar('train/Good_student', Good_student, iter_num)

            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_seg1 : %f v_loss_seg_dice1 : %f v_loss_seg2 : %f v_loss_seg_dice2 : %f r_supervised_loss : %f r_loss_seg1 : %f r_loss_seg_dice1 : %f r_loss_seg2 : %f r_loss_seg_dice2 : %f loss_contrastive1: %f loss_contrastive2: %f v_loss: %f r_loss: %f Good_student: %f ' %
                (iter_num,
                 v_supervised_loss.item(), v_loss_seg1.item(), v_loss_seg_dice1.item(), v_loss_seg2.item(),
                 v_loss_seg_dice2.item(),
                 r_supervised_loss.item(), r_loss_seg1.item(), r_loss_seg_dice1.item(), r_loss_seg2.item(),
                 r_loss_seg_dice2.item(), loss_contrastive1.item(), loss_contrastive2.item(), v_loss.item(),
                 r_loss.item(), Good_student))

            if iter_num % 2500 == 0 and iter_num != 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            # if iter_num % 10 == 0:



            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            break


    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))

    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))

    writer.close()



