import os
import argparse
import torch
from models.vnet import VNet
from models.ResNet34 import Resnet34
from test_util import test_all_case


parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='../dataset/LA/LA_Data/', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='./dataset/LA/LA_Data/', help='Name of Experiment')
parser.add_argument('--root_path', type=str, default='./dataset/LA/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default="C3S3", help='model_name')
parser.add_argument('--gpu', type=str,  default='6', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "checkpoints/" + FLAGS.model + "/"
test_save_path = "model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

# with open(FLAGS.root_path + '/../Flods/test0.list', 'r') as f:
with open(FLAGS.root_path + 'Flods/test0.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]

def create_model(name='vnet'):
    if name == 'vnet':
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        model = torch.nn.DataParallel(model)
    if name == 'resnet34':
        net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        model = torch.nn.DataParallel(model)
    return model

def test_calculate_metric(epoch_num):
    vnet = create_model(name='vnet')
    resnet = create_model(name='resnet34')

    v_save_mode_path = os.path.join(snapshot_path, 'vnet_iter_' + str(epoch_num) + '.pth')
    vnet.load_state_dict(torch.load(v_save_mode_path))
    print("init weight from {}".format(v_save_mode_path))
    vnet.eval()

    r_save_mode_path = os.path.join(snapshot_path, 'resnet_iter_' + str(epoch_num) + '.pth')
    resnet.load_state_dict(torch.load(r_save_mode_path))
    print("init weight from {}".format(r_save_mode_path))
    resnet.eval()

    avg_metric = test_all_case(vnet, resnet, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    iters = 6000
    metric = test_calculate_metric(iters)
    print('iter:', iter)
    print(metric)