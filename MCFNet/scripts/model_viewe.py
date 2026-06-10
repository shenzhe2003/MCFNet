from model.DNANet.utils import *
from model.DNANet.metric import *
from model.DNANet.loss import *
from model.DNANet.parse_args_train import  parse_args
args = parse_args()
if args.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus) # import torch 之前设置 os.environ["CUDA_VISIBLE_DEVICES"] 才生效
# 数据集
from model.DNANet.load_param_data import  load_dataset,load_param
from torch.utils.data import DataLoader

from DMFNet.model.DNANet.model_DFMNet import  Res_CBAM_block
from DMFNet.model.DNANet.model_DFMNet import  DNANet
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model.DNANet.utils import *
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet')
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet)')


    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset name:  NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50, 10000_100(for NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')


    args = parser.parse_args()

    return args

class Model_Viewer:
    def __init__(self, args):
        # 定义SummaryWriter
        self.writer = SummaryWriter('runs/network_view')

        # 定义输入数据
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


        self.trainset = TrainSet(dataset_dir, img_id=train_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.train_data_loader = DataLoader(dataset=self.trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        self.dataiter = iter(self.train_data_loader)
        self.image_input, labels = next(self.dataiter)
        self.image_input = self.image_input.cuda()

        # 定义模型
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        self.net = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        self.net = self.net.cuda()
 
    def visualize_model(self):
        self.writer.add_graph(self.net, self.image_input)
        print("模型可视化输出结束。")
        self.writer.close()


if __name__ == '__main__':
    args = parse_args()
    viewer = Model_Viewer(args=args)
    viewer.visualize_model()
    print("程序结束。")


    





