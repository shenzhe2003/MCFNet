from model.DNANet.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='MCFNet')
    parser.add_argument('--model', type=str, default='',
                        help='model name: MCFNet')
    parser.add_argument('--logname', type=str, default='')
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=bool, default=True, help='True or False')
    parser.add_argument('--scale', type=float, default='-1.0')
    parser.add_argument('--ca', type=str, default='MSFFN', help='')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name:  NUDT-SIRST, NUAA-SIRST, NUST-SIRST, IRSTD-1k')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default='path to datasets')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='1_1')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--train_base_size', type=int, default=256,
                        help='base image size 256 or 512')
    parser.add_argument('--test_base_size', type=int, default=256,
                        help='base image size 256 or 512')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size 256 or 512')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')
    parser.add_argument('--warm_epoch', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=800, metavar='N')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                        training')
    parser.add_argument('--test_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                        testing')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gpus', type=str, default='0')
    return args