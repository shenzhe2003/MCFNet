from model.Net.utils import *
from model.Net.metrics import *
from model.Net.loss import *
from model.Net.load_param_data import  load_dataset, load_param
from model.Net.parse_args_main import  parse_args
import os
import subprocess
args = parse_args()
if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
  
# CHOOSE DIFFERENT MODELS
if args.model == "MCFNet":
    from model.Net.model_MCFNet import MCFNet 
else:
    exit("no model!")

def Loss(pred, target):
    return SoftIoULoss(pred, target)+bce_loss(pred, target)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PD_FA = PD_FA(1, 10, cfg)
        self.warm_epoch = args.warm_epoch or 100
        self.down = nn.MaxPool2d(2, 2)
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            if args.dataset == 'IRSTD-1k':
                args.split_method = '4_1'
            train_img_ids, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSet(dataset_dir, img_id=train_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        testset         = TestSet (dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        self.test_data  = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False)
        self.scale = args.scale
        if args.model   == 'MNCFet':  
            model       = MCFNet(deep_supervision=args.deep_supervision, scale=self.scale)
            
        model           = model.cuda()
        model.apply(weights_init_xavier)        
        self.model      = model
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-10)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()
        self.best_iou       = 0
        self.best_pd        = 0
        self.best_fa        = 1.0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]
        
    def training(self,epoch):
            tbar = tqdm(self.train_data)
            self.model.train()
            losses = AverageMeter()
            tag = True
            for i, (data, data_1, labels) in enumerate(tbar):
                data   = data.cuda()
                data_1 = data_1.cuda()
                labels = labels.cuda()
                if epoch>self.warm_epoch:
                    tag = True
                masks, pred = self.model(data, data_1, tag)
                loss = Loss(pred, labels)
                for j in range(len(masks)):
                    loss = loss + Loss(masks[j], labels)*(0.5**(j+1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), pred.size(0))
                tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
            self.train_loss = losses.avg  
            
    def testing (self, epoch, size):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.ROC.reset()
        self.mIoU.reset()
        self.pdfa.reset()   
        losses = AverageMeter()
        tag = True
        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if epoch>self.warm_epoch:
                    tag = True
                _, pred = self.model(data, data, tag)
                loss = Loss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                self.pdfa.update(pred, labels, (size, size), threshold=threshold)
                _, mean_IoU = self.mIoU.get()
                fa, pd = self.pdfa.get()
                tbar.set_description('Epoch %d, test loss %.4f, IoU: %.4f, pd: %.4f, fa: %.2f' % (epoch, losses.avg, mean_IoU, pd, fa))
            test_loss=losses.avg      
        if mean_IoU > self.best_iou:
            self.best_iou = mean_IoU
        if fa < self.best_fa:
            self.best_fa = fa
        if pd > self.best_pd:
            self.best_pd = pd   
             
def get_gpu_info():
    # 获取当前使用的 GPU 信息
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=index,memory.used --format=csv,nounits", shell=True).decode('utf-8').strip().split('\n')[1]
    gpu_index, memory_used = map(int, gpu_info.split(', '))
    return gpu_index, memory_used 

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.scheduler.step()
        trainer.testing(epoch, args.crop_size)        
    print('iou:' + str(trainer.best_iou), 'pd:' + str(trainer.best_pd), 'fa:' + str(trainer.best_fa))
    writer.close()  
    
if __name__ == "__main__":
    main(args)
