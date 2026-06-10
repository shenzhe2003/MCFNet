from model.DNANet.utils import *
from model.DNANet.metric import *
from model.DNANet.loss import *
from model.DNANet.load_param_data import  load_dataset, load_param
from model.DNANet.parse_args_train import  parse_args
# import torch 之前设置 os.environ["CUDA_VISIBLE_DEVICES"] 才生效
args = parse_args()
if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
writer = SummaryWriter('./runs/learning_rate_experiment')
  
if args.model == 'DMFNet':
    from model.DNANet.model_DMFNet import  *
elif args.model == 'LGMFNet':
    from model.DNANet.model_LGMFNet import  *
elif args.model == 'LGMFNet_MSFFN':
    from model.DNANet.model_LGMFNet_MSFFN import  *   
elif args.model == 'LGMFNet_AyB':
    from model.DNANet.model_LGMFNet_AyB import  *
elif args.model == 'LGMFNet_DDC_AyB':
    from model.DNANet.model_LGMFNet_DDC_AyB import unet_resnet_18
elif args.model == "LGMFNet_DEC_AyB":
    from model.DNANet.model_LGMFNet_DEC_AyB import unet_resnet_18
elif args.model == "LGMFNet_DEC_AyB_MSFFN":
    from model.DNANet.model_LGMFNet_DEC_AyB_MSFFN import unet_resnet_18   
elif args.model == "LGMFNet_SCSA_AyB":
    from model.DNANet.model_LGMFNet_SCSA_AyB import unet_resnet_18    
elif args.model == "LGMFNet_HWDC_AyB":
    from model.DNANet.model_LGMFNet_HWDC_AyB import unet_resnet_18
elif args.model == "LGMFNet_DEC_HWDC_AyB":
    from model.DNANet.model_LGMFNet_DEC_HWDC_AyB import unet_resnet_18
elif args.model == "LGMFNet_DEC_HFF":
    from model.DNANet.model_LGMFNet_DEC_HFF import unet_resnet_18 
else:
    exit("no model!")


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.nIoU = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5) # 计算nIoU
        self.mIoU = mIoU(1)
        self.pdfa = PD_FA()
        
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            if args.dataset == 'IRSTD-1k':
                args.split_method = '800_200'
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSet(dataset_dir, img_id=train_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        testset         = TestSet (dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        self.test_data  = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False)

        
        model       = unet_resnet_18(deep_supervision=args.deep_supervision)
        model = nn.DataParallel(model)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-4)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        self.best_iou       = 0
        self.best_niou      = 0
        self.best_pd        = 0
        self.best_fa        = 1.0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    def training(self,epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, (data, data_1, labels) in enumerate(tbar):
            data   = data.cuda()
            data_1 = data_1.cuda()
            labels = labels.cuda()
            # 判断是否使用深度监督，如果是
            if args.deep_supervision == True:
                preds= self.model(data, data_1)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
                pred = preds[-1]
            else:
                pred = self.model(data, data_1)
                loss = SoftIoULoss(pred, labels)
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
        self.nIoU.reset()
        self.pdfa.reset()   
        
        losses = AverageMeter()
        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data, data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1] # 针对深度监督而有的
                else:
                    pred = self.model(data, data)
                    loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                self.nIoU.update(pred, labels)
                self.pdfa.update((pred[0, 0, :, :] > 0.3).cpu(), labels[0, 0, :, :], (size,size))
                
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IoU = self.mIoU.get()
                _, nIoU = self.nIoU.get()
                fa, pd = self.pdfa.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f, nIoU: %.4f, pd: %.6f, fa: %.10f' % (epoch, losses.avg, mean_IoU, nIoU, pd, fa))
            test_loss=losses.avg

        save_model(nIoU, self.best_niou, mean_IoU, self.best_iou, pd, self.best_pd, fa, self.best_fa, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())

         # 指标更新
        if mean_IoU > self.best_iou:
            self.best_iou = mean_IoU
        if nIoU > self.best_niou:
            self.best_niou = nIoU 
        if fa < self.best_fa:
            self.best_fa = fa
        if pd > self.best_pd:
            self.best_pd = pd    
            
def main(args):
    # 打印实验配置
    args_dict = vars(args)
    print("实验配置：")
    for key in args_dict:
        print(f"{key}: {args_dict[key]}")
    # 训练
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        # 打印并记录每个参数组的学习率
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            lr = param_group['lr']
            print(f"Epoch {epoch + 1}, Param Group {i + 1}, Learning Rate: {lr}")
        # 将学习率写入 TensorBoard
        writer.add_scalar(f'Learning_Rate/Param_Group_{i + 1}', lr, epoch)
        trainer.testing(epoch, args.crop_size)
    print('best_iou:' + str(trainer.best_iou), 'best_niou:' + str(trainer.best_niou), 'best_pd:' + str(trainer.best_pd), 'best_fa:' + str(trainer.best_fa))
    # 关闭 writer
    writer.close()



if __name__ == "__main__":
    main(args)


