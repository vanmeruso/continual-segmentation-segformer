import os
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from utils import eval_seg
from torch.utils import data
import torch.nn.functional as F

from utils import ext_transforms as et
from datasets import Cityscapes, Acdc

from core.model import WeTr
import torch.distributed as dist
from utils.optimizer import PolyWarmupAdamW
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets import imutils

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='./configs/cityscapes.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--gpu_ids", type=list, default=[0,1],help="GPU ID")

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


   



def get_dataset(opts):

    train_transform = et.ExtCompose([
        et.ExtRandomCrop(size=(opts.dataset.crop_size[0], opts.dataset.crop_size[0] ), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    

    if opts.dataset.name == 'cityscapes':
        train_dst = Cityscapes(root=opts.dataset.data_root,
                                split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.dataset.data_root,
                                split='val', transform=val_transform)
    else:
        train_dst = Acdc(root=opts.dataset.data_root,
                                split='train', transform=train_transform)
        val_dst = Acdc(root=opts.dataset.data_root,
                                split='val', transform=val_transform)
        
    dataset_dict = {}
    dataset_dict['train'] = train_dst
    
    dataset_dict['test'] = val_dst
    
    return dataset_dict

def validate(model=None, criterion=None, device = None, data_loader=None):

    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
       for inputs, labels  in data_loader:
            
    
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs,
                                            size=labels.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(
                torch.argmax(resized_outputs,
                             dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = eval_seg.scores(gts, preds)

    return val_loss.cpu().numpy() / float(len(data_loader)), score

def train(opts):
    """
        
        opts.random_seed = 1
    """
    num_workers = 4
    #writer = SummaryWriter('runs/segformer')
    
    torch.cuda.set_device(args.gpu_ids[args.local_rank])
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
  
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.deterministic = True
    
    dataset_dict = get_dataset(opts)
    train_sampler = DistributedSampler(dataset_dict['train'],shuffle=True)

    train_loader = data.DataLoader(
        dataset_dict['train'], 
        batch_size=opts.dataset.batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True, 
        prefetch_factor=4)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.dataset.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
   
    
    model = WeTr(backbone=opts.backbone,
                num_classes=opts.dataset.num_classes,
                embedding_dim=768,
                pretrained=True)
    
    param_groups = model.get_param_groups()
    model.to(device)
    
    model = DistributedDataParallel(model, device_ids=[args.gpu_ids[args.local_rank]], find_unused_parameters=True)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=opts.dataset.ignore_index)
    criterion = criterion.to(device)

    train_sampler.set_epoch(0)
    max_iters = opts.train.epochs * len(train_loader)
    val_interval = max(100, max_iters // 100)

    if args.local_rank==0:
        print("==============================================")
        print("  Device: %s" % device)
        print( "  opts : ")
        print(opts)
        print("==============================================")
    
        print("Dataset: %s, Train set: %d, Test set: %d" %
          (opts.dataset.name, len(train_sampler), len(dataset_dict['test'])))

        print(f"train epoch : {opts.train.epochs} , iterations : {max_iters} , val_interval : {val_interval}")
    
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": opts.optimizer.learning_rate,
                "weight_decay": opts.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": opts.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": opts.optimizer.learning_rate * 10,
                "weight_decay": opts.optimizer.weight_decay,
            },
        ],
        lr = opts.optimizer.learning_rate,
        weight_decay = opts.optimizer.weight_decay,
        betas = opts.optimizer.betas,
        warmup_iter = opts.scheduler.warmup_iter,
        max_iter = max_iters,
        warmup_ratio = opts.scheduler.warmup_ratio,
        power = opts.scheduler.power
    )
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, path)
        
        if args.local_rank==0:
            print("Model saved as %s" % path)
    
    #utils.mkdir('checkpoints')    



    cur_epochs = 0
    
    for n_iter in range(max_iters):
        try:
            inputs, labels = next(train_loader_iter)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            inputs, labels = next(train_loader_iter)
            cur_epochs += 1
        
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        outputs = model(inputs)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = criterion(outputs, labels.type(torch.long))
        
        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()
        
        if (n_iter+1) % opts.train.log_iters == 0 and args.local_rank==0:
            delta, eta = cal_eta(time0, n_iter+1, max_iters)
            lr = optimizer.param_groups[0]['lr']
            print("[Epochs: %d Iter: %d] Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(cur_epochs, n_iter+1, delta, eta, lr, seg_loss.item())) 
            #writer.add_scalar('loss/train', seg_loss.item(), n_iter+1)
            #writer.add_scalar('lr/train', lr, n_iter+1)
            #record_inputs, record_outputs, record_labels = imutils.tensorboard_image(inputs, outputs, labels)
            #writer.add_image("input/train", record_inputs, n_iter+1)
            #writer.add_image("output/train", record_outputs, n_iter+1)
            #writer.add_image("label/train", record_labels, n_iter+1)
            
        if (n_iter+1) % val_interval == 0:
            if args.local_rank==0:
                print('Validating...')
            val_loss, val_score = validate(model=model, criterion=criterion, device= device, data_loader=test_loader)    
            if args.local_rank==0:
                print("valloss : %f, mIOU: %f"%(val_loss, val_score['Mean IoU']))
            #writer.add_scalars('val/train', {"Pixel Accuracy": val_score["Pixel Accuracy"],
                                           # "Mean Accuracy": val_score["Mean Accuracy"],
                                           # "Mean IoU": val_score["Mean IoU"]}, n_iter+1)
            
    print("end")
    return True

if __name__ == "__main__":
    
    args = parser.parse_args()
    opts = OmegaConf.load(args.config)
    
    train(opts=opts)