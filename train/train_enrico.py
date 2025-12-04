import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import sys
active_root = os.environ['ACTIVE_ROOT']
sys.path.append(active_root)
import argparse
from tqdm import tqdm

from dataset.enrico_dataset import get_dataset, EnricoCollect
from model.dense_model import SinglemodalTransformer, MultimodalTransformer, MultimodalTransformerWF, MultimodalTransformerTokenF, MultimodalTransformerUniTokenFusionV3
from model.model_util import *
from acm_utilize.yml_parser import YmlConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, warmup_lr_start=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr_start = warmup_lr_start
        self.epoch = 0
        
    def step(self):
        if self.epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr_start + (self.base_lr - self.warmup_lr_start) * (self.epoch / self.warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (0.5 * (1.0 + np.cos(np.pi * progress)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1
    
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--path', type=str, default="model_ckpt")
parser.add_argument('--cpt_name', type=str, default="enrico_img_wire")
parser.add_argument('--enrico_path', type=str, default='data/enrico')
parser.add_argument('--train', default=True, type=str2bool)
parser.add_argument('--model_config', default='config/enrico_config.yml', type=str)

args = parser.parse_args()

def finetune_epoch(model,
                   loss_fn,
                   optimizer,
                   schedular,
                   dataloader: DataLoader,
                   epoch_index = 0,
                   training = True,
                   device = 'cpu',
                   logger = None,
                   accumulation_steps = 1):
    # train_sampler = torch.util
    loader = dataloader
    losses = []
    pred_accs = []
    pbar = tqdm(enumerate(loader), total = len(loader))
    model.train(training)
    batch_sizes = []
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        y = dbatch.pop('target')
        dbatch.pop('idx')
        x = {kk: dbatch[kk].to(device) for kk in dbatch}
        y = y.to(device)

        batch_sizes.append(y.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(training):
            if isinstance(model, MultimodalTransformerUniTokenFusionV3):
                logits, mspc_logits = model(x, epoch_id=epoch_index)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accumulation_steps
            else:
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accumulation_steps
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())

        if training:
            # model.zero_grad()
            loss.backward()
            # scaler.scale(loss).backward()
            if (it + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                # scaler.step(optimizer)
                schedular.step()
                model.zero_grad()
            # scaler.update()
        else:
            acc = (logits.argmax(dim=1) == y).float().sum().item()
            pred_accs.append(acc)
        pbar.set_description(f"epoch {epoch_index} iter {it}: training loss {loss.item() * accumulation_steps:.5f}.")
    if not training:
        avg_acc = np.sum(pred_accs) / np.sum(batch_sizes)
        avg_loss = np.sum(losses) / len(losses)
        if logger:
            logger.add_scalar_auto('Eval acc', avg_acc)
            logger.add_scalar_auto('Eval loss', avg_loss)
        else:
            print('Eval acc', avg_acc)
            print('Eval loss', avg_loss)
        return avg_acc

def prediction(model,
                   loss_fn,
                   optimizer,
                   schedular,
                   dataloader: DataLoader,
                   epoch_index = 0,
                   training = True,
                   device = 'cpu',
                   logger = None,
                   accumulation_steps = 1):
    # train_sampler = torch.util
    loader = dataloader
    losses = []
    pred_accs = []
    pbar = tqdm(enumerate(loader), total = len(loader))
    model.train(training)
    batch_sizes = []
    predict_dict = {}
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        y = dbatch.pop('target')
        bidx = dbatch.pop('idx')
        x = {kk: dbatch[kk].to(device) for kk in dbatch}
        y = y.to(device)

        batch_sizes.append(y.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(training):
            if isinstance(model, MultimodalTransformerUniTokenFusionV3):
                logits, mspc_logits = model(x, epoch_id=epoch_index)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accumulation_steps
            else:
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accumulation_steps
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())

        if training:
            # model.zero_grad()
            loss.backward()
            # scaler.scale(loss).backward()
            if (it + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                # scaler.step(optimizer)
                schedular.step()
                model.zero_grad()
            # scaler.update()
        else:
            acc = (logits.argmax(dim=1) == y).float().sum().item()
            pred_result = (logits.argmax(dim=1) == y).bool()
            for i, idx in enumerate(bidx):
                predict_dict[idx.item()] = pred_result[i].item()
            # print(predict_dict)
            pred_accs.append(acc)
        pbar.set_description(f"epoch {epoch_index} iter {it}: training loss {loss.item() * accumulation_steps:.5f}.")
    if not training:
        avg_acc = np.sum(pred_accs) / np.sum(batch_sizes)
        avg_loss = np.sum(losses) / len(losses)
        if logger:
            logger.add_scalar_auto('Eval acc', avg_acc)
            logger.add_scalar_auto('Eval loss', avg_loss)
        else:
            print('Eval acc', avg_acc)
            print('Eval loss', avg_loss)
        return avg_acc, predict_dict

def input_dim(input_axis, freq_bands, channels):
    return input_axis * ((freq_bands * 2) + 1) + channels

def main(args):
    model_config = YmlConfig(args.model_config)

    modality = 'all' if len(model_config.obj.modality) > 1 else list(model_config.obj.modality.keys())[0]

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    train_dataset, valid_dataset, test_dataset = get_dataset(args.enrico_path, modalities=modality, cut_into=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=EnricoCollect())
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, collate_fn=EnricoCollect())
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, collate_fn=EnricoCollect())

    if modality != 'all':
        modality_config = model_config.parse_to_modality(model_config.obj.modality[modality])
        model = SinglemodalTransformer(
            device,
            modality_config,
            **model_config.obj.network
        )
    else:
        modality_config = {}
        for kk in model_config.obj.modality:
            modality_config[kk] = model_config.parse_to_modality(
                model_config.obj.modality[kk]
            )
        model = eval(model_config.obj.network_type)(
            device,
            modality_config,
            **model_config.obj.network
        )
        # model = MultimodalTransformer(
        #     device,
        #     modality_config,
        #     **model_config.obj.network
        # )
    model.to(device)
    if not hasattr(args, "fusion_type"):
        try:
            ft = model_config.obj.fusion_type
        except Exception:
            ft = "na"
        args.fusion_type = ft

    model_dumper = ModelDumper(args.path, args.seed, args.cpt_name, model_config.obj.modality, args)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    schedular = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)

    loss_fn = torch.nn.CrossEntropyLoss()
    b_acc = 0.0

    if args.train:
        for epoch in range(args.epochs):
            finetune_epoch(model, loss_fn, optimizer, schedular,
                           train_dataloader, epoch, True, device)

            v_acc = finetune_epoch(model, loss_fn, optimizer, schedular, valid_dataloader, epoch, False, device=device)
            print("Current ACC:{:.5f}".format(v_acc))
            if v_acc >= b_acc:
                b_acc = v_acc
                model_dumper.dump(model)
    model.load_state_dict(torch.load(model_dumper.model_path))
    t_acc, pred_dict = prediction(model, loss_fn, optimizer, schedular, test_dataloader, 0, False, device=device)
    print("Final ACC: {:.5f}".format(t_acc))
    model_dumper.dump_json(pred_dict)

if __name__ == '__main__':
    print("Training Arguments : {}".format(args))
    set_seed(args.seed)
    main(args)