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
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset.mimiciv_dataset import MIMICIVDataset, get_dataset
import time
from model.dense_model import *
from model.model_util import *
from acm_utilize.yml_parser import YmlConfig
from accelerate import Accelerator
from thop import profile
import pickle
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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

def network_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--path', type=str, default="model_ckpt")
parser.add_argument('--cpt_name', type=str, default="enrico_img")
parser.add_argument('--dataset_path', type=str, default='data/enrico')
parser.add_argument('--train', default=True, type=str2bool)
parser.add_argument('--model_config', default='config/multi_modal_config.yml', type=str)
parser.add_argument('--modality_latent_len', default=-1, type=int)
parser.add_argument('--fusion_type', default='None', type=str)
# parser.add_argument('--modality', default='screenImg', type=str, choices=['all', 'screenImg', 'screenWireframeImg'])

args = parser.parse_args()


def finetune_epoch(model, 
                   loss_fn, 
                   optimizer, 
                   schedular, 
                   dataloader: DataLoader, 
                   epoch_index = 0, 
                   training = True, 
                   device = 'cpu',
                   accelerator = None,
                   logger = None,
                   accumulation_steps = 1):
    # train_sampler = torch.util
    loader = dataloader
    losses = []
    pred_accs = []
    pbar = tqdm(enumerate(loader), total = len(loader))
    model.train(training)
    batch_sizes = []
    ground_truth = []
    predict_result = []
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        y = dbatch.pop('target')
        dbatch.pop('idx')
        x = {kk: dbatch[kk].to(device) for kk in dbatch}
        y = y.to(device)
        
        batch_sizes.append(y.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(training):
            logits, mspc_x = model(x, epoch_index)
            # print(logits, y)
            loss = (loss_fn(logits.squeeze(), y)+model.limoe_loss * 0.1+model.get_router_loss() * 0.1) / accumulation_steps
            loss += sum([loss_fn(mspc_x[mk].squeeze(), y) for mk in mspc_x]) / accumulation_steps
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())
        predict_result.append(torch.max(logits, 1)[1].cpu())
        ground_truth.append(y.cpu())
        if training:
            # model.zero_grad()
            # loss.backward()
            if loss == torch.nan:
                continue
            accelerator.backward(loss)
            # scaler.scale(loss).backward()
            if (it + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                # scaler.step(optimizer)
                if schedular is not None:
                    schedular.step()
                model.zero_grad()
            # scaler.update()
            acc = (torch.max(logits, 1)[1] == y).float().sum().item()
            pred_accs.append(acc)
        else:
            acc = (torch.max(logits, 1)[1] == y).float().sum().item()
            pred_accs.append(acc)
        pbar.set_description(f"epoch {epoch_index} iter {it}: training loss {loss.item() * accumulation_steps:.5f}.")
    if not training:
        avg_acc = np.sum(pred_accs) / np.sum(batch_sizes)
        avg_loss = np.sum(losses) / len(losses)
        y_true_np = torch.cat(ground_truth, dim=0).numpy()
        y_pred_np = torch.cat(predict_result, dim=0).numpy()
        precision = precision_score(y_true_np, y_pred_np, average='macro')
        recall = recall_score(y_true_np, y_pred_np, average='macro')
        f1 = f1_score(y_true_np, y_pred_np, average='macro')
        if logger:
            logger.add_scalar_auto('Eval acc', avg_acc)
            logger.add_scalar_auto('Eval loss', avg_loss)
            logger.add_scalar_auto('Eval precision', precision)
            logger.add_scalar_auto('Eval recall', recall)
            logger.add_scalar_auto('Eval f1', f1)
        else:
            print('Eval acc', avg_acc)
            print('Eval loss', avg_loss)
            print(f"Eval - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return avg_acc, precision, recall, f1
    
    else:
        avg_loss = np.sum(losses) / len(losses)
        avg_acc = np.sum(pred_accs) / np.sum(batch_sizes)
        y_true_np = torch.cat(predict_result, dim=0).numpy()
        y_pred_np = torch.cat(ground_truth, dim=0).numpy()
        precision = precision_score(y_true_np, y_pred_np, average='macro')
        recall = recall_score(y_true_np, y_pred_np, average='macro')
        f1 = f1_score(y_true_np, y_pred_np, average='macro')
        print(f"Training - Precision: {precision:.4}, Recall: {recall:.4}, F1: {f1:.4}, Loss: {avg_loss:.4}")
        return avg_acc, precision, recall, f1

def entropy_calc(prob):
    return -sum([x*np.log(x) for x in prob])

def uncertainty_ensemble(all_predict_logit):
    n_instance = 0
    modality_list = []
    for mm in all_predict_logit:
        all_predict_logit[mm] = torch.cat(all_predict_logit[mm], dim=0)
        n_instance = len(all_predict_logit[mm])
        modality_list.append(mm)
        # print(all_predict_logit[mm].shape)
    ret_predict = []
    mm_select = []
    for i in range(n_instance):
        uc_vector = [entropy_calc(all_predict_logit[mm][i]) for mm in modality_list]
        confident_mm = modality_list[np.argmin(uc_vector)]
        ret_predict.append(all_predict_logit[confident_mm][i].unsqueeze(0))
        mm_select.append(confident_mm)
    
    ret_predict = torch.cat(ret_predict, dim=0)
    # print(ret_predict.shape)
    ret_predict = torch.max(ret_predict, 1)[1].cpu().numpy()
    return ret_predict

def uncertainty_ensemblev2(all_predict_logit):
    n_instance = 0
    modality_list = []
    for mm in all_predict_logit:
        all_predict_logit[mm] = torch.cat(all_predict_logit[mm], dim=0)
        n_instance = len(all_predict_logit[mm])
        modality_list.append(mm)
        # print(all_predict_logit[mm].shape)
    ret_predict = []
    mm_select = []
    for i in range(n_instance):
        uc_vector = [all_predict_logit[mm][i].argmax() for mm in modality_list]
        if np.sum(uc_vector) >= len(uc_vector)-np.sum(uc_vector):
            ret_predict.append(1)
        else:
            ret_predict.append(0)
    ret_predict = torch.from_numpy(np.array(ret_predict))
    return ret_predict

def calculate_entropy(logits: torch.Tensor):
    prob = logits / logits.sum(dim=-1, keepdim=True)
    return -(prob * prob.log()).sum(-1)

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
    ground_truth = []
    predict_result = []
    predict_logits = {}
    grad_cosines = []
    predict_entropy = []
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        y = dbatch.pop('target')
        bidx = dbatch.pop('idx')
        x = {kk: dbatch[kk].to(device) for kk in dbatch}
        y = y.to(device)
        
        batch_sizes.append(y.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(False):
            logits, mspc_x = model(x, epoch_index)
            predict_entropy.append(calculate_entropy(logits=logits))
            loss = loss_fn(logits.squeeze(1), y) / accumulation_steps
            loss += sum([loss_fn(mspc_x[mk].squeeze(1), y) for mk in mspc_x]) / accumulation_steps
            loss_dict = {
                mk: loss_fn(mspc_x[mk].squeeze(1), y) for mk in mspc_x
            }

            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())
        predict_result.append(torch.max(logits, 1)[1].cpu())
        for mk in mspc_x:
            if mk not in predict_logits:
                predict_logits[mk] = []
            predict_logits[mk].append(mspc_x[mk].softmax(dim=-1).cpu())
        if 'mm' not in predict_logits:
            predict_logits['mm'] = []
        predict_logits['mm'].append(logits.softmax(dim=-1).cpu())
            
        ground_truth.append(y.cpu())
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
            acc = (torch.max(logits, 1)[1] == y).float().sum().item()
            pred_result = (logits.argmax(dim=1) == y).bool()
            # ground_truth.append(y.cpu())
            # predict_result.append(pred_result.cpu())
            for i, idx in enumerate(bidx):
                predict_dict[idx.item()] = pred_result[i].item()
            # print(predict_dict)
            pred_accs.append(acc)
        pbar.set_description(f"epoch {epoch_index} iter {it}: training loss {loss.item() * accumulation_steps:.5f}.")
    if not training:
        avg_acc = np.sum(pred_accs) / np.sum(batch_sizes)
        avg_loss = np.sum(losses) / len(losses)
        y_true_np = torch.cat(ground_truth, dim=0).numpy()
        y_pred_np = torch.cat(predict_result, dim=0).numpy()
        y_pred_np = uncertainty_ensemblev2(predict_logits)
        predict_entropy_np = torch.cat(predict_entropy, dim=0).cpu().detach().numpy()
        precision = precision_score(y_true_np, y_pred_np, average='macro')
        recall = recall_score(y_true_np, y_pred_np, average='macro')
        f1 = f1_score(y_true_np, y_pred_np, average='macro')
        print(f"Training - Precision: {precision}, Recall: {recall}, F1: {f1}, Entropy: {np.mean(predict_entropy_np):.4f}")
        if logger:
            logger.add_scalar_auto('Eval acc', avg_acc)
            logger.add_scalar_auto('Eval loss', avg_loss)
        else:
            print('Eval acc', avg_acc)
            print('Eval loss', avg_loss)
        return (avg_acc, precision, recall, f1), predict_dict
    else:
        y_true_np = torch.cat(predict_result, dim=0).numpy()
        y_pred_np = torch.cat(ground_truth, dim=0).numpy()
        precision = precision_score(y_true_np, y_pred_np, average='macro')
        recall = recall_score(y_true_np, y_pred_np, average='macro')
        f1 = f1_score(y_true_np, y_pred_np, average='macro')
        print(f"Training - Precision: {precision}, Recall: {recall}, F1: {f1}")

def input_dim(input_axis, freq_bands, channels):
    return input_axis * ((freq_bands * 2) + 1) + channels

def main(args):
    model_config = YmlConfig(args.model_config)
    
    # modality = 'all' if len(model_config.obj.modality) > 1 else list(model_config.obj.modality.keys())[0]
    modality = model_config.obj.modality.keys()
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    train_dataset, valid_dataset, test_dataset = get_dataset(args.dataset_path, modalities=modality)

    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    if args.modality_latent_len > 0:
        for kk in model_config.obj.modality:
            model_config.obj.modality[kk]['modality_latent_len'] = args.modality_latent_len
    
    if args.fusion_type != 'None':
        model_config.obj.network.fusion_type = args.fusion_type
        args.fusion_type = model_config.obj.network.fusion_type
    else:
        args.fusion_type = model_config.obj.network.fusion_type
        
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
    
    model.to(device)
    # model.apply(network_init)
    model_dumper = ModelDumper(args.path, args.seed, args.cpt_name, model_config.obj.modality, args)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.001)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=args.lr)
    accelerator = Accelerator(mixed_precision='fp16')
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]).to(device))
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = DiceLoss()
    # accelerator.prepare(
    #     model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    # )
    b_f1 = 0.0
    
    if args.train:
        for epoch in range(args.epochs):
            s_time = time.time()
            v_acc, precision, recall, f1 = finetune_epoch(model, loss_fn, optimizer, None,
                           train_dataloader, epoch, True, device, accelerator=accelerator)
            print(time.time()-s_time)
            if epoch > (args.epochs * 0.5):
                v_acc, precision, recall, f1_v = finetune_epoch(model, loss_fn, optimizer, None, valid_dataloader, epoch, False, device=device, accelerator=accelerator)
                print("Current ACC:{:.5f}".format(v_acc))
                if f1_v >= b_f1:
                    b_f1 = f1_v
                    model_dumper.dump(model)
            #     checkpoint_index += 1
            #     # raw_model = model.module if hasattr(model, "module") else model
            #     cpt_name = "{}_{}.pth".format(args.cpt_name, checkpoint_index) # cpt_name_id.pth
            #     b_cpt_name = cpt_name
            #     print("Save checkpoint, path:{}, file_name:{}".format(args.path, cpt_name))
            #     torch.save(model.state_dict(), os.path.join(args.path, cpt_name))
            #     model.save(os.path.join(args.path, cpt_name))
    model.load_state_dict(torch.load(model_dumper.model_path))
    t_metrix, pred_dict = prediction(model, loss_fn, optimizer, schedular, test_dataloader, args.epochs, False, device=device)
    print("Final - ACC: {:.5f}".format(t_metrix[0]))
    print(f"Final - Precision: {t_metrix[1]}")
    print(f"Final - Recall: {t_metrix[2]}")
    print(f"Final - F1: {t_metrix[3]}")
    model_dumper.dump_json(pred_dict)
    model_dumper.dump_results(
        {
            'Acc': t_metrix[0],
            'Precision': t_metrix[1],
            'Recall': t_metrix[2],
            'F1': t_metrix[3]
        }
    )

if __name__ == '__main__':
    print("Training Arguments : {}".format(args))
    set_seed(args.seed)
    main(args)
