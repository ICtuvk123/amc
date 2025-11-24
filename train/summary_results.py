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

from model.dense_model import *
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
# parser.add_argument('--modality', default='screenImg', type=str, choices=['all', 'screenImg', 'screenWireframeImg'])
parser.add_argument('--modality_latent_len', default=-1, type=int)
parser.add_argument('--fusion_type', default='None', type=str)
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
            logits = model(x)
            
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accumulation_steps
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())
        predict_result.append((logits.argmax(dim=1) == y).cpu())
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
            acc = (logits.argmax(dim=1) == y).float().sum().item()
            pred_accs.append(acc)
        pbar.set_description(f"epoch {epoch_index} iter {it}: training loss {loss.item() * accumulation_steps:.5f}.")
    if not training:
        avg_acc = np.sum(pred_accs) / np.sum(batch_sizes)
        avg_loss = np.sum(losses) / len(losses)
        y_true_np = torch.cat(ground_truth, dim=0).numpy()
        y_pred_np = torch.cat(predict_result, dim=0).numpy()
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
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
        y_true_np = torch.cat(predict_result, dim=0).numpy()
        y_pred_np = torch.cat(ground_truth, dim=0).numpy()
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
        print(f"Training - Precision: {precision}, Recall: {recall}, F1: {f1}")
        return None, None, None, None

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
    # scaler = torch.cuda.amp.GradScaler()
    for it, dbatch in pbar:
        y = dbatch.pop('target')
        bidx = dbatch.pop('idx')
        x = {kk: dbatch[kk].to(device) for kk in dbatch}
        y = y.to(device)
        
        batch_sizes.append(y.shape[0])
        # print(x[0], y[0])
        with torch.set_grad_enabled(training):
            logits = model(x)
            
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accumulation_steps
            losses.append(loss.item())
            if logger:
                logger.add_scalar_auto('Training Loss', loss.item())
        predict_result.append((logits.argmax(dim=1) == y).cpu())
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
            acc = (logits.argmax(dim=1) == y).float().sum().item()
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
        y_true_np = torch.cat(predict_result, dim=0).numpy()
        y_pred_np = torch.cat(ground_truth, dim=0).numpy()
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
        print(f"Training - Precision: {precision}, Recall: {recall}, F1: {f1}")
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
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
        print(f"Training - Precision: {precision}, Recall: {recall}, F1: {f1}")

def input_dim(input_axis, freq_bands, channels):
    return input_axis * ((freq_bands * 2) + 1) + channels

def main(args):
    model_config = YmlConfig(args.model_config)
    
    if args.modality_latent_len > 0:
        for kk in model_config.obj.modality:
            model_config.obj.modality[kk]['modality_latent_len'] = args.modality_latent_len

    if args.fusion_type != 'None':
        model_config.obj.network.fusion_type = args.fusion_type
        
    if args.fusion_type != 'None':
        model_config.obj.network.fusion_type = args.fusion_type
        args.fusion_type = model_config.obj.network.fusion_type
    else:
        args.fusion_type = model_config.obj.network.fusion_type
    # modality = 'all' if len(model_config.obj.modality) > 1 else list(model_config.obj.modality.keys())[0]
    modality = model_config.obj.modality.keys()
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # train_dataset, valid_dataset, test_dataset = get_dataset(args.dataset_path, modalities=modality)

    # train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)
    # valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    # modality_config = {}
    # for kk in model_config.obj.modality:
    #     modality_config[kk] = model_config.parse_to_modality(
    #         model_config.obj.modality[kk]
    #     )
    # model = MultimodalTransformer(
    #     device,
    #     modality_config,
    #     **model_config.obj.network
    # )
    # model = eval(model_config.obj.network_type)(
    #         device,
    #         modality_config,
    #         **model_config.obj.network
    #     )
    all_results = []
    for i in [123, 132, 213, 231, 321]: #, 213, 231, 321
        model_dumper = ModelDumper(args.path, i, args.cpt_name, model_config.obj.modality, args)
        all_results.append(model_dumper.load_results())
    
    display_results = {}
    for i, item in enumerate(all_results):
        for key in item:
            if key not in display_results:
                display_results[key] = []
            display_results[key].append(item[key])
    # print(display_results)
    tmp_results = {}
    for key in display_results:
        tmp_results[key] = display_results[key]
        tmp_results[f"{key}-mean"] = np.mean(display_results[key]) * 100
        tmp_results[f"{key}-std"] = np.std(display_results[key]) * 100
        # print(f"{key}: {np.mean(display_results[key])}, std: {np.std(display_results[key])}")

    model_dumper = ModelDumper(args.path, 123, args.cpt_name, model_config.obj.modality, args)
    model_dumper.dump_json_cross_seeds(tmp_results)

if __name__ == '__main__':
    print("Training Arguments : {}".format(args))
    set_seed(args.seed)
    main(args)
