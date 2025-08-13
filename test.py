import numpy as np
import torch
import argparse
import torch.nn as nn
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader

from albumentations.augmentations import transforms
from albumentations import RandomRotate90, Resize, HorizontalFlip
from albumentations.core.composition import Compose
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from hausdorff import hausdorff_distance
from thop import profile
from DualDecoder.vision_transformer import CSwinUnet as ViT_seg
from datasets.dataset_BUSI import BUSI, load_dataset
import os
import time
import matplotlib.pyplot as plt
import pylab as pl
from config import get_config
import math
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test_metrics', type=str, default='Dice', choices=['Dice', 'All'], help="start epoch for test")

parser.add_argument('--root_path', type=str, help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='BUSI', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', default='./checkpoint', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=False, default='./configs/cswin_tiny_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = "./data/Thyroid Dataset/tn3k"

pre_model_path = "./checkpoint/TN3K"

save_output_path = "./data/TN3K/outputs"
os.makedirs(save_output_path, exist_ok=True)

val_transform = Compose([
    Resize(224, 224),
    transforms.Normalize(),
])

def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        from itertools import ifilterfalse
        from math import isnan
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def get_DC(pred, target):
    smooth = 1e-6
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()
    return float(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def get_specificity(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    tn = torch.sum((pred == 0) & (target == 0)).float()
    fp = torch.sum((pred == 1) & (target == 0)).float()
    return (tn + smooth) / (tn + fp + smooth)

def get_sensitivity(predict, target):
    smooth = 1e-6
    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)
    intersection = float((target * predict).sum())
    return (intersection + smooth) / (float(target.sum()) + smooth)

def get_HD(pred, target):
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0 # 或者其他合适的值，例如NaN
    HD = hausdorff_distance(torch.squeeze(pred).cpu().numpy(), torch.squeeze(target).cpu().numpy(), distance="euclidean")
    return HD

def calculate_metrics(pred, target):
    pred_bin = (pred > 0.5).cpu().numpy().astype(np.uint8)
    target_bin = target.cpu().numpy().astype(np.uint8)
    
    metrics = {}
    try:
        tn, fp, fn, tp = confusion_matrix(target_bin.flatten(), pred_bin.flatten()).ravel()
    except ValueError:
        if np.all(pred_bin == 0) and np.all(target_bin == 0):
            tn, fp, fn, tp = pred_bin.size, 0, 0, 0
        elif np.all(pred_bin == 1) and np.all(target_bin == 1):
            tn, fp, fn, tp = 0, 0, 0, pred_bin.size
        elif np.all(pred_bin == 0):
            tn, fp, fn, tp = np.sum(target_bin == 0), 0, np.sum(target_bin == 1), 0
        elif np.all(pred_bin == 1):
            tn, fp, fn, tp = 0, np.sum(target_bin == 0), 0, np.sum(target_bin == 1)
        else:
            tn, fp, fn, tp = confusion_matrix(target_bin.flatten(), pred_bin.flatten()).ravel()

    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    metrics['precision'] = tp / (tp + fp + 1e-6)
    metrics['recall'] = tp / (tp + fn + 1e-6)
    metrics['specificity'] = tn / (tn + fp + 1e-6)
    metrics['dice'] = 2 * tp / (2 * tp + fp + fn + 1e-6)
    metrics['iou'] = tp / (tp + fp + fn + 1e-6)
    metrics['auc'] = roc_auc_score(target_bin.flatten(), pred.cpu().flatten()) if len(np.unique(target_bin)) > 1 else 0.5
    metrics['hd'] = get_HD(pred, target)
    return metrics

def calculate_flops_params(model, input_shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops, params

def predice():
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    model.load_from(config)
    
    us_dataset = BUSI(base_dir=root_path, split="test", transform=val_transform)
    dataloaders = DataLoader(us_dataset, batch_size=1, drop_last=True)
    imgs = load_dataset(os.path.join(root_path, "dataset.json"), split='test')

    model_path = os.path.join(pre_model_path, f'epoch_{epoch_num}.pth')
        
    print(f"\nTesting model from epoch {epoch_num}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    current_metrics_list = defaultdict(list)
    
    with torch.no_grad():
        for count, sample in enumerate(dataloaders):
            x, y_tensor = sample['image'].to(device), sample['label'].to(device)
            outputs = model(x)
            probabilities = torch.softmax(outputs, dim=1)
            pred_tensor = torch.argmax(probabilities, dim=1)

            pred_np = torch.squeeze(pred_tensor).cpu().numpy().astype(np.uint8) ##### MODIFIED #####
            y_np = (torch.squeeze(y_tensor).cpu().numpy() > 0.5).astype(np.uint8)   ##### NEW #####

            # 将预测图像转换为三通道（如果需要） ##### NEW #####
            if len(pred_np.shape) == 2:
                pred_color = cv2.cvtColor(pred_np * 255, cv2.COLOR_GRAY2BGR)
            else:
                pred_color = pred_np

            # 检测 Ground Truth 的边缘 ##### NEW #####
            gt_blur = cv2.GaussianBlur(y_np, (5, 5), 0)
            contours, _ = cv2.findContours(gt_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 在预测图像上用红色标记 Ground Truth 的边缘 ##### NEW #####
            cv2.drawContours(pred_color, contours, -1, (0, 0, 255), 1) # BGR红色, 线宽为1

            save_name = os.path.basename(imgs[count]['mask'])
            cv2.imwrite(os.path.join(save_output_path, save_name), pred_color)

            metrics = calculate_metrics(pred_tensor, y_tensor)
            for k, v in metrics.items():
                current_metrics_list[k].append(v)
        
        avg_metrics = {k: np.mean(v) for k, v in current_metrics_list.items()}
        std_metrics = {k: np.std(v) for k, v in current_metrics_list.items()}

    total_flops, total_params = calculate_flops_params(model)
    print(f"\nModel Statistics:")
    print(f"Total FLOPs: {total_flops / 1e9:.2f}G")
    print(f"Total Params: {total_params / 1e6:.2f}M")

    print("=" * 50)
    print("\nDetailed Metrics (Mean ± Std Dev):")
    print("-" * 40)
    for metric, value in avg_metrics.items():
        std_dev = std_metrics.get(metric, 0.0)
        print(f"{metric:12s}: {value:.4f} ± {std_dev:.4f}")
    print("-" * 40)

if __name__ == '__main__':
    predice()