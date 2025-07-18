r""" cadet training code """
import argparse
import os

import torch.nn as nn
import torch

from common.logger import AverageMeter
from common.evaluation import Evaluator
from common import config
from common import utils
from data.dataset import CSDataset
from models import create_model

def get_parser():
    parser = argparse.ArgumentParser(description='cadet for Curvilinear Structure Segmentation')
    parser.add_argument('--config', type=str, default='config/UNet_DRIVE.yaml', help='Model config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def main():
    global args
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    # Model initialization
    model = create_model(args)

    print("=> creating model ...")
    print("Classes: {}".format(args.classes))

    # Device setup
    print('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = model.cuda()
        model = nn.DataParallel(model)
        print('Use GPU Parallel.')
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model

    if args.weight:
        if os.path.isfile(args.weight):
            print("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weight '{}'".format(args.weight))
        else:
            print("=> no weight found at '{}'".format(args.weight))
    else:
        raise RuntimeError("Please support weight.")

    Evaluator.initialize()

    # Dataset initialization
    CSDataset.initialize(datapath=args.datapath)
    dataloader_val = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size_val,
                                                args.nworker,
                                                'val',
                                                'same',
                                                None)

    with torch.no_grad():
        val_loss_dict, val_f1, val_pr, val_r, val_quality, val_cor, val_com, val_ap = evaluate(model, dataloader_val)

    print('F1: {:.2f} Precision: {:.2f} Recall: {:.2f} AP: {:.2f}'.format(val_f1.item(), val_pr.item(), val_r.item(), val_ap.item()*100))
    print('Quality: {:.2f} Correctness: {:.2f} Completeness: {:.2f}'.format(val_quality.item(), val_cor.item(), val_com.item()))
    print('==================== Finished Testing ====================')

def evaluate(model, dataloader):
    r""" Eval cadet """
    # Force randomness during training / freeze randomness during testing
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Forward pass
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch

        # 修改2：根据 args.dual_decoder 判断是否启用双解码器
        if args.dual_decoder:
            # 双解码器模式，返回三个结果
            output_dict, output_dict2, merged_dict = model(batch)
            out = merged_dict['output']  # 修改3：评估时只使用 merge 的结果
        else:
            # 单解码器模式，返回一个结果
            output_dict = model(batch)
            out = output_dict['output']

        pred_mask = torch.where(out >= 0.5, 1, 0)

        # 2. Compute loss & update model parameters
        # 修改4：根据 args.dual_decoder 判断是否启用双解码器
        if args.dual_decoder:
            # 双解码器模式，计算三个损失
            loss_dict1 = model.module.compute_objective(output_dict, batch) if torch.cuda.device_count() > 1 else model.compute_objective(output_dict, batch_dict=batch)
            loss_dict2 = model.module.compute_objective(output_dict2, batch) if torch.cuda.device_count() > 1 else model.compute_objective(output_dict2, batch_dict=batch)
            loss_dict_merged = model.module.compute_objective(merged_dict, batch, multi_layer=False) if torch.cuda.device_count() > 1 else model.compute_objective(merged_dict, batch_dict=batch)

            # 合并三个损失
            total_loss = loss_dict1['total_loss'] + loss_dict2['total_loss'] + loss_dict_merged['total_loss']
            loss_dict = {
                'total_loss': total_loss,
                'loss1': loss_dict1['total_loss'],
                'loss2': loss_dict2['total_loss'],
                'loss_merged': loss_dict_merged['total_loss']
            }
        else:
            # 单解码器模式，计算一个损失
            loss_dict = model.module.compute_objective(output_dict, batch) if torch.cuda.device_count() > 1 else model.compute_objective(output_dict, batch_dict=batch)

        # 3. Evaluate prediction
        f1, pr, r, quality, cor, com = Evaluator.classify_prediction(pred_mask.clone(), batch)
        ap = Evaluator.compute_ap(out.clone(), batch)  # 新增AP计算
        average_meter.update(f1, pr, r, quality, cor, com, loss_dict, ap)  # 传入AP值

    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    f1 = average_meter.compute_f1()
    pr = average_meter.compute_precision()
    r = average_meter.compute_recall()
    quality = average_meter.compute_quality()
    cor = average_meter.compute_correctness()
    com = average_meter.compute_completeness()
    ap = average_meter.compute_ap()  # 获取平均AP

    return avg_loss_dict, f1, pr, r, quality, cor, com, ap

if __name__ == '__main__':
    main()