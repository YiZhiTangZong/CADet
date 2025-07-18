import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

r""" cadet可视化代码（适配实际数据结构版） """
import argparse
import os
import numpy as np
from PIL import Image
import torch

from common import config
from common import utils
from data.dataset import CSDataset
from models import create_model

def get_parser():
    parser = argparse.ArgumentParser(description='cadet Visualization')
    parser.add_argument('--config', type=str, default='/autodl-tmp/cadet/config/test/cadet_crackls315.yaml', help='模型配置文件')
    parser.add_argument('--output', type=str, default='visualize/cadet_crackls315', help='输出目录')
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.output = args.output
    return cfg

def generate_comparison_mask(true_mask, pred_mask):
    """生成三色对比掩码（适配实际形状）"""
    assert true_mask.shape == pred_mask.shape, f"形状不匹配: 真值{true_mask.shape} vs 预测{pred_mask.shape}"
    
    comparison = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    
    true_bool = (true_mask > 0.5)
    pred_bool = (pred_mask > 0.5)
    
    # 正确检测（白色）
    comparison[np.logical_and(true_bool, pred_bool)] = [255, 255, 255]
    # 漏检（红色）
    comparison[np.logical_and(true_bool, ~pred_bool)] = [255, 0, 0]
    # 误检（蓝色）
    comparison[np.logical_and(~true_bool, pred_bool)] = [0, 0, 255]
    
    return comparison

def save_results(batch, pred, output_dir, idx):
    """保存结果并生成拼接图"""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    # 处理原始图像
    img_tensor = batch['img'][0].cpu().float()
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = (img * STD + MEAN) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 处理预测结果
    true_mask = batch['anno_mask'][0].cpu().numpy().squeeze()
    pred_mask = pred[0].cpu().numpy().squeeze()
    comparison = generate_comparison_mask(true_mask, pred_mask)

    # 转换为PIL图像
    img_pil = Image.fromarray(img)
    comp_pil = Image.fromarray(comparison)

    # 创建拼接图像
    combined = Image.new('RGB', (img_pil.width + comp_pil.width, img_pil.height))
    combined.paste(img_pil, (0, 0))
    combined.paste(comp_pil, (img_pil.width, 0))

    # 保存所有结果
    # img_pil.save(f"{output_dir}/original_{idx:03d}.png")
    # comp_pil.save(f"{output_dir}/comparison_{idx:03d}.png")
    combined.save(f"{output_dir}/combined_{idx:03d}.png")

def main():
    args = get_parser()
    os.makedirs(args.output, exist_ok=True)
    
    model = create_model(args).cuda()
    
    if args.weight and os.path.isfile(args.weight):
        print(f"加载权重: {args.weight}")
        checkpoint = torch.load(args.weight, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError("未提供有效权重文件")
    
    CSDataset.initialize(datapath=args.datapath)
    dataloader = CSDataset.build_dataloader(
        args.benchmark,
        1,
        args.nworker,
        'val',
        False,
        None
    )
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= 40: break
            
            batch = utils.to_cuda({
                'img': batch['img'],
                'anno_mask': batch['anno_mask']
            })
            if args.dual_decoder:
                output = model(batch)[2]['output']
            else:
                output = model(batch)['output']

            save_results(batch, output, args.output, idx)
    
    print(f"结果已保存至: {os.path.abspath(args.output)}")

if __name__ == '__main__':
    main()