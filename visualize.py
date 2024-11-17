import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
from importlib import import_module
#
import torch
from torch.utils.data import DataLoader
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    #
    parser.add_argument("--seq_id", default=-1, type=int, help="Selected sequence ID")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle order")
    parser.add_argument("--visualizer", default="", type=str, help="Type of visualizer")
    parser.add_argument("--show_conditioned", action="store_true", help="Show missed sample only")
    return parser.parse_args()


def main():
    """
    主函数，负责根据命令行参数执行模型的加载、训练、验证或测试。
    """
    # 解析命令行参数
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    # 根据参数选择使用CPU还是GPU
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    # 生成当前日期和时间的字符串，用于日志等
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 解析可视化工具的参数，并加载相应的可视化模块
    vis_file, vis_name = args.visualizer.split(':')
    print('[Loader] load visualizer {} from {}'.format(vis_name, vis_file))
    vis = getattr(import_module(vis_file), vis_name)()

    # 根据参数选择不同的执行模式：训练、验证或测试
    if args.mode != 'test':
        # 初始化数据加载器
        loader = Loader(args, device, is_ddp=False)
        # 加载模型和数据集
        print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
        (train_set, val_set), net, _, _, evaluator = loader.load()
        net.eval()  # 将模型的一些层（如 Dropout 层和 BatchNorm 层）的行为从训练模式切换到评估模式

        # 根据模式选择数据集
        if args.mode == 'train':
            dataloader = DataLoader(train_set,
                                    batch_size=1,
                                    shuffle=args.shuffle,
                                    num_workers=0,
                                    collate_fn=train_set.collate_fn,
                                    drop_last=False)
        elif args.mode == 'val':
            dataloader = DataLoader(val_set,
                                    batch_size=1,
                                    shuffle=args.shuffle,
                                    num_workers=0,
                                    collate_fn=val_set.collate_fn,
                                    drop_last=False)

        # 对数据集进行评估
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                if args.seq_id == -1:
                    data_in = net.pre_process(data)
                    out = net(data_in)
                    post_out = net.post_process(out)
                    torch.cuda.synchronize()    # # 等待所有CUDA操作完成，确保结果的正确性

                    eval_out = evaluator.evaluate(post_out, data)

                    if args.show_conditioned:
                        if eval_out['mr_k'] == 0.0:
                            continue

                    print(f'\n\nSequence ID: {data["SEQ_ID"][0]}')
                    print(f'Evaluation Metrics:')
                    for name, val in eval_out.items():
                        print('-- {}: {:.4}'.format(name, val))

                    vis.draw_once(post_out, data, eval_out, show_map=True, split=args.mode)

                else:
                    if args.seq_id == data['SEQ_ID'][0]:
                        data_in = net.pre_process(data)
                        out = net(data_in)
                        post_out = net.post_process(out)

                        eval_out = evaluator.evaluate(post_out, data)

                        print(f'\n\nSequence ID: {data["SEQ_ID"][0]}')
                        print(f'Evaluation Metrics:')
                        for name, val in eval_out.items():
                            print('-- {}: {:.4}'.format(name, val))

                        vis.draw_once(post_out, data, eval_out, show_map=True, split=args.mode)
                        break

    else:
        # 测试模式
        loader = Loader(args, device, is_ddp=False)
        print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
        test_set, net, _, _, _ = loader.load()
        net.eval()

        dl_test = DataLoader(test_set,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False,
                             collate_fn=test_set.collate_fn)

        with torch.no_grad():
            for i, data in enumerate(tqdm(dl_test)):
                data_in = net.pre_process(data)
                out = net(data_in)
                post_out = net.post_process(out)

                vis.draw_once(post_out, data, {}, show_map=True, test_mode=True, split=args.mode)

    print('\nExit...')


if __name__ == "__main__":
    main()
