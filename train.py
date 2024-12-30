import os
import sys
import time
import subprocess
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
#
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader # py stl
#
from loader import Loader   # 自己写的函数
from utils.logger import Logger # 自己写的函数，主要是对SummaryWriter的封装
from utils.utils import AverageMeterForDict
from utils.utils import save_ckpt, set_seed


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation intervals")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--rank_metric", required=False, type=str, default="brier_fde_k", help="Ranking metric")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()

    faulthandler.enable()   # 启用 Python 的 faulthandler 模块。该模块用于在 Python 程序崩溃时（例如发生段错误或非法指令），自动打印出当前的栈跟踪信息
    start_time = time.time()
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, log_dir=log_dir, enable_flags={'writer': args.logger_writer})    # 自己写的函数，主要是对SummaryWriter的封装
    # log basic info
    logger.log_basics(args=args, datetime=date_str)

    loader = Loader(args, device, is_ddp=False) # 自己写的函数
    if args.resume:
        logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()


    # torch.utils.data.DataLoader(
    #     dataset,  # 数据集，实现了 __getitem__() 和 __len__() 方法的数据集对象。
    #     batch_size=1,  # 每个batch的样本数，默认是1。
    #     shuffle=False,  # 是否在每个epoch开始时打乱数据顺序，默认不打乱。
    #     sampler=None,  # 自定义从数据集中抽取样本的方法，如果指定了sampler，则shuffle必须为False。
    #     batch_sampler=None,  # 自定义batch的抽样方法，如果指定了这个参数，则不能指定batch_size, shuffle, sampler和drop_last。
    #     num_workers=0,  # 使用多少个子进程来加载数据，默认是0，表示在主进程中加载。
    #     collate_fn=None,  # 合并样本列表成mini-batch的函数，默认是将样本堆叠起来。
    #     pin_memory=False,  # 如果True，则会将数据加载到pinned memory中，这样可以加速从CPU向GPU的传输。
    #     drop_last=False,  # 如果数据集大小不能被batch size整除，是否丢弃最后一个不完整的batch，默认不丢弃。
    #     timeout=0,  # 如果使用了num_workers>0，此参数用于设置获取下一个batch的最大等待时间，默认是0，表示无限制。
    #     worker_init_fn=None,  # 如果不是None，将在每个worker初始化时调用此函数。
    #     multiprocessing_context=None,  # 设置多进程上下文环境。
    #     generator=None,  # 用于随机数生成的torch.Generator对象。
    #     prefetch_factor=2,  # 每个worker预先加载的数据量，默认值为2。
    #     persistent_workers=False,  # 如果为True， DataLoader的子进程在迭代结束时不销毁，这可以在多个epoch间减少启动开销。
    # )
    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          shuffle=True,
                          num_workers=8,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          pin_memory=True)  # py stl
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        pin_memory=True)

    niter = 0
    best_metric = 1e6
    rank_metric = args.rank_metric  # 从命令行参数中获取用于评估模型性能的排名指标，并将其赋值给变量 rank_metric。这个指标将在后续的验证过程中用于比较和保存最佳模型
    net_name = loader.network_name()

    for epoch in range(args.train_epoches):
        logger.print('\nEpoch {}'.format(epoch))
        torch.cuda.empty_cache()    # 释放未被占用但不可用的缓存内存
        torch.cuda.reset_peak_memory_stats()    # 重置当前设备的峰值内存使用统计信息，方便后续监控和调试

        # * Train
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()    # train_loss_meter用于跟踪训练过程中每个批次的损失值，在训练循环中，train_loss_meter 用于累积和更新每个批次的损失值
        train_eval_meter = AverageMeterForDict()
        net.train() # 更改net mode：启用梯度计算、激活 Dropout 和 BatchNorm 层

        # 1. 遍历数据：通过 enumerate 函数获取每个批次的数据及其索引。
        # 2. 进度条显示：使用 tqdm 库显示进度条，控制台会实时更新训练进度。disable=args.no_pbar 控制是否显示进度条，ncols=80 设置进度条宽度为80个字符。
        for i, data in enumerate(tqdm(dl_train, disable=args.no_pbar, ncols=80)):
            
            # data_in如下：
            # actors: [108， 14， 48]。这个108是变化的actor数量，14是actor的feature数量，48是actor的obs长度
            # actor_idcs: 4组(args.train_batch_size)，0~20， 21~39， 40~65， 66~107。共108个actor
            # lanes: [256, 10, 16]。这个256是变化的lane数量，10是lane的feature数量，16是lane的obs长度
            # lane_idcs: 4组(args.train_batch_size), 0~109, 110~172, 173~198, 199~255。共256个lane
            # rpe: 4组(args.train_batch_size)，[5, 131, 131], [5, 82, 82], [5, 52, 52], [5, 99, 99]。actors和lanes的全连接GNN
            data_in = net.pre_process(data)

            # out如下：
            # 3组数据，分别是res_cls, res_reg, res_aux
            # res_cls有4组(args.train_batch_size)数据，维度是[N_{actor}, n_mod]：[21, 6], [19, 6], [26, 6], [42, 6]
            # res_reg有4组(args.train_batch_size)数据，维度是[N_{actor}, n_mod, pred_len, 2]: [21, 6, 60, 2], [19, 6, 60, 2], [26, 6, 60, 2], [42, 6, 60, 2]
            # res_aux有4组(args.train_batch_size)数据...
            out = net(data_in)

            loss_out = loss_fn(out, data)

            post_out = net.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            lr = optimizer.step()

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)
            niter += args.train_batch_size
            logger.add_dict(loss_out, niter, prefix='train/')

        # print('epoch: {}, lr: {}'.format(epoch, lr))
        optimizer.step_scheduler()
        max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20

        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}, peak mem: {} MB'.
                     format(loss_avg, (time.time() - epoch_start) / 60.0, lr, max_memory))
        logger.print('-- ' + train_eval_meter.get_info())

        logger.add_scalar('train/lr', lr, it=epoch)
        logger.add_scalar('train/max_mem', max_memory, it=epoch)
        for key, elem in train_eval_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)

        if ((epoch + 1) % args.val_interval == 0) or epoch > int(args.train_epoches / 2):
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val, disable=args.no_pbar, ncols=80)):
                    data_in = net.pre_process(data)
                    out = net(data_in)
                    loss_out = loss_fn(out, data)

                    post_out = net.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                    val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                logger.print('-- ' + val_eval_meter.get_info())

                for key, elem in val_loss_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                for key, elem in val_eval_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                if (epoch >= args.train_epoches / 2):
                    if val_eval_meter.metrics[rank_metric].avg < best_metric:
                        model_name = date_str + '_{}_best.tar'.format(net_name)
                        save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
                        best_metric = val_eval_meter.metrics[rank_metric].avg
                        print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, rank_metric, best_metric, epoch))

        if int(100 * epoch / args.train_epoches) in [20, 40, 60, 80]:
            model_name = date_str + '_{}_ckpt_epoch{}.tar'.format(net_name, epoch)
            save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
            logger.print('Save the model to {}'.format('saved_models/' + model_name))

    logger.print("\nTraining completed in {:.2f} mins".format((time.time() - start_time) / 60.0))

    # save trained model
    model_name = date_str + '_{}_epoch{}.tar'.format(net_name, args.train_epoches)
    save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
    print('Save the model to {}'.format('saved_models/' + model_name))

    print('\nExit...\n')


if __name__ == "__main__":
    main()
