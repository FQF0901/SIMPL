"""This script is for dataset preprocessing."""

import os
from os.path import expanduser
import time
from typing import Any, Dict, List, Tuple
import random
#
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import pickle as pkl
#
from argo_preprocess import ArgoPreproc

_FEATURES_SMALL_SIZE = 1024


def parse_arguments() -> Any:
    """
    Parse command line arguments：解析命令行参数。
    
    此函数使用argparse模块设置并解析命令行参数，允许通过命令行指定数据目录、保存目录、模式、轨迹观测长度、预测时间范围、调试模式等参数。
    
    返回:
        Any: 返回解析后的参数对象，其中包含所有指定参数的值。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="存储序列（CSV文件）的目录",
    )
    parser.add_argument(
        "--save_dir",
        default="./dataset_argo/features/",
        type=str,
        help="存储计算特征的目录",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help="训练/验证/测试模式",
    )
    parser.add_argument(
        "--obs_len",
        default=20,
        type=int,
        help="轨迹观测长度",
    )
    parser.add_argument(
        "--pred_len",
        default=30,
        type=int,
        help="预测时间范围",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="如果为真，使用小部分数据",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="如果为真，启用调试模式",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="如果为真，启用可视化",
    )
    return parser.parse_args()


def load_seq_save_features(args: Any, start_idx: int, batch_size: int, sequences: List[str],
                           save_dir: str, thread_idx: int) -> None:
    """
    加载序列数据，计算特征并保存。

    参数:
    - args (Any): 配置参数对象。
    - start_idx (int): 批处理的起始索引。
    - batch_size (int): 批处理大小。
    - sequences (List[str]): 序列文件列表。
    - save_dir (str): 保存目录路径。
    - thread_idx (int): 线程索引。

    返回:
    - None
    """

    # 初始化数据预处理对象，设置 verbose 为 False 以抑制不必要的输出
    dataset = ArgoPreproc(args, verbose=False)

    # 从 start_idx 开始遍历批处理中的序列
    for seq in sequences[start_idx:start_idx + batch_size]:
        if not seq.endswith(".csv"):
            continue  # 跳过非 CSV 文件

        # 提取序列 ID 和路径
        seq_id = int(seq.split(".")[0])
        seq_path = f"{args.data_dir}/{seq}"

        # 读取 CSV 文件
        df = pd.read_csv(seq_path, dtype={'TIMESTAMP': str,
                                          'TRACK_ID': str,
                                          'OBJECT_TYPE': str,
                                          'X': float,
                                          'Y': float,
                                          'CITY_NAME': str})

        # 处理数据并获取特征和表头
        data, headers = dataset.process(seq_id, df)

        # 如果不是调试模式，则将数据保存为 Pickle 文件
        if not args.debug:
            data_df = pd.DataFrame(data, columns=headers)
            filename = '{}'.format(data[0][0])
            data_df.to_pickle(f"{save_dir}/{filename}.pkl")  # compression='gzip'

    # 打印完成信息
    print('Finish computing {} - {}'.format(start_idx, start_idx + batch_size))


if __name__ == "__main__":
    """
    加载序列并保存计算的特征。
    """
    # 记录执行开始时间
    start = time.time()
    # 1.解析命令行参数
    args = parse_arguments()

    # 2.获取数据目录中的所有文件列表
    sequences = os.listdir(args.data_dir)
    # 可选地打乱文件列表
    # random.shuffle(sequences)

    # 根据是否启用小数据集选项确定处理的序列数量
    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)
    # 截取序列列表到所需的数量
    sequences = sequences[:num_sequences]
    print("Num of sequences: ", num_sequences)

    # 3.不debug的话全面调用CPU
    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1

    # 计算每个进程处理的批处理大小
    batch_size = np.max([int(np.ceil(num_sequences / n_proc)), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    # 创建保存目录
    save_dir = args.save_dir + f"{args.mode}"
    os.makedirs(save_dir, exist_ok=True)
    print('save processed dataset to {}'.format(save_dir))

    # 并行处理每个批处理
    Parallel(n_jobs=n_proc)(delayed(load_seq_save_features)(args, i, batch_size, sequences, save_dir, k)
                            for i, k in zip(range(0, num_sequences, batch_size), range(len(range(0, num_sequences, batch_size)))))
    
    # 4.输出处理完成的时间
    print(f"Preprocess for {args.mode} set completed in {(time.time()-start)/60.0:.2f} mins")
