import os
import random
from typing import Any, Dict, List, Tuple, Union
import argparse
from importlib import import_module
#
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
#
from utils.optimizer import Optimizer
from utils.evaluator import TrajPredictionEvaluator


class Loader:
    '''
        Get and return dataset, network, loss_fn, optimizer, evaluator
    '''

    def __init__(self, args, device, is_ddp=False, world_size=1, local_rank=0, verbose=True):
        self.args = args
        self.device = device
        self.is_ddp = is_ddp
        self.world_size = world_size
        self.local_rank = local_rank
        self.resume = False
        self.verbose = verbose

        self.print('[Loader] load adv_cfg from {}'.format(self.args.adv_cfg_path))
        # 使用 import_module 函数根据 self.args.adv_cfg_path 指定的路径动态导入AdvCfg 类并实例化赋值给 self.adv_cfg
        # AdvCfg()是个自定义的类，在simpl_av2_cfg.py
        self.adv_cfg = import_module(self.args.adv_cfg_path).AdvCfg()

    def print(self, info):
        if self.verbose:
            print(info)

    def set_resmue(self, model_path):
        self.resume = True
        if not model_path.endswith(".tar"):
            assert False, "Model path error - '{}'".format(model_path)
        self.ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

    def load(self):
        # dataset
        dataset = self.get_dataset()
        # network
        model = self.get_model()
        # loss_fn
        loss_fn = self.get_loss_fn()
        # optimizer
        optimizer = self.get_optimizer(model)
        # evaluator
        evaluator = self.get_evaluator()

        return dataset, model, loss_fn, optimizer, evaluator

    def get_dataset(self):
        """
        根据配置加载数据集。

        本函数根据配置文件中的数据集配置，动态导入数据集模块并创建训练、验证或测试数据集对象。
        不同模式下（训练、验证、测试）加载的数据集不同，以适应不同的使用场景。

        Returns:
            - 在训练或验证模式下，返回训练数据集和验证数据集的实例。
            - 在测试模式下，返回测试数据集的实例。
        """
        # 获取数据集配置
        data_cfg = self.adv_cfg.get_dataset_cfg()

        # 解析数据集配置，获取数据集文件名和数据集名称
        ds_file, ds_name = data_cfg['dataset'].split(':')
        self.print('[Loader] load dataset {} from {}'.format(ds_name, ds_file))

        # 根据配置获取训练、验证和测试数据集的目录路径
        train_dir = self.args.features_dir + 'train/'
        val_dir = self.args.features_dir + 'val/'
        test_dir = self.args.features_dir + 'test/'

        # 根据运行模式选择性地加载训练、验证或测试数据集
        if self.args.mode == 'train' or self.args.mode == 'val':
            # 导入模块：使用 import_module 根据 ds_file 指定的文件路径，动态导入指定的模块 ds_file: simpl.av2_dataset
            # 获取数据集类：通过 getattr 从导入的模块中获取名为 ds_name 的数据集类 ds_name: Av2Dataset。这貌似是个自定义的类，在av2_dataset.py里
            # 从simpl/av2_dataset.py里倒入AV2Dataset类
            train_set = getattr(import_module(ds_file), ds_name)(train_dir,
                                                                    mode='train',
                                                                    obs_len=data_cfg['g_obs_len'],
                                                                    pred_len=data_cfg['g_pred_len'],
                                                                    verbose=self.verbose,
                                                                    aug=self.args.data_aug)
            val_set = getattr(import_module(ds_file), ds_name)(val_dir,
                                                                mode='val',
                                                                obs_len=data_cfg['g_obs_len'],
                                                                pred_len=data_cfg['g_pred_len'],
                                                                verbose=self.verbose,
                                                                aug=False)
            return train_set, val_set
        elif self.args.mode == 'test':
            test_set = getattr(import_module(ds_file), ds_name)(test_dir,
                                                                    mode='test',
                                                                    obs_len=data_cfg['g_obs_len'],
                                                                    pred_len=data_cfg['g_pred_len'],
                                                                    verbose=self.verbose)
            return test_set
        else:
            # 如果模式未知，抛出异常
            assert False, "Unknown mode"

    def get_model(self):
        """
        获取并配置神经网络模型。

        该方法根据配置初始化神经网络模型，加载模型参数（如果需要恢复训练），并根据是否使用分布式数据并行（DDP）进行相应的配置。

        返回:
            model: 配置好的神经网络模型实例。
        """

        # 获取网络配置信息
        net_cfg = self.adv_cfg.get_net_cfg()
        net_file, net_name = net_cfg['network'].split(':')  # simpl.simpl， Simpl

        # 打印加载的网络信息
        self.print('[Loader] load network {} from {}'.format(net_name, net_file))
        model = getattr(import_module(net_file), net_name)(net_cfg, self.device)    # 从simpl/simpl.py里导入Simpl类

        # 打印网络参数统计信息
        # 1. model.parameters() 返回模型中所有可训练参数的生成器。
        # 2. p.numel() 返回每个参数张量中的元素个数。
        # 3. sum() 函数对所有参数的元素个数求和，得到总的参数数量。
        total_num = sum(p.numel() for p in model.parameters())
        self.print('[Loader] network params:')
        self.print('-- total: {}'.format(total_num))

        # 统计并打印每个子网的参数数量：从模型的参数名称中提取子网名称，并存入列表 subnets
        subnets = list()
        for name, param in model.named_parameters():    # model.named_parameters() 返回一个生成器，生成模型中所有参数的名称和参数对象
            subnets.append(name.split('.')[0])
        subnets = list(set(subnets))    # 使用集合去重后，再将结果转换为列表
        for subnet in subnets:
            numelem = 0 # 对于每个子网络，初始化计数器 numelem 为 0
            for name, param in model.named_parameters():
                if name.startswith(subnet): # 遍历模型的所有参数，如果参数名称以当前子网络名称开头，则累加该参数的元素数量到 numelem
                    numelem += param.numel()
            self.print('-- {} {}'.format(subnet, numelem))

        # 如果需要恢复训练，则加载模型参数
        if self.resume:
            model.load_state_dict(self.ckpt["state_dict"])

        # 根据是否使用分布式数据并行（DDP）进行模型配置
        if self.is_ddp:
            # 将模型转换为同步批量归一化，并移动到指定设备
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            # 使用DDP封装模型
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            # 直接将模型移动到指定设备
            model = model.to(self.device)

        return model

    def get_loss_fn(self):
        loss_cfg = self.adv_cfg.get_loss_cfg()
        loss_file, loss_name = loss_cfg['loss_fn'].split(':')   # simpl.av2_loss_fn, LossFunc

        self.print('[Loader] loading loss {} from {}'.format(loss_name, loss_file))
        # 1. 使用 import_module 导入 loss_file 指定的模块。
        # 2. 使用 getattr 从导入的模块中获取名为 loss_name 的类。
        # 3. 调用该类的构造函数，传入 loss_cfg 和 self.device 参数，创建实例并赋值给 loss。
        loss = getattr(import_module(loss_file), loss_name)(loss_cfg, self.device)  # 从simpl/av2_loss_fn.py 中导入 LossFunc 类
        return loss

    def get_optimizer(self, model):
        opt_cfg = self.adv_cfg.get_opt_cfg()

        if opt_cfg['lr_scale_func'] == 'linear':
            opt_cfg['lr_scale'] = self.world_size
        elif opt_cfg['lr_scale_func'] == 'sqrt':
            opt_cfg['lr_scale'] = np.sqrt(self.world_size)
        else:
            opt_cfg['lr_scale'] = 1.0

        optimizer = Optimizer(model, opt_cfg)

        if self.resume:
            optimizer.load_state_dict(self.ckpt["opt_state"])

        return optimizer

    def get_evaluator(self):
        eval_cfg = self.adv_cfg.get_eval_cfg()
        eval_file, eval_name = eval_cfg['evaluator'].split(':')

        evaluator = getattr(import_module(eval_file), eval_name)(eval_cfg)
        return evaluator

    def network_name(self):
        _, net_name = self.adv_cfg.get_net_cfg()['network'].split(':')
        return net_name
