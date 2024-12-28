import imp
import os
import subprocess
import sys
import datetime
import logging
import torch
import numpy as np
#
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, date_str, enable=True, log_dir='', enable_flags={'writer': True}):
        """
        初始化Logger类的构造函数。

        参数:
        - date_str: 日期字符串，用于日志文件命名。
        - enable: 布尔值，表示是否启用日志记录功能。
        - log_dir: 日志文件保存的目录路径。如果未指定，则使用默认路径。
        - enable_flags: 字典，包含不同日志记录功能的启用标志。默认启用'writer'功能。
        """
        # 设置日志记录的全局启用状态
        self.enable = enable
        # 初始化功能启用标志，确保每个标志都根据全局启用状态进行调整
        self.enable_flags = enable_flags
        for k, v in self.enable_flags.items():  # k = 'writer', v = 'True'
            self.enable_flags[k] = v and self.enable

        # 保存日期字符串，用于后续的日志信息
        self.date_str = date_str
        # 打印Logger初始化信息，包括日期和功能启用标志
        self.print('[Logger] {} - logger enable_flags: {}'.format(self.date_str, self.enable_flags))

        # 根据提供的log_dir参数确定日志文件的保存路径
        if log_dir == '':
            self.log_dir = "log/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.log_dir = log_dir

        # 如果启用了'writer'功能，则创建SummaryWriter实例并配置日志文件
        if self.enable_flags['writer']:
            self.writer = SummaryWriter(self.log_dir)

            # 定义日志文件的文件名和格式
            filename = 'log/log_{}.log'.format(self.date_str)
            # 清除现有的日志处理器，避免重复记录
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            # 配置日志记录的基本设置，包括格式、级别和文件模式
            logging.basicConfig(filename=filename,
                                format='%(asctime)s:%(message)s', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S')

    def log_basics(self, args, datetime):
        # log basic info
        self.print("Args: {}\n".format(args))
        _, ret = subprocess.getstatusoutput("echo $PWD")
        self.print("Project Path: {}".format(ret))
        self.print("Datetime: {}\n".format(datetime))
        _, ret = subprocess.getstatusoutput("git log -n 1")
        self.print("Commit Msg: {}\n".format(ret))

        self.print("======================================\n")

    def add_scalar(self, title, value, it):
        if self.enable_flags['writer']:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.writer.add_scalar(title, value, it)

    def add_dict(self, data, it, prefix=''):
        if self.enable_flags['writer']:
            for key, val in data.items():
                title = prefix + key
                self.add_scalar(title, val, it)

    def print(self, info):
        if self.enable:
            print(info)
            logging.info(info)
