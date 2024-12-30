from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long


class LossFunc(nn.Module):  # 继承自 LossFunc 类
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device

        self.yaw_loss = config['yaw_loss']  # bool, 用于决定是否计算方向角损失（yaw angle loss）
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")    # 使用 SmoothL1Loss 损失函数，它对异常值更鲁棒

    def forward(self, out, data):
        """
        输入：
            out: 3组数据，分别是res_cls, res_reg, res_aux
                res_cls有4组(args.train_batch_size)数据，维度是[N_{actor}, n_mod]: [21, 6], [19, 6], [26, 6], [42, 6]
                res_reg有4组(args.train_batch_size)数据，维度是[N_{actor}, n_mod, pred_len, 2]: [21, 6, 60, 2], [19, 6, 60, 2], [26, 6, 60, 2], [42, 6, 60, 2]
                res_aux有4组(args.train_batch_size)数据...
            data: 12个数据
                BATCH_SIZE = 4
                SEQ_ID: 4个，如'f8be58f6-cc75-40fa-956c-9536f9397392'等
                CITY_NAME: 4个，如'washington-dc', 'pittsburgh'等
                ORIG
                ROT
                TRAJS: TRAJS_POS_OBS, TRAJS_ANG_OBS, TRAJS_VEL_OBS, TRAJS_TYPE, PAD_OBS; 
                        TRAJS_POS_FUT, TRAJS_ANG_FUT, TRAJS_VEL_FUT, PAD_FUT; 
                        TRAJS_CTRS, TRAJS_VECS, TRAJS_TID, TRAJS_CAT, TRAIN_MASK, YAW_LOSS_MASK
                LANE_GRAPH
                RPE
                ACTORS
                ACTOR_IDCS
                LANES
                LANE_IDCS
        Algo:
            1. 首先提取未来轨迹、填充标志、分类、回归和辅助信息。
            2. 然后应用训练掩码以过滤无效样本。根据 yaw_loss 是否为真，选择调用 pred_loss_with_yaw 或 pred_loss 来计算损失，并最终返回总损失
        输出：
            loss_out: 损失函数输出，包含类别损失、位置损失、方向角损失、总损失
        """
        traj_fut = [x['TRAJS_POS_FUT'] for x in data["TRAJS"]]
        traj_fut = gpu(traj_fut, device=self.device)    # [21, 60, 2], [19, 60, 2], [26, 60, 2], [42, 60, 2]

        # PAD_FUT 表示未来轨迹的填充标志。由于不同样本的未来轨迹长度可能不同，为了使所有样本具有相同的维度，通常会对较短的轨迹进行填充。
        # PAD_FUT 用于标识哪些时间步是填充的（即无效的），哪些是有效的。在计算损失时，可用于忽略填充的时间步，确保只对有效的时间步进行计算，从而避免引入不必要的误差
        pad_fut = [x['PAD_FUT'] for x in data["TRAJS"]]
        pad_fut = to_long(gpu(pad_fut, device=self.device)) # [21, 60], [19, 60], [26, 60], [42, 60]

        cls, reg, aux = out # 解包模型输出，获取分类、回归和辅助任务的结果

        # apply training mask
        # print('\n-- original')
        # print('cls:', [x.shape for x in cls])
        # print('reg:', [x.shape for x in reg])
        # print('vel:', [x.shape for x in vel])
        # print('traj_fut:', [x.shape for x in traj_fut])
        # print('ang_fut:', [x.shape for x in ang_fut])
        # print('pad_fut:', [x.shape for x in pad_fut])
        # print('yaw_loss_mask: ', [x.shape for x in yaw_loss_mask])

        # TRAIN_MASK 是一个布尔掩码，用于标识哪些样本或时间步是用于训练的。
        # 这可能是基于某些条件生成的，例如某些样本可能被标记为验证或测试样本，或者某些时间步的数据可能是无效的（如缺失值或填充值）
        train_mask = [x["TRAIN_MASK"] for x in data["TRAJS"]]
        train_mask = gpu(train_mask, device=self.device)    # 貌似全true, [21], [19], [26], [42]
        # print('train_mask:', [x.shape for x in train_mask])
        # print('whitelist num: ', [x.sum().item() for x in train_mask])

        cls = [x[train_mask[i]] for i, x in enumerate(cls)] # [21, 6], [19, 6], [26, 6], [42, 6]
        reg = [x[train_mask[i]] for i, x in enumerate(reg)] # [21, 6, 60, 2], [19, 6, 60, 2], [26, 6, 60, 2], [42, 6, 60, 2]
        traj_fut = [x[train_mask[i]] for i, x in enumerate(traj_fut)]   # [21, 60, 2], [19, 60, 2], [26, 60, 2], [42, 60, 2]
        pad_fut = [x[train_mask[i]] for i, x in enumerate(pad_fut)] # [21, 60], [19, 60], [26, 60], [42, 60]

        # print('-- masked')
        # print('cls:', [x.shape for x in cls])
        # print('reg:', [x.shape for x in reg])
        # print('vel:', [x.shape for x in vel])
        # print('traj_fut:', [x.shape for x in traj_fut])
        # print('ang_fut:', [x.shape for x in ang_fut])
        # print('pad_fut:', [x.shape for x in pad_fut])
        # print('yaw_loss_mask: ', [x.shape for x in yaw_loss_mask])

        if self.yaw_loss:
            # yaw angle GT
            ang_fut = [x['TRAJS_ANG_FUT'] for x in data["TRAJS"]]
            ang_fut = gpu(ang_fut, device=self.device)  # [21, 60, 2], [19, 60, 2], [26, 60, 2], [42, 60, 2]
            # for yaw loss
            yaw_loss_mask = gpu([x["YAW_LOSS_MASK"] for x in data["TRAJS"]], device=self.device)    # [21], [19], [26], [42]
            # collect aux info
            vel = [x[0] for x in aux]   # [21, 6, 60, 2], [19, 6, 60, 2], [26, 6, 60, 2], [42, 6, 60, 2]
            # apply train mask
            vel = [x[train_mask[i]] for i, x in enumerate(vel)] # [21, 6, 60, 2], [19, 6, 60, 2], [26, 6, 60, 2], [42, 6, 60, 2]
            ang_fut = [x[train_mask[i]] for i, x in enumerate(ang_fut)] # [21, 60, 2], [19, 60, 2], [26, 60, 2], [42, 60, 2]
            yaw_loss_mask = [x[train_mask[i]] for i, x in enumerate(yaw_loss_mask)] # [21], [19], [26], [42]

            loss_out = self.pred_loss_with_yaw(cls, reg, vel, traj_fut, ang_fut, pad_fut, yaw_loss_mask)
            loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"] + loss_out["yaw_loss"]
        else:
            loss_out = self.pred_loss(cls, reg, traj_fut, pad_fut)
            loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"]

        return loss_out

    def pred_loss_with_yaw(self,
                           cls: List[torch.Tensor],
                           reg: List[torch.Tensor],
                           vel: List[torch.Tensor],
                           gt_preds: List[torch.Tensor],
                           gt_ang: List[torch.Tensor],
                           pad_flags: List[torch.Tensor],
                           yaw_flags: List[torch.Tensor]):
        """
        带Yaw角度损失的预测损失：pred_loss_with_yaw，计算分类损失、回归损失和方向角损失。
        1. 分类损失：基于最小距离模式与其他模式之间的差异；
        2. 回归损失：基于预测轨迹与真实轨迹之间的差异；
        3. 方向角损失：基于预测速度方向与真实方向之间的余弦相似度
        """
        # 将输入的列表形式的张量拼接成一个大的张量，并指定维度0进行拼接
        cls = torch.cat([x for x in cls], dim=0)                     # [108, 6]，108是n_agents, 6是n_mod
        reg = torch.cat([x for x in reg], dim=0)                     # [108, 6, 60, 2]，60是n_preds， 2是xy
        vel = torch.cat([x for x in vel], dim=0)                     # [108, 6, 60, 2]，2是velocity_x/y
        gt_preds = torch.cat([x for x in gt_preds], dim=0)           # [108, 60, 2]
        gt_ang = torch.cat([x for x in gt_ang], dim=0)               # [108, 60, 2]
        has_preds = torch.cat([x for x in pad_flags], dim=0).bool()  # [108, 60]，表示哪些时间步是有效的（非填充）
        has_yaw = torch.cat([x for x in yaw_flags], dim=0).bool()    # [108]，表示哪些样本需要计算方向角损失

        loss_out = dict()  # 初始化一个字典来存储损失值
        num_modes = self.config["g_num_modes"]  # 获取配置中的模式数量: 6
        num_preds = self.config["g_pred_len"]   # 获取配置中的预测步数: 60
        # assert(has_preds.all())

        # 计算每个样本在预测序列中的最后一个有效时间步
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds) # [108, 60]
        max_last, last_idcs = last.max(1)  # 找到每个样本的最大有效时间步及其索引, [108], [108]
        mask = max_last > 1.0  # 过滤掉没有足够有效时间步的样本，有true有false, [108]

        # 应用mask过滤无效样本
        cls = cls[mask]
        reg = reg[mask]
        vel = vel[mask]
        gt_preds = gt_preds[mask]
        gt_ang = gt_ang[mask]
        has_preds = has_preds[mask]
        has_yaw = has_yaw[mask]
        last_idcs = last_idcs[mask]

        # 提取预测轨迹的前两维坐标，用于后续计算
        _reg = reg[..., 0:2].clone()  # for WTA strategy

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)  # 创建一个索引张量
        dist = []  # 存储每个模式下的距离
        for j in range(num_modes):
            # 计算每个模式下预测轨迹与真实轨迹在最后一个有效时间步的距离
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)  # 将所有模式的距离拼接成一个张量
        min_dist, min_idcs = dist.min(1)  # 找到每个样本的最小距离及其对应的模式索引
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)

        # 计算分类损失
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls  # 计算最小距离模式与其他模式之间的差异
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)  # 标记最小距离小于阈值的样本
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]  # 标记其他模式的距离大于忽略阈值的样本
        mgn = mgn[mask0 * mask1]  # 只保留满足条件的差异
        mask = mgn < self.config["mgn"]  # 标记差异小于边际的样本
        num_cls = mask.sum().item()  # 统计满足条件的样本数量
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)  # 计算分类损失
        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss  # 应用权重系数

        # 计算回归损失
        reg = reg[row_idcs, min_idcs]  # 选择最小距离模式的预测轨迹
        num_reg = has_preds.sum().item()  # 统计有效时间步的数量
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)  # 计算回归损失
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss  # 应用权重系数

        # 计算方向角损失 ~ yaw loss
        vel = vel[row_idcs, min_idcs]  # select the best mode, keep identical to reg， 选择最小距离模式的速度方向

        # 筛选出需要计算方向角损失的时间步
        _has_preds = has_preds[has_yaw].view(-1)
        _v1 = vel[has_yaw].view(-1, 2)[_has_preds]  # 预测的速度方向
        _v2 = gt_ang[has_yaw].view(-1, 2)[_has_preds]  # 真实的速度方向
        # print('_has_preds: ', _has_preds.shape)
        # print('_v1: ', _v1.shape)
        # print('_v2: ', _v2.shape)
        # ang diff loss use cosine similarity

        # 使用余弦相似度计算方向角损失
        cos_sim = torch.cosine_similarity(_v1, _v2)  # [-1, 1] 范围内的余弦相似度
        # print('cos_sim: ', cos_sim.shape, cos_sim[:100])
        loss_out["yaw_loss"] = ((1 - cos_sim) / 2).mean()  # 将余弦相似度转换为 [0, 1] 范围并求均值

        return loss_out  # 返回包含分类损失、回归损失和方向角损失的字典
    def pred_loss(self,
                  cls: List[torch.Tensor],
                  reg: List[torch.Tensor],
                  gt_preds: List[torch.Tensor],
                  pad_flags: List[torch.Tensor]):
        """
        预测损失：pred_loss，计算分类损失和回归损失，适用于不需要方向角损失的情况。
        1. 分类损失基于最小距离模式与其他模式之间的差异；
        2. 回归损失基于预测轨迹与真实轨迹之间的差异；
        """
        cls = torch.cat([x for x in cls], 0)                        # [98, 6]
        reg = torch.cat([x for x in reg], 0)                        # [98, 6, 60, 2]
        gt_preds = torch.cat([x for x in gt_preds], 0)              # [98, 60, 2]
        has_preds = torch.cat([x for x in pad_flags], 0).bool()     # [98, 60]

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = self.config["g_pred_len"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        _reg = reg[..., 0:2].clone()  # for WTA strategy

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        num_cls = mask.sum().item()
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss

        reg = reg[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss

        return loss_out

    def print(self):
        print('\nloss_fn config:', self.config)
        print('loss_fn device:', self.device)
