import os
import sys


class AdvCfg():
    def __init__(self):
        self.g_cfg = dict()
        self.g_cfg['g_num_modes'] = 6   # 预测的模式数量，这里是6个
        self.g_cfg['g_obs_len'] = 20    # 观察序列的长度，这里是20
        self.g_cfg['g_pred_len'] = 30   # 预测序列的长度，这里是30

    def get_dataset_cfg(self):
        data_cfg = dict()
        data_cfg['dataset'] = "simpl.av1_dataset:ArgoDataset"

        data_cfg.update(self.g_cfg)  # append global config
        return data_cfg

    def get_net_cfg(self):  # get_net_cfg 方法返回一个字典，包含网络的配置信息
        net_cfg = dict()
        net_cfg["network"] = "simpl.simpl:Simpl"
        net_cfg["init_weights"] = False # 是否初始化权重，这里是 False
        net_cfg["in_actor"] = 3 # in_actor, d_actor, n_fpn_scale, in_lane, d_lane: 网络输入和中间层的维度
        net_cfg["d_actor"] = 128
        net_cfg["n_fpn_scale"] = 3
        net_cfg["in_lane"] = 10
        net_cfg["d_lane"] = 128

        net_cfg["d_rpe_in"] = 5 # d_rpe_in, d_rpe, d_embed, n_scene_layer, n_scene_head, dropout, update_edge: 网络内部的一些参数
        net_cfg["d_rpe"] = 128
        net_cfg["d_embed"] = 128
        net_cfg["n_scene_layer"] = 4
        net_cfg["n_scene_head"] = 8
        net_cfg["dropout"] = 0.1
        net_cfg["update_edge"] = True

        net_cfg["param_out"] = 'none'  # bezier/monomial/none
        net_cfg["param_order"] = 5     # 5-th order polynomials

        net_cfg.update(self.g_cfg)  # append global config
        return net_cfg

    def get_loss_cfg(self):
        loss_cfg = dict()
        loss_cfg["loss_fn"] = "simpl.av1_loss_fn:LossFunc"
        loss_cfg["cls_coef"] = 0.1
        loss_cfg["reg_coef"] = 0.9
        loss_cfg["mgn"] = 0.2
        loss_cfg["cls_th"] = 2.0
        loss_cfg["cls_ignore"] = 0.2

        loss_cfg.update(self.g_cfg)  # append global config
        return loss_cfg

    def get_opt_cfg(self):
        opt_cfg = dict()
        opt_cfg['opt'] = 'adam' # opt: 指定了优化器的类型，这里是 adam
        opt_cfg['weight_decay'] = 0.0
        opt_cfg['lr_scale_func'] = 'none'  # none/sqrt/linear

        # scheduler
        opt_cfg['scheduler'] = 'polyline'   # 学习率调度器的类型，这里是 polyline

        if opt_cfg['scheduler'] == 'cosine':
            opt_cfg['init_lr'] = 6e-4
            opt_cfg['T_max'] = 50
            opt_cfg['eta_min'] = 1e-5
        elif opt_cfg['scheduler'] == 'cosine_warmup':
            opt_cfg['init_lr'] = 1e-3
            opt_cfg['T_max'] = 50
            opt_cfg['eta_min'] = 1e-4
            opt_cfg['T_warmup'] = 5
        elif opt_cfg['scheduler'] == 'step':
            opt_cfg['init_lr'] = 1e-3
            opt_cfg['step_size'] = 40
            opt_cfg['gamma'] = 0.1
        elif opt_cfg['scheduler'] == 'polyline':
            opt_cfg['init_lr'] = 1e-4
            opt_cfg['milestones'] = [0, 5, 35, 40]
            opt_cfg['values'] = [1e-4, 1e-3, 1e-3, 1e-4]

        opt_cfg.update(self.g_cfg)  # append global config
        return opt_cfg

    def get_eval_cfg(self):
        eval_cfg = dict()
        eval_cfg['evaluator'] = 'utils.evaluator:TrajPredictionEvaluator'
        eval_cfg['data_ver'] = 'av1'
        eval_cfg['miss_thres'] = 2.0

        eval_cfg.update(self.g_cfg)  # append global config
        return eval_cfg