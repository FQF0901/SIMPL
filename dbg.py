import numpy as np
import matplotlib.pyplot as plt

# 定义总训练步数
total_steps = 100

# 定义初始学习率和最小学习率
initial_lr = 0.1
min_lr = 0.001

# 定义不同的调度器参数
cosine_params = {
    'T_max': total_steps,
    'eta_min': min_lr,
    'initial_lr': initial_lr
}

cosine_warmup_params = {
    'T_warmup': 10,
    'T_max': total_steps,
    'eta_min': min_lr,
    'initial_lr': initial_lr
}

step_params = {
    'step_size': 20,
    'gamma': 0.1,
    'initial_lr': initial_lr
}

polyline_params = {
    'milestones': [0, 25, 50, 75, 100],
    'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]
}

# 定义学习率调度器函数
def cosine_annealing(step, T_max, eta_min, initial_lr):
    return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * step / T_max)) / 2

def cosine_annealing_with_warmup(step, T_warmup, T_max, eta_min, initial_lr):
    if step < T_warmup:
        return initial_lr * (step / T_warmup)
    else:
        return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * (step - T_warmup) / (T_max - T_warmup))) / 2

def step_decay(step, step_size, gamma, initial_lr):
    return initial_lr * (gamma ** (step // step_size))

def polyline_scheduler(step, milestones, values):
    for i in range(len(milestones) - 1):
        if milestones[i] <= step < milestones[i + 1]:
            t = (step - milestones[i]) / (milestones[i + 1] - milestones[i])
            return values[i] + t * (values[i + 1] - values[i])
    return values[-1]

# 计算学习率
steps = np.arange(total_steps)
lrs_cosine = [cosine_annealing(s, **cosine_params) for s in steps]
lrs_cosine_warmup = [cosine_annealing_with_warmup(s, **cosine_warmup_params) for s in steps]
lrs_step = [step_decay(s, **step_params) for s in steps]
lrs_polyline = [polyline_scheduler(s, **polyline_params) for s in steps]

# 绘制学习率曲线
plt.figure(figsize=(12, 8))

plt.plot(steps, lrs_cosine, label='Cosine Annealing')
plt.plot(steps, lrs_cosine_warmup, label='Cosine Annealing with Warmup')
plt.plot(steps, lrs_step, label='Step Decay')
plt.plot(steps, lrs_polyline, label='Polyline Scheduler')

plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedulers')
plt.legend()
plt.grid(True)
plt.show()