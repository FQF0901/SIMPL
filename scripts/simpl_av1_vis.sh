# CUDA_VISIBLE_DEVICES=0 是一个环境变量设置，用于指定哪些GPU设备对当前进程可见。这里的 0 表示只使用第一个GPU设备（即索引为0的GPU）
# --model_path 参数指定了要加载的模型文件路径。在这个例子中，模型文件是 saved_models/simpl_av1_ckpt.tar【虽然 .tar 文件本身并不进行压缩，但通常会与压缩算法结合使用，在机器学习和深度学习中，模型权重和其他相关文件经常被打包成 .tar 文件以便于管理和分发】
# --visualizer 参数指定了使用的可视化类。这里的格式是 module_name:class_name，表示从 simpl.av1_visualizer 模块中导入 Visualizer 类来执行可视化任务
# --seq_id 参数指定了要可视化的序列ID。-1 通常表示处理所有序列，而不是特定的一个序列
CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --features_dir data_argo/features/ \
  --mode val \
  --use_cuda \
  --model_path saved_models/simpl_av1_ckpt.tar \
  --adv_cfg_path config.simpl_cfg \
  --visualizer simpl.av1_visualizer:Visualizer \
  --seq_id -1