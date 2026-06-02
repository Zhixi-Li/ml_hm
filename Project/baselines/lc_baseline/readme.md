# LC Baseline Optimization Notes

## 优化思路

本实现选择 LC（Local Construction）/POMO 路线作为主提交范式。相比 GP 的全局 heatmap 预测，LC 自回归构造 tour 时天然支持不同节点规模，因此更适合同时兼顾 TSP-50、OOD TSP-50 和 TSP-100 的评测要求。

主要改动如下：

- **距离感知 Encoder**：将原来的标准 `nn.MultiheadAttention` 替换为自定义 multi-head attention，在 `QK/sqrt(d)` 注意力分数中加入由距离矩阵产生的 learnable bias。这样模型不只依赖坐标 embedding，也能直接利用 TSP 的局部几何结构。
- **距离感知 Decoder**：在下一节点选择 logits 中加入当前节点到候选节点的归一化距离惩罚，形式为 `score - alpha * dist`，其中 `alpha` 为可学习正参数。该项鼓励模型优先考虑短边，但仍允许神经策略在必要时选择长边。
- **跨规模环境支持**：`TSPEnv.load_problems_manual()` 会根据输入距离矩阵自动同步 `node_cnt` 和 `pomo_size`。因此评估 TSP-100 时会完整解码 100 步，并使用 100 条 POMO rollout。
- **D4 几何增强**：训练时随机使用单位正方形上的 8 种对称变换，包括翻转、交换坐标轴及其组合。TSP 最优 tour 在这些变换下保持等价，可提升几何泛化。
- **混合规模 curriculum**：前期训练 TSP-50，后期混合采样 TSP-50/TSP-75/TSP-100，增强跨规模泛化。大规模 batch 会自动降低以控制显存。
- **三验证集加权选优**：训练时同时验证 `tsp50_uniform`、`tsp50_ood`、`tsp100_uniform`，按项目说明中的权重 `0.6/0.2/0.2` 计算 weighted validation score，并保存最佳模型。

所有外部提交接口保持兼容：

- `model/lc_model.py` 中仍导出 `LCModel`。
- `model/tsp_env.py` 中仍导出 `TSPEnv`。
- `evaluate_lc.py` 中保留全局 `model_params` 和 `env_params`。
- 最佳权重保存到 `checkpoints/best_model.pth`。

## 关键文件

- `model/lc_model.py`：distance-aware encoder/decoder 及 `LCModel` 接口。
- `model/tsp_env.py`：POMO 环境、动态 TSP-50/TSP-100 规模同步。
- `train_lc.py`：D4 增强、curriculum、加权验证、checkpoint 保存。
- `evaluate_lc.py`：三类公开验证集评估。

## 训练命令

在课程环境中安装 `requirements.txt` 后，从本目录执行：

```bash
cd Project/baselines/lc_baseline
python train_lc.py
```

训练脚本会保存：

- `checkpoints/<timestamp>/best_model.pth`：本次运行的最佳权重。
- `checkpoints/best_model.pth`：提交评测脚本会加载的权重。

如需调整训练预算，可直接修改 `train_lc.py` 顶部配置：

- `EPOCHS`
- `BATCHES_PER_EPOCH`
- `BATCH_SIZE`
- `CURRICULUM_START_EPOCH`
- `LEARNING_RATE`

## 验证命令

训练完成并生成 `checkpoints/best_model.pth` 后，从本目录执行：

```bash
cd Project/baselines/lc_baseline
python evaluate_lc.py
```

默认会依次评估：

- `../../data/val/tsp50_uniform_val_128.txt`
- `../../data/val/tsp50_ood_val_16.txt`
- `../../data/val/tsp100_uniform_val_16.txt`

输出包括平均 tour cost、reference optimal cost、average gap、总耗时和单实例平均耗时。

如果只想评估单个文件，可在 `evaluate_lc.py` 中将：

```python
EVALUATE_ALL_VAL = True
```

改为：

```python
EVALUATE_ALL_VAL = False
```

然后修改 `TEST_DATA_PATH` 为目标验证文件。


