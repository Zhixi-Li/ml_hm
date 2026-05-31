# 课程项目：深度学习求解简单组合路径规划问题

## 1. 问题定义

**旅行商问题 (Traveling Salesman Problem, TSP)** 是组合优化（Combinatorial Optimization, CO）领域的经典NP-hard问题。
给定 $n$ 个城市的坐标集合 $\mathcal{V} = \{v_1, v_2, \ldots, v_n\}$，$v_i \in [0,1]^2$，
目标是找到一条访问每个城市恰好一次并返回出发点的最短哈密顿回路：

$$\tau^* = \arg\min_{\tau \in \Tau} \sum_{i=1}^{n} d(v_{\tau_i}, v_{\tau_{i+1}})$$

其中 $\Pi_n$ 是 $n$ 个城市所有排列的集合，$d(\cdot, \cdot)$ 为欧氏距离，$\tau_{n+1} = \tau_1$。

本项目聚焦于 **TSP-50**（$n = 50$），城市坐标服从 $[0,1]^2$ 上的均匀分布（训练阶段）。

---

## 2. 方法背景与项目任务

传统精确或启发式算法（如Concorde/LKH）能保证解质量，但对大规模实例计算代价极高。神经组合优化领域（Neural Combinatorial Optimization, NCO）应运而生，其核心目标就是通过深度学习技术，利用数据驱动方法，使模型学习问题结构和解分布的规律，避免传统算法对每个问题实例都无记忆地执行迭代程序，转向利用GPU并行优势，在效率提升的同时追求与传统精确求解器的解质量最优差距（Optimality gap）不断缩小。本项目提供两种主流的机器学习范式作为baseline：

### 2.1 技术路线一：GP (Global Prediction) — 基于非自回归监督学习的全局解预测

**核心思想**：设计神经网络学习，利用高质量监督信号，让模型学习一个从图结构到选边概率的映射。推理时，再贪心地、不违反约束地从预测的概率分数中解码出一条可行回路作为最终解。

**数学形式化**：
- 输入图 $G = (\mathcal{V}, \mathcal{E})$，节点特征 $\mathbf{X} \in \mathbb{R}^{n \times 2}$，边特征 $\mathbf{E} \in \mathbb{R}^{n \times n}$（距离矩阵）
- GNN 编码器 $f_\theta$：输出每条边属于最优解的预测 $\hat{H}_{ij}$，通常称为一个概率热图（heatmap）
- 监督信号：最优（或高质量）tour ($\tau^*$) 对应的边标签 $H^*_{ij} \in \{0, 1\}$，即 $H^*_{ij}=1$表示边$(v_i,v_j)\in\tau^*$。
- 训练目标（参考）：

    - **二元交叉熵（BCE）**：将边标签建模为伯努利分布，最大化正确分类的对数似然：
    $$\mathcal{L}_{BCE}(\theta) = -\frac{1}{|\mathcal{E}|}\sum_{(i,j)\in\mathcal{E}} \left[ H^*_{ij} \log \hat{H}_{ij} + (1 - H^*_{ij})\log(1 - \hat{H}_{ij}) \right]$$
    其中 $\hat{H}_{ij} = softmax(f_\theta(\mathbf{X}, \mathbf{E})_{ij}) \in (0,1)$ 为预测概率。

    - **均方误差（MSE）**：直接回归边标签，使网络输出实值 logit $\hat{z}_{ij}$：
    $$\mathcal{L}_{MSE}(\theta) = \frac{1}{|\mathcal{E}|}\sum_{(i,j)\in\mathcal{E}} \left( H^*_{ij} - \hat{z}_{ij} \right)^2$$

- 推理阶段：从概率热图 $\hat{\mathbf{H}}$ 出发，使用**贪心解码**逐步构造tour

**简易的GNN层更新规则**（Gated Graph ConvNet）：
$$\mathbf{e}_{ij}^{(l+1)} = \text{ReLU}\left(\mathbf{A}\mathbf{h}_i^{(l)} + \mathbf{B}\mathbf{h}_j^{(l)} + \mathbf{C}\mathbf{e}_{ij}^{(l)}\right)$$
$$\sigma_{ij}^{(l+1)} = \text{sigmoid}\left(\mathbf{e}_{ij}^{(l+1)}\right)$$
$$\mathbf{h}_i^{(l+1)} = \text{ReLU}\left(\mathbf{U}\mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \sigma_{ij}^{(l+1)} \odot \mathbf{V}\mathbf{h}_j^{(l)}\right)$$

---
### 2.2 技术路线二：LC (Local Construction) — 基于自回归强化学习的局部解构造

**核心思想**：将优化问题建模为马尔科夫决策过程，利用无监督的强化学习，使模型学习预测“下一节点选择”，端到端地构造tour的策略。

**数学形式化**：
- 策略 $\pi_\theta(a_t | s_t)$：在当前部分tour状态 $s_t$ 下，选择下一个城市 $a_t$ 的概率分布
- 状态 $s_t$：已访问城市集合 $\mathcal{S}_t$，当前节点 $v_{a_t}$，可行动作集合 $\mathcal{V} \setminus \mathcal{S}_t$
- 轨迹 $\tau = (a_1, a_2, \ldots, a_n)$ 对应一个完整tour
- 奖励（每个episode结束后给出）：

$$R(\tau) = -\sum_{t=1}^{n} d(v_{a_t}, v_{a_{t+1}})$$

- 训练目标（REINFORCE）：

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[(R(\tau) - b(\tau)) \nabla_\theta \log p_\theta(\tau)\right]$$

其中 $b(\tau)$ 为baseline，用于减小方差。**POMO** (Policy Optimization with Multiple Optima) 利用TSP的对称性（任意节点均可为起点），并行采样 $m = n$ 条轨迹，以均值作为baseline：

$$b = \frac{1}{m}\sum_{k=1}^{m} R(\tau^{(k)})$$



**Attention Model**：Transformer-based encoder-decoder架构

**Encoder**（对所有节点并行编码，输出上下文感知的节点嵌入）：

- **输入投影**：将原始坐标 $\mathbf{x}_i \in \mathbb{R}^2$ 线性映射到 $d$-维嵌入空间：
$$\mathbf{h}_i^{(0)} = \mathbf{W}^{in}\mathbf{x}_i + \mathbf{b}^{in}, \quad \mathbf{h}_i^{(0)} \in \mathbb{R}^d$$

- **多头自注意力（Multi-Head Self-Attention）**：在第 $l$ 层，每个头 $m$ 分别计算查询、键、值向量：
$$\mathbf{q}_i^{(m)} = \mathbf{W}_Q^{(m)}\mathbf{h}_i^{(l)}, \quad \mathbf{k}_j^{(m)} = \mathbf{W}_K^{(m)}\mathbf{h}_j^{(l)}, \quad \mathbf{v}_j^{(m)} = \mathbf{W}_V^{(m)}\mathbf{h}_j^{(l)}$$
- 注意力权重与加权聚合：
$$\alpha_{ij}^{(m)} = \frac{\exp\!\left(\mathbf{q}_i^{(m)\top}\mathbf{k}_j^{(m)} / \sqrt{d_k}\right)}{\sum_{j'}\exp\!\left(\mathbf{q}_i^{(m)\top}\mathbf{k}_{j'}^{(m)} / \sqrt{d_k}\right)}, \quad \mathbf{o}_i^{(m)} = \sum_j \alpha_{ij}^{(m)}\mathbf{v}_j^{(m)}$$
- 多头输出拼接并投影：$\mathbf{o}_i = \mathbf{W}^O[\mathbf{o}_i^{(1)}\|\cdots\|\mathbf{o}_i^{(M)}]$

- **残差连接 + 层归一化 + 前馈网络（FFN）**：
$$\mathbf{h}_i^{(l+\frac{1}{2})} = \text{LayerNorm}\!\left(\mathbf{h}_i^{(l)} + \mathbf{o}_i\right)$$
$$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\!\left(\mathbf{h}_i^{(l+\frac{1}{2})} + \text{FFN}(\mathbf{h}_i^{(l+\frac{1}{2})})\right)$$
经过 $L$ 层后得到最终节点嵌入 $\{\mathbf{h}_i\}_{i=1}^n$，全局图嵌入 $\bar{\mathbf{h}} = \frac{1}{n}\sum_i \mathbf{h}_i$。

**Decoder**（自回归逐步选择下一节点）：

- 每步 $t$，以当前节点嵌入 $\mathbf{h}_{a_{t-1}}$ 与图均值 $\bar{\mathbf{h}}$ 拼接构造查询上下文：
$$\mathbf{q}_t = \mathbf{W}_q [\mathbf{h}_{a_{t-1}} \| \bar{\mathbf{h}}] \in \mathbb{R}^d$$

- 对所有未访问节点 $j \notin \mathcal{S}_t$，计算键向量并得到注意力得分：
$$\mathbf{k}_j = \mathbf{W}_k \mathbf{h}_j, \qquad u_{j} = C \cdot \tanh\!\left(\frac{\mathbf{q}_t^\top \mathbf{k}_j}{\sqrt{d}}\right)$$

- 对已访问节点屏蔽（设为 $-\infty$），再做 softmax 得到选择概率：
$$\pi_\theta(a_t = j \mid s_t) = \frac{\exp(u_j)}{\sum_{j' \notin \mathcal{S}_t} \exp(u_{j'})}, \quad j \notin \mathcal{S}_t$$

训练时按概率采样（exploration），推理时取 argmax（greedy decoding）。

### 2.3 项目任务

**任务要求**：选择上述两种范式之一（GP 或 LC），在 baseline 基础上进行改进，最终目标是尽可能**提升求解质量**，同时兼顾**推理效率**与**跨分布/跨规模泛化性**。改进可在以下维度自由探索（仅供参考，不限于此）：

- **基础调参**：模型规模（层数、隐藏维度等）、学习率、batch size、学习率调度等
- **架构改进**：改进特征表示（如注入拓扑特征、位置编码）、改进注意力分数（如融合距离矩阵）、替换或增强编码器/解码器结构、引入残差/稠密连接等
- **训练策略**：数据增强（利用TSP的几何性质等）、课程学习（Curriculum Learning）、损失函数设计优化、引入扩散生成机制等
- **泛化增强**：数据分割分治、跨规模混合训练、实例归一化、引入局部几何信息等
- **范式融合**：考虑GNN/Transformer两类模型架构的优势互补，设计改良的融合架构；考虑训练机制的互补，如二阶段训练；考虑数据利用效率的提高，如专家信号引导强化学习等。

鼓励自由探索和创新，失败尝试同样有价值，可在技术报告中详细记录实验过程与分析。

---

## 3. 代码结构

```
Project/
├── INSTRUCTION.md           # 说明文档
├── data/
│   ├── train/
│   │   └── tsp50_uniform_train_128k.txt    # 训练集 (128k instances, uniform分布)
│   └── val/
│       ├── tsp50_uniform_val_128.txt       # 标准验证集 (128 instances, uniform分布)
│       ├── tsp50_ood_val_16.txt       # OOD测试集 (16 instances, 未知分布)
│       └── tsp100_uniform_val_16.txt       # Cross-scale测试集 (16 instances, TSP-100)
└── baselines/
    ├── gp_baseline/              # GP范式 (Supervised Learning)
    │   ├── model/
    │   │   ├── __init__.py       # 导出 GPModel
    │   │   ├── gnn_encoder.py    # GNN编码器
    │   │   ├── gnn_decoder.py    # 贪心解码器
    │   │   └── gp_model.py       # 完整模型 (encode + decode + loss)
    │   ├── utils/
    │   │   └── dataset.py        # 数据读取与预处理
    │   ├── checkpoints/          # 权重保存目录
    │   ├── evaluate_gp.py        # 学生自测脚本
    │   └── train_gp.py           # 训练脚本
    │
    └── lc_baseline/              # LC范式 (Reinforcement Learning)
        ├── model/
        │   ├── __init__.py       # 导出 LCModel, TSPEnv
        │   ├── lc_model.py       # Transformer encoder + Attention decoder
        │   └── tsp_env.py        # TSP环境 (RL交互接口 + 数据在线生成)
        ├── checkpoints/          # 权重保存目录
        ├── evaluate_lc.py        # 学生自测脚本
        └── train_lc.py           # 训练脚本
```

---

## 4. 环境配置

```bash
conda create --name tsp_project python=3.8
conda activate tsp_project

pip install torch==2.1.0
pip install scipy==1.10.1
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install wandb==0.16.3 tensordict==0.2.0 pytorch-lightning==2.1.0 einops==0.8.0
pip install ml4co-kit==0.3.3
```

如果使用 `torch==1.13.0+cu117` 环境，可改用以下安装命令：

```bash
conda create --name tsp_project python=3.8
conda activate tsp_project

pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install scipy==1.10.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install wandb==0.16.3 tensordict==0.2.0 pytorch-lightning==2.1.0 einops==0.8.0
pip install ml4co-kit==0.3.3
```

---

## 5. 运行Baseline

### 5.1 GP Baseline

```bash
cd baselines/gp_baseline
python train_gp.py # 训练配置直接在 train_gp.py 顶部修改
python evaluate_gp.py #自测脚本

```

### 5.2 LC Baseline

```bash
cd baselines/lc_baseline
python train_lc.py # 无需训练数据，在线生成随机实例
python evaluate_lc.py #自测脚本
```

---

## 6. 提交接口

评测脚本会在外部自动导入你的模型并进行测试。你需要选定一种范式进行修改调优，并**严格保证接口不变**：

### 6.1 GP 范式接口规范

**必须保证** `model/gp_model.py` 中的 `GPModel` 类接口与以下示例完全一致：

```python
import torch
import torch.nn as nn
from typing import List
import numpy as np

class GPModel(nn.Module):
    """GP Model for TSP
    """
    def __init__(self, 
        num_layers: int, 
        hidden_dim: int, 
        aggregation: str,
        ...
    ):
        super().__init__()
        # Your encoder/decoder initialization
        self.encoder = ...  # GNN encoder
        self.decoder = ...  # Greedy decoder
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass (training)
        
        Args:
            coords: (B, V, 2) node coordinates in [0,1]^2
        Returns:
            logits: (B, V, V, 2) edge logits (2-channel: not-in-tour, in-tour)
        """
        return self.encoder(coords)
    
    def solve(self, coords: torch.Tensor) -> List[np.ndarray]:
        """Inference interface (evaluation)
        
        Args:
            coords: (B, V, 2) node coordinates in [0,1]^2
        Returns:
            tours: List of B tours, each a numpy array of shape (V+1,)
                   starting and ending at node 0
        
        Example:
            >>> model = GPModel(num_layers=6, hidden_dim=128, aggregation='sum')
            >>> coords = torch.rand(4, 50, 2)  # 4 instances, 50 nodes
            >>> tours = model.solve(coords)
            >>> len(tours)  # 4
            >>> tours[0].shape  # (51,)
            >>> tours[0][0] == tours[0][-1] == 0  # True
        """
        self.eval()
        with torch.no_grad():
            logits = self.encoder(coords)  # (B, V, V, 2)
            tours = self.decoder.decode(logits, coords)
        return tours
```

**`model/__init__.py` 必须导出 `GPModel`**：
```python
from .gp_model import GPModel

__all__ = ['GPModel']
```

### 6.2 LC 范式接口规范

**必须保证** `model/lc_model.py` 中的 `LCModel` 和 `model/tsp_env.py` 中的 `TSPEnv` 接口与以下示例完全一致：

**`model/lc_model.py`**:
```python
import torch
import torch.nn as nn

class LCModel(nn.Module):
    """LC Model for TSP (POMO-style)
    """
    def __init__(self, embedding_dim, sqrt_embedding_dim, num_att_layers,
                 qkv_dim, sqrt_qkv_dim, num_heads, logit_clipping,
                 ff_hidden_dim, eval_type):
        super().__init__()
        self.eval_type = eval_type
        # Your encoder/decoder initialization
        self.encoder = ...
        self.decoder = ...
    
    def pre_forward(self, reset_state):
        """Encode the problem instance before decoding
        
        Args:
            reset_state: dict with 'coords' (B, V, 2) and 'dist' (B, V, V)
        """
        # Encode once, cache embeddings for decoding
        ...
    
    def forward(self, state):
        """Decode one step
        
        Args:
            state: dict with current partial tour state
        Returns:
            selected: (B, pomo) selected node indices
            prob: (B, pomo) selection probabilities (or None if first step)
        """
        ...
```

**`model/tsp_env.py`**:
```python
import torch

class TSPEnv:
    """TSP Environment for POMO
    
    Args:
        task: 'TSP'
        node_cnt: Number of nodes (e.g., 50)
        pomo_size: Number of parallel rollouts (typically == node_cnt)
    """
    def __init__(self, task: str, node_cnt: int, pomo_size: int):
        self.node_cnt = node_cnt
        self.pomo_size = pomo_size
        ...
    
    def load_problems(self, batch_size: int):
        """Generate random TSP instances
        
        Args:
            batch_size: Number of instances to generate
        """
        ...
    
    def load_problems_manual(self, problems: torch.Tensor, coords: torch.Tensor):
        """Load external TSP instances
        
        Args:
            problems: (B, V, V) distance matrices
            coords: (B, V, 2) node coordinates
        """
        ...
    
    def reset(self):
        """Reset environment, return initial state
        
        Returns:
            reset_state: dict with 'coords' and 'dist'
            reward: None
            done: False
        """
        ...
    
    def pre_step(self):
        """Initialize POMO rollouts (all nodes as starting points)
        
        Returns:
            state: dict with partial tour state
            reward: None
            done: False
        """
        ...
    
    def step(self, selected):
        """Execute one decoding step
        
        Args:
            selected: (B, pomo) selected node indices
        Returns:
            state: updated state dict
            reward: (B, pomo) negative tour lengths (if done), else None
            done: bool, True if all tours complete
        """
        ...
```

**`model/__init__.py` 必须导出 `LCModel` 和 `TSPEnv`**：
```python
from .lc_model import LCModel
from .tsp_env import TSPEnv

__all__ = ['LCModel', 'TSPEnv']
```

### 6.3 权重文件

最终权重必须保存至 `checkpoints/best_model.pth`（评测脚本会自动加载此文件）。

---

## 7. 评测指标

最终在三类隐藏测试集上进行评测（不可见）：

| 测试集 | 分布 | 权重（暂定） |
|-|-|-|
| In-distribution | TSP-50 (Uniform)  | 60% |
| Cross-distribution | TSP-50 (Unseen distribution)  | 20% |
| Cross-scale | TSP-100 (Uniform) | 20% |

**主要指标**: Optimality Gap (%)，即与Concorde最优解的相对差距：
$$\text{Gap} = \frac{L(\hat{\pi}) - L(\pi^*)}{L(\pi^*)} \times 100\%$$

其中 $L(\pi) = \sum_{i=1}^n d(v_{\pi_i}, v_{\pi_{i+1}})$ 为tour总长度。

**次要指标**: Solving Time，即对全部测试实例完成推理所需的总时长（秒），衡量模型在实际部署场景下的效率。评测时统一在相同硬件环境下单进程顺序推理，不计数据加载时间。

**样例测试集**：为便于学生自测泛化性能，提供了3个样例测试集（位于 `data/val/`）：
- `tsp50_uniform_val_128.txt`：128个TSP-50 Uniform分布实例，用于测试同分布求解能力
- `tsp50_ood_val_16.txt`：16个TSP-50 未知分布实例，用于测试OOD泛化能力
- `tsp100_uniform_val_16.txt`：16个TSP-100 Uniform分布实例，用于测试跨规模泛化能力

这些样例集仅供参考，最终评测使用的隐藏测试集规模更大、分布更多样。

**关于范式间的公平性**：由于GP 与 LC 两种范式在架构选择、训练成本、推理机制上存在内生差异，因此，**绝对指标数值不是唯一的评分判据**，评分时会综合考虑：

- 相对于范式内baseline的求解精度、效率、泛化性能
- 方法设计的合理性与创新性
- 技术报告的质量

换言之，在同一范式内做出显著改进、并能清晰阐述设计动机与实验分析的提交，将获得充分认可。

---

## 8. 提交要求

### 8.1 代码文件

**GP 范式**：
```
baselines/gp_baseline/
├── model/
│   ├── __init__.py              # 必须导出 GPModel
│   ├── gnn_encoder.py           # 可修改
│   ├── gnn_decoder.py           # 可修改
│   └── gp_model.py              # GPModel 接口保持不变（见第 6 节）
├── utils/
│   └── dataset.py               # 可修改
├── checkpoints/
│   └── best_model.pth           # 必须提供，评测脚本自动加载
├── evaluate_gp.py               # 学生自测脚本，必须定义 model_params 字典
├── train_gp.py                  # 训练脚本（可复现训练过程）
└── README.md                    # 见 8.2 节要求
```

**重要**：`evaluate_gp.py` 中必须定义全局变量 `model_params` 字典，包含所有模型初始化参数。例如：
```python
model_params = {
    'num_layers': 6,
    'hidden_dim': 128,
    'aggregation': 'sum'
}
```
外部评测脚本会自动读取此配置。如果新增模型参数，只需在此字典中添加即可。

**LC 范式**：
```
baselines/lc_baseline/
├── model/
│   ├── __init__.py              # 必须导出 LCModel, TSPEnv
│   ├── lc_model.py              # LCModel 接口保持不变（见第 6 节）
│   └── tsp_env.py               # TSPEnv 接口保持不变（见第 6 节）
├── checkpoints/
│   └── best_model.pth           # 必须提供，评测脚本自动加载
├── evaluate_lc.py               # 学生自测脚本，必须定义 model_params 和 env_params 字典
├── train_lc.py                  # 训练脚本（可复现训练过程）
└── README.md                    # 见 8.2 节要求
```

**重要**：`evaluate_lc.py` 中必须定义全局变量 `model_params` 和 `env_params` 字典。例如：
```python
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** 0.5,
    'num_att_layers': 3,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** 0.5,
    'num_heads': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

env_params = {
    'task': 'TSP',
    'node_cnt': 50,
    'pomo_size': 50
}
```
外部评测脚本会自动读取这些配置。如果新增模型参数，只需在相应字典中添加即可。

### 8.2 技术报告（Report.pdf）

建议包含以下内容：

1. **方法概述**：简述所选范式及核心改进思路
2. **模型架构**：描述网络结构与关键设计决策，与 baseline 的差异
3. **训练配置**：完整列出超参数（batch size、learning rate、epochs、hidden dim 等）
4. **实验结果**：
   - 在验证集上的 Optimality Gap 与 baseline 的对比
   - 如做了多组消融/对比实验，可以列出相应结果和分析（可选）
5. **设计动机与分析**：解释"为什么这样改"，实验现象与理论预期是否吻合，失败尝试同样有价值
6. **改进方向**：简述未来可继续探索的方向

> **评分说明**：技术报告质量与实验分析深度是重要评分维度。在同一范式内做出有据可查的改进、并能清晰阐述设计动机的提交，将获得充分认可。

---

## 9. 参考文献

- [GP Baseline] *An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem*. arXiv:1906.01227
- [LC Baseline] *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning*. NeurIPS 2020

其他参考补充：

**GP:**
- *Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances*. AAAI 2021
- *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization*. NeurIPS 2023
- *Fast t2t: Optimization consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization*. NeurIPS 2024

**LC**
- *Attention, Learn to Solve Routing Problems!* ICLR 2019
- *Matrix encoding networks for neural combinatorial optimization*. NeurIPS 2021
- *Sym-nco: Leveraging symmetricity for neural combinatorial optimization* NeurIPS 2022


**工具文档**
- ML4CO-Kit: https://ml4co-kit.readthedocs.io/en/latest/
