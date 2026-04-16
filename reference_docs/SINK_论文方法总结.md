# SINK.pdf 中关于 sink token 检测与注意力头筛选的方法总结

本文对应论文是 **"See What You Are Told: Visual Attention Sink"**。论文核心观点是：

- 多模态大模型里有一部分视觉 token 会像语言模型里的 `BOS`、标点那样，拿到很多注意力，但本身几乎没有有效语义。
- 这类 token 被论文称为 **visual sink tokens**。
- 识别出这些 sink token 后，可以进一步判断哪些注意力头真正关注了图像中的有效信息，从而筛选出 **image-centric heads**。

---

## 1. sink token 的检测思路

### 1.1 观察出发点

论文先通过注意力可视化发现：

- 某些与问题无关的图像区域 token 会持续拿到较大的注意力权重；
- 这种现象和语言模型中的 attention sink 很像；
- 这些“无关但高注意力”的视觉 token，隐藏状态中会在少数固定维度上出现异常大的激活。

因此，论文不是直接用“注意力大”来定义 sink token，而是结合了 **隐藏状态中的 sink dimension 激活** 来做检测。

### 1.2 sink dimensions 的来源

论文认为，sink token 会在少数固定维度 `D_sink` 上出现大幅激活，这些维度来自底座语言模型本身，是一种继承来的性质。

例如论文报告：

- `LLaVA-1.5-7B` 对应 `D_sink = {1415, 2533}`
- `Qwen2-VL-7B` 对应 `D_sink = {458, 2570}`
- `InternVL2-8B` 对应 `D_sink = {3584}`

注意：

- 这里是论文中给出的 **Qwen2-VL-7B** 结果；
- 对 **Qwen2.5-VL** 是否完全相同，论文没有直接给出，需要你单独验证。

### 1.3 sink dimension value 的定义

给定某个 token 的隐藏状态 `x ∈ R^D`，论文定义：

```text
phi(x) = max_{d in D_sink} | x[d] / RMS(x) |
```

其中：

- `x[d]` 表示隐藏状态在第 `d` 个维度上的值；
- `RMS(x)` 是对整条隐藏向量做均方根归一化；
- 只取 `D_sink` 中最大的那个归一化激活值。

直觉上就是：

- 如果一个 token 在 sink 维度上异常激活；
- 那么它更可能是 sink token。

### 1.4 sink token 的判定规则

论文定义某层 `l` 中的 sink token 索引集合为：

```text
I_hat^l = { j in I | phi(x_j^(l-1)) >= tau }
```

视觉 sink token 则是：

```text
I_hat_vis^l = I_hat^l ∩ I_vis
```

其中：

- `I` 是全部 token；
- `I_vis` 是视觉 token 集合；
- `tau` 是阈值。

论文实验里使用：

```text
tau = 20
```
---

## 2. 注意力头筛选的方法

### 2.1 为什么要筛选头

论文后续提出的 VAR（Visual Attention Redistribution）会把分配给 sink token 的部分注意力重新分给更有意义的视觉 token。

但论文发现：

- 如果对 **所有注意力头** 都这样做，性能会大幅下降；
- 因为并不是每个 head 都负责图像信息处理；
- 所以必须先筛出真正“看图”的那些 head，也就是 **image-centric heads**。

### 2.2 第一层过滤：先去掉几乎不看图的 head

对每个 layer 的每个 head，先计算它分给全部视觉 token 的总注意力。

如果这个总量小于：

```text
0.2
```

就直接丢弃，不参与后续筛选。

原因很直接：

- 连图像 token 都几乎不看，这种 head 不可能是 image-centric head。

### 2.3 核心指标：visual non-sink ratio

论文认为，一个 head 是否真的在看“有效图像内容”，关键不在于它给视觉 token 的总注意力有多大，而在于：

- 这些视觉注意力里，有多少是落在 **visual non-sink tokens** 上，而不是落在 sink token 上。

于是定义：

```text
r_i^(l,h) =
sum_{j in I_vis \ I_hat_vis^l} alpha_{i,j}^(l,h)
/
sum_{j in I_vis} alpha_{i,j}^(l,h)
```

其中：

- `alpha_{i,j}^(l,h)` 是第 `l` 层第 `h` 个头里，文本 token `i` 对视觉 token `j` 的注意力权重；
- 分子是分给 **非 sink 视觉 token** 的注意力；
- 分母是分给 **全部视觉 token** 的注意力。

这个比值越大，表示：

- 该 head 虽然在看图像；
- 而且主要在看 **有意义的图像 token**；
- 因此更像是真正负责视觉信息整合的 head。

### 2.4 image-centric head 的判定规则

论文直接设定阈值 `rho`：

```text
r_i^(l,h) >= rho
```

满足该条件的 head 被选为 **image-centric heads**。

论文中的经验值：

- 通用视觉语言任务：`rho = 0.8`
- 视觉幻觉任务：`rho = 0.5`
- 强视觉理解任务：`rho = 0.9`

另外，论文还特别说明：

- **最后一层不做修改**，因为最后一层可能承担更专门的功能。

---

## 3. 这套方法背后的整体思路

可以把论文思路概括成下面这条链路：

1. **先识别“高注意力但低语义”的视觉 token**
   不是看 attention map 是否亮，而是看 token 是否在 `D_sink` 上有异常激活。

2. **把视觉 token 分成 sink / non-sink**
   sink token 代表“吸走注意力预算但没有真正提供图像语义”的部分。

3. **用 non-sink 占比衡量 head 是否真的理解图像**
   如果一个 head 给视觉 token 很多注意力，但大多数都给了 sink token，那么它其实并不真正关注图像内容。

4. **只在 image-centric heads 上动手**
   这样可以避免误伤那些并不负责视觉融合的 head。

5. **把 sink 上浪费掉的注意力预算转移到 non-sink token**
   论文把这一步称为 Visual Attention Redistribution。

换句话说，论文的关键思想不是“把所有视觉注意力都加大”，而是：

- 先区分 **假视觉关注** 和 **真视觉关注**；
- 再只增强真正有信息的那部分视觉注意力。

---

## 4. 如果你要复现，最小可执行步骤

### 4.1 检测 visual sink token

对某层输入隐藏状态 `x^(l-1)`：

1. 准备模型对应的 `D_sink`
2. 对每个视觉 token 计算：

```text
phi(x) = max_{d in D_sink} | x[d] / RMS(x) |
```

3. 用阈值判断：

```text
phi(x) >= 20
```

4. 满足条件的视觉 token 记为 visual sink token

### 4.2 筛选 image-centric heads

对每个文本 token `i`、每层 `l`、每个 head `h`：

1. 先算该 head 对全部视觉 token 的总注意力
2. 若总注意力 `< 0.2`，直接跳过
3. 否则计算：

```text
r_i^(l,h) =
sum(attn to visual non-sink tokens)
/
sum(attn to all visual tokens)
```

4. 若 `r_i^(l,h) >= rho`，则选为 image-centric head

### 4.3 参数建议

可先直接照论文默认值起步：

- `tau = 20`
- `visual-attention prefilter = 0.2`
- `rho = 0.8` 作为通用初始值

如果你的目标更偏：

- 抑制幻觉，可尝试更低的 `rho`
- 强依赖细粒度视觉理解，可尝试更高的 `rho`

---

## 5. 对你当前项目更有用的实现提醒

如果你要把这篇论文的方法迁移到自己的可视化或分析代码里，最重要的是先确认两件事：

1. **Qwen2.5-VL 的 `D_sink` 是否与 Qwen2-VL-7B 一致**
   论文只明确给出了 `Qwen2-VL-7B` 的 `D_sink = {458, 2570}`，不能直接无条件套用到 Qwen2.5-VL。

2. **你统计的是哪一层、哪一类 token**
   论文的方法是层相关的，`visual sink token` 也是按层定义的，不是一次性全局固定集合。

---

## 6. 一句话总结

这篇论文的核心方法就是：

**先用 hidden state 在 sink dimensions 上的异常激活检测 visual sink token，再用“注意力落到 visual non-sink token 的比例”筛选真正关注图像内容的 attention heads。**
