**先说结论**
按你现在这条脚本 [run_stage1_isic2019lt_factors.sh](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/scripts/run_stage1_isic2019lt_factors.sh) 和当前配置 [configs.yaml](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/config/configs.yaml)，你实际在跑的不是“完整原始 MRC-VFC 双分支 Stage1”，而是这条更精简的路径：

- 输入图像
- 生成两种增强视图：`strong` 给 student，`weak` 给 teacher
- `VAVAEStudentVAE` 编码成 32 维特征
- 线性分类头输出 8 类 logits
- 用 EMA teacher 做 logit 蒸馏、feature 蒸馏、CKA 结构蒸馏
- 同时对 student 自己做带权重 CE 分类
- 周期性在 `val/test` 上评估 student encoder + 线性头

也就是说，你现在的 Stage1 本质上是：

**“一个 VA-VAE encoder + linear classifier 的自蒸馏表征学习器”**

而不是：
- ResNet 主干训练
- Mix 分支
- AuxVAE 重建
- Gaussian prior 真正参与反传

这些在你当前脚本下都没有真正进入主训练目标。

---

**1. 全流程闭环：从原始图像到结果输出**

**1) 读 split csv**
训练脚本按 `factor=100/200/500` 选对应训练集 csv，同一套 `validation.csv/testing.csv` 做评估。  
见 [run_stage1_isic2019lt_factors.sh](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/scripts/run_stage1_isic2019lt_factors.sh#L29) 和 [lt_split.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/utils/lt_split.py)

作用：
- 决定这次实验用哪些图像、哪些标签
- 控制 long-tail 强度

---

**2) 数据集读取**
`ISICDataset` 从 csv 读：
- `image` 列：图像 id
- 后面的 one-hot 标签列：通过 `argmax` 变成类别 id

见 [dataset.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/data/dataset.py#L9)

作用：
- 把“文件名 + one-hot 标签”变成 PyTorch 可用样本

输出单样本形式：
- 图像：PIL RGB
- 标签：`torch.long` 标量，例如 `5`

---

**3) 训练增强：一张图变两张视图**
当前训练 `transform` 返回的是：

- `strong_augmentation`
- `weak_augmentation`

见 [transforms.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/data/transforms.py#L66)

具体操作：
- resize 到 `image_size`
- 翻转
- 强增强还会加模糊、形变、颜色偏移、亮度对比度、grid dropout
- 然后 ImageNet 均值方差归一化

见 [transforms.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/data/transforms.py#L23)

作用：
- `strong` 让 student 学到更鲁棒特征
- `weak` 给 teacher，提供相对稳定的监督目标

生活类比：
- 同一个病人拍两张图，一张正常拍，一张故意加些扰动
- 你希望模型知道“虽然拍法不同，病还是同一个病”

---

**4) DataLoader 打包**
Stage1 主入口里：

- 训练集 `drop_last=True`
- `batch_size` 来自脚本，当前默认是 `4`
- `grad_accum_steps=4`

见 [stage1.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/stage1.py#L137)

作用：
- 每次真正在 GPU 上前向的是 4 张
- 累积 4 次梯度再更新一次参数，相当于优化器视角接近 16 张

---

**5) 模型构建：当前只保留 student/teacher encoder 分支**
在你的当前脚本里：

- `kd_only=True`
- `kd_teacher_source=lite`
- `student_source=vavae`
- `mix_enable=False`
- `show_teacher_metrics=False`

所以 ResNet 主干被直接跳过。  
见 [stage1.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/stage1.py#L183)

然后真正构建的是：
- `lite_vae`：实际上是 `VAVAEStudentVAE`
- `lite_classifier`：一个线性层
- `lite_vae_teacher` / `lite_classifier_teacher`：student 的 EMA teacher 副本

见 [stage1.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/stage1.py#L260)

这里变量名 `lite_vae` 有历史遗留意味，但你当前 `student_source=vavae`，它实际就是 VA-VAE student。

---

**6) Student encoder 前向**
`VAVAEStudentVAE.encode(x)` 做的是：

1. 如果输入不是 `224x224`，先内部 resize 到 `224x224`
2. 过 encoder
3. `quant_conv`
4. 切成 `mu_map` 和 `logvar_map`
5. 做全局池化，得到 `mu`、`logvar`
6. 重参数化采样得到 `z`

见 [vavae_teacher.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/vavae_teacher.py#L384)

作用：
- `mu`：稳定的语义中心特征
- `logvar`：每一维的不确定性
- `z`：从分布里随机采的样本

你当前分类默认用的是 `mu`，不是 `z`。  
见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L83)

---

**7) 分类头**
`Linear(32, 8)`  
见 [classifiers.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/classifiers.py#L4)

作用：
- 把 32 维表征映射到 8 类 logits

数学形式：

```math
\text{logits} = W \mu + b
```

---

**8) Teacher 分支**
当前 teacher 不是外部预训练 vavae teacher，而是：

- student 当前参数的一份 EMA 副本
- 输入弱增强图像
- 走同样的 encoder + classifier

见 [stage1.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/stage1.py#L411)

作用：
- 给 student 一个更平滑、更稳定的目标
- 避免 student 只盯着 one-hot 标签学

生活类比：
- student 是正在写作业的学生
- teacher 是“过去一段时间平均水平的自己”
- 每次不拿自己当前最冲动的答案监督自己，而拿“更稳的平均答案”来纠偏

---

**9) 损失计算**
你当前实际有效的损失项是：

```math
L
=
L_{cls}
+
0.5 L_{kd\_logit}
+
0.5 L_{kd\_feat}
+
1.0 L_{cka}
```

其中：

**(a) 带权重 CE**
student 自己的分类损失：

```math
L_{cls} = - w_y \log p_y
```

实现路径：
- `stage1_cls_loss_type=ce`
- `use_class_weight=True`

见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L162)

作用：
- 保证 student 至少学会分类
- 对尾类给更大梯度

**(b) Logit KD**
teacher 和 student 的 soft logits 做 KL：

```math
L_{kd\_logit}
=
KL\left(
\text{softmax}(z_t/T)\ \|\ \text{softmax}(z_s/T)
\right)\cdot T^2
```

见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L743)

作用：
- 不只学“正确类”
- 还学 teacher 认为哪些类相似

**(c) Feature KD**
student 特征和 teacher 特征做 MSE：

```math
L_{kd\_feat} = \| \hat f_s - \hat f_t \|_2^2
```

见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L764)

作用：
- 直接对齐表征空间

**(d) CKA 结构损失**
不是只对齐每个样本特征，而是对齐“这一批样本之间的关系结构”。

```math
L_{cka} = 1 - \text{CKA}(F_s, F_t)
```

见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L774)

作用：
- 保持整批样本的几何关系一致

生活类比：
- `feat MSE`：要求每个学生站到对应位置
- `CKA`：要求整个班级的队形也尽量一样

---

**10) 反向传播与 EMA 更新**
总损失回传到：
- `VAVAEStudentVAE`
- `Linear classifier`

teacher 不参与梯度更新，而是每次优化器 step 后做 EMA 更新。  
见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L798)

---

**11) 评估**
如果 `lite_eval_enable=True`，会在 `val/test` 上走：

- test transform
- student encoder
- `mu`
- linear classifier
- softmax
- 指标计算：`acc/f1/auc/bac/sens/spec`

见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L891)

---

**12) 保存输出**
当前 Stage1 会保存：

- `litevae_epoch_k_.pth`
- `lite_classifier_epoch_k_.pth`
- `litevae_latest.pth`
- `lite_classifier_latest.pth`
- `lite_gaussian_prior_latest.pth`

见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L1140)

这里的 `lite_gaussian_prior_latest.pth` 不是“训练损失启用后的先验模型”，而是**按类统计好的特征均值/方差文件**。

---

**2. 单样本调试式追踪：按你当前实际代码路径**

你要求假设输入一张 `224x224x3` 图片。  
但要注意：**你当前脚本实际先把图像 resize 到 256，再在 VAVAE 内部 resize 回 224。**

所以真实路径是：

### 2.1 原始输入
- 原图：`(H, W, 3)`，例如 `(原始尺寸, 原始尺寸, 3)`
- 读入后是 PIL RGB

### 2.2 训练增强输出
`Transforms(size=args.image_size)` 里 `image_size` 当前来自 config，默认是 `256`。

所以单样本经过增强后：

- `strong_img`: `(3, 256, 256)`
- `weak_img`: `(3, 256, 256)`

数值范围变化：
- 原图像素是 `[0,255]`
- `ToTensor/ToTensorV2` 后变成 `[0,1]`
- Normalize 后大约落在 `[-2.1, 2.6]`

更具体地：
- R 通道最小约 `(0 - 0.485) / 0.229 ≈ -2.12`
- B 通道最大约 `(1 - 0.406) / 0.225 ≈ 2.64`

---

### 2.3 DataLoader batch=1 时
如果只取 1 张图：

- `img`: `(1, 3, 256, 256)`  强增强
- `ema_img`: `(1, 3, 256, 256)` 弱增强
- `label`: `(1,)`

---

### 2.4 进入 student encoder 前的内部 resize
你当前 student 配置：
- `vavae_student_input_size=224`
- `vavae_student_resize_input=True`

所以进入 encoder 前先变成：

- `x`: `(1, 3, 224, 224)`

见 [vavae_teacher.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/vavae_teacher.py#L385)

---

### 2.5 过 `_Encoder`
当前结构参数：
- `ch=128`
- `ch_mult=(1,1,2,2,4)`
- `num_res_blocks=2`
- `attn_levels=(4,)`

对应 shape 变化如下：

**输入**
- `(1, 3, 224, 224)`

**conv_in**
- `3 -> 128`
- 输出：`(1, 128, 224, 224)`

**Level 0**
- 两个 ResBlock，通道保持 128
- 输出：`(1, 128, 224, 224)`
- Downsample
- 输出：`(1, 128, 112, 112)`

**Level 1**
- 两个 ResBlock，通道仍 128
- 输出：`(1, 128, 112, 112)`
- Downsample
- 输出：`(1, 128, 56, 56)`

**Level 2**
- ResBlock: `128 -> 256`
- ResBlock: `256 -> 256`
- 输出：`(1, 256, 56, 56)`
- Downsample
- 输出：`(1, 256, 28, 28)`

**Level 3**
- 两个 ResBlock，通道 256
- 输出：`(1, 256, 28, 28)`
- Downsample
- 输出：`(1, 256, 14, 14)`

**Level 4**
- ResBlock: `256 -> 512`
- ResBlock: `512 -> 512`
- 此层有 attention
- 输出：`(1, 512, 14, 14)`

**Mid**
- ResBlock
- Attention
- ResBlock
- 输出仍是：`(1, 512, 14, 14)`

**conv_out**
- `512 -> 64`，因为 `2 * z_channels = 64`
- 输出：`(1, 64, 14, 14)`

见 [vavae_teacher.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/vavae_teacher.py#L106)

---

### 2.6 `quant_conv` 和拆分 `mu/logvar`
`quant_conv` 是 `1x1 conv`，通道仍然 64：

- `moments`: `(1, 64, 14, 14)`

然后切成两半：

- `mu_map`: `(1, 32, 14, 14)`
- `logvar_map`: `(1, 32, 14, 14)`

见 [vavae_teacher.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/vavae_teacher.py#L388)

---

### 2.7 池化成向量
因为当前 `pool="avg"`：

- `mu = GAP(mu_map)` -> `(1, 32)`
- `logvar = GAP(logvar_map)` -> `(1, 32)`

---

### 2.8 重参数化
```math
z = \mu + \epsilon \odot \exp(0.5\log\sigma^2)
```

输出：
- `z`: `(1, 32)`

见 [vavae_teacher.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/vavae_teacher.py#L371)

但你当前分类默认用的是 `mu`，所以真正送进分类头的是：

- `feat = mu`: `(1, 32)`

---

### 2.9 线性分类头
`Linear(32, 8)`：

- 输入：`(1, 32)`
- 输出 logits：`(1, 8)`

见 [classifiers.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/models/classifiers.py#L4)

然后 softmax 后：
- 概率向量：`(1, 8)`

这就是单张图的最终类别分布。

---

**3. Teacher 分支的单样本流**
teacher 分支输入的是 `ema_img`，流程和 student 一样：

- `(1, 3, 256, 256)`
- 内部 resize 到 `(1, 3, 224, 224)`
- 编码成 `(1, 32)`
- 分类成 `(1, 8)`

区别只有：
- 输入增强更弱
- 参数来自 EMA teacher
- 不反传

---

**4. 这些操作为什么有意义**

**卷积**
像一个“滑动的小滤镜”，在图像上到处看局部模式。  
类比：
- 医生不是一眼看整张片子，而是先看局部纹理、边缘、颜色块

**下采样**
把大图逐步压缩，保留重要结构，减少计算。  
类比：
- 从“看像素”切换到“看器官结构”

**ResBlock**
让网络更深但不容易学坏。  
类比：
- 每一层不是推翻前一层，而是在前一层基础上微调

**Attention**
在最深层 `14x14` 特征图上，让每个位置都能“看见”其他位置。  
类比：
- 模型不只看局部斑点，还会问：这个斑点和整张皮损其他区域的关系是什么？

**mu / logvar / z**
不是只输出一个特征，而是输出一个“分布”：
- `mu`：中心判断
- `logvar`：不确定性
- `z`：从这个不确定性里采样的实例

类比：
- 医生不是只说“我觉得是黑色素瘤”
- 而是说“我大致判断在这个范围，但还存在这些不确定性”

---

**5. 你当前这条脚本真正优化的是什么**
因为 `kd_only=True`，当前不会优化 ResNet/MRC 那套 `base_loss`，而是主要优化这几项：

```math
L = L_{lite\_ce} + 0.5L_{kd\_logit} + 0.5L_{kd\_feat} + 1.0L_{cka}
```

在代码里最终累加见 [train.py](/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/train.py#L798)

所以你可以把当前 Stage1 理解成：

- 主任务：student 自己要会分类
- 辅任务：student 的输出、特征、结构都尽量向 teacher 靠拢

---

**6. 如果真的只设 Batch Size = 1，会发生什么**
从 shape 上完全能跑通：

- CE 可以算
- KD logit 可以算
- KD feat MSE 可以算

但有一个重要例外：

**CKA 在 batch=1 时几乎没有意义，甚至会退化。**

因为 CKA 本质上比较的是“一批样本之间的结构关系”。  
当 batch 只有 1：
- 没有“样本之间的关系”
- 居中后信息几乎消失
- 结构损失会非常不稳定或接近退化

所以：
- 你让我用 `B=1` 做 shape 调试是可以的
- 但它不代表这套训练在 `B=1` 下是合理工作点

这也是为什么你当前真实脚本用：
- `batch_size=4`
- `grad_accum_steps=4`

---

**7. 当前 Stage1 跑完到底得到什么**
你当前得到的是三类核心产物：

1. **编码器参数**
- `litevae_latest.pth`
- 实际上是 `VAVAEStudentVAE` 的 encoder + quant_conv 参数

2. **分类头参数**
- `lite_classifier_latest.pth`

3. **按类统计的高斯信息**
- `lite_gaussian_prior_latest.pth`

这三个东西分别回答：

- 怎么把图像压成 32 维语义向量
- 怎么把 32 维向量映射成 8 类
- 每一类的 32 维向量大概落在哪、散成什么样

---

**8. 你现在有哪些模块可以考虑删除**
如果你确认后续实验都沿着当前这条 `vavae self-distill` 路线走，可以考虑“下线”这些模块：

**当前脚本下未实际使用的**
- ResNet backbone 路径
- `ProbabilityLoss`
- `BatchLoss`
- `ChannelLoss`
- `mix_enable` 整套 mix 分支
- `aux_vae` 分支
- Stage1 中真正参与反传的 `gaussian_prior_loss` 路径
- `show_teacher_metrics` 相关 ResNet 评估路径

**不建议删的**
- `VAVAEStudentVAE`
- `Linear`
- `ISICDataset`
- `Transforms`
- `lite_gaussian_prior_latest.pth` 的统计保存逻辑
- Stage2 的高斯建模和采样逻辑

原因很简单：
- 前一组是“当前实验不走的历史路径”
- 后一组是你当前主线还在用的

---

**最短总结**
你当前脚本的数据闭环是：

```text
csv -> 读图 -> 强/弱增强 -> student/teacher 双视图
-> VAVAE encoder -> 32维 mu 特征
-> 线性分类头 -> 8类 logits
-> weighted CE + KD(logit/feat/CKA)
-> 反传更新 student
-> EMA 更新 teacher
-> 周期性 val/test 评估
-> 保存 encoder / classifier / 类高斯统计
```

如果你愿意，我下一步可以继续给你两样东西：

1. 一张“你当前 Stage1 实际执行路径”的简化结构图  
2. 一份“哪些文件/模块可以安全裁剪，哪些暂时不要动”的具体删减清单