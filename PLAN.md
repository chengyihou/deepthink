# 最简 OpenMax 改造方案（按你现有代码结构，走标准 OpenMax 思路）

## Summary
保留你当前的训练流程、模型结构和 `Softmax` 损失不变，只把开集识别部分从“类中心最小余弦距离阈值”改成“训练后基于 `logit` 空间的 MAV + Weibull 拟合 + 测试时 top-k 重校准”。

默认按标准 OpenMax 的核心做法落地：
- `MAV` 用每类训练集中“预测正确样本”的 `logit` 均值
- 距离用 `Euclidean`
- 每类只拟合尾部最大 `tailsize` 个距离
- 测试时只重校准 top-`alpha` 个类
- 削减掉的已知类分数全部分给 `unknown`
- 最终在 `C + 1` 维上做 OpenMax 概率并输出 `-1` 作为未知类

## Key Changes
### 1. 训练部分保持不动，只新增 OpenMax 统计阶段
在 [train.py](/Users/yiweicheng/xiaozu/xiangmu/OpenMax/ARPL_test/train.py) 新增一个训练后统计函数，建议命名为 `fit_openmax_stats(...)`。

这个函数做下面几件事：
1. `net.eval()`，遍历 `trainloader`
2. 前向拿到 `feat, logits = net(data, True)`
3. 用 `pred = logits.argmax(dim=1)`，只保留 `pred == labels` 的样本
4. 按类别收集这些“正确分类样本”的原始 `logits`
5. 对每个类 `c`：
   - 计算 `MAV_c = mean(logits_c, dim=0)`，维度是 `(num_classes,)`
   - 计算每个样本到 `MAV_c` 的欧氏距离 `d = ||logits - MAV_c||_2`
   - 取最大的 `tailsize` 个距离作为 tail
   - 用 `scipy.stats.weibull_min.fit(tail, floc=0)` 拟合 Weibull，保存 `(shape, loc, scale)`
6. 保存统计结果，推荐一个文件 `openmax_stats.npz`，内容至少包含：
   - `mavs`: `(C, C)`
   - `weibull_shape`: `(C,)`
   - `weibull_loc`: `(C,)`
   - `weibull_scale`: `(C,)`
   - `alpha`
   - `tailsize`

默认参数直接定死即可：
- `alpha = 3`
- `tailsize = 20`

边界处理也直接定死：
- 某类正确分类样本数少于 `5`，这一类不拟合 Weibull，测试时该类的 Weibull 分数固定为 `0`
- 某类样本数在 `5 <= n < tailsize` 时，用全部距离拟合

### 2. 新增 OpenMax 重校准和测试函数
在 [eval.py](/Users/yiweicheng/xiaozu/xiangmu/OpenMax/ARPL_test/eval.py) 新增两层逻辑。

先写一个单样本或批量可用的重校准函数，建议命名为 `openmax_recalibrate(logits, stats, alpha)`：
1. 输入原始 `logits`，不要先过 `softmax`
2. 对每个样本：
   - 找到 `top-alpha` 类，按 logit 从大到小排序
   - 对每个入选类 `c`：
     - 计算它到自身 `MAV_c` 的欧氏距离 `d_c`
     - 用该类的 Weibull CDF 得到尾部概率 `w_c`
     - 用 rank 权重做衰减，权重固定为：
       - 第 1 名：`1.0`
       - 第 2 名：`(alpha-1)/alpha`
       - 第 3 名：`(alpha-2)/alpha`
     - 重校准该类分数：
       - `revised_c = logit_c * (1 - rank_weight * w_c)`
   - 未入选的类分数保持不变
   - `unknown_score = sum(original_topk - revised_topk)`
3. 拼出 `C + 1` 维向量：
   - 前 `C` 维是 revised known logits
   - 最后一维是 `unknown_score`
4. 对这 `C + 1` 维做 softmax，得到 OpenMax 概率

然后新增 `test_openmax(...)`：
- 已知集 `testloader`：
  - 前向取 `feat, logits = net(data, True)`
  - 保留你现在的 loss 统计方式，继续 `criterion(x, logits, labels)` 算测试损失
  - 预测时不再用 `criterion` 输出的 softmax，而是直接走 `openmax_recalibrate`
  - 若 `argmax` 落在最后一维，则预测标签记为 `-1`，否则为对应已知类
- 未知集 `outloader`：
  - 同样走 `openmax_recalibrate`
- 输出保持和你现在一致：
  - `acc`
  - `AUROC`
  - `f1_macro`
  - `confusion_matrix`
  - `test_x`、`out_x`、`labels_testloader`

AUROC 的定义直接定为：
- `y_true`: 已知类为 `0`，未知类为 `1`
- `y_score`: OpenMax 概率中的 `P(unknown)`

这样就不再依赖当前“最小距离越大越未知”的 AUROC 逻辑。

### 3. 主流程只加一条 OpenMax 分支，不动现有训练主干
在 [OSR.py](/Users/yiweicheng/xiaozu/xiangmu/OpenMax/ARPL_test/OSR.py) 做三处改动。

第一处，新增参数：
- `--osr-method`，默认直接设成 `openmax`
- `--openmax-alpha`，默认 `3`
- `--openmax-tailsize`，默认 `20`

第二处，训练模式：
- 训练循环照旧
- 最后一个 epoch 结束后，调用 `fit_openmax_stats(...)`
- 保存 `openmax_stats.npz`

第三处，评估模式：
- 加载模型后，不再读取 `centers.csv`
- 改为读取 `openmax_stats.npz`
- 调用 `test_openmax(...)`
- 你原来的 `test_center_Dist(...)` 和 `test_center_Dist_thr_from_train(...)` 保留，避免影响旧实验

## Test Plan
1. 闭集行为不应被训练阶段改坏。
   - 训练 loss 和保存模型逻辑保持原样
   - 不允许因为引入 OpenMax 而影响模型收敛流程

2. `openmax_stats.npz` 必须能正确生成。
   - 每个类都有一个 `MAV`
   - 每个可拟合类都有一组 Weibull 参数
   - 样本不足的类按默认规则跳过，不报错

3. 已知类测试时，近 `MAV` 样本应主要保留已知类预测。
   - `acc` 可直接与当前 closed-set 预测比较
   - 少量已知类被拒识是允许的，但不能大面积塌掉

4. 未知类测试时，`P(unknown)` 应显著高于已知类。
   - AUROC 直接用 `unknown` 概率计算
   - 混淆矩阵里应能看到 `-1` 预测

5. 兼容性检查。
   - `ConvNet` 和 `NSRFF` 这两条模型分支都应能拿到 `(feat, logits)`
   - 评估入口 `--eval` 仍可直接运行
   - 结果 CSV 结构不变，未知类仍然写成 `-1`

## Assumptions
- 这是“最简可用 OpenMax”，不是官方源码一比一复刻。
- `MAV` 放在 `logit` 空间，不放在 embedding 空间。
- 距离固定用 `Euclidean`，不做 `eucos`，因为这是在你现有结构下最省事且更接近 OpenMax 的实现。
- 训练阶段不引入任何新的损失或联合优化，OpenMax 只作为训练后统计和测试时后处理。
- 当前环境里 `scipy` 实际未安装；实现前需要先补齐 `scipy`，否则 Weibull 拟合无法按这个方案落地。
