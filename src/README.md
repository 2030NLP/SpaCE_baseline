# 文件夹说明
```
.
├─data (储存输入文件与输出结果)
├─scripts (Linux环境下可直接运行的脚本，请在项目根目录下运行，如`sh scripts/task1_train.sh`)
└─src (baseline与评测代码)
   ├─task1 (task1的评测代码与baseline)
   ├─task2 (task2的评测代码与旧baseline，模型占空间较小，分数较低)
   ├─task2_new (task2的评测代码与新baseline，模型占空间较大，分数稍高)
   ├─task3_extractive (task3的评测代码与事件抽取式baseline)
   └─task3_generative (task3的评测代码与Seq2Seq生成式baseline)
```

# Baseline代码文件说明
analyze.py: 对数据集特征的分析

data.py: 读取数据集并转化为TensorDataset以便于训练与评测

evaluate.py: 简单评估模型表现

examine.py: 对生成的预测结果文件进行最终评分

model.py: baseline模型代码

predict.py: 使用训练好的模型预测评测集与测试集的结果

train.py: 对模型进行训练

utils.py: 模型运行时依赖的函数

# 其他提示
scripts目录下脚本中的`CUDA_VISIBLE_DEVICES=4`代表使用的显卡编号，请根据自己计算资源的情况进行修改（如果不与其他人共享计算资源也可删去）。