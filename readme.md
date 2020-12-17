1. 在eval_4.py中进行配置（加粗为可能需要修改的）

变量  | 意义
------------- | -------------
**data_dir_dic**| 数据集映射，ds_type为train，dev，test, idx为哪一折|
**pretrain_map**| 预训练模型的映射
**pretrain_model**| 训练使用的预训练模型
ir_type |选择要训练或验证的数据集，来自data_dir_dic的键值
**gpu** |训练使用的gpu，用“，”分开
**dev_gpu**| 验证使用的gpu
output| 模型输出文件夹
data_dir| 数据集文件夹和缓存文件夹
model_type| 使用的模型类型，来自config的MODEL_CLASSES定义
**models**| 验证时使用的每折模型，来自trained_models.py中定义

其他：**max_seq_len, lr, epoch, p_num(topk段落)，batch_size, grad_acc(梯度累计)**|

2. 运行请使用
    * 训练\
      修改eval_4.py的main中为train(), 设置超参数和数据集路径和预训练模型路径后，执行：
        ```commandline
        python eval_4.py
        ```
    * 验证\
      修改eval_4.py的main中为dev_all_folds(), 并修改models为想要验证和测试的五折模型，执行：
        ```commandline
        python eval_4.py
        ```
