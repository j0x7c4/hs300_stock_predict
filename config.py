# coding:utf-8
from data_utils import *
import os
root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'data')
model_root = os.path.join(root, 'models')

print("data root:", data_root)
print("model root:", model_root)

# -------------------参数配置----------------- #
class Arg:
    def __init__(self):
        # 训练集数据存放路径
        self.train_dir = os.path.join(data_root,"train_mix-17-18.csv")
        # 测试集数据存放路径
        self.test_dir = os.path.join(data_root, "train_mix-1904.csv")
        # 更新数据存放路径
        self.new_dir = os.path.join(data_root, 'train_mix-19.csv')
        # 要预测的数据存放路径
        self.predict_dir = os.path.join(data_root, '000001.csv')
        # 模型存放路径
        self.train_model_dir = os.path.join(root, 'models')
        # fining-tune模型存放路径
        self.fining_turn_model_dir = os.path.join(model_root, 'new_logfile',
                'finet')
        # 训练图存放路径
        self.train_graph_dir = os.path.join(model_root, 'graph', 'train_270')
        # 验证loss存放路径
        self.val_graph_dir = os.path.join(model_root, 'graph', 'val_270')
        # 模型名称
        self.model_name = 'model-270-17-19'
        self.model_name_ft = 'model-ft-01-03'
        self.rnn_unit = 128     # 隐层节点数
        self.input_size = 6     # 输入维度（既用几个特征）
        self.output_size = 6    # 输出维度（既使用分类类数预测）
        self.layer_num = 3      # 隐藏层层数
        self.lr = 0.0006         # 学习率
        self.time_step = 20     # 时间步长
        self.epoch = 50         # 训练次数
        self.epoch_fining = 30  # 微调的迭代次数
        # 单只股票的长度（同一数据集股票长度应处理等长）
        # self.stock_len = get_data_len(os.path.join(data_root,
        #     '399300_1904.csv'))
        self.stock_len = 4
        # 更新后单只股票的长度（同一数据集股票长度应处理等长）
        # self.stock_len_new = get_data_len(os.path.join(data_root,
        #     '399300_190103.csv'))
        self.stock_len_new = 4
        self.batch_size = 1  # batch_size
        self.ratio = 0.8        # 训练集验证集比例
