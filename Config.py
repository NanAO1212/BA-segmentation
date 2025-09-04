import os

class Config:
    def __init__(self):
        self.is_local = not self.is_autodl()

        if self.is_local:
            self.data_path = "data_samples"  # 本地用小数据集
            self.batch_size = 4
            self.num_workers = 2
        else:
            self.data_path = "data"          # AutoDL用完整数据集
            self.batch_size = 16
            self.num_workers = 8

    def is_autodl(self):
        # 检测是否在AutoDL环境
        return os.path.exists('/root/autodl-tmp')
