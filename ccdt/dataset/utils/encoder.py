# 计算机登录用户: jk
# 系统日期: 2023/5/17 9:53
# 项目名称: async_ccdt
# 开发者: zhanyong
import json
import numpy as np


class Encoder(json.JSONEncoder):
    """
    labelme数据保存编码实现类
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(Encoder, self).default(obj)
