# 训练命令
# 填写对应的权重文件和超参数文件
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("D:/software/pycharm_project/original_ultralytics-main/datasets_shan/yolov8n.yaml").train(**{'cfg':'/software/pycharm_project/original_ultralytics-main/ultralytics/cfg/default.yaml'})


