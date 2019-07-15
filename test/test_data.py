import sys
sys.path.append('./auxiliary/')

from dataset_surreal import *
import pointcloud_processor

datas = torch.load("./data/datas_surreal_train.pth")


def is_dataset_centered():
	for i in range(-10,10):
		data, _,_,_ = datas[i]
		points, _, _ = pointcloud_processor.center_bounding_box(points)
		points, translation , _ =    pointcloud_processor.center_bounding_box(points)
		if abs(translation[0]) > 0.01 or abs(translation[1]) < 0.01 or abs(translation[2]) < 0.01:
			return False
	return True