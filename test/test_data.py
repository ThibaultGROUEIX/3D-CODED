import sys
sys.path.append('./auxiliary/')

from dataset_surreal import *
import pointcloud_processor

datas = torch.load("./data/datas_surreal_train.pth")


def test_is_dataset_centered():
	for i in range(-10,10):
		points = datas[i]
		points, _, _ = pointcloud_processor.center_bounding_box(points)
		points, translation , _ =    pointcloud_processor.center_bounding_box(points)
		assert abs(translation[0]) < 0.01 and abs(translation[1]) < 0.01 and abs(translation[2]) < 0.01, "Problem in training data centering"
