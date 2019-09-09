#TODO for what i want
# Currently eval is done online so not super useful

import os
import pandas as pd
import sys


class MakeTable(object):
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.results = pd.read_csv(path, header=0)

    def get_type_of_properties(self, list_of_properties):
        self.results = self.results[list_of_properties]

    def get_equal_properties(self, list_of_properties):
        for key in list_of_properties:
            self.results = self.results[self.results[key] == list_of_properties[key]]

    def get_positive_properties(self, list_of_properties):
        for key in list_of_properties:
            self.results = self.results[self.results[key] > list_of_properties[key]]

    def sort(self, list_of_properties):
        self.results = self.results.sort_values(list_of_properties)

    def save(self, name):
        self.results.to_csv(name + ".csv", index=False)
        html_str = self.results.to_html(table_id="example").replace("&lt;", "<").replace("&gt;", ">")
        # html_str contient le html 'brut'
        with open(os.path.join("html", name + ".html"), "w") as fo:
            # Tu le dump dans un fichier
            fo.write(html_str)

    def remove_fail_experiments_line(self, list):
        for i in list:
            self.results = self.results.drop(i)


todelete = []


def make_table_2(path):
    print("Making table 2 from result files")
    table = MakeTable(path)
    table.remove_fail_experiments_line(todelete)
    table.get_positive_properties({"iou_NN_NN_emsemble": 0, "num_shots_eval": 10000})
    table.get_type_of_properties(
        ["timing", "part_supervision", "cat", "both_cycle_ours_emsemble", "iou_NN_NN_emsemble", "iou_NN_ours_emsemble",
         "iou_chamfer_ours_emsemble", "iou_cosine_ours_emsemble", "oracle_NN_NN", "oracle_ours_ours"])
    table.sort(["part_supervision", "cat"])
    table.save("table_2")


def make_table_3(path):
    print("Making table 3 from result files")
    table = MakeTable(path)
    table.remove_fail_experiments_line(todelete)
    table.get_positive_properties({"iou_NN_NN_emsemble": 0})
    table.get_equal_properties({"cat": "Car", "part_supervision": 0, "num_shots_eval": 100})
    table.get_type_of_properties(
        ["timing", "knn", "cat", "both_cycle_ours", "iou_NN_ours", "iou_NN_NN", "iou_chamfer_ours", "iou_cosine_ours",
         "oracle_ours_ours", "oracle_NN_NN", "lambda_chamfer", "lambda_cycle_2"])
    table.sort(["knn"])
    table.save("table_3")


def make_table_1(path):
    print("Making table 1 from result files")
    table = MakeTable(path)
    table.remove_fail_experiments_line(todelete)
    table.get_positive_properties({"iou_NN_NN": 0})
    table.get_equal_properties({"part_supervision": 0, "num_shots_eval": 10})
    table.get_type_of_properties(
        ["timing", "cat", "part_supervision", "iou_NN_ICP_NN", "iou_NN_atlasPatch", "iou_NN_atlasSphere",
         "both_cycle_ours", "iou_NN_ours", "iou_chamfer_ours", "iou_cosine_ours", "oracle_ours_ours", "num_shots_eval"])
    table.sort(["cat"])
    table.save("table_1")


def make_table_4(path):
    print("Making table 4: A ablation study on cars with full-shot without ensemble")
    table = MakeTable(path)
    table.remove_fail_experiments_line(todelete)
    table.get_positive_properties({"iou_NN_NN_emsemble": 0})
    table.get_equal_properties({"cat": "Car", "part_supervision": 0, "num_shots_eval": 5000})
    table.get_type_of_properties(
        ["timing", "knn", "cat", "both_cycle_ours", "iou_NN_ours", "iou_NN_NN", "iou_chamfer_ours", "iou_cosine_ours",
         "oracle_ours_ours", "oracle_NN_NN", "lambda_chamfer", "lambda_cycle_2"])
    table.sort(["knn"])
    table.save("table_4")


def make_table_5(path):
    print("Making table 5 : A ablation study on all cats with full-shot with ensemble")
    table = MakeTable(path)
    table.remove_fail_experiments_line(todelete)
    table.get_positive_properties({"iou_NN_NN_emsemble": 0})
    table.get_equal_properties({"num_shots_eval": 5000})
    table.get_type_of_properties(
        ["timing", "part_supervision", "cat", "both_cycle_ours_emsemble", "iou_NN_NN_emsemble", "iou_NN_ours_emsemble",
         "iou_chamfer_ours_emsemble", "iou_cosine_ours_emsemble", "oracle_NN_NN", "oracle_ours_ours", "lambda_chamfer", "lambda_cycle_2", "knn"])
    table.sort(["part_supervision", "cat"])
    table.save("table_5")


if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except:
        path = "results.csv"

    make_table_2(path)
    make_table_3(path)
    make_table_1(path)
    make_table_4(path)
    make_table_5(path)