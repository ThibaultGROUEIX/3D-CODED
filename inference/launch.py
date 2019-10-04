import argparse
import os
import gpustat
import time

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="inference", choices=['training', 'inference', ''])
    opt = parser.parse_args()
    return opt

opt = parser()

class Experiments(object):
    def __init__(self):
        self.inference = {
            0: "python inference/script.py --dir_name learning_elementary_structure_trained_models/0point_translation --HR 1 --faust INTER",
            1: "python inference/script.py --dir_name learning_elementary_structure_trained_models/1patch_deformation --HR 1 --faust INTER",
            2: "python inference/script.py --dir_name learning_elementary_structure_trained_models/2point_translation_and_patch_deformation --HR 1 --faust INTER",
            3: "python inference/script.py --dir_name learning_elementary_structure_trained_models/3D_CODED --HR 1 --faust INTER",
            4: "python inference/script.py --dir_name learning_elementary_structure_trained_models/0point_translation --HR 1 --faust INTRA",
            5: "python inference/script.py --dir_name learning_elementary_structure_trained_models/1patch_deformation --HR 1 --faust INTRA",
            6: "python inference/script.py --dir_name learning_elementary_structure_trained_models/2point_translation_and_patch_deformation --HR 1 --faust INTRA",
            7: "python inference/script.py --dir_name learning_elementary_structure_trained_models/3D_CODED --HR 1 --faust INTRA",
        }
        self.trainings = {
            0: "python training/train.py --id 0 --point_translation 0 --patch_deformation 1",
            1: "python training/train.py --id 1 --point_translation 0 --patch_deformation 1",
            2: "python training/train.py --id 2 --point_translation 0 --patch_deformation 0",
            3: "python training/train.py --id 3 --point_translation 1 --patch_deformation 1",
        }

exp = Experiments()

def get_first_available_gpu():
    """
    Check if a gpu is free and returns it
    :return: gpu_id
    """
    query = gpustat.new_query()
    for gpu_id in range(len(query)):
        gpu = query[gpu_id]
        if gpu.memory_used < 700:
            has = os.system("tmux has-session -t " + f"GPU{gpu_id}" + " 2>/dev/null")
            if not int(has)==0:
                return gpu_id
    return -1


def job_scheduler(dict_of_jobs):
    """
    Launch Tmux session each time it finds a free gpu
    :param dict_of_jobs:
    """
    keys = list(dict_of_jobs.keys())
    while len(keys) > 0:
        job_key = keys.pop()
        job = dict_of_jobs[job_key]
        while get_first_available_gpu() < 0:
            print("Waiting to find a GPU for ", job)
            time.sleep(30) # Sleeps for 30 sec
        gpu_id = get_first_available_gpu()
        cmd = f"conda activate pytorch-3D-CODED; CUDA_VISIBLE_DEVICES={gpu_id} {job} 2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t GPU{gpu_id}"
        CMD = f'tmux new-session -d -s GPU{gpu_id} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)
        time.sleep(15)  # Sleeps for 30 sec


if not os.path.exists("./data/datas_surreal_train.pth"):
            os.system("chmod +x ./data/download_dataset.sh")
            os.system("./data/download_dataset.sh")
            os.system("mv *.pth data/")

if not os.path.exists("./data/template/template.ply"):
    os.system("chmod +x ./data/download_template.sh")
    os.system("./data/download_template.sh")

if not os.path.exists("log_terminals"):
    print("Creating log_terminals folder")
    os.mkdir("log_terminals")


if opt.mode == "training":
    print("training mode")
    job_scheduler(exp.trainings)
if opt.mode == "inference":
    if not os.path.exists("learning_elementary_structure_trained_models/0point_translation/network.pth"):
        os.system("chmod +x ./inference/download_trained_models.sh")
        os.system("./inference/download_trained_models.sh")
    print("inference mode")
    job_scheduler(exp.inference)