import argparse
import os
import gpustat
import time



class Experiments(object):
    def __init__(self):
        self.inference = {
            # 0: "python inference/script.py --id 0 --randomize 0 --LR_input 0 --model_path ./log/2019-09-05T21:21:12.069673/network_last.pth",
            # 1: "python inference/script.py --id 1 --randomize 0 --LR_input 0 --model_path ./log/2019-09-05T21:23:32.544225/network_last.pth",
            # 2: "python inference/script.py --id 2 --randomize 0 --LR_input 0 --model_path ./log/2019-09-05T22:46:04.314684/network_last.pth",
            3: "python inference/script.py --id 3 --randomize 0 --LR_input 0",
            # 4: "python inference/script.py --id 4 --randomize 1 --LR_input 1",
            # 5: "python inference/script.py --id 5 --randomize 1 --LR_input 1",
            # 6: "python inference/script.py --id 6 --randomize 1 --LR_input 1",
            # 7: "python inference/script.py --id 7 --randomize 1 --LR_input 1",
            # 8: "python inference/script.py --id 8 --randomize 1 --LR_input 1",
            # 9: "python inference/script.py --id 9 --randomize 1 --LR_input 1",
        }
        self.trainings = {
            0: "python training/train_sup_2.py --id 0",
            1: "python training/train_sup_2.py --id 1",
            2: "python training/train_sup_2.py --id 2",
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
        if gpu.memory_used < 20:
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


job_scheduler(exp.inference)
# job_scheduler(exp.trainings)