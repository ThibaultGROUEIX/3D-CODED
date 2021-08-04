import torch
import torch.optim as optim
import my_utils
import json
import visualization
import os
import meter
from termcolor import colored
import pandas as pd
import time

class AbstractTrainer(object):
    def __init__(self, opt):
        super(AbstractTrainer, self).__init__()
        self.start_time = time.time()
        self.opt = opt
        self.git_repo_path = ""
        self.start_visdom()
        self.get_log_paths()
        self.init_meters()
        self.reset_epoch()

        self.save_examples = False
        my_utils.print_arg(self.opt)

    def commit_experiment(self):
        pass

    def get_current_commit(self):
        """
        This helps reproduce results as all results will be associated with a commit
        :return:
        """
        with open("commit.txt", 'r') as f:
            current_commit = f.read()
            print("git repo path : ", self.git_repo_path)
        return self.git_repo_path + current_commit[:-1]

    def init_save_dict(self, opt):
        self.local_dict_to_save_experiment = opt.__dict__
        self.local_dict_to_save_experiment["commit"] = self.get_current_commit()

    def save_new_experiments_results(self):
        """
        This fonction should be called exactly once per experiment and avoid conflicts with other experiments
        :return:
        """
        if os.path.exists('results.csv'):
            self.results = pd.read_csv('results.csv', header=0)
        else:
            columns = []
            self.results = pd.DataFrame(columns=columns)
        self.update_results()
        self.results.to_csv('results.csv', index=False) # Index=False avoids the proliferation of indexes

    def update_results(self):
        self.end_time = time.time()
        self.local_dict_to_save_experiment["timing"] = self.end_time - self.start_time
        self.results = self.results.append(self.local_dict_to_save_experiment, ignore_index=True)
        # self.results.drop(self.results.columns[self.results.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

        # Code snippet from Yana Hasson
        # results est un dataframe pandas, dest_folder est un dossier qui contient index.html
        # le replace est au cas où tu as des tags html dans le contenu de tes colonnes (sinon il ne fait rien)
        html_str = self.results.to_html(table_id="example").replace("&lt;", "<").replace("&gt;", ">")
        # html_str contient le html 'brut'
        with open(os.path.join(self.opt.dest_folder, "raw.html"), "w") as fo:
            # Tu le dump dans un fichier
            fo.write(html_str)

    def start_visdom(self):
        self.visualizer = visualization.Visualizer(self.opt.port, self.opt.env)

    def get_log_paths(self):
        """
        Get paths to save and reload networks
        :return:
        """
        if not os.path.exists("log"):
            print("Creating log folder")
            os.mkdir("log")
        if not os.path.exists(self.opt.dir_name):
            print("creating folder  ", self.opt.dir_name)
            os.mkdir(self.opt.dir_name)

        self.opt.logname = os.path.join(self.opt.dir_name, "log.txt")
        self.opt.checkpointname = os.path.join(self.opt.dir_name, 'optimizer_last.pth')

        # # If a network is already created in th
        self.opt.reload = False
        if os.path.exists(os.path.join(self.opt.dir_name, "network.pth")):
            print("Going to reload experiment")
            self.opt.reload = True

    def init_meters(self):
        self.log = meter.Logs()

    def print_loss_info(self):
        pass

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        pass

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.opt.reload:
            self.optimizer.load_state_dict(torch.load(f'{self.opt.checkpointname}'))
            my_utils.yellow_print("Reloaded optimizer")

    def build_dataset_train(self):
        """
        Create training dataset
        """
        pass

    def build_dataset_test(self):
        """
        Create testing dataset
        """
        pass


    def build_losses(self):
        pass

    def save_network(self):
        print("saving net...")
        torch.save(self.network.state_dict(), f"{self.opt.dir_name}/network.pth")
        torch.save(self.optimizer.state_dict(), f"{self.opt.dir_name}/optimizer_last.pth")

    def dump_stats(self):
        """
        Save stats at each epoch
        """
        log_table = {
            "epoch": self.epoch + 1,
            "lr": self.opt.lrate,
            "env": self.opt.env,
        }
        log_table.update(self.log.current_epoch)
        print(log_table)
        with open(self.opt.logname, "a") as f:  # open and append
            f.write("json_stats: " + json.dumps(log_table) + "\n")
        self.local_dict_to_save_experiment.update(self.log.current_epoch)

        self.opt.start_epoch = self.epoch
        with open(os.path.join(self.opt.dir_name, "options.json"), "w") as f:  # open and append
            f.write(json.dumps(self.opt.__dict__))


    def print_iteration_stats(self, loss):
        """
        print stats at each iteration
        """
        current_time = time.time()
        ellpased_time = current_time - self.start_train_time
        total_time_estimated = self.opt.nepoch * len(self.dataloader_train) * ellpased_time / (0.00001 + self.iteration + 1.0 * self.epoch * self.len_dataset/self.opt.batch_size) # regle de 3
        ETL = total_time_estimated - ellpased_time
        print(
            f"\r["
            + colored(f"{self.epoch}", "cyan")
            + f": "
            + colored(f"{self.iteration}", "red")
            + "/"
            + colored(f"{len(self.dataloader_train)}", "red")
            + "] train loss:  "
            + colored(f"{loss.item()} ", "yellow")
            + colored(f"Ellapsed Time: {ellpased_time/60/60}h ", "cyan")
            + colored(f"ETL: {ETL/60/60}h", "red"),
            end="",
        )


    def train_iteration(self):
        pass

    def learning_rate_scheduler(self):
        """
        Defines the learning rate schedule
        """
        if self.epoch == self.opt.lr_decay_1:
            self.opt.lrate = self.opt.lrate / 10.0
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.epoch == self.opt.lr_decay_2:
            self.opt.lrate = self.opt.lrate / 10.0
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)

    def train_epoch(self):
        pass

    def test_iteration(self):
        pass

    def test_epoch(self):
        pass

    def increment_epoch(self):
        self.epoch = self.epoch + 1

    def increment_iteration(self):
        self.iteration = self.iteration + 1

    def reset_iteration(self):
        self.iteration = 0

    def reset_epoch(self):
        self.epoch = self.opt.start_epoch
