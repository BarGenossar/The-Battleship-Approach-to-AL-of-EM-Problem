from itertools import combinations, groupby
import networkx as nx
import random
import time
import pickle
from LSH_graph_Bar import LSH_graph
import argparse
import json
import torch
import os


class TopKSelection:
    def __init__(self, task, SourceTask, k, iteration, mode, files_path,
                 orig_train_path, seed, iterations, output_path, criterion='pagerank', intent_num=0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.task = task
        self.source_task = SourceTask
        self.k = k
        self.iter = iteration
        self.mode = mode
        self.files_path = files_path
        self.orig_train = orig_train_path
        self.seed = seed
        self.iterations = iterations
        self.intent = intent_num
        self.output_path = output_path
        self.criterion = criterion
        self.original_input = self.get_original_input(self.orig_train)
        self.available_pool_ids = self.get_available_pool_ids()
        self.available_pool, self.pool_to_original = self.get_available_pool()
        self.current_train_ids = self.get_new_train_ids()
        self.current_train = self.get_new_train()
        self.write_pairs2file('pool')
        self.write_pairs2file('train')
        # self.pairs_ids, self.ids_pairs = self.get_pairs_ids()

    def get_available_pool_ids(self):
        if self.mode == "all_D" or "only_selected" in self.mode:
            return None
        if self.iter == 0:
            available_pool_ids = set([idx for idx in range(len(self.original_input))])
            output = open(self.files_path + 'available_pool_ids.pkl', 'wb')
            pickle.dump(available_pool_ids, output)
            output.close()
        else:
            pkl_file = open(self.files_path + 'available_pool_ids.pkl', 'rb')
            available_pool_ids = pickle.load(pkl_file)
            pkl_file.close()
        return available_pool_ids

    def get_available_pool(self):
        if self.mode == "all_D" or "only_selected" in self.mode:
            return None, None
        available_pool = []
        pool_to_original = dict()
        pool_counter = 0
        for idx, pair in enumerate(self.original_input):
            if idx in self.available_pool_ids:
                available_pool.append(pair)
                pool_to_original[pool_counter] = idx
                pool_counter += 1
        self.save_to_pkl(pool_to_original, "pool_to_original_dict")
        return available_pool, pool_to_original

    def update_train_pkl(self, selected_k):
        new_train = selected_k
        if self.iter > 1:
            pkl_file = open(self.files_path + 'current_train.pkl', 'rb')
            previous_train = pickle.load(pkl_file)
            pkl_file.close()
            new_train.update(previous_train)
        output = open(self.files_path + 'current_train.pkl', 'wb')
        pickle.dump(new_train, output)
        output.close()
        return new_train

    def update_pool_pkl(self, selected_k):
        pkl_file = open(self.files_path + 'available_pool_ids.pkl', 'rb')
        available_pool_ids = pickle.load(pkl_file)
        pkl_file.close()
        available_pool_ids = available_pool_ids.difference(selected_k)
        pkl_file = open(self.files_path + 'available_pool_ids.pkl', 'wb')
        pickle.dump(available_pool_ids, pkl_file)
        pkl_file.close()
        self.available_pool_ids = available_pool_ids
        self.available_pool, self.pool_to_original = self.get_available_pool()

    def get_new_train_ids(self):
        if "only_selected" in self.mode:
            return self.find_all_selected()
        elif self.mode == "all_D" or self.iter == 0:
            return None
        elif self.mode == "random":
            selected_samples = set(random.sample(range(0, len(self.available_pool_ids)), self.k))
        else:
            selected_samples = self.find_top_k()
        selected_samples = {self.pool_to_original[idx] for idx in selected_samples}
        updated_train_ids = self.update_train_pkl(selected_samples)
        self.update_pool_pkl(selected_samples)
        self.save_to_pkl(selected_samples, "selected_k_pool_to_original")
        return updated_train_ids

    def find_all_selected(self):
        selected_ids = set()
        base_mode = self.mode.split("/")[0]
        current_path = self.output_path + base_mode + "/pkl_files/selected_k_pool_to_original_iter"
        for current_iter in range(1, iterations + 1):
            selected_ids.update(self.read_selected_k_pkl_file(current_iter, current_path))
        return selected_ids

    def read_selected_k_pkl_file(self, current_iter, current_path):
        pkl_file = open(current_path + str(current_iter) + "_seed" +
                        str(self.seed) + ".pkl", 'rb')
        required_file = pickle.load(pkl_file)
        pkl_file.close()
        return required_file

    def read_source_dataset(self, source_task):
        source_dataset_file = self.find_file(source_task)
        source_dataset = open(source_dataset_file, "r", encoding="utf-8")
        source_lines = source_dataset.readlines()
        source_dataset.close()
        current_train = []
        for pair in source_lines:
            current_train.append(pair)
        return current_train

    @staticmethod
    def find_file(source_task):
        configs = json.load(open('configs.json'))
        configs = {conf['name']: conf for conf in configs}
        config = configs[source_task]
        source_dataset_file = config['trainset']
        return source_dataset_file

    def get_new_train(self):
        if self.mode == "all_D":
            return self.original_input
        elif "only_selected" in self.mode:
            return self.read_only_selected()
        current_train = self.read_source_dataset(self.source_task)
        if self.iter >= 1:
            for idx, pair in enumerate(self.original_input):
                if idx in self.current_train_ids:
                    current_train.append(pair)
            random.seed(self.seed)
            random.shuffle(current_train)
        return current_train

    def read_only_selected(self):
        current_train = []
        for idx, pair in enumerate(self.original_input):
            if idx in self.current_train_ids:
                current_train.append(pair)
        return current_train


    def get_pairs_ids(self):
        pairs_ids_dict, ids_pairs_dict = dict(), dict()
        for idx, pair in enumerate(self.original_input):
            pairs_ids_dict[pair] = idx
            ids_pairs_dict[idx] = pair
        return pairs_ids_dict, ids_pairs_dict

    def find_top_k(self):
        poolers_path = self.define_poolers_path()
        poolers_path_available_pool = poolers_path.replace("data", "output")
        poolers_path_current_train = poolers_path_available_pool.replace("available_pool", "current_train")
        LSH_graph_obj = LSH_graph([poolers_path_available_pool, poolers_path_current_train],
                                  self.k, self.seed, self.files_path, self.output_path, self.iter,
                                  self.criterion, self.mode)
        return LSH_graph_obj.get_selected_k

    def define_poolers_path(self):
        task = self.task.split('/')[1]
        source_task = self.output_path.split('/')[-2]
        if "train" + str(self.intent) + ".txt" in self.orig_train:
            poolers_path = self.orig_train.replace("train" + str(self.intent) + ".txt",
                                                   source_task + "/" + self.mode + "/"
                                                   + task + "_available_pool" + str(self.intent)
                                                   + "_iter" + str(self.iter - 1)
                                                   + "_train_output_seed" + str(self.seed) + ".txt")
        else:
            poolers_path = self.orig_train.replace("train.txt",
                                                   source_task + "/" + self.mode + "/"
                                                   + task + "_available_pool" + str(self.intent)
                                                   + "_iter" + str(self.iter - 1)
                                                   + "_train_output_seed" + str(self.seed) + ".txt")
            if poolers_path[-4:] != ".txt":
                poolers_path = poolers_path.split(".txt", 1)[0]
                poolers_path += '.txt'
        # poolers_path = poolers_path.replace('er_magellan/', '')
        return poolers_path

    def write_pairs2file(self, pairs_type):
        if (self.mode == "all_D" or "only_selected" in self.mode) and \
                pairs_type == 'pool':
            return
        source_task = self.output_path.split('/')[-2]
        documentation_path = self.files_path + source_task + "/" + self.mode + "/"
        if not os.path.exists(documentation_path):
            os.makedirs(documentation_path)
        if pairs_type == 'pool':
            pairs = self.available_pool
            new_file1 = open(self.files_path + 'available_pool.txt', "w", encoding="utf-8")
            new_file2 = open(documentation_path + 'available_pool_iter' + str(self.iter) +
                             '_seed' + str(self.seed) + '.txt', "w", encoding="utf-8")
        else:
            pairs = self.current_train
            new_file1 = open(self.files_path + 'current_train.txt', "w", encoding="utf-8")
            if self.mode == "all_D":
                new_file2 = open(documentation_path + 'all_D.txt', "w", encoding="utf-8")
            else:
                new_file2 = open(documentation_path + 'current_train_iter' + str(self.iter) +
                                 '_seed' + str(self.seed) + '.txt', "w", encoding="utf-8")
        for pair in pairs:
            new_file1.write(pair)
            new_file2.write(pair)
        new_file1.close()
        new_file2.close()
        return

    @staticmethod
    def get_original_input(orig_train_path):
        training_file = open(orig_train_path, "r", encoding="utf-8")
        training_lines = training_file.readlines()
        training_file.close()
        return training_lines

    def save_to_pkl(self, file, file_name):
        path = self.output_path + self.mode + "/pkl_files/"
        if not os.path.exists(path):
            os.makedirs(path)
        output = open(path + file_name + '_iter' + str(self.iter) +
                      '_seed' + str(self.seed) + '.pkl', 'wb')
        pickle.dump(file, output)
        output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Walmart-Amazon")
    parser.add_argument("--source_task", type=str, default="Structured/Amazon-Google")
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--k_size", type=int, default=100)
    parser.add_argument("--iter_num", type=int, default=0)
    parser.add_argument("--mode", type=str, default="top_k_threshold/only_selected")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--criterion", type=str, default="pagerank")
    parser.add_argument("--output_path", type=str, default="output/er_magellan/Structured/Walmart-Amazon/Amazon-Google/")
    start = time.time()
    hp = parser.parse_args()

    task = hp.task
    source_task = hp.source_task
    intent = hp.intent
    k_size = hp.k_size
    iter_num = hp.iter_num
    selection_mode = hp.mode
    seed = hp.seed

    # if "only_selected" in mode then iterations = 0 in the call for main.sh but in this file it must be
    # the original number of iterations
    iterations = hp.iterations
    criterion = hp.criterion
    output_path = hp.output_path

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}

    path = configs[task + str(intent)]['path']
    orig_train = configs[task + str(intent)]['trainset']
    source_task += str(intent)
    top_k_manager = TopKSelection(task, source_task, k_size, iter_num, selection_mode,
                                  path, orig_train, seed, iterations, output_path, criterion)
    end = time.time()

    print(f'The process took :{round(end - start, 2)} seconds')
