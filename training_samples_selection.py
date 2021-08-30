from itertools import combinations, groupby
import networkx as nx
import random
import time
import pickle
from LSH_graph_Bar import LSH_graph
import argparse
import json
import torch

class TopKSelection:
    def __init__(self, task, SourceTask, k, iteration, mode, files_path,
                 orig_train_path, seed, criterion='pagerank', intent_num=0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.task = task
        self.source_task = SourceTask
        self.k = k
        self.iteration = iteration
        self.mode = mode
        self.files_path = files_path
        self.orig_train = orig_train_path
        self.seed = seed
        self.intent = intent_num
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
        if self.iteration == 0:
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
        available_pool = []
        pool_to_original = dict()
        pool_counter = 0
        for idx, pair in enumerate(self.original_input):
            if idx in self.available_pool_ids:
                available_pool.append(pair)
                pool_to_original[pool_counter] = idx
                pool_counter += 1
        return available_pool, pool_to_original

    def update_train_pkl(self, selected_k):
        new_train = selected_k
        if self.iteration > 1:
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
        if self.iteration == 0:
            return None
        elif self.mode == "random":
            selected_samples = set(random.sample(range(0, len(self.available_pool_ids)), self.k))
        else:
            selected_samples = self.find_top_k()
        selected_samples = {self.pool_to_original[idx] for idx in selected_samples}
        updated_train_ids = self.update_train_pkl(selected_samples)
        self.update_pool_pkl(selected_samples)
        return updated_train_ids

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
        current_train = self.read_source_dataset(self.source_task)
        if self.iteration >= 1:
            for idx, pair in enumerate(self.original_input):
                if idx in self.current_train_ids:
                    current_train.append(pair)
        random.seed(1)
        random.shuffle(current_train)
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
                                  self.k, self.seed, self.files_path, self.iteration, self.criterion)
        return LSH_graph_obj.get_selected_k

    def define_poolers_path(self):
        task = self.task.split('/')[1]
        if "train" + str(self.intent) + ".txt" in self.orig_train:
            poolers_path = self.orig_train.replace("train" + str(self.intent) + ".txt",
                                                   task + "_available_pool" + str(self.intent)
                                                   + "_iter" + str(self.iteration - 1)
                                                   + "_train_output_seed" + str(self.seed) + ".txt")
        else:
            poolers_path = self.orig_train.replace("train.txt",
                                                   task + "_available_pool" + str(self.intent)
                                                   + "_iter" + str(self.iteration - 1)
                                                   + "_train_output_seed" + str(self.seed) + ".txt")
            if poolers_path[-4:] != ".txt":
                poolers_path = poolers_path.split(".txt", 1)[0]
                poolers_path += '.txt'
        poolers_path = poolers_path.replace('er_magellan/', '')
        return poolers_path

    def write_pairs2file(self, pairs_type):
        if pairs_type == 'pool':
            pairs = self.available_pool
            new_file = open(self.files_path + 'available_pool.txt', "w", encoding="utf-8")
        else:
            pairs = self.current_train
            new_file = open(self.files_path + 'current_train.txt', "w", encoding="utf-8")
        for pair in pairs:
            new_file.write(pair)
        new_file.close()
        return

    @staticmethod
    def get_original_input(orig_train_path):
        training_file = open(orig_train_path, "r", encoding="utf-8")
        training_lines = training_file.readlines()
        training_file.close()
        return training_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="WDC/wdc_shoes_title_small")
    parser.add_argument("--source_task", type=str, default="WDC/wdc_computers_title_small")
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--k_size", type=int, default=200)
    parser.add_argument("--iter_num", type=int, default=2)
    parser.add_argument("--mode", type=str, default="top_k")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--criterion", type=str, default="pagerank")
    start = time.time()
    hp = parser.parse_args()

    task = hp.task
    source_task = hp.source_task
    intent = hp.intent
    k_size = hp.k_size
    iter_num = hp.iter_num
    selection_mode = hp.mode
    seed = hp.seed
    criterion = hp.criterion

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}

    path = configs[task + str(intent)]['path']
    orig_train = configs[task + str(intent)]['trainset']
    source_task += str(intent)
    top_k_manager = TopKSelection(task, source_task, k_size, iter_num, selection_mode, path, orig_train, seed, criterion)
    end = time.time()

    print(f'The process took :{round(end - start, 2)} seconds')

