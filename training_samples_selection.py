from itertools import combinations, groupby
import networkx as nx
import random
import time
import pickle
from battleships import battleships_graph
from DTAL import DTAL
import argparse
import json
import torch
import os
import re


class TopKSelection:
    def __init__(self, task, SourceTask, k, iteration, mode, files_path,
                 orig_train_path, seed, iterations, output_path, from_iter,
                 criterion='pagerank', intent_num=0, replace_param=1,
                 without_DA=True):
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
        self.intent = intent_num  # Irrelevant
        self.output_path = output_path
        self.criterion = criterion
        self.replaced_samples_size = replace_param * k # Irrelevant
        self.weak_supervision = "_ws_" in mode or mode == "top_k_DTAL"
        self.from_iter = from_iter
        self.without_da = without_DA
        self.original_input = self.get_original_input(self.orig_train)
        self.handle_from_iter()
        self.available_pool_ids = self.get_available_pool_ids()
        self.available_pool, self.pool_to_original, self.pool_labels_dict = self.get_available_pool()
        self.current_train_ids, self.high_confidence_positive, self.high_confidence_negative = self.handle_new_train_ids()
        self.D_prime_neg_labels = self.get_D_prime_neg_labels()
        self.removed_from_D_prime = self.get_removed_idxs()
        self.current_train = self.get_new_train()
        self.write_pairs2file('pool')
        self.write_pairs2file('train')
        self.create_weak_file()

    def handle_new_train_ids(self):
        if "DTAL" in self.mode:
            return self.get_new_train_ids_DTAL()
        else:
            return self.get_new_train_ids()

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
            return None, None, None
        available_pool = []
        pool_to_original = dict()
        pool_counter = 0
        pool_labels_dict = {0: [], 1: []}
        for idx, pair in enumerate(self.original_input):
            if idx in self.available_pool_ids:
                available_pool.append(pair)
                label = int(re.sub("[^0-9]", "", pair[-2]))
                pool_labels_dict[label].append(idx)
                pool_to_original[pool_counter] = idx
                pool_counter += 1
        self.save_to_pkl(pool_to_original, "pool_to_original_dict")
        return available_pool, pool_to_original, pool_labels_dict

    def handle_from_iter(self):
        if self.from_iter == 0 or self.from_iter != self.iter:
            return
        available_pool = set([idx for idx in range(len(self.original_input))])
        if "DTAL" in self.mode:
            files_list = ["selected_k_pool_to_original_iter", "high_confidence_negative_k_pool_to_original_iter",
                          "high_confidence_positive_k_pool_to_original_iter"]
        else:
            files_list = ["selected_k_pool_to_original_iter", "ws_neg_cands_pool_to_original_iter",
                          "ws_pos_cands_pool_to_original_iter"]
        pkl_output_path = self.output_path + self.mode + '/pkl_files/'
        selected_ids = set()
        weak_ids = set()
        for curr_iter in range(1, self.iter):
            for file in files_list:
                pkl_file = open(pkl_output_path + file + str(curr_iter) + '_seed' + str(self.seed) + '.pkl', 'rb')
                file_ids = pickle.load(pkl_file)
                pkl_file.close()
                selected_ids.update(file_ids)
                if self.weak_supervision:
                    weak_ids.update(file_ids)
        available_pool = available_pool.difference(selected_ids)
        output = open(self.files_path + 'available_pool_ids.pkl', 'wb')
        output2 = open(self.files_path + 'weak_ids.pkl', 'wb')
        pickle.dump(available_pool, output)
        pickle.dump(weak_ids, output2)
        output.close()
        output2.close()
        self.update_current_train_from_iter()
        return

    def update_current_train_from_iter(self):
        current_path = self.files_path + self.source_task.split('/')[1][:-1] + '/' + self.mode + '/'
        correct_file = open(current_path + "current_train_iter" + str(self.iter - 1) + "_seed" + str(self.seed) + '.txt',
                            "r", encoding="utf-8")
        file_lines = correct_file.readlines()
        correct_file.close()
        new_current_train = open(self.files_path + 'current_train.txt', "w", encoding="utf-8")
        for pair in file_lines:
            new_current_train.write(pair)
        new_current_train.close()
        return

    def update_train_pkl(self, new_samples):
        new_train = new_samples
        if self.iter >= 1:
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
        self.available_pool, self.pool_to_original, pool_labels_dict = self.get_available_pool()

    def update_weak_pkl(self, ws_pos_cands, ws_neg_cands):
        new_weaks = ws_pos_cands.union(ws_neg_cands)
        if self.iter >= 1:
            pkl_file = open(self.files_path + 'weak_ids.pkl', 'rb')
            previous_weaks = pickle.load(pkl_file)
            pkl_file.close()
            new_weaks.update(previous_weaks)
        output = open(self.files_path + 'weak_ids.pkl', 'wb')
        pickle.dump(new_weaks, output)
        output.close()
        return new_weaks

    def create_weak_file(self):
        if self.iter == 0:
            dummy = set()
            output = open(self.files_path + 'weak_ids.pkl', 'wb')
            output2 = open(self.files_path + 'weak_ids_current_train.pkl', 'wb')
            pickle.dump(dummy, output)
            pickle.dump(dummy, output2)
            output.close()
        return

    def get_new_train_ids(self):
        ws_pos_cands, ws_neg_cands = set(), set()
        random.seed(self.seed)
        if "only_selected" in self.mode:
            return self.find_all_selected(), None, None
        # elif self.mode == "all_D" or self.iter == 0:
        #     return None, None, None
        elif self.mode == "all_D":
            return None, None, None
        elif self.iter == 0:
            # We assume that we start with k/2 positives and k/2 negatives
            selected_samples = set(random.sample(self.pool_labels_dict[0], int(self.k/2)))
            selected_samples.update(set(random.sample(self.pool_labels_dict[1], int(self.k / 2))))
        elif self.mode == "random":
            selected_samples = set(random.sample(range(0, len(self.available_pool_ids)), self.k))
        else:
            selected_samples, ws_pos_cands, ws_neg_cands = self.find_top_k()
        selected_samples = {self.pool_to_original[idx] for idx in selected_samples}
        if self.weak_supervision:
            ws_pos_cands = {self.pool_to_original[idx] for idx in ws_pos_cands}
            ws_neg_cands = {self.pool_to_original[idx] for idx in ws_neg_cands}
        new_samples = set.union(selected_samples, ws_pos_cands, ws_neg_cands)
        updated_train_ids = self.update_train_pkl(new_samples)
        self.update_pool_pkl(new_samples)
        self.update_weak_pkl(ws_pos_cands, ws_neg_cands)
        self.save_to_pkl(selected_samples, "selected_k_pool_to_original")
        self.save_to_pkl(ws_pos_cands, "ws_pos_cands_pool_to_original")
        self.save_to_pkl(ws_neg_cands, "ws_neg_cands_pool_to_original")
        return updated_train_ids, None, None

    def get_new_train_ids_DTAL(self):
        if self.iter == 0:
            if self.without_da:
                selected_samples = set(random.sample(self.pool_labels_dict[0], int(self.k / 2)))
                selected_samples.update(set(random.sample(self.pool_labels_dict[1], int(self.k / 2))))
                high_confidence_positive, high_confidence_negative = set(), set()
            else:
                return None, None, None
        else:
            selected_samples, high_confidence_positive, high_confidence_negative = self.top_k_DTAL()
            if self.weak_supervision:
                high_confidence_positive = {self.pool_to_original[idx] for idx in high_confidence_positive}
                high_confidence_negative = {self.pool_to_original[idx] for idx in high_confidence_negative}
            else:
                high_confidence_positive = set()
                high_confidence_negative = set()
        selected_samples = {self.pool_to_original[idx] for idx in selected_samples}
        self.save_to_pkls([selected_samples, high_confidence_positive, high_confidence_negative],
                          ["selected_k_pool_to_original", "high_confidence_positive_k_pool_to_original",
                           "high_confidence_negative_k_pool_to_original"])
        new_samples = set.union(selected_samples, high_confidence_positive, high_confidence_negative)
        updated_train_ids = self.update_train_pkl(new_samples)
        self.update_pool_pkl(new_samples)
        return updated_train_ids, high_confidence_positive, high_confidence_negative

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

    def read_source_dataset(self):
        current_train = []
        source_lines = self.get_source_lines()
        if self.without_da:
            return []
        if "D_rep" not in self.mode or self.iter == 0:
            for pair in source_lines:
                current_train.append(pair)
        else:
            for id_val, pair in enumerate(source_lines):
                if id_val not in self.removed_from_D_prime:
                    current_train.append(pair)
        return current_train

    @staticmethod
    def find_file(source_task):
        configs = json.load(open('configs.json'))
        configs = {conf['name']: conf for conf in configs}
        config = configs[source_task]
        source_dataset_file = config['trainset']
        return source_dataset_file

    def get_D_prime_neg_labels(self):
        if "D_rep" not in self.mode:
            return None
        D_prime_neg_idx = set()
        source_lines = self.get_source_lines()
        for id_val, line in enumerate(source_lines):
            if int(re.sub("[^0-9]", "", line[-2])) == 0:
                D_prime_neg_idx.add(id_val)
        return D_prime_neg_idx

    def get_source_lines(self):
        source_dataset_file = self.find_file(self.source_task)
        source_dataset = open(source_dataset_file, "r", encoding="utf-8")
        source_lines = source_dataset.readlines()
        source_dataset.close()
        return source_lines

    def get_removed_idxs(self):
        if self.iter == 0 or "D_rep" not in self.mode:
            return None
        else:
            current_path = self.output_path + self.mode + "/pkl_files/removed_from_D_prime.pkl"
            if self.iter == 1:
                return self.initialize_removed_idxs(current_path)
            else:
                return self.update_removed_idxs(current_path)

    def initialize_removed_idxs(self, current_path):
        random.seed(self.seed)
        removed_idxs = set(random.sample(self.D_prime_neg_labels, self.replaced_samples_size))
        self.update_removed_file(current_path, removed_idxs)
        return removed_idxs

    @staticmethod
    def update_removed_file(current_path, removed_idxs):
        output = open(current_path, 'wb')
        pickle.dump(removed_idxs, output)
        output.close()
        return

    def update_removed_idxs(self, current_path):
        random.seed(self.seed)
        pkl_file = open(current_path, 'rb')
        removed_idxs = pickle.load(pkl_file)
        pkl_file.close()
        candidates_for_removal = {idx for idx in self.D_prime_neg_labels if idx not in removed_idxs}
        just_removed = set(random.sample(candidates_for_removal, self.replaced_samples_size))
        removed_idxs.update(just_removed)
        self.update_removed_file(current_path, removed_idxs)
        return removed_idxs

    def get_new_train(self):
        if self.mode == "all_D":
            return self.original_input
        elif "only_selected" in self.mode:
            return self.read_only_selected()
        current_train = self.read_source_dataset()
        if self.iter >= 1 or self.without_da:
            if self.weak_supervision:
                current_train = self.get_current_train_ws(current_train)
            else:
                current_train = self.get_current_train_without_ws(current_train)
        return current_train

    def get_current_train_ws(self, current_train):
        weak_ids_orig = self.find_weak_ids()
        weak_ids_tmp = set()
        counter = 0
        for idx, pair in enumerate(self.original_input):
            if idx in self.current_train_ids:
                current_train.append(self.modify_pair(idx, pair))
                if idx in weak_ids_orig:
                    weak_ids_tmp.add(counter)
                counter += 1
        orig_mapping = self.create_orig_mapping(current_train)
        random.seed(self.seed)
        random.shuffle(current_train)
        self.convert_weak_ids(current_train, orig_mapping, weak_ids_tmp)
        return current_train

    def get_current_train_without_ws(self, current_train):
        for idx, pair in enumerate(self.original_input):
            if idx in self.current_train_ids:
                current_train.append(self.modify_pair(idx, pair))
        random.seed(self.seed)
        random.shuffle(current_train)
        return current_train

    def convert_weak_ids(self, current_train, orig_mapping, weak_ids_tmp):
        weak_idx_current_train = set()
        for new_idx, pair in enumerate(current_train):
            if orig_mapping[pair] in weak_ids_tmp:
                weak_idx_current_train.add(new_idx)
        output = open(self.files_path + 'weak_ids_current_train.pkl', 'wb')
        pickle.dump(weak_idx_current_train, output)
        output.close()
        return

    @staticmethod
    def create_orig_mapping(current_train):
        orig_mapping = dict()
        for idx, pair in enumerate(current_train):
            orig_mapping[pair] = idx
        return orig_mapping

    def find_weak_ids(self):
        pkl_file = open(self.files_path + 'weak_ids.pkl', 'rb')
        weak_ids = pickle.load(pkl_file)
        pkl_file.close()
        return weak_ids

    def modify_pair(self, idx, pair):
        if self.mode == "top_k_DTAL":
            if idx in self.high_confidence_positive:
                pair = pair[:-2] + "1" + pair[-1:]
            elif idx in self.high_confidence_negative:
                pair = pair[:-2] + "0" + pair[-1:]
        return pair

    def read_only_selected(self):
        return [pair for idx, pair in enumerate(self.original_input)
                if idx in self.current_train_ids]

    def get_pairs_ids(self):
        pairs_ids_dict, ids_pairs_dict = dict(), dict()
        for idx, pair in enumerate(self.original_input):
            pairs_ids_dict[pair] = idx
            ids_pairs_dict[idx] = pair
        return pairs_ids_dict, ids_pairs_dict

    def top_k_DTAL(self):
        poolers_path = self.define_poolers_path()
        poolers_path_available_pool = poolers_path.replace("data", "output")
        DTAL_obj = DTAL(poolers_path_available_pool, self.k)
        return DTAL_obj.likely_false, DTAL_obj.high_confidence_positive, DTAL_obj.high_confidence_negative

    def find_top_k(self):
        ws_pos_cands, ws_neg_cands = set(), set()
        poolers_path = self.define_poolers_path()
        # Path to available pool (including last iteration predictions)
        poolers_path_available_pool = poolers_path.replace("data", "output")
        # Path to current train (including source), (including last iteration predictions)
        poolers_path_current_train = poolers_path_available_pool.replace("available_pool", "current_train")
        alpha = float(self.mode.split("alpha=")[1].split("_")[0])
        beta = float(self.mode.split("beta=")[1].split("_")[0])
        graph_obj = battleships_graph([poolers_path_available_pool, poolers_path_current_train],
                                      self.k, self.seed, self.files_path, self.output_path, self.iter,
                                      self.criterion, self.mode, alpha, beta)
        selected_k = graph_obj.get_selected_k
        if self.weak_supervision:
            ws_pos_cands, ws_neg_cands = graph_obj.get_weakly_supervised
        return selected_k, ws_pos_cands, ws_neg_cands

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

    def save_to_pkls(self, files_list, file_names_list):
        path = self.output_path + self.mode + "/pkl_files/"
        if not os.path.exists(path):
            os.makedirs(path)
        for file, file_name in zip(files_list, file_names_list):
            output = open(path + file_name + '_iter' + str(self.iter) +
                          '_seed' + str(self.seed) + '.pkl', 'wb')
            pickle.dump(file, output)
            output.close()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Walmart-Amazon")
    parser.add_argument("--source_task", type=str, default="Structured/Walmart-Amazon")
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--k_size", type=int, default=100)
    parser.add_argument("--iter_num", type=int, default=1)
    parser.add_argument("--mode", type=str, default="battleships_ws_b_alpha=0.0")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--criterion", type=str, default="pagerank")
    parser.add_argument("--output_path", type=str, default="output/er_magellan/Structured/Walmart-Amazon/Walmart-Amazon/")
    parser.add_argument("--from_iter", type=int, default=0)
    start = time.time()
    hp = parser.parse_args()


    task = hp.task
    source_task = hp.source_task
    intent = hp.intent
    k_size = hp.k_size
    iter_num = hp.iter_num
    selection_mode = hp.mode
    seed = hp.seed
    from_iter = hp.from_iter

    iterations = hp.iterations
    criterion = hp.criterion
    output_path = hp.output_path

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    path = configs[task + str(intent)]['path']
    orig_train = configs[task + str(intent)]['trainset']
    source_task += str(intent)
    if "dummy" not in selection_mode:
        top_k_manager = TopKSelection(task, source_task, k_size, iter_num, selection_mode,
                                      path, orig_train, seed, iterations, output_path,
                                      from_iter, criterion)
    else:
        "dummy mode"
        print()
    end = time.time()

    print(f'The process took :{round(end - start, 2)} seconds')
