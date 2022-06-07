import numpy as np
import networkx as nx
from itertools import repeat
from itertools import combinations
from collections import Counter
import torch
import re
import json
from math import log2
from scipy import spatial
import random
import os
import pickle
from scipy import spatial
import multiprocessing
from collections import defaultdict
import time
import faiss


class battleships_graph:
    def __init__(self, poolers_paths, k, seed, files_path, output_path, iteration, criterion='pagerank',
                 mode='top_k', weights_type='with_threshold',
                 lsh_iterations=10, dim=768,
                 min_cc_ratio=0.03, max_cc_ratio=0.15,
                 nearest_param=25, treat_weak_labels=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.poolers_paths = poolers_paths
        self.files_path = files_path
        self.output_path = output_path
        self.iter = iteration
        self.k = k
        self.weights_type = weights_type
        self.dim = dim
        self.lsh_iterations = lsh_iterations
        self.criterion = criterion
        self.mode = mode
        self.nearest_param = nearest_param
        self.treat_weak_labels = treat_weak_labels
        self.poolers, self.available_pool_size = self.create_poolers()
        self.weak_ids = self.find_weaks()
        self.selection_param = float(mode.split("=")[1])
        self.training_labels = self.create_labels()
        self.pool_predictions, self.confidence_dict = self.create_predictions()
        self.weak_labels_confidence = self.create_weak_labels_confidence()
        self.pos_labels_ids = {pooler_id for pooler_id in self.training_labels.keys()
                               if self.training_labels[pooler_id]}
        self.neg_labels_ids = {pooler_id for pooler_id in self.training_labels.keys()
                               if not self.training_labels[pooler_id]}
        self.pos_preds_ids = {pooler_id for pooler_id in self.pool_predictions.keys()
                              if self.pool_predictions[pooler_id]}
        self.neg_preds_ids = {pooler_id for pooler_id in self.pool_predictions.keys()
                              if not self.pool_predictions[pooler_id]}
        self.pos_budget = min(max(round(self.k * (0.85 - 0.05 * iteration)), round(0.5 * self.k)), len(self.pos_preds_ids))
        self.min_cc_ratio = min_cc_ratio
        self.max_cc_ratio = max_cc_ratio
        # self.min_val = int(min_cc_ratio * len(self.pos_preds_ids))  # The actual size of the smallest possible CC
        # self.max_val = int(max_cc_ratio * len(self.pos_preds_ids))  # The actual size of the largest possible CC
        # self.graph, self.connected_components, self.ccs_available_pool_sizes = self.from_lsh2graph(self.poolers.keys())
        self.pos_graph, self.pos_connected_components, self.pos_ccs_available_pool_sizes = self.from_lsh2graph_type(1)
        self.neg_graph, self.neg_connected_components, self.neg_ccs_available_pool_sizes = self.from_lsh2graph_type(0)
        self.het_graph, self.het_connected_components, self.het_ccs_available_pool_sizes = self.from_lsh2graph_type(2)
        self.validate_connected_components()
        # self.pos_old_connected_components = self.clean_old_train_pos()
        # self.neg_old_connected_components = self.clean_old_train_neg()
        # self.positive_graph_ids, self.positive_CCs_size = self.calc_CCs_type_size(1)
        # self.negative_graph_ids, self.negative_CCs_size = self.calc_CCs_type_size(0)
        # self.positive_budget_dict, self.negative_budget_dict = self.distribute_budget()
        self.positive_budget_dict = self.distribute_budget(1)
        self.negative_budget_dict = self.distribute_budget(0)
        self.selected_k, self.pos_uncertainty, self.neg_uncertainty, self.votes_dict = self.calc_criterion()
        self.ws_pos_cands, self.ws_neg_cands = self.find_weakly_supervised()
        # self.selected_k = self.calc_criterion_pos()

    def create_poolers(self):
        """
        Create a mapping from row indices to CLS vectors, starting from the available pool, followed by the current
        train. Return the mapping and the size of the dictionary prefix containing the available pool.
        """
        poolers_dict = self.create_poolers_available_pool()
        available_pool_size = len(poolers_dict)
        poolers_dict = self.create_poolers_current_train(poolers_dict, available_pool_size)
        self.save_to_pkl([poolers_dict], ["poolers"])
        return poolers_dict, available_pool_size

    def create_poolers_available_pool(self):
        """
        Load the available pool CLS vectors and save a mapping from row indices (in the pooler file) to the CLSs
        """

        """
        pooler_path = self.poolers_paths[0].replace(".txt", "_poolers.pkl")
        pkl_file = open(pooler_path, 'rb')
        poolers_dict = pickle.load(pkl_file)
        pkl_file.close()
        """
        pooler_path = self.poolers_paths[0]
        poolers_dict = dict()
        # self.poolers_paths[0] is the poolers file of the available pool
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
            poolers_dict[id_val] = np.array(list(map(float, pooler.split(' '))))
        preds_file.close()

        return poolers_dict

    def find_weaks(self):
        pkl_file = open(self.files_path + 'weak_ids_current_train.pkl', 'rb')
        weak_ids = pickle.load(pkl_file)
        pkl_file.close()
        weak_ids = {weak_id + self.available_pool_size for weak_id in weak_ids}
        return weak_ids

    def create_poolers_current_train(self, poolers_dict, available_pool_size):
        """
        Load the current train CLS vectors and insert them to the end of poolers dict.
        """
        # self.poolers_paths[1] is the poolers file of the current train
        pooler_path = self.poolers_paths[1]
        """
        pooler_path = self.poolers_paths[1].replace(".txt", "_poolers.pkl")
        pkl_file = open(pooler_path, 'rb')
        tmp_poolers_dict = pickle.load(pkl_file)
        pkl_file.close()
        poolers_dict.update({id_val + available_pool_size: pooler
                            for id_val, pooler in tmp_poolers_dict.items()})
        """
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
            poolers_dict[id_val + available_pool_size] = np.array(list(map(float, pooler.split(' '))))
        preds_file.close()

        return poolers_dict

    def create_labels(self):
        """
        Create a mapping from row indices to label, staring from the available pool labeled 2 for unknown label,
        followed by the current train with the given labels.
        """
        # for each pair in the available we assign the label 2 (unknown)
        labels_dict = {id_val: 2 for id_val in range(self.available_pool_size)}
        labels_file = open(self.files_path + 'current_train.txt', "r", encoding="utf-8")
        lines_labels = labels_file.readlines()
        labels_file.close()
        for id_val, line in enumerate(lines_labels):
            labels_dict[id_val + self.available_pool_size] = int(re.sub("[^0-9]", "", line[-2]))
        return labels_dict

    def create_predictions(self):
        """
        Create a mapping from row indices to PREDICTED label, for the available pool.
        """
        """
        dict_list = []
        file_suffux_list = ["preds", "conf"]
        for curr_suffix in file_suffux_list:
            curr_path = self.poolers_paths[0].replace(".txt", "_" + curr_suffix + ".pkl")
            pkl_file = open(curr_path, 'rb')
            curr_dict = pickle.load(pkl_file)
            pkl_file.close()
            dict_list.append(curr_dict)

        """
        preditions_dict, confidence_dict = dict(), dict()
        pooler_path = self.poolers_paths[0]
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            preditions_dict[id_val] = int(re.sub("[^0-9]", "", line.split("\"match\"")[1][3]))
            confidence_dict[id_val] = float(re.sub("[^0-9.]", "", line.split("match_confidence")[1].split("pooler")[0]))
            if preditions_dict[id_val] != 0 and preditions_dict[id_val] != 1:
                pass
        preds_file.close()

        return preditions_dict, confidence_dict

    def create_weak_labels_confidence(self):
        """
        Create a mapping from row indices to PREDICTED label, for the available pools.
        """
        confidence_dict = dict()
        indent = self.available_pool_size
        pooler_path = self.poolers_paths[1]
        labels_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = labels_file.readlines()
        for id_val, line in enumerate(lines_preds):
            confidence_dict[id_val + indent] = float(re.sub("[^0-9.]", "",
                                                            line.split("match_confidence")[1].split("pooler")[0]))
        labels_file.close()
        return confidence_dict


    # def create_weak_labels_confidence(self):
    #
    #     # Create a mapping from row indices to PREDICTED label, for the available pools.
    #     indent = self.available_pool_size
    #     # conf_path = self.poolers_paths[1].replace(".txt", "_conf.pkl")
    #     conf_path = self.poolers_paths[1]
    #     # pkl_file = open(conf_path, 'rb')
    #     # tmp_conf_dict = pickle.load(pkl_file)
    #     pkl_file.close()
    #     # confidence_dict = {id_val + indent: pooler for id_val, pooler in tmp_conf_dict.items()}
    #
    #     """
    #     labels_file = open(pooler_path, "r", encoding="utf-8")
    #     lines_preds = labels_file.readlines()
    #     for id_val, line in enumerate(lines_preds):
    #         confidence_dict[id_val + indent] = float(re.sub("[^0-9.]", "",
    #                                                         line.split("match_confidence")[1].split("pooler")[0]))
    #     labels_file.close()
    #     """
    #     return confidence_dict

    # def from_lsh2graph(self, poolers_ids):
    #     buckets2poolers = self.create_buckets(poolers_ids)
    #     final_buckets2poolers, bucket_parents = self.iteative_bucketing(buckets2poolers)
    #     graph = self.initialize_graph(poolers_ids)
    #     # graph = self.create_graph_edges(graph, lsh_iterations, final_buckets2poolers,
    #     #                                 edges_threshold, sim_threshold)
    #     graph = self.graph_with_threshold(graph, final_buckets2poolers, self.sim_threshold)
    #     connected_components = self.create_connected_components(graph)
    #     light_conncted_components = self.get_light_connected_components(connected_components)
    #     ccs_available_pool_sizes = self.calc_CCS_available_pool_sizes(connected_components)
    #     self.save_to_pkl([final_buckets2poolers, light_conncted_components,
    #                       ccs_available_pool_sizes, buckets2poolers, bucket_parents],
    #                      ["final_buckets2poolers", "orig_connected_components(light)",
    #                       "ccs_available_pool_sizes", "orig_buckets2poolers", "bucket_parents"])
    #     return graph, connected_components, ccs_available_pool_sizes

    def find_rel_ids_min_max(self, label_type):
        if label_type == 2:
            rel_ids = {pooler_id for pooler_id in self.poolers.keys()}
            min_val = int(0.25 * self.min_cc_ratio * len(self.poolers))
            max_val = int(0.25 * self.max_cc_ratio * len(self.poolers))
        else:
            rel_ids = self.pos_preds_ids if label_type == 1 else self.neg_preds_ids
            min_val = int(self.min_cc_ratio * len(self.pos_preds_ids)) if label_type == 1 \
                else int(self.min_cc_ratio * len(self.neg_preds_ids))
            max_val = int(self.max_cc_ratio * len(self.pos_preds_ids)) if label_type == 1 \
                else int(self.max_cc_ratio * len(self.neg_preds_ids))
        return rel_ids, min_val, max_val

    def from_lsh2graph_type(self, label_type):
        rel_ids, min_val, max_val = self.find_rel_ids_min_max(label_type)
        suffix = str(label_type)
        buckets2poolers = self.create_buckets(rel_ids, min_val)
        final_buckets2poolers, bucket_parents = self.iteative_bucketing(buckets2poolers, min_val, max_val)
        graph = self.initialize_graph(rel_ids)
        graph = self.connect_nodes(graph, final_buckets2poolers, label_type)
        connected_components = self.create_connected_components(graph)
        light_conncted_components = self.get_light_connected_components(connected_components)
        ccs_available_pool_sizes = self.calc_CCS_available_pool_sizes(connected_components)
        self.save_to_pkl([final_buckets2poolers, light_conncted_components,
                          ccs_available_pool_sizes, buckets2poolers, bucket_parents],
                         ["final_buckets2poolers" + suffix, "orig_connected_components(light)" + suffix,
                          "ccs_available_pool_sizes" + suffix, "orig_buckets2poolers" + suffix,
                          "bucket_parents" + suffix])
        return graph, connected_components, ccs_available_pool_sizes

    # def validate_lsh(self, ccs_available_pool_sizes):
    #     legits_counter = 0
    #     for cc_size in ccs_available_pool_sizes.values():
    #         if self.min_val < cc_size < self.max_val:
    #             legits_counter += 1
    #     if legits_counter >= 1:
    #         return True
    #     else:
    #         return False

    # def create_buckets(self, lsh_iterations, rel_poolers):
    #     vectors_num = log2(len(rel_poolers))
    #     buckets2poolers_dict = dict()
    #     poolers2buckets_dict = {pooler_id: [] for pooler_id in rel_poolers.nodes()}
    #     for iteration in range(lsh_iterations):
    #         buckets2poolers_dict[iteration] = dict()
    #         random_vecs = np.random.randn(vectors_num, self.dim)
    #         for pooler_id, pooler_vec in poolers.items():
    #             bucket_id = self.classify_pooler(pooler_vec, random_vecs)
    #             if bucket_id not in buckets2poolers_dict[iteration].keys():
    #                 buckets2poolers_dict[iteration][bucket_id] = [pooler_id]
    #             else:
    #                 buckets2poolers_dict[iteration][bucket_id].append(pooler_id)
    #             poolers2buckets_dict[pooler_id].append(bucket_id)
    #     return buckets2poolers_dict, poolers2buckets_dict

    def create_buckets(self, rel_poolers_ids, min_val):
        """
        Perform LSH iteration: generate vector_num random hyperplanes and classify each pooler to a bucket
        according to its spatial representation, with respect to the intersection of the vectors.
        Return a dictionary maps from buckets_ids to poolers_indices.
        """
        if len(rel_poolers_ids) == 0:
            return dict()
        rel_poolers = {pooler_id: pooler for pooler_id, pooler in
                       self.poolers.items() if pooler_id in rel_poolers_ids}
        vectors_num = max(int(log2((len(rel_poolers_ids)) / max(min_val, 1))), 1)
        buckets2poolers_dict = defaultdict(list)
        random_vecs = np.random.randn(vectors_num, self.dim)
        for pooler_id, pooler_vec in rel_poolers.items():
            bucket_id = self.classify_pooler(pooler_vec, random_vecs)
            buckets2poolers_dict[bucket_id].append(pooler_id)
        return buckets2poolers_dict

    # @staticmethod
    # def clean_buckets2poolers(buckets2poolers, final_buckets2poolers):
    #     for bucket_id in buckets2poolers.keys():
    #         if bucket_id in final_buckets2poolers.keys():
    #             buckets2poolers.pop(bucket_id)
    #     return buckets2poolers

    def iteative_bucketing(self, buckets2poolers, min_val, max_val):
        lsh_iter = 0
        final_buckets2poolers = dict()
        bucket_parents = dict()
        while lsh_iter < self.lsh_iterations and len(buckets2poolers) > 0:
            buckets2poolers, final_buckets2poolers, bucket_parents = self.handle_legit_buckets(buckets2poolers,
                                                                                               final_buckets2poolers,
                                                                                               bucket_parents,
                                                                                               str(lsh_iter), min_val,
                                                                                               max_val)
            buckets2poolers, final_buckets2poolers, bucket_parents = self.handle_large_bucket(buckets2poolers,
                                                                                              final_buckets2poolers,
                                                                                              bucket_parents,
                                                                                              str(lsh_iter), min_val,
                                                                                              max_val)
            rel_poolers_ids = self.find_rel_poolers(buckets2poolers)
            if len(rel_poolers_ids):
                lsh_iter += 1
                buckets2poolers = self.create_buckets(rel_poolers_ids, min_val)
            else:
                break
        final_buckets2poolers, bucket_parents = self.merge_buckets2poolers(final_buckets2poolers,
                                                                           buckets2poolers,
                                                                           bucket_parents,
                                                                           str(lsh_iter))
        return final_buckets2poolers, bucket_parents

    @ staticmethod
    def clean_buckets2poolers(buckets2poolers):
        if len(buckets2poolers) > 0:
            merged_bucket = []
            buckets2poolers_copy = buckets2poolers.copy()
            for bucket_id, bucket in buckets2poolers_copy.items():
                merged_bucket.extend(bucket)
                buckets2poolers.pop(bucket_id)
            buckets2poolers['final_lsh_iter'] = merged_bucket
        return buckets2poolers

    def merge_buckets2poolers(self, final_buckets2poolers, buckets2poolers, bucket_parents, lsh_iter):
        new_bucket_id = len(final_buckets2poolers)
        buckets2poolers = self.clean_buckets2poolers(buckets2poolers)
        for bucket_id, bucket in buckets2poolers.items():
            final_buckets2poolers['lshIter' + lsh_iter + '_' + str(new_bucket_id)] = bucket
            bucket_parents['lshIter' + lsh_iter + '_' + str(new_bucket_id)] = bucket_id
            new_bucket_id += 1
        return final_buckets2poolers, bucket_parents

    @staticmethod
    def find_rel_poolers(buckets2poolers):
        rel_poolers_ids = set()
        for bucket in buckets2poolers.values():
            rel_poolers_ids.update(bucket)
        return rel_poolers_ids

    def handle_legit_buckets(self, buckets2poolers, final_buckets2poolers, bucket_parents, lsh_iter, min_val, max_val):
        buckets2poolers_copy = buckets2poolers.copy()
        new_bucket_id = len(final_buckets2poolers)
        for bucket_id, bucket in buckets2poolers_copy.items():
            if min_val < len(bucket) < max_val:
                final_buckets2poolers['lshIter' + lsh_iter + '_' + str(new_bucket_id)] = bucket
                bucket_parents['lshIter' + lsh_iter + '_' + str(new_bucket_id)] = bucket_id
                buckets2poolers.pop(bucket_id)
                new_bucket_id += 1
        return buckets2poolers, final_buckets2poolers, bucket_parents

    def handle_large_bucket(self, buckets2poolers, final_buckets2poolers, bucket_parents, lsh_iter, min_val, max_val):
        buckets2poolers_copy = buckets2poolers.copy()
        new_bucket_id = len(final_buckets2poolers)
        internal_bucket_id = 0
        for bucket_id, bucket in buckets2poolers_copy.items():
            if len(bucket) >= max_val:
                new_buckets = self.create_buckets(bucket, min_val)
                buckets2poolers.pop(bucket_id)
                for bucket_id2, bucket2 in new_buckets.items():
                    if min_val < len(bucket) < max_val:
                        final_buckets2poolers['lshIter' + lsh_iter + '_' + str(new_bucket_id)] = bucket2
                        bucket_parents['lshIter' + lsh_iter + '_' + str(new_bucket_id)] = bucket_id
                        new_bucket_id += 1
                    else:
                        buckets2poolers[str(internal_bucket_id) + '_' + bucket_id2] = bucket2
                        internal_bucket_id += 1
        return buckets2poolers, final_buckets2poolers, bucket_parents

    @staticmethod
    def create_connected_components(graph):
        """
        Generate indexed connected components
        """
        graphs_dict = dict()
        connected_components = nx.connected_components(graph)
        for graph_id, cc in enumerate(connected_components):
            graphs_dict[graph_id] = graph.subgraph(cc)
        return graphs_dict

    def validate_connected_components(self):
        # self.fix_large_connected_components()
        # self.fix_small_connected_components(0)
        # self.fix_small_connected_components(1)
        # self.fix_small_connected_components_het()
        light_conncted_components_pos = self.get_light_connected_components(self.pos_connected_components)
        light_conncted_components_neg = self.get_light_connected_components(self.neg_connected_components)
        self.save_to_pkl([light_conncted_components_pos, light_conncted_components_neg,
                          self.pos_connected_components, self.neg_connected_components],
                         ["final_connected_components(light1)", "final_connected_components(light0)",
                          "final_connected_components_pos", "final_connected_components_neg"])
        return

    def get_light_connected_components(self, connected_components):
        """
        Generate a dictionary with connected components ids as keys and corresponding poolers vectors as values
        """
        light_dict = dict()
        for graph_id, graph in connected_components.items():
            light_dict[graph_id] = [self.poolers[pooler_id] for pooler_id in graph.nodes()]
        return light_dict

    def calc_CCS_available_pool_sizes(self, connected_components):
        """
        For each CC - calc the number of nodes originally from available_pool
        """
        ccs_available_pool_sizes = dict()
        for graph_id, graph in connected_components.items():
            ccs_available_pool_sizes[graph_id] = len([pooler_id for pooler_id in graph.nodes() if
                                                      pooler_id < self.available_pool_size])
        return ccs_available_pool_sizes

    # def fix_large_connected_components(self):
    #     ccs_copy = self.connected_components.copy()
    #     ccs_available_pool_sizes_copy = self.ccs_available_pool_sizes.copy()
    #     flag = 0
    #     for graph_id, graph in ccs_copy.items():
    #         if ccs_available_pool_sizes_copy[graph_id] > self.max_val:
    #             new_connected_components = self.decompose_graph(graph)
    #             self.connected_components.pop(graph_id)
    #             self.ccs_available_pool_sizes.pop(graph_id)
    #             self.update_connected_components(new_connected_components)
    #             flag = 1
    #     if flag == 1:
    #         self.fix_large_connected_components()
    #     return

    # def decompose_graph(self, graph):
    #     # vectors_num = int(log2(graph.number_of_nodes() / self.min_cc_param)) + 1
    #     poolers = {pooler_id: self.poolers[pooler_id] for pooler_id in graph.nodes()}
    #     new_graph, connected_components, ccs_available_pool_sizes = self.from_lsh2graph(graph.nodes())
    #
    #     buckets2poolers_dict, poolers2buckets_dict = self.create_buckets(graph.nodes(), lsh_iterations,
    #                                                                      vectors_num, poolers)
    #     new_graph = self.initialize_graph(poolers.keys())
    #     new_graph = self.create_graph_edges(new_graph, lsh_iterations, buckets2poolers_dict,
    #                                         edges_threshold, self.adapted_sim_threshold)
    #     connected_components = self.create_connected_components(new_graph)
    #     return connected_components
    #
    # def update_connected_components(self, new_connected_components):
    #     current_size = len(self.connected_components)
    #     for graph_id, graph in new_connected_components.items():
    #         self.connected_components[graph_id + current_size] = graph
    #         self.ccs_available_pool_sizes[graph_id + current_size] = len([pooler_id for pooler_id in
    #                                                                       graph.nodes() if pooler_id <
    #                                                                       self.available_pool_size])
    #     return

    def fix_small_connected_components(self, label_type):
        min_val = int(self.min_cc_ratio * len(self.pos_preds_ids)) if label_type == 1 \
            else int(self.min_cc_ratio * len(self.neg_preds_ids))
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 \
            else self.neg_connected_components.copy()
        ccs_available_pool_copy = self.pos_ccs_available_pool_sizes.copy() if label_type == 1 \
            else self.neg_ccs_available_pool_sizes.copy()
        ccs_centroids = self.calc_centroids(ccs_copy)
        small_connected_components = {graph_id: graph for graph_id, graph in ccs_copy.items()
                                      if ccs_available_pool_copy[graph_id] < min_val}
        legit_connected_components = {graph_id for graph_id in ccs_copy.keys()
                                      if graph_id not in small_connected_components.keys()}
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            closest_graphs = pool.starmap(self.find_closest_cc, zip(small_connected_components.keys(),
                                                                    repeat(ccs_centroids),
                                                                    repeat(legit_connected_components)))
        closest_graphs_dict = {graph_id: closest_graph for graph_id, closest_graph in closest_graphs}
        for graph_id, graph in small_connected_components.items():
            closest_graph_id = closest_graphs_dict[graph_id]
            ccs_copy = self.connect_small_to_legit(graph, closest_graph_id, ccs_copy)
            ccs_copy.pop(graph_id)
        self.update_rel_CCs(ccs_copy, label_type)
        return

    def fix_small_connected_components_het(self):
        min_val = int(0.25 * self.min_cc_ratio * len(self.poolers))
        ccs_copy = self.het_connected_components.copy()
        ccs_centroids = self.calc_centroids(ccs_copy)
        small_connected_components = {graph_id: graph for graph_id, graph in ccs_copy.items()
                                      if len(graph) < min_val}
        legit_connected_components = {graph_id for graph_id in ccs_copy.keys()
                                      if graph_id not in small_connected_components.keys()}
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            closest_graphs = pool.starmap(self.find_closest_cc, zip(small_connected_components.keys(),
                                                                    repeat(ccs_centroids),
                                                                    repeat(legit_connected_components)))
        closest_graphs_dict = {graph_id: closest_graph for graph_id, closest_graph in closest_graphs}
        for graph_id, graph in small_connected_components.items():
            closest_graph_id = closest_graphs_dict[graph_id]
            ccs_copy = self.connect_small_to_legit(graph, closest_graph_id, ccs_copy)
            ccs_copy.pop(graph_id)
        self.update_rel_CCs(ccs_copy, 2)
        return

    def calc_centroids(self, ccs_copy):
        ccs_centroids = dict.fromkeys(ccs_copy.keys())
        for graph_id, graph in ccs_copy.items():
            poolers_list = [self.poolers[pooler_id] for pooler_id in graph.nodes()]
            ccs_centroids[graph_id] = np.array(list(np.mean(poolers_list, axis=0)))
        return ccs_centroids

    def update_rel_CCs(self, rel_CCs, label_type):
        if label_type == 1:
            self.pos_connected_components = rel_CCs
        elif label_type == 0:
            self.neg_connected_components = rel_CCs
        else:
            # label_type = 2
            self.het_connected_components = rel_CCs
        return

    @staticmethod
    def find_closest_cc(graph_id, ccs_centroids, legit_connected_components):
        current_centroid = ccs_centroids[graph_id]
        distances = [(cc_id, round(spatial.distance.cosine(current_centroid, ccs_centroids[cc_id]), 3))
                     for cc_id in legit_connected_components]
        return graph_id, min(distances, key=lambda x: x[1])[0]

    def connect_small_to_legit(self, graph, closest_graph_id, rel_CCs):
        # for each node from the small graph connect it with an edge to a
        # random node from the closest legit graph
        closest_graph = rel_CCs[closest_graph_id].copy()
        np.random.seed(seed=self.seed)
        selected_nodes = np.random.choice(closest_graph.nodes(), graph.number_of_nodes())
        pairs = set(zip(graph.nodes(), selected_nodes))
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            pairs_weight = list(pool.map(self.calc_pair_weight, pairs))
        new_pairs_weight = [(pair[0], pair[1], weight) for pair, weight in pairs_weight]
        closest_graph.add_weighted_edges_from(new_pairs_weight)
        rel_CCs[closest_graph_id] = closest_graph
        return rel_CCs

    def assign_cc_labels(self, pos_threshold):
        """
        Assign a label for each CC. positive label for containing sufficient ratio of positive samples,
        negative otherwise.
        """
        connected_components_labels = dict()
        # labels_weight = 1 - 0.05 * self.iter
        labels_weight = 0.5
        for graph_id, graph in self.connected_components.items():
            cc_labels = {node_id: self.training_labels[node_id] for node_id in graph.nodes()}
            cc_predictions = {node_id: self.pool_predictions[node_id] for node_id in graph.nodes()
                              if node_id in self.pool_predictions.keys()}
            labels_counter = Counter(cc_labels.values())
            predictions_counter = Counter(cc_predictions.values())
            expectation = labels_weight * labels_counter[1] + (1 - labels_weight) * predictions_counter[1]
            try:
                if expectation >= pos_threshold * graph.number_of_nodes():
                    connected_components_labels[graph_id] = 1
                else:
                    connected_components_labels[graph_id] = 0
            except:
                connected_components_labels[graph_id] = 0
        ccs_labels_counter = Counter(connected_components_labels.values())
        if ccs_labels_counter[1] < 2:
            # If there are less than two potentially positive connected components
            # we call this function again with a more tolerant threshold
            connected_components_labels = self.assign_cc_labels(0.9 * pos_threshold)
        self.save_to_pkl([connected_components_labels], ["connected_components_labels"])
        return connected_components_labels

    def calc_CCs_type_size(self, cc_label):
        """
        Return a list with the CC_ids with cc_label.
        In addition, return the sum of nodes in those CC.
        """
        relevant_graph_ids = [graph_id for graph_id in self.connected_components.keys()
                              if self.cc_labels[graph_id] == cc_label]
        relevant_size = sum([len(self.connected_components[graph_id]) for graph_id in
                             self.connected_components.keys() if graph_id in relevant_graph_ids])
        return relevant_graph_ids, relevant_size

    # def distribute_budget(self):
    #     positive_budget_dict = self.create_budget_dict(1, self.positive_graph_ids, self.positive_CCs_size)
    #     negative_budget_dict = self.create_budget_dict(0, self.negative_graph_ids, self.negative_CCs_size)
    #     return positive_budget_dict, negative_budget_dict

    def distribute_budget(self, label_type, ws_candidates=None, ws=False):
        cc_copy = self.pos_connected_components if label_type == 1 else self.neg_connected_components
        if not ws:
            rel_budget = self.pos_budget if label_type == 1 else self.k - self.pos_budget
        else:
            rel_budget = min(int(self.k / 2), len(self.pos_preds_ids.intersection(ws_candidates)))
        total_elements = sum([len(cc) for cc in cc_copy.values()])
        budget_dict = dict()
        total_used = 0
        for graph_id, graph_elements in cc_copy.items():
            relative_share = len(graph_elements) / total_elements
            budget = int(relative_share * rel_budget)
            budget_dict[graph_id] = budget
            total_used += budget
        if total_used < rel_budget:
            budget_dict = self.assign_residue(budget_dict, rel_budget - total_used,
                                              list(cc_copy.keys()))
        return budget_dict

    # def create_budget_dict(self, label, rel_graph_ids, rel_ccs_size):
    #     """
    #     Split the label-relative budget to the corresponded-labeled CCs with respect to the relative size.
    #     """
    #     budget_dict = dict()
    #     total_used = 0
    #     rel_share = self.pos_budget if label == 1 else 1 - self.pos_budget
    #     total_budget = round(self.k * rel_share)
    #     for graph_id, graph_label in self.cc_labels.items():
    #         if graph_id in rel_graph_ids:
    #             relative_share = len(self.connected_components[graph_id]) / rel_ccs_size
    #             budget = int(relative_share * total_budget)
    #             budget_dict[graph_id] = budget
    #             total_used += budget
    #     if total_used < total_budget:
    #         budget_dict = self.assign_residue(budget_dict, total_budget - total_used, rel_graph_ids)
    #     return budget_dict

    @staticmethod
    def assign_residue(budget_dict, residue, rel_graph_ids):
        """
        Given the budget dict and a budget residue, split the residue randomly between the CCs
        """
        chosen_graph_ids = random.choices(rel_graph_ids, k=residue)
        for graph_id in chosen_graph_ids:
            budget_dict[graph_id] += 1
        return budget_dict

    @staticmethod
    def classify_pooler(pooler, random_vecs):
        r"""
        Given a pooler vector, find its bucket ID.
        This is done by multiplying the pooler with each one of the random vectors
        and assigning the result to a single bit (1 if the result is positive,
        0 otherwise). In the end we get a *vectors_num* length binary string which
        represents the bucket ID of the pooler.
        """
        bucket_id = ''
        for rand_vec in random_vecs:
            bucket_id += '1' if rand_vec.dot(pooler) > 0 else '0'
        return bucket_id

    # def create_graph_funcs_dict(self):
    #     return {'without': self.graph_without,
    #             'with_threshold': self.graph_with_threshold,
    #             # 'relative_adjs': self.graph_relative_adjs,
    #             # 'cos_sim': self.graph_cos_sim(),
    #             # 'relative_adjs_cos_sim': self.graph_relative_adjs_cos_sim()
    #             }

    @staticmethod
    def initialize_graph(poolers_ids):
        """
        Initial a networkx graph
        """
        graph = nx.Graph()
        graph.add_nodes_from(poolers_ids)
        return graph

    # def create_graph_edges(self, graph, lsh_iterations, buckets2poolers, edges_threshold, sim_threshold):
    #     funcs_dict = self.create_graph_funcs_dict()
    #     return funcs_dict[self.weights_type](graph, lsh_iterations, buckets2poolers,
    #                                          edges_threshold, sim_threshold)
    #
    # def graph_without(self, graph, lsh_iterations, buckets2poolers, edges_threshold=None):
    #     for iteration in range(lsh_iterations):
    #         for bucket_id in buckets2poolers[iteration].keys():
    #             current_bucket = buckets2poolers[iteration][bucket_id]
    #             graph.add_edges_from(list(combinations(current_bucket, 2)))
    #     return graph

    def add_automatic_edges(self, pooler_id, edges_set, automatic_edges_num, neighbors, bucket2orig):
        added = 0
        for neighbor in neighbors[0][1:]:
            if added >= automatic_edges_num:
                break
            if (neighbor, pooler_id) not in edges_set \
                    and min(bucket2orig[neighbor], bucket2orig[pooler_id]) < self.available_pool_size:
                edges_set.add((bucket2orig[pooler_id], bucket2orig[neighbor]))
                added += 1
        return edges_set

    def update_candidate_neighbors_dict(self, pooler_id, edges_set, automatic_edges_num,
                                        neighbors, dists, candidate_neighbors_dict, bucket2orig):
        for neighbor, dist in zip(neighbors[0][automatic_edges_num + 1:], dists[0][automatic_edges_num + 1:]):
            if (pooler_id, neighbor) not in edges_set and (neighbor, pooler_id) not in edges_set:
                if (neighbor, pooler_id) not in candidate_neighbors_dict.keys() \
                        and min(bucket2orig[neighbor], bucket2orig[pooler_id]) < self.available_pool_size:
                    candidate_neighbors_dict[(bucket2orig[pooler_id], bucket2orig[neighbor])] = dist
        return candidate_neighbors_dict

    def update_edges_and_candidates(self, pooler_vec, index, candidate_neighbors_size, pooler_id,
                                    edges_set, automatic_edges_num, candidate_neighbors_dict, bucket2orig):
        query_pooler = np.expand_dims(pooler_vec, axis=0)
        dists, neighbors = index.search(query_pooler, candidate_neighbors_size)
        edges_set = self.add_automatic_edges(pooler_id, edges_set, automatic_edges_num, neighbors, bucket2orig)
        candidate_neighbors_dict = self.update_candidate_neighbors_dict(pooler_id, edges_set,
                                                                        automatic_edges_num, neighbors,
                                                                        dists, candidate_neighbors_dict,
                                                                        bucket2orig)
        return edges_set, candidate_neighbors_dict

    def process_candidates(self, candidate_neighbors_dict, edges_ratio, edges_set):
        edges_limit = int(edges_ratio * len(candidate_neighbors_dict))
        for counter, pair in enumerate(candidate_neighbors_dict.keys()):
            if counter > edges_limit:
                break
            edges_set.add((pair[0], pair[1]))
        return edges_set

    def create_bucket_edges(self, bucket_ids, label_type, automatic_edges_num=5, edges_ratio=0.05):
        automatic_edges_num = automatic_edges_num if label_type < 2 else 3 * automatic_edges_num
        rel_poolers = np.array([self.poolers[pooler_id] for pooler_id in bucket_ids], dtype="float32")
        bucket2orig = {idx: pooler_id for idx, pooler_id in enumerate(bucket_ids)}
        d = len(self.poolers[0])
        index = faiss.IndexFlatL2(d)
        index.add(rel_poolers)
        candidate_neighbors_size = min(self.k, len(bucket_ids))
        edges_set = set()
        candidate_neighbors_dict = dict()
        for pooler_id, pooler_vec in enumerate(rel_poolers):
            edges_set, candidate_neighbors_dict = self.update_edges_and_candidates(pooler_vec, index,
                                                                                   candidate_neighbors_size,
                                                                                   pooler_id, edges_set,
                                                                                   automatic_edges_num,
                                                                                   candidate_neighbors_dict,
                                                                                   bucket2orig)
        if label_type < 2:
            candidate_neighbors_dict = {k: v for k, v in sorted(candidate_neighbors_dict.items(),
                                                                key=lambda item: item[1])}
            edges_set = self.process_candidates(candidate_neighbors_dict, edges_ratio, edges_set)
        return edges_set

    @staticmethod
    def create_final_edge_set(edges_set_per_bucket):
        final_edge_set = set()
        for bucket_edges in edges_set_per_bucket:
            final_edge_set.update(bucket_edges)
        return final_edge_set

    def connect_nodes(self, graph, buckets2poolers, label_type):
        """
        Add edges between nodes only when at least one of them belongs to the available pool,
        and within the same bucket.
        """
        # pairs_set = self.create_pairs_set(buckets2poolers)
        # pairs_list = [(pair[0], pair[1], self.calc_pair_weight(pair)) for pair in pairs_set if
        #               min(pair[0], pair[1]) < self.available_pool_size]
        edges_set_per_bucket = set()
        # for bucket in buckets2poolers.values():
        #     edges_set.update(list(self.create_bucket_edges(bucket)))
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            edges_set_per_bucket = list(pool.starmap(self.create_bucket_edges, zip(buckets2poolers.values(),
                                                                                   repeat(label_type))))
        final_edge_set = self.create_final_edge_set(edges_set_per_bucket)
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            final_pairs_weight = list(pool.map(self.calc_pair_weight, final_edge_set))
        # final_pairs_weight = [(pair[0], pair[1], weight) for pair, weight in pairs_weight if
        #                       min(pair[0], pair[1]) < self.available_pool_size]
        # pairs_list_final = final_pairs_weight.copy()
        graph.add_weighted_edges_from(final_pairs_weight)
        return graph


    def graph_with_threshold(self, graph, buckets2poolers, sim_threshold):
        """
        Add edges between nodes only when at least one of them belongs to the available pool,
        and within the same bucket.
        """
        pairs_set = self.create_pairs_set(buckets2poolers)
        # pairs_list = [(pair[0], pair[1], self.calc_pair_weight(pair)) for pair in pairs_set if
        #               min(pair[0], pair[1]) < self.available_pool_size]
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            pairs_weight = list(pool.map(self.calc_pair_weight, pairs_set))
        final_pairs_weight = [(pair[0], pair[1], weight) for pair, weight in pairs_weight if
                              min(pair[0], pair[1]) < self.available_pool_size]
        pairs_list_final = final_pairs_weight.copy()
        graph.add_weighted_edges_from(pairs_list_final)
        return graph

    def calc_pair_weight(self, pair):
        """
        calculate cosing similarity between a pair.
        """
        pooler1 = self.poolers[pair[0]]
        pooler2 = self.poolers[pair[1]]
        weight = max(round(1 - spatial.distance.cosine(pooler1, pooler2), 3), 0)
        return pair[0], pair[1], weight

    @staticmethod
    def create_pairs_set(buckets2poolers):
        """
        Generate a set of nodes pairs from each bucket
        """
        pairs_set = set()
        for bucket in buckets2poolers.values():
            pairs_set.update(list(combinations(bucket, 2)))
        return pairs_set

    def clean_old_train_pos(self):
        # Every connected components in old_connected_components includes the training samples in
        # addition to the available pool. They are essential for the uncertainty calculation.
        removal_dict = dict()
        old_connected_components = self.pos_connected_components.copy()
        for graph_id in self.pos_connected_components.keys():
            current_graph = self.pos_connected_components[graph_id].copy()
            removal_dict[graph_id] = [node_id for node_id in current_graph.nodes()
                                      if node_id >= self.available_pool_size]
        for graph_id in removal_dict.keys():
            current_graph = self.pos_connected_components[graph_id].copy()
            current_graph.remove_nodes_from(removal_dict[graph_id])
            self.pos_connected_components[graph_id] = current_graph
        self.save_to_pkl([self.pos_connected_components], ["final_clean_connected_components"])
        return old_connected_components

    def clean_old_train_neg(self):
        # Every connected components in old_connected_components includes the training samples in
        # addition the the available pool. They are essential for the uncertainty calculation.
        removal_dict = dict()
        old_connected_components = self.neg_connected_components.copy()
        for graph_id in self.neg_connected_components.keys():
            current_graph = self.neg_connected_components[graph_id].copy()
            removal_dict[graph_id] = [node_id for node_id in current_graph.nodes()
                                      if node_id >= self.available_pool_size]
        for graph_id in removal_dict.keys():
            current_graph = self.neg_connected_components[graph_id].copy()
            current_graph.remove_nodes_from(removal_dict[graph_id])
            self.neg_connected_components[graph_id] = current_graph
        self.save_to_pkl([self.neg_connected_components], ["final_clean_connected_components"])
        return old_connected_components

    def calc_criterion(self):
        pos_centrality = self.calc_centrality(1)
        neg_centrality = self.calc_centrality(0)
        pos_uncertainty, neg_uncertainty, votes_dict = self.calc_uncertainty()
        pos_selected = self.find_candidates(pos_centrality, pos_uncertainty, 1)
        neg_selected = self.find_candidates(neg_centrality, neg_uncertainty, 0)
        selected_k = pos_selected + neg_selected
        self.save_to_pkl([pos_centrality, neg_centrality, pos_uncertainty, neg_uncertainty, selected_k],
                         ["pos_centrality", "neg_centrality", "pos_uncertainty", "neg_uncertainty", "selected_k"])
        return selected_k, pos_uncertainty, neg_uncertainty, votes_dict

    # def calc_criterion_pos(self):
    #     pos_centrality = self.calc_centrality(self.connected_components.keys())
    #     pos_uncertainty = self.calc_uncertainty(self.connected_components.keys())
    #     pos_selected = self.find_candidates(pos_centrality, pos_uncertainty,
    #                                         self.connected_components.keys(),
    #                                         self.positive_budget_dict, True)
    #     neg_ids = [pooler_id for pooler_id in self.pool_predictions.keys() if not self.pool_predictions[pooler_id]]
    #     np.random.seed(seed=self.seed)
    #     neg_selected = list(np.random.choice(neg_ids, round(self.k * (1 - self.pos_budget)), replace=False))
    #     selected_k = pos_selected + neg_selected
    #     self.save_to_pkl([pos_centrality, pos_uncertainty, selected_k],
    #                      ["pos_centrality", "pos_uncertainty", "selected_k"])
    #     return selected_k

    def calc_centrality(self, label_type):
        """
        Perform the require centrality calculation.
        """
        if self.criterion == 'bc':
            return self.calc_betweenness_centrality(label_type)
        elif self.criterion == 'pagerank':
            return self.calc_pagerank_centrality(label_type)

    def calc_betweenness_centrality(self, label_type):
        # selected_samples = list()
        bc_dict = dict()
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 \
            else self.neg_connected_components.copy()
        for graph_id in ccs_copy.keys():
            bc_values = nx.betweenness_centrality(ccs_copy[graph_id], normalized=True, weight='weight')
            bc_dict[graph_id] = self.rank_it(bc_values)
            # selected_samples.extend(sorted(bc_dict, key=bc_dict.get, reverse=True)
            #                         [:relevant_budget_dict[graph_id]])
        return bc_dict

    def calc_pagerank_centrality(self, label_type):
        """
        Create a dict contains the relevant CC_ids as keys, and dicts, containing their nodes' pagerank ranking,
        as values.
        """
        # selected_samples = list()
        pagerank_dict = dict()
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 \
            else self.neg_connected_components.copy()
        for graph_id in ccs_copy.keys():
            flag = 0
            tolerance = 1e-06
            while not flag:
                try:
                    pagerank_values = nx.pagerank(ccs_copy[graph_id], tol=tolerance, weight='weight')
                    flag = 1
                except:
                    tolerance *= 2
            pagerank_dict[graph_id] = self.rank_it(pagerank_values)
        return pagerank_dict

    def calc_uncertainty(self):
        """
        Perform the required uncertainty calculation.
        """
        entropy_dict, votes_dict = self.calc_neighbors_uncertainty()
        pos_uncertainty = self.create_uncertainty_dict(1, entropy_dict)
        neg_uncertainty = self.create_uncertainty_dict(0, entropy_dict)
        return pos_uncertainty, neg_uncertainty, votes_dict

    def calc_neighbors_uncertainty(self, entropy_param=0.5):
        final_entropy_dict, uncertainty_dict, votes_dict = dict(), dict(), dict()
        conf_dict = self.create_conf_dict()
        for graph_id, graph in self.het_connected_components.items():
            for pooler_id in graph:
                if pooler_id >= self.available_pool_size:
                    continue
                regular_entropy = self.calc_entropy_scalar(conf_dict[pooler_id])
                votes_values = self.crete_votes_dict(pooler_id, graph, conf_dict)
                neighborhood_entropy = self.calc_neighborhood_entropy(votes_values)
                votes_dict[pooler_id] = votes_values
                final_entropy_dict[pooler_id] = entropy_param * regular_entropy + \
                                                (1 - entropy_param) * neighborhood_entropy
        return final_entropy_dict, votes_dict

    def crete_votes_dict(self,pooler_id, graph, conf_dict):
        votes_values = {0: 0, 1: 0}
        for neighbor in graph[pooler_id]:
            weight = graph[pooler_id][neighbor]['weight']
            if neighbor < self.available_pool_size:
                votes_values[self.pool_predictions[neighbor]] += weight * conf_dict[neighbor]
            else:
                votes_values[self.training_labels[neighbor]] += weight * conf_dict[neighbor]
        return votes_values


    def create_uncertainty_dict(self, label_type, entropy_dict):
        uncertainty_dict = dict()
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 else self.neg_connected_components.copy()
        for graph_id, graph in ccs_copy.items():
            current_entropy_dict = {pooler_id: entropy_dict[pooler_id] for pooler_id in graph}
            uncertainty_dict[graph_id] = self.rank_it(current_entropy_dict)
        return uncertainty_dict

    @staticmethod
    def calc_entropy_scalar(conf_val):
        try:
            entropy = -conf_val * log2(conf_val) - (1 - conf_val) * log2(1 - conf_val)
            return entropy
        except:
            return 0

    @staticmethod
    def calc_neighborhood_entropy(pooler_votes_values):
        try:
            p = pooler_votes_values[1] / (pooler_votes_values[1] + pooler_votes_values[0])
            entropy = -p * log2(p) - (1 - p) * log2(1 - p)
            return entropy
        except:
            return 0


    # def calc_neighbors_uncertainty(self):
    #     votes_values, entropy_dict, uncertainty_dict = dict(), dict(), dict()
    #     conf_dict = self.create_conf_dict()
    #     for graph_id, graph in self.het_connected_components.items():
    #         rel_poolers = [pooler_id for pooler_id in graph if pooler_id < self.available_pool_size]
    #         with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
    #             entropy_vals = pool.starmap(self.calc_pooler_entropy, zip(rel_poolers,
    #                                                                       repeat(graph),
    #                                                                       repeat(conf_dict)))
    #         entropy_dict.update({pooler_id: entropy_val for pooler_id, entropy_val in entropy_vals})
    #     return entropy_dict
    #
    # def calc_pooler_entropy(self, pooler_id, graph, conf_dict):
    #     votes_values = dict()
    #     votes_values[0] = 0
    #     votes_values[1] = 0
    #     for neighbor in graph[pooler_id]:
    #         weight = graph[pooler_id][neighbor]['weight']
    #         if neighbor < self.available_pool_size:
    #             votes_values[self.pool_predictions[neighbor]] += weight * conf_dict[neighbor]
    #         else:
    #             votes_values[self.training_labels[neighbor]] += weight * conf_dict[neighbor]
    #     return pooler_id, self.calc_entropy(votes_values)


    @staticmethod
    def rank_it(input_dict):
        """
        Create a dictionary that contains the poolers id as keys, and their rank w.r.t the measurement as values.
        """
        sorted_items = sorted(input_dict.items(), key=lambda item: item[1], reverse=True)
        rank, count, previous, result = 0, 0, None, dict()
        for pooler_id, measurement_val in sorted_items:
            count += 1
            if measurement_val != previous:
                rank += count
                previous = measurement_val
                count = 0
            result[pooler_id] = rank
        return result

    def find_candidates(self, cands_centrality, cands_uncertainty, label_type):
        final_cands = []
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 else self.neg_connected_components.copy()
        relevant_budget_dict = self.positive_budget_dict if label_type == 1 else self.negative_budget_dict
        for graph_id in ccs_copy.keys():
            weighted_ranking = dict()
            for pooler_id in ccs_copy[graph_id]:
                centrality_val = cands_centrality[graph_id][pooler_id]
                uncertainty_val = cands_uncertainty[graph_id][pooler_id]
                pooler_rank = self.selection_param * centrality_val + (1 - self.selection_param) * uncertainty_val
                weighted_ranking[pooler_id] = pooler_rank
            sorted_items = sorted(weighted_ranking.items(), key=lambda item: item[1])
            cc_cands = [item[0] for item in sorted_items[:relevant_budget_dict[graph_id]]]
            final_cands.extend(cc_cands)
        return final_cands

    def find_weakly_supervised(self):
        ws_candidates = {pooler_id for pooler_id in range(self.available_pool_size)}
        ws_candidates = ws_candidates.difference(self.selected_k)
        if "ws_k" in self.mode:
            pos_sorted_items, neg_sorted_items = self.calc_Kasai_ws(ws_candidates)
        elif "ws_b" in self.mode:
            return self.calc_battleships_ws(ws_candidates)
        else:
            pos_sorted_items, neg_sorted_items = {}, {}  # Without weak supervision

        # ws_pos_expectation = self.calc_pos_expectation(ws_candidates)
        # pos_sorted_items = sorted(ws_pos_expectation['pos_predicted'].items(), key=lambda item: item[1], reverse=True)
        # neg_sorted_items = sorted(ws_pos_expectation['neg_predicted'].items(), key=lambda item: item[1], reverse=False)
        #
        # # Sometimes ws_pos_cands will be smaller than k/2 (as it bounded by min(k/2, len(self.pos_preds_ids)).
        # # Hence, in order to obtain balanced sampling we take len(ws_pos_cands) from self.neg_preds_ids

        ws_pos_cands = set([item[0] for item in pos_sorted_items[:round(self.k / 2)]])
        if len(ws_pos_cands) > 0:
            ws_neg_cands = set([item[0] for item in neg_sorted_items[:len(ws_pos_cands)]])
        else:
            ws_neg_cands = set()
        return ws_pos_cands, ws_neg_cands

    def calc_Kasai_ws(self, ws_candidates):
        conf_dict = self.create_conf_dict()
        pos_dict = {pooler_id: conf_dict[pooler_id] for pooler_id in ws_candidates
                    if self.pool_predictions[pooler_id]}
        neg_dict = {pooler_id: conf_dict[pooler_id] for pooler_id in ws_candidates
                    if not self.pool_predictions[pooler_id]}
        pos_sorted_items = sorted(pos_dict.items(), key=lambda item: item[1], reverse=True)
        neg_sorted_items = sorted(neg_dict.items(), key=lambda item: item[1], reverse=True)
        return pos_sorted_items, neg_sorted_items

    def calc_battleships_ws(self, ws_candidates):
        ws_cands_pos = self.calc_battleships_ws_by_type(ws_candidates, 1)
        ws_cands_neg = self.calc_battleships_ws_by_type(ws_candidates, 0)
        return ws_cands_pos, ws_cands_neg

    def calc_battleships_ws_by_type(self, ws_candidates, label_type):
        budget_dict = self.distribute_budget(label_type, ws_candidates, True)
        uncertainty_values = self.pos_uncertainty if label_type == 1 else self.neg_uncertainty
        final_cands = []
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 else self.neg_connected_components.copy()
        for graph_id in ccs_copy.keys():
            available_budget = budget_dict[graph_id]
            sorted_uncertainty = sorted(uncertainty_values[graph_id].items(), key=lambda item: item[1], reverse=True)
            curr_idx = 0
            cc_cands = []
            while available_budget and curr_idx < len(sorted_uncertainty):
                curr_pooler = sorted_uncertainty[curr_idx][0]
                if curr_pooler in ws_candidates and \
                        self.votes_dict[curr_pooler][label_type] > self.votes_dict[curr_pooler][1-label_type]:
                    cc_cands.append(sorted_uncertainty[curr_idx][0])
                    available_budget -= 1
                curr_idx += 1
            final_cands.extend(cc_cands)
        return set(final_cands)

    # def calc_pos_expectation(self, ws_candidates):
    #     ws_distances = self.calc_ws_distances(ws_candidates)
    #     ws_distances = self.sort_distances(ws_distances)
    #     pos_expectation_dict = {'pos_predicted': dict(), 'neg_predicted': dict()}
    #     conf_dict = self.create_conf_dict()
    #     for pooler_id, sorted_dists in ws_distances.items():
    #         exp_val = sum([self.training_labels[neighbor[0]] * neighbor[1] * conf_dict[neighbor[0]]
    #                        for neighbor in sorted_dists])
    #         if pooler_id in self.pos_preds_ids:
    #             pos_expectation_dict['pos_predicted'][pooler_id] = exp_val
    #         else:
    #             pos_expectation_dict['neg_predicted'][pooler_id] = exp_val
    #     return pos_expectation_dict

    def create_conf_dict(self):
        if self.treat_weak_labels:
            conf_dict = {pooler_id: 1 if pooler_id not in self.weak_ids else self.weak_labels_confidence[pooler_id]
                         for pooler_id in range(self.available_pool_size, len(self.poolers))}
        else:
            conf_dict = {pooler_id: 1 if pooler_id not in self.weak_ids else 0 for
                         pooler_id in range(self.available_pool_size, len(self.poolers))}
        conf_dict.update({pooler_id: self.confidence_dict[pooler_id] for pooler_id in
                          range(self.available_pool_size)})
        return conf_dict


    def calc_ws_distances(self, ws_candidates):
        distances = {pooler_id: dict() for pooler_id in ws_candidates}
        labeled_poolers = {labeled_id: self.poolers[labeled_id] for labeled_id in
                           range(self.available_pool_size, len(self.poolers))}
        for pooler_id in distances.keys():
            for labeled_id in range(self.available_pool_size, len(self.poolers)):
                distances[pooler_id][labeled_id] = spatial.distance.cosine(self.poolers[pooler_id],
                                                                           labeled_poolers[labeled_id])
        del labeled_poolers
        return distances

    def sort_distances(self, ws_distances):
        pooler_ids = {pooler_id for pooler_id in ws_distances.keys()}
        sorted_distances = dict()
        for pooler_id in pooler_ids:
            sorted_distances[pooler_id] = [(k, v) for k, v in sorted(ws_distances[pooler_id].items(),
                                                                     key=lambda item: item[1],
                                                                     reverse=True)][:self.nearest_param]
        return sorted_distances

    def save_to_pkl(self, files_list, file_names_list):
        path = self.output_path + self.mode + "/pkl_files/"
        if not os.path.exists(path):
            os.makedirs(path)
        for file, file_name in zip(files_list, file_names_list):
            output = open(path + file_name + '_iter' + str(self.iter) +
                          '_seed' + str(self.seed) + '.pkl', 'wb')
            pickle.dump(file, output)
            output.close()
        return

    @property
    def get_selected_k(self):
        return self.selected_k

    @property
    def get_weakly_supervised(self):
        return self.ws_pos_cands, self.ws_neg_cands
