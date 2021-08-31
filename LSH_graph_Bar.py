import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter
import torch
import re
import json
from math import log2
from scipy import spatial
import random
import time


class LSH_graph:
    def __init__(self, poolers_paths, k, seed, files_path, iteration, criterion='pagerank',
                 weights_type='with_threshold', vectors_num=12,
                 lsh_iterations=1, dim=768, pos_threshold_cond=0.9,
                 pos_budget=0.5, edges_threshold=0.75, sim_threshold=0.85,
                 adapted_sim_threshold=0.9, min_cc_ratio=1/100, max_cc_ratio=1/20,
                 selection_param=0.5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.poolers_paths = poolers_paths
        self.files_path = files_path
        self.iter = iteration
        self.k = k
        self.weights_type = weights_type
        self.dim = dim
        self.vectors_num = vectors_num
        self.lsh_iterations = lsh_iterations
        self.pos_budget = pos_budget
        self.edges_threshold = edges_threshold
        self.sim_threshold = sim_threshold
        self.adapted_sim_threshold = adapted_sim_threshold
        self.criterion = criterion
        self.poolers, self.available_pool_size = self.create_poolers()
        self.min_cc_param = int(min_cc_ratio * k)
        self.max_cc_param = int(max_cc_ratio * k)
        self.selection_param = selection_param
        self.training_labels = self.create_labels()
        self.poolers_ids = self.create_poolers_ids_list()
        self.pool_predictions = self.create_predictions()
        self.buckets2poolers, self.poolers2buckets = self.create_buckets(self.poolers, self.lsh_iterations,
                                                                         self.vectors_num, self.poolers)
        self.graph = self.initialize_graph(self.poolers_ids)
        self.graph = self.create_graph_edges(self.graph, self.lsh_iterations,
                                             self.buckets2poolers, self.edges_threshold, self.sim_threshold)
        self.connected_components = self.create_connected_components(self.graph)
        self.ccs_available_pool = self.calc_CCS_available_pool()
        self.connected_components = self.validate_connected_components()
        self.cc_labels = self.assign_cc_labels(pos_threshold_cond)
        self.old_connected_components = self.clean_old_train()
        self.positive_graph_ids, self.positive_CCs_size = self.calc_CCs_type_size(1)
        self.negative_graph_ids, self.negative_CCs_size = self.calc_CCs_type_size(0)
        self.positive_budget_dict, self.negative_budget_dict = self.distribute_budget()
        self.selected_k = self.calc_criterion()

    def create_poolers(self):
        poolers_dict = self.create_poolers_available_pool()
        available_pool_size = len(poolers_dict)
        poolers_dict = self.create_poolers_current_train(poolers_dict, available_pool_size)
        return poolers_dict, available_pool_size

    def create_poolers_available_pool(self):
        poolers_dict = dict()
        # self.poolers_paths[0] is the poolers file of the available pool
        pooler_path = self.poolers_paths[0]
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
            poolers_dict[id_val] = list(map(float, pooler.split(' ')))
        preds_file.close()
        return poolers_dict

    def create_poolers_current_train(self, poolers_dict, available_pool_size):
        # self.poolers_paths[1] is the poolers file of the current train
        pooler_path = self.poolers_paths[1]
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
            poolers_dict[id_val + available_pool_size] = list(map(float, pooler.split(' ')))
        preds_file.close()
        return poolers_dict

    def create_poolers_ids_list(self):
        return [pooler_id for pooler_id in range(len(self.poolers))]

    def create_labels(self):
        # for each pair in the available we assign the label 2 (unknown)
        labels_dict = {id_val: 2 for id_val in range(self.available_pool_size)}
        labels_file = open(self.files_path + 'current_train.txt', "r", encoding="utf-8")
        lines_labels = labels_file.readlines()
        labels_file.close()
        for id_val, line in enumerate(lines_labels):
            labels_dict[id_val + self.available_pool_size] = int(re.sub("[^0-9]", "", line[-2]))
        return labels_dict

    def create_predictions(self):
        preditions_dict = dict()
        pooler_path = self.poolers_paths[0]
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            preditions_dict[id_val] = int(re.sub("[^0-9]", "", line.split("match")[1]))
        preds_file.close()
        return preditions_dict

    def create_buckets(self, rel_nodes, lsh_iterations, vectors_num, poolers):
        buckets2poolers_dict = dict()
        poolers2buckets_dict = {pooler_id: [] for pooler_id in rel_nodes}
        for iteration in range(lsh_iterations):
            buckets2poolers_dict[iteration] = dict()
            random_vecs = np.random.randn(vectors_num, self.dim)
            for pooler_id, pooler_vec in poolers.items():
                bucket_id = self.classify_pooler(pooler_vec, random_vecs)
                if bucket_id not in buckets2poolers_dict[iteration].keys():
                    buckets2poolers_dict[iteration][bucket_id] = [pooler_id]
                else:
                    buckets2poolers_dict[iteration][bucket_id].append(pooler_id)
                poolers2buckets_dict[pooler_id].append(bucket_id)
        return buckets2poolers_dict, poolers2buckets_dict

    @ staticmethod
    def create_connected_components(graph):
        graphs_dict = dict()
        connected_components = nx.connected_components(graph)
        for graph_id, cc in enumerate(connected_components):
            graphs_dict[graph_id] = graph.subgraph(cc)
        return graphs_dict

    def validate_connected_components(self):
        self.fix_large_connected_components()
        self.fix_small_connected_components()
        return self.connected_components

    def calc_CCS_available_pool(self):
        ccs_available_pool = dict()
        for graph_id, graph in self.connected_components.items():
            ccs_available_pool[graph_id] = len([pooler_id for pooler_id in graph.nodes() if
                                                pooler_id < self.available_pool_size])
        return ccs_available_pool

    def fix_large_connected_components(self):
        ccs_copy = self.connected_components.copy()
        ccs_available_pool_copy = self.ccs_available_pool.copy()
        max_val = self.max_cc_param * self.available_pool_size / self.k
        flag = 0
        for graph_id, graph in ccs_copy.items():
            if ccs_available_pool_copy[graph_id] > max_val:
                new_connected_components = self.decompose_graph(graph)
                self.connected_components.pop(graph_id)
                self.ccs_available_pool.pop(graph_id)
                self.update_connected_components(new_connected_components)
                flag = 1
        if flag == 1:
            self.fix_large_connected_components()
        return

    def decompose_graph(self, graph):
        vectors_num = int(log2(graph.number_of_nodes() / self.min_cc_param)) + 1
        poolers = {pooler_id: self.poolers[pooler_id] for pooler_id in graph.nodes()}
        lsh_iterations = 1
        edges_threshold = 0.5
        buckets2poolers_dict, poolers2buckets_dict = self.create_buckets(graph.nodes(), lsh_iterations,
                                                                         vectors_num, poolers)
        new_graph = self.initialize_graph(poolers.keys())
        new_graph = self.create_graph_edges(new_graph, lsh_iterations, buckets2poolers_dict,
                                            edges_threshold, self.adapted_sim_threshold)
        connected_components = self.create_connected_components(new_graph)
        return connected_components

    def update_connected_components(self, new_connected_components):
        current_size = len(self.connected_components)
        for graph_id, graph in new_connected_components.items():
            self.connected_components[graph_id + current_size] = graph
            self.ccs_available_pool[graph_id + current_size] = len([pooler_id for pooler_id in
                                                                    graph.nodes() if pooler_id <
                                                                    self.available_pool_size])
        return

    def fix_small_connected_components(self):
        ccs_copy = self.connected_components.copy()
        ccs_available_pool_copy = self.ccs_available_pool.copy()
        ccs_centroids = self.calc_centroids()
        min_val = self.min_cc_param * self.available_pool_size / self.k
        small_connected_components = {graph_id: graph for graph_id, graph in ccs_copy.items()
                                      if ccs_available_pool_copy[graph_id] < min_val}
        legit_connected_components = {graph_id for graph_id in ccs_copy.keys()
                                      if graph_id not in small_connected_components.keys()}
        for graph_id, graph in small_connected_components.items():
            closest_graph_id = self.find_closest_cc(graph_id, ccs_centroids,
                                                    legit_connected_components)
            self.connect_small_to_legit(graph_id, graph, closest_graph_id)
            self.connected_components.pop(graph_id)
            self.ccs_available_pool.pop(graph_id)
        return

    def calc_centroids(self):
        ccs_centroids = dict.fromkeys(self.connected_components.keys())
        for graph_id, graph in self.connected_components.items():
            poolers_list = [self.poolers[pooler_id] for pooler_id in graph.nodes()]
            ccs_centroids[graph_id] = list(np.mean(poolers_list, axis=0))
        return ccs_centroids

    @staticmethod
    def find_closest_cc(graph_id, ccs_centroids, legit_connected_components):
        current_centroid = np.array(ccs_centroids[graph_id])
        # Create a list of pairs (cc_id, dist) where dist is the distance
        # from the current_centroid to the centroid of cc_id
        distances = [(cc_id, round(spatial.distance.cosine(current_centroid, np.array(ccs_centroids[cc_id])), 3))
                     for cc_id in legit_connected_components]
        return sorted(distances, key=lambda x: x[1])[0][0]

    def connect_small_to_legit(self, graph_id, graph, closest_graph_id):
        closest_graph = self.connected_components[closest_graph_id].copy()
        # for each node from the small graph connect it with an edge to a
        # random node from the closest legit graph
        selected_nodes = np.random.choice(closest_graph.nodes(), graph.number_of_nodes())
        edges_list = []
        for node1, node2 in zip(graph.nodes(), selected_nodes):
            edges_list.append((node1, node2, self.calc_pair_weight((node1, node2))))
        closest_graph.add_weighted_edges_from(edges_list)
        self.connected_components[closest_graph_id] = closest_graph
        return

    def assign_cc_labels(self, pos_threshold):
        connected_components_labels = dict()
        for graph_id, graph in self.connected_components.items():
            cc_labels = {node_id: self.training_labels[node_id] for node_id in graph.nodes()}
            labels_counter = Counter(cc_labels.values())
            try:
                if labels_counter[1] / (labels_counter[0] + labels_counter[1]) >= pos_threshold:
                    connected_components_labels[graph_id] = 1
                else:
                    connected_components_labels[graph_id] = 0
            except:
                # This clause exists for cases where only labels_counter[0]>0
                connected_components_labels[graph_id] = 0
        ccs_labels_counter = Counter(connected_components_labels.values())
        if ccs_labels_counter[1] < 2:
            # If there are less than two potentially positive connected components
            # we call this function again with a more tolerant threshold
            connected_components_labels = self.assign_cc_labels(0.9 * pos_threshold)
        return connected_components_labels

    def calc_CCs_type_size(self, cc_label):
        relevant_graph_ids = [graph_id for graph_id in self.connected_components.keys()
                              if self.cc_labels[graph_id] == cc_label]
        relevant_size = sum([len(self.connected_components[graph_id]) for graph_id in
                        self.connected_components.keys() if graph_id in relevant_graph_ids])
        return relevant_graph_ids, relevant_size

    def distribute_budget(self):
        positive_budget_dict = self.create_budget_dict(1, self.positive_graph_ids, self.positive_CCs_size)
        negative_budget_dict = self.create_budget_dict(0, self.negative_graph_ids, self.negative_CCs_size)
        return positive_budget_dict, negative_budget_dict

    def create_budget_dict(self, label, rel_graph_ids, rel_ccs_size):
        budget_dict = dict()
        total_used = 0
        rel_share = self.pos_budget if label == 1 else 1 - self.pos_budget
        total_budget = round(self.k * rel_share)
        for graph_id, graph_label in self.cc_labels.items():
            if graph_id in rel_graph_ids:
                relative_share = len(self.connected_components[graph_id]) / rel_ccs_size
                budget = int(relative_share * total_budget)
                budget_dict[graph_id] = budget
                total_used += budget
        if total_used < total_budget:
            budget_dict = self.assign_residue(budget_dict, total_budget - total_used, rel_graph_ids)
        return budget_dict

    @staticmethod
    def assign_residue(budget_dict, residue, rel_graph_ids):
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

    def create_graph_funcs_dict(self):
        return {'without': self.graph_without,
                'with_threshold': self.graph_with_threshold,
                # 'relative_adjs': self.graph_relative_adjs,
                # 'cos_sim': self.graph_cos_sim(),
                # 'relative_adjs_cos_sim': self.graph_relative_adjs_cos_sim()
                }

    @staticmethod
    def initialize_graph(poolers_ids):
        graph = nx.Graph()
        graph.add_nodes_from(poolers_ids)
        return graph

    def create_graph_edges(self, graph, lsh_iterations, buckets2poolers, edges_threshold, sim_threshold):
        funcs_dict = self.create_graph_funcs_dict()
        return funcs_dict[self.weights_type](graph, lsh_iterations, buckets2poolers,
                                             edges_threshold, sim_threshold)

    def graph_without(self, graph, lsh_iterations, buckets2poolers, edges_threshold=None):
        for iteration in range(lsh_iterations):
            for bucket_id in buckets2poolers[iteration].keys():
                current_bucket = buckets2poolers[iteration][bucket_id]
                graph.add_edges_from(list(combinations(current_bucket, 2)))
        return graph

    def graph_with_threshold(self, graph, lsh_iterations, buckets2poolers, edges_threshold, sim_threshold):
        pairs_dict = self.create_pairs_dict(lsh_iterations, buckets2poolers)
        threshold = edges_threshold * lsh_iterations
        pairs_list = [(pair[0], pair[1], self.calc_pair_weight(pair)) for pair in
                      pairs_dict.keys()]
        # pairs_list = [(pair[0], pair[1], self.calc_pair_weight(pair)) for pair in
        #               pairs_dict.keys() if pairs_dict[pair] >= sim_threshold]
        # pairs_list = [(pair[0], pair[1]) for pair in pairs_dict.keys() if pairs_dict[pair] >= threshold]
        pairs_list_final = [pair for pair in pairs_list if pair[2] > sim_threshold]
        graph.add_weighted_edges_from(pairs_list_final)
        # graph.add_edges_from(pairs_list)
        return graph

    def calc_pair_weight(self, pair):
        pooler1 = self.poolers[pair[0]]
        pooler2 = self.poolers[pair[1]]
        return round(1 - spatial.distance.cosine(np.array(pooler1), np.array(pooler2)), 3)

    def graph_relative_adjs(self):
        pairs_dict = self.create_pairs_dict()
        pairs_list = [(key[0], key[1], val/self.lsh_iterations)
                      for key, val in pairs_dict.items()]
        self.graph.add_weighted_edges_from(pairs_list)
        return self.graph

    @staticmethod
    def create_pairs_dict(lsh_iterations, buckets2poolers):
        pairs_dict = dict()
        for iteration in range(lsh_iterations):
            for bucket_id in buckets2poolers[iteration].keys():
                current_bucket = buckets2poolers[iteration][bucket_id]
                pairs = list(combinations(current_bucket, 2))
                for pair in pairs:
                    # There is no need to check the inverse order of the pair's elements
                    # since the ordinality of the pooler IDs is guaranteed
                    if pair in pairs_dict.keys():
                        pairs_dict[pair] += 1
                    else:
                        pairs_dict[pair] = 1
        return pairs_dict

    def clean_old_train(self):
        # Every connected components in old_connected_components includes the training samples in
        # addition the the available pool. They are essential for the uncertainty calculation.
        removal_dict = dict()
        old_connected_components = self.connected_components.copy()
        for graph_id in self.connected_components.keys():
            current_graph = self.connected_components[graph_id].copy()
            removal_dict[graph_id] = [node_id for node_id in current_graph.nodes()
                                      if node_id >= self.available_pool_size]
        for graph_id in removal_dict.keys():
            current_graph = self.connected_components[graph_id].copy()
            current_graph.remove_nodes_from(removal_dict[graph_id])
            self.connected_components[graph_id] = current_graph
        return old_connected_components

    def calc_criterion(self):
        pos_centrality = self.calc_centrality(self.positive_graph_ids)
        neg_centrality = self.calc_centrality(self.negative_graph_ids)
        pos_uncertainty = self.calc_uncertainty(self.positive_graph_ids)
        neg_uncertainty = self.calc_uncertainty(self.negative_graph_ids)
        pos_selected = self.find_candidates(pos_centrality, pos_uncertainty,
                                            self.positive_graph_ids, self.positive_budget_dict)
        neg_selected = self.find_candidates(neg_centrality, neg_uncertainty,
                                            self.negative_graph_ids, self.negative_budget_dict)
        selected_k = pos_selected + neg_selected
        return selected_k

    def calc_centrality(self, relevant_graph_ids):
        if self.criterion == 'bc':
            return self.calc_betweenness_centrality(relevant_graph_ids)
        elif self.criterion == 'pagerank':
            return self.calc_pagerank_centrality(relevant_graph_ids)

    def calc_betweenness_centrality(self, relevant_graph_ids):
        # selected_samples = list()
        bc_dict = dict()
        for graph_id in relevant_graph_ids:
            bc_values = nx.betweenness_centrality(self.connected_components[graph_id],
                                                          normalized=True, weight='weight')
            bc_dict[graph_id] = self.rank_it(bc_values)
            # selected_samples.extend(sorted(bc_dict, key=bc_dict.get, reverse=True)
            #                         [:relevant_budget_dict[graph_id]])
        return bc_dict

    def calc_pagerank_centrality(self, relevant_graph_ids):
        # selected_samples = list()
        pagerank_dict = dict()
        for graph_id in relevant_graph_ids:
            pagerank_values = nx.pagerank(self.connected_components[graph_id], weight='weight')
            pagerank_dict[graph_id] = self.rank_it(pagerank_values)

            # selected_samples.extend(sorted(pagerank_dict, key=pagerank_dict.get, reverse=True)
            #                         [:relevant_budget_dict[graph_id]])

        return pagerank_dict

    def calc_uncertainty(self, relevant_graph_ids):
        votes_values, entropy_dict, uncertainty_dict = dict(), dict(), dict()
        for graph_id in relevant_graph_ids:
            for pooler_id in self.connected_components[graph_id]:
                votes_values[pooler_id] = dict()
                votes_values[pooler_id][0] = 0
                votes_values[pooler_id][1] = 0
                for neighbor in self.old_connected_components[graph_id][pooler_id]:
                    weight = self.old_connected_components[graph_id][pooler_id][neighbor]['weight']
                    if neighbor < self.available_pool_size:
                        votes_values[pooler_id][self.pool_predictions[neighbor]] += weight
                    else:
                        votes_values[pooler_id][self.training_labels[neighbor]] += weight
                entropy_dict[pooler_id] = self.calc_entropy(votes_values[pooler_id])
            uncertainty_dict[graph_id] = self.rank_it(entropy_dict)
        return uncertainty_dict

    @staticmethod
    def calc_entropy(votes_values):
        try:
            p = votes_values[1] / (votes_values[1] + votes_values[0])
            entropy = -p * log2(p) if p < 1 else p * log2(p)
            return entropy
        except:
            return 0

    @staticmethod
    def rank_it(input_dict):
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

    def find_candidates(self, cands_centrality, cands_uncertainty, relevant_graph_ids, relevant_budget_dict):
        final_cands = []
        for graph_id in relevant_graph_ids:
            weighted_ranking = dict()
            for pooler_id in self.connected_components[graph_id]:
                centrality_val = cands_centrality[graph_id][pooler_id]
                uncertainty_val = cands_uncertainty[graph_id][pooler_id]
                pooler_rank = self.selection_param * centrality_val + (1 - self.selection_param) * uncertainty_val
                weighted_ranking[pooler_id] = pooler_rank
            sorted_items = sorted(weighted_ranking.items(), key=lambda item: item[1])
            cc_cands = [item[0] for item in sorted_items[:relevant_budget_dict[graph_id]]]
            final_cands.extend(cc_cands)
        return final_cands

    @property
    def get_selected_k(self):
        return self.selected_k


#
# if __name__ == "__main__":
#     configs = json.load(open('configs.json'))
#     configs = {conf['name']: conf for conf in configs}
#
#     task = "Structured/Walmart-Amazon"
#
#     intent = 0
#     path = configs[task + str(intent)]['path']
#     iteration = 1
#     seed = 1
#     orig_train = configs[task + str(intent)]['trainset']
#     task = task.split('/')[1]
#     poolers_path = orig_train.replace("train" + str(intent) + ".txt",
#                                            "available_pool" + str(intent)
#                                            + "_iter" + str(iteration)
#                                            + "_train_output_seed" + str(seed) + ".txt")
#     poolers_path = poolers_path.replace('er_magellan/', '')
#     poolers_path_available_pool = poolers_path.replace("data", "output")
#     poolers_path_current_train = poolers_path_available_pool.replace("available_pool", "current_train")
#
#     k = 300
#     dim = 768
#     rand_vectors_num = 10
#     LSH_iterations = 5
#     LSH_obj = LSH_graph([poolers_path_available_pool, poolers_path_current_train],
#                         k, 1, path, 'bc')













