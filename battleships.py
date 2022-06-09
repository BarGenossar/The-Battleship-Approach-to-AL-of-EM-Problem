import numpy as np
import networkx as nx
from itertools import repeat
import torch
import re
from math import log2
import random
import os
import pickle
from scipy import spatial
import multiprocessing
from collections import defaultdict
import faiss


class battleships_graph:
    def __init__(self, poolers_paths, k, seed, files_path,
                 output_path, iteration, criterion='pagerank',
                 mode='top_k', lsh_iterations=10, dim=768,
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
        self.pos_graph, self.pos_connected_components, self.pos_ccs_available_pool_sizes = self.from_lsh2graph_type(1)
        self.neg_graph, self.neg_connected_components, self.neg_ccs_available_pool_sizes = self.from_lsh2graph_type(0)
        self.het_graph, self.het_connected_components, self.het_ccs_available_pool_sizes = self.from_lsh2graph_type(2)
        self.validate_connected_components()
        self.positive_budget_dict = self.distribute_budget(1)
        self.negative_budget_dict = self.distribute_budget(0)
        self.selected_k, self.pos_uncertainty, self.neg_uncertainty, self.votes_dict = self.calc_criterion()
        self.ws_pos_cands, self.ws_neg_cands = self.find_weakly_supervised()

    def create_poolers(self):
        poolers_dict = self.create_poolers_available_pool()
        available_pool_size = len(poolers_dict)
        poolers_dict = self.create_poolers_current_train(poolers_dict, available_pool_size)
        self.save_to_pkl([poolers_dict], ["poolers"])
        return poolers_dict, available_pool_size

    def create_poolers_available_pool(self):
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
        pooler_path = self.poolers_paths[1]
        preds_file = open(pooler_path, "r", encoding="utf-8")
        lines_preds = preds_file.readlines()
        for id_val, line in enumerate(lines_preds):
            pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
            poolers_dict[id_val + available_pool_size] = np.array(list(map(float, pooler.split(' '))))
        preds_file.close()

        return poolers_dict

    def create_labels(self):
        labels_dict = {id_val: 2 for id_val in range(self.available_pool_size)}
        labels_file = open(self.files_path + 'current_train.txt', "r", encoding="utf-8")
        lines_labels = labels_file.readlines()
        labels_file.close()
        for id_val, line in enumerate(lines_labels):
            labels_dict[id_val + self.available_pool_size] = int(re.sub("[^0-9]", "", line[-2]))
        return labels_dict

    def create_predictions(self):
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

    def create_buckets(self, rel_poolers_ids, min_val):
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

    @staticmethod
    def handle_legit_buckets(buckets2poolers, final_buckets2poolers, bucket_parents, lsh_iter, min_val, max_val):
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
        graphs_dict = dict()
        connected_components = nx.connected_components(graph)
        for graph_id, cc in enumerate(connected_components):
            graphs_dict[graph_id] = graph.subgraph(cc)
        return graphs_dict

    def validate_connected_components(self):
        light_conncted_components_pos = self.get_light_connected_components(self.pos_connected_components)
        light_conncted_components_neg = self.get_light_connected_components(self.neg_connected_components)
        self.save_to_pkl([light_conncted_components_pos, light_conncted_components_neg,
                          self.pos_connected_components, self.neg_connected_components],
                         ["final_connected_components(light1)", "final_connected_components(light0)",
                          "final_connected_components_pos", "final_connected_components_neg"])
        return

    def get_light_connected_components(self, connected_components):
        light_dict = dict()
        for graph_id, graph in connected_components.items():
            light_dict[graph_id] = [self.poolers[pooler_id] for pooler_id in graph.nodes()]
        return light_dict

    def calc_CCS_available_pool_sizes(self, connected_components):
        ccs_available_pool_sizes = dict()
        for graph_id, graph in connected_components.items():
            ccs_available_pool_sizes[graph_id] = len([pooler_id for pooler_id in graph.nodes() if
                                                      pooler_id < self.available_pool_size])
        return ccs_available_pool_sizes

    def update_rel_CCs(self, rel_CCs, label_type):
        if label_type == 1:
            self.pos_connected_components = rel_CCs
        elif label_type == 0:
            self.neg_connected_components = rel_CCs
        else:
            self.het_connected_components = rel_CCs
        return

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

    @staticmethod
    def assign_residue(budget_dict, residue, rel_graph_ids):
        chosen_graph_ids = random.choices(rel_graph_ids, k=residue)
        for graph_id in chosen_graph_ids:
            budget_dict[graph_id] += 1
        return budget_dict

    @staticmethod
    def classify_pooler(pooler, random_vecs):
        bucket_id = ''
        for rand_vec in random_vecs:
            bucket_id += '1' if rand_vec.dot(pooler) > 0 else '0'
        return bucket_id

    @staticmethod
    def initialize_graph(poolers_ids):
        graph = nx.Graph()
        graph.add_nodes_from(poolers_ids)
        return graph

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

    @staticmethod
    def process_candidates(candidate_neighbors_dict, edges_ratio, edges_set):
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
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            edges_set_per_bucket = list(pool.starmap(self.create_bucket_edges, zip(buckets2poolers.values(),
                                                                                   repeat(label_type))))
        final_edge_set = self.create_final_edge_set(edges_set_per_bucket)
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            final_pairs_weight = list(pool.map(self.calc_pair_weight, final_edge_set))
        graph.add_weighted_edges_from(final_pairs_weight)
        return graph

    def calc_pair_weight(self, pair):
        """
        calculate cosing similarity between a pair.
        """
        pooler1 = self.poolers[pair[0]]
        pooler2 = self.poolers[pair[1]]
        weight = max(round(1 - spatial.distance.cosine(pooler1, pooler2), 3), 0)
        return pair[0], pair[1], weight

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

    def calc_centrality(self, label_type):
        """
        Perform the require centrality calculation.
        """
        if self.criterion == 'bc':
            return self.calc_betweenness_centrality(label_type)
        elif self.criterion == 'pagerank':
            return self.calc_pagerank_centrality(label_type)

    def calc_betweenness_centrality(self, label_type):
        bc_dict = dict()
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 \
            else self.neg_connected_components.copy()
        for graph_id in ccs_copy.keys():
            bc_values = nx.betweenness_centrality(ccs_copy[graph_id], normalized=True, weight='weight')
            bc_dict[graph_id] = self.rank_it(bc_values)
        return bc_dict

    def calc_pagerank_centrality(self, label_type, tolerance=1e-06):
        pagerank_dict = dict()
        ccs_copy = self.pos_connected_components.copy() if label_type == 1 \
            else self.neg_connected_components.copy()
        for graph_id in ccs_copy.keys():
            flag = 0
            while not flag:
                try:
                    pagerank_values = nx.pagerank(ccs_copy[graph_id], tol=tolerance, weight='weight')
                    flag = 1
                except:
                    tolerance *= 2
            pagerank_dict[graph_id] = self.rank_it(pagerank_values)
        return pagerank_dict

    def calc_uncertainty(self):
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
            pos_sorted_items, neg_sorted_items = self.calc_DTAL_ws(ws_candidates)
        elif "ws_b" in self.mode:
            return self.calc_battleships_ws(ws_candidates)
        else:
            pos_sorted_items, neg_sorted_items = {}, {}  # Without weak supervision

        # Sometimes ws_pos_cands will be smaller than k/2 (as it is bounded by min(k/2, len(self.pos_preds_ids)).
        # Hence, in order to obtain balanced sampling we take only len(ws_pos_cands) samples from self.neg_preds_ids
        ws_pos_cands = set([item[0] for item in pos_sorted_items[:round(self.k / 2)]])
        if len(ws_pos_cands) > 0:
            ws_neg_cands = set([item[0] for item in neg_sorted_items[:len(ws_pos_cands)]])
        else:
            ws_neg_cands = set()
        return ws_pos_cands, ws_neg_cands

    def calc_DTAL_ws(self, ws_candidates):
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
