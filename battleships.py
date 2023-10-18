import numpy as np
import networkx as nx
from itertools import repeat
import torch
import re
from math import log2, ceil, sqrt
import random
import os
import pickle
from scipy import spatial
import multiprocessing
from collections import defaultdict
import faiss
from kneed import KneeLocator
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_score
import time



class battleships_graph:
    def __init__(self, poolers_paths, k, seed, files_path,
                 output_path, iteration, criterion='pagerank',
                 mode='top_k', alpha=0.5, beta=0.5, dim=768,
                 min_cc_ratio=0.1, max_cc_ratio=0.15,
                 nn_param=15, treat_weak_labels=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.poolers_paths = poolers_paths
        self.files_path = files_path
        self.output_path = output_path
        self.iter = iteration
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.dim = dim
        self.criterion = criterion
        self.mode = mode
        self.nn_param = nn_param
        self.treat_weak_labels = treat_weak_labels
        self.poolers, self.available_pool_size = self.create_poolers()
        self.normalized_poolers = self.normalize_poolers()
        self.weak_ids = self.find_weaks()
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
        self.pos_budget = min(max(round(self.k * (0.8 - 0.05 * iteration)), round(0.5 * self.k)), len(self.pos_preds_ids))
        self.min_cc_ratio = min_cc_ratio
        self.max_cc_ratio = max_cc_ratio
        self.neg_graph, self.neg_connected_components, self.neg_ccs_available_pool_sizes = self.cluster_and_graph(0)
        self.pos_graph, self.pos_connected_components, self.pos_ccs_available_pool_sizes = self.cluster_and_graph(1)
        self.het_graph, self.het_connected_components, self.het_ccs_available_pool_sizes = self.cluster_and_graph(2)
        # self.validate_connected_components()
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

    def normalize_poolers(self):
        normalized_poolers = dict()
        for pooler_id, pooler in self.poolers.items():
            normalized_poolers[pooler_id] = sqrt(sum(pooler ** 2))
        return normalized_poolers


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
            min_val = int(self.min_cc_ratio * len(self.poolers))
            max_val = int(self.max_cc_ratio * len(self.poolers))
        else:
            rel_ids = self.pos_preds_ids if label_type == 1 else self.neg_preds_ids
            min_val = int(self.min_cc_ratio * len(self.pos_preds_ids)) if label_type == 1 \
                else int(self.min_cc_ratio * len(self.neg_preds_ids))
            max_val = int(self.max_cc_ratio * len(self.pos_preds_ids)) if label_type == 1 \
                else int(self.max_cc_ratio * len(self.neg_preds_ids))
        return rel_ids, min_val, max_val

    def cluster_and_graph(self, label_type):
        rel_ids, min_val, max_val = self.find_rel_ids_min_max(label_type)
        suffix = str(label_type)
        clusters2poolers = self.create_clusters(rel_ids, min_val, max_val)
        graph = self.initialize_graph(rel_ids)
        graph = self.connect_nodes(graph, clusters2poolers, label_type)
        connected_components = self.create_connected_components(graph)
        # light_conncted_components = self.get_light_connected_components(connected_components)
        ccs_available_pool_sizes = self.calc_CCS_available_pool_sizes(connected_components)
        # self.save_to_pkl([clusters2poolers, light_conncted_components, ccs_available_pool_sizes],
        #                  ["clusters2poolers" + suffix, "connected_components(light)" + suffix,
        #                   "ccs_available_pool_sizes" + suffix])
        return graph, connected_components, ccs_available_pool_sizes

    def create_clusters(self, rel_ids, min_val, max_val):
        rel_poolers = np.array([pooler for pooler_id, pooler in self.poolers.items()
                                if pooler_id in rel_ids])
        if len(rel_poolers):
            ids_mapping = self.rel2orig(rel_ids)
            k, cluster_labels = self.find_optimal_k(rel_ids, rel_poolers, min_val, max_val)
        else:
            return dict()

        # kmeans_model = KMeansConstrained(n_clusters=k,
        #                                  size_min=min_val,
        #                                  size_max=max_val,
        #                                  random_state=kmeans_seeds_num+1).fit(rel_poolers)
        # cluster_labels = kmeans_model.labels_
        clusters2poolers = defaultdict(list)
        for ind, clus in enumerate(cluster_labels):
            clusters2poolers[clus].append(ids_mapping[ind])
        return clusters2poolers

    def find_optimal_k(self, rel_ids, rel_poolers, min_val, max_val):
        start = time.time()
        total_size = len(rel_ids)
        min_k = ceil(total_size / max_val)
        max_k = int(total_size / min_val)
        k_list = list(range(min_k, min(max_k + 1, min_k + 6)))
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            scores = list(pool.starmap(self.calc_k_means, zip(k_list,
                                                              repeat(rel_poolers),
                                                              repeat(min_val),
                                                              repeat(max_val))))
        sse_vals = [score_vals[1] for score_vals in scores]
        sil_vals = [score_vals[2] for score_vals in scores]
        cluster_labels = [score_vals[3] for score_vals in scores]
        kn = KneeLocator(k_list, sse_vals, curve='convex', direction='decreasing')
        end = time.time()
        total_time = round(end - start, 2)
        print(f"k means took : {total_time} seconds")
        if kn.knee is not None:
            return kn.knee, cluster_labels[k_list.index(kn.knee)]
        else:
            selected_ind = int(np.argmax(sil_vals))
            return k_list[selected_ind], cluster_labels[selected_ind]


    @staticmethod
    def calc_k_means(k, poolers, min_k, max_k):
        # print(f"k:{k}, max_k:{max_k}, min_k:{min_k}, size:{len(poolers)}")
        kmeans_model = KMeansConstrained(n_clusters=k,
                                         size_min=min_k,
                                         size_max=max_k,
                                         random_state=0).fit(poolers)
        clusters = kmeans_model.labels_
        sse_score = kmeans_model.inertia_
        sil_score = silhouette_score(poolers, clusters, metric='l2')
        return k, sse_score, sil_score, clusters

    @staticmethod
    def rel2orig(rel_ids):
        ids_mapping = dict()
        for curr_ind, pooler_id in enumerate(rel_ids):
            ids_mapping[curr_ind] = pooler_id
        return ids_mapping

    @staticmethod
    def create_connected_components(graph):
        graphs_dict = dict()
        connected_components = nx.connected_components(graph)
        for graph_id, cc in enumerate(connected_components):
            graphs_dict[graph_id] = graph.subgraph(cc)
        return graphs_dict

    # def validate_connected_components(self):
    #     light_conncted_components_pos = self.get_light_connected_components(self.pos_connected_components)
    #     light_conncted_components_neg = self.get_light_connected_components(self.neg_connected_components)
    #     self.save_to_pkl([light_conncted_components_pos, light_conncted_components_neg,
    #                       self.pos_connected_components, self.neg_connected_components],
    #                      ["final_connected_components(light1)", "final_connected_components(light0)",
    #                       "final_connected_components_pos", "final_connected_components_neg"])
    #     return

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
        if len(rel_graph_ids):
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

    def add_automatic_edges(self, pooler_id, edges_dict, neighbors, dists, bucket2orig):
        added = 0
        for neighbor, dist in zip(neighbors[0][1:], dists[0][1:]):
            if added >= self.nn_param:
                break
            orig_pooler, orig_neighbor = bucket2orig[pooler_id], bucket2orig[neighbor]
            if min(orig_neighbor, orig_pooler) < self.available_pool_size:
                denominator = self.normalized_poolers[orig_pooler] * self.normalized_poolers[orig_neighbor]
                edges_dict[(orig_neighbor, orig_pooler)] = max(dist / denominator, 0)
                added += 1
        return edges_dict

    def update_candidate_neighbors_dict(self, pooler_id, neighbors, dists, candidate_neighbors_dict, bucket2orig, edges_dict):
        for neighbor, dist in zip(neighbors[0][self.nn_param + 1:], dists[0][self.nn_param + 1:]):
            orig_pooler, orig_neighbor = bucket2orig[pooler_id], bucket2orig[neighbor]
            if min(orig_pooler, orig_neighbor) < self.available_pool_size and \
                    (orig_neighbor, orig_pooler) not in edges_dict.keys() and \
                    (orig_pooler, orig_neighbor) not in edges_dict.keys():
                denominator = self.normalized_poolers[orig_pooler] * self.normalized_poolers[orig_neighbor]
                candidate_neighbors_dict[(orig_neighbor, orig_pooler)] = max(dist / denominator, 0)
        return candidate_neighbors_dict

    def update_edges_and_candidates(self, pooler_vec, index, candidate_neighbors_size, pooler_id,
                                    edges_dict, candidate_neighbors_dict, bucket2orig):
        query_pooler = np.expand_dims(pooler_vec, axis=0)
        dists, neighbors = index.search(query_pooler, candidate_neighbors_size)
        edges_dict = self.add_automatic_edges(pooler_id, edges_dict, neighbors, dists, bucket2orig)
        candidate_neighbors_dict = self.update_candidate_neighbors_dict(pooler_id, neighbors, dists,
                                                                        candidate_neighbors_dict, bucket2orig, edges_dict)
        return edges_dict, candidate_neighbors_dict

    def process_candidates(self, candidate_neighbors_dict, edges_ratio, edges_dict):
        edges_limit = int(edges_ratio * len(candidate_neighbors_dict))
        counter = 0
        for pair in candidate_neighbors_dict.keys():
            if counter > edges_limit:
                break
            else:
                edges_dict[pair] = candidate_neighbors_dict[pair]
                counter += 1
        return edges_dict

    def create_bucket_edges(self, bucket_ids, label_type, edges_ratio=0.03):
        rel_poolers = np.array([self.poolers[pooler_id] for pooler_id in bucket_ids], dtype="float32")
        bucket2orig = {idx: pooler_id for idx, pooler_id in enumerate(bucket_ids)}
        d = len(self.poolers[0])
        index = faiss.IndexFlatIP(d)
        index.add(rel_poolers)
        candidate_neighbors_size = min(self.k, len(bucket_ids))
        edges_dict = dict()
        candidate_neighbors_dict = dict()
        for pooler_id, pooler_vec in enumerate(rel_poolers):
            edges_dict, candidate_neighbors_dict = self.update_edges_and_candidates(pooler_vec, index,
                                                                                    candidate_neighbors_size,
                                                                                    pooler_id, edges_dict,
                                                                                    candidate_neighbors_dict,
                                                                                    bucket2orig)
        if label_type < 2:
            candidate_neighbors_dict = {k: v for k, v in sorted(candidate_neighbors_dict.items(),
                                                                key=lambda item: item[1], reverse=True)}
            edges_dict = self.process_candidates(candidate_neighbors_dict, edges_ratio, edges_dict)
        weighted_edges = [(pair[0], pair[1], weight) for pair, weight in edges_dict.items()]
        return weighted_edges

    @staticmethod
    def create_final_edge_list(edges_list_per_bucket):
        final_edge = []
        for bucket_edges in edges_list_per_bucket:
            final_edge.extend(bucket_edges)
        return final_edge

    def connect_nodes(self, graph, buckets2poolers, label_type):
        start = time.time()
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() / 3) - 1) as pool:
            edges_per_bucket = list(pool.starmap(self.create_bucket_edges, zip(buckets2poolers.values(),
                                                                               repeat(label_type))))
        final_edge_list = self.create_final_edge_list(edges_per_bucket)
        graph.add_weighted_edges_from(final_edge_list)
        end = time.time()
        total_time = round(end - start, 2)
        print(f"Edge creation took: {total_time} seconds")
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
        start = time.time()
        pos_centrality = self.calc_centrality(1)
        neg_centrality = self.calc_centrality(0)
        pos_uncertainty, neg_uncertainty, votes_dict = self.calc_uncertainty()
        pos_selected = self.find_candidates(pos_centrality, pos_uncertainty, 1)
        neg_selected = self.find_candidates(neg_centrality, neg_uncertainty, 0)
        selected_k = pos_selected + neg_selected
        self.save_to_pkl([pos_centrality, neg_centrality, pos_uncertainty, neg_uncertainty, selected_k],
                         ["pos_centrality", "neg_centrality", "pos_uncertainty", "neg_uncertainty", "selected_k"])
        end = time.time()
        total_time = round(end - start, 2)
        print(f"Ranking took: {total_time} seconds")
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

    def calc_neighbors_uncertainty(self):
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
                final_entropy_dict[pooler_id] = self.beta * regular_entropy + \
                                                (1 - self.beta) * neighborhood_entropy
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
                pooler_rank = self.alpha * centrality_val + (1 - self.alpha) * uncertainty_val
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
            # Sometimes ws_pos_cands will be smaller than k/2 (as it is bounded by min(k/2, len(self.pos_preds_ids)).
            # Hence, in order to obtain balanced sampling we take only len(ws_pos_cands) samples from self.neg_preds_ids
            ws_pos_cands = set([item[0] for item in pos_sorted_items[:round(self.k / 2)]])
            if len(ws_pos_cands) > 0:
                ws_neg_cands = set([item[0] for item in neg_sorted_items[:len(ws_pos_cands)]])
            else:
                ws_neg_cands = set()
            return ws_pos_cands, ws_neg_cands
        elif "ws_b" in self.mode:
            return self.calc_battleships_ws(ws_candidates)
        else:
            return {}, {}  # Without weak supervision



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
