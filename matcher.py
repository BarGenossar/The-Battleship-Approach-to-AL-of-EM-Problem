import torch
import torch.nn as nn
import os
import numpy as np
import random
import json
import jsonlines
import csv
import re
import time
import argparse
import sys
import traceback

from torch.utils import data
from tqdm import tqdm
from apex import amp
from scipy.special import softmax

sys.path.insert(0, "Snippext_public")
from snippext.model import MultiTaskNet
from ditto.exceptions import ModelNotFoundError
from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *


def to_str(row, summarizer=None, max_len=256, dk_injector=None):
    """Serialize a data entry

    Args:
        row (Dictionary): the data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """
    # if the entry is already serialized
    if isinstance(row, str):
        return row
    content = ''
    for attr in row.keys():
        content += 'COL %s VAL %s ' % (attr, row[attr])

    if summarizer is not None:
        content = summarizer.transform(content, max_len=max_len)

    if dk_injector is not None:
        content = dk_injector.transform(content)

    return content


def classify(sentence_pairs, config, model, file_type, seed, lm='distilbert',
             max_len=256, test_poolers=True, trained_model=None):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of tuples of str): the sentence pairs
        config (dict): the model configuration
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length

    Returns:
        list of float: the scores of the pairs
    """
    inputs = []
    for (sentA, sentB) in sentence_pairs:
        inputs.append(sentA + '\t' + sentB)

    dataset = DittoDataset(inputs, config['vocab'], config['name'], seed, lm=lm, max_len=max_len)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=16,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)

    # prediction
    Y_logits = []
    Y_poolers = []
    Y_hat = []
    with torch.no_grad():
        # print('Classification')
        if file_type == 'train' or test_poolers:
            for i, batch in enumerate(iterator):
                words, x, is_heads, tags, mask, y, seqlens, taskname = batch
                if trained_model:
                    intent = taskname[0][-1]
                    taskname = trained_model.split('/')
                    taskname = taskname[0] + '/' + taskname[1] + str(intent)
                else:
                    taskname = taskname[0]
                logits, _, y_hat, poolers = model(x, y, task=taskname, get_enc=True)  # y_hat: (N, T)
                poolers = poolers.cpu().numpy().tolist()
                poolers = [[round(elem, 4) for elem in tensor] for tensor in poolers]
                Y_logits += logits.cpu().numpy().tolist()
                Y_poolers += poolers
                Y_hat.extend(y_hat.cpu().numpy().tolist())
        else:
            for i, batch in enumerate(iterator):
                words, x, is_heads, tags, mask, y, seqlens, taskname = batch
                taskname = taskname[0]
                logits, _, y_hat, _ = model(x, y, task=taskname, get_enc=True)  # y_hat: (N, T)
                Y_logits += logits.cpu().numpy().tolist()
                Y_hat.extend(y_hat.cpu().numpy().tolist())
    results = []
    for i in range(len(inputs)):
        pred = dataset.idx2tag[Y_hat[i]]
        results.append(pred)

    return results, Y_logits, Y_poolers


def predict(input_path, output_path, output_path_file, config, model, file_type, seed,
            intent=1,
            batch_size=1024,
            summarizer=None,
            lm='distilbert',
            max_len=256,
            dk_injector=None,
            trained_model=None):

    pairs = []
    def process_batch(rows, pairs, writer, trained_model=None, test_poolers=True):
        try:
            predictions, logits, poolers = classify(pairs, config, model, file_type,
                                                    seed, lm=lm, max_len=max_len,
                                                    trained_model=trained_model)
        except:
            # ignore the whole batch
            return
        scores = softmax(logits, axis=1)
        if file_type == 'train' or test_poolers:
            for row, pred, score, pooler in zip(rows, predictions, scores, poolers):
                output = {'left': row[0], 'right': row[1],
                          'match': pred,
                          'match_confidence': round(score[int(pred)], 4),
                          'pooler': pooler}
                writer.write(output)
        else:
            for row, pred, score in zip(rows, predictions, scores):
                output = {'left': row[0], 'right': row[1],
                          'match': pred,
                          'match_confidence': round(score[int(pred)], 4)}
                writer.write(output)

    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    # input_path = input_path.replace('.txt', str(intent) + ".txt")
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    # batch processing
    start_time = time.time()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with jsonlines.open(input_path) as reader, \
            jsonlines.open(output_path_file, mode='w') as writer:
        pairs = []
        rows = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append((to_str(row[0], summarizer, max_len, dk_injector),
                          to_str(row[1], summarizer, max_len, dk_injector)))
            rows.append(row)
            if len(pairs) == batch_size:
                process_batch(rows, pairs, writer, trained_model)
                pairs.clear()
                rows.clear()

        if len(pairs) > 0:
            process_batch(rows, pairs, writer, trained_model)

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], lm, str(dk_injector != None), str(summarizer != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))


def load_model(task, path, lm, use_gpu, seed, intent, fp16=True, trained_model=None):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models
    model_name = task.split('/')[1]
    full_path = task[:-1] + '/' + model_name

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    if trained_model:
        checkpoint = os.path.join(path, '%s.pt' % trained_model)
    else:
        checkpoint = os.path.join(path, '%s.pt' % full_path)
    if not os.path.exists(checkpoint):
        raise ModelNotFoundError(checkpoint)



    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    if trained_model:
        tmp_config = trained_model.split('/')
        tmp_config = tmp_config[0] + '/' + tmp_config[1] + str(intent)
        model = MultiTaskNet([configs[tmp_config]], seed, device, True, lm=lm)
    else:
        model = MultiTaskNet([config], seed, device, True, lm=lm)
    saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state)

    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return config, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/Walmart-Amazon')
    parser.add_argument("--input_path", type=str, default='data/er_magellan/Structured/Walmart-Amazon/')
    parser.add_argument("--output_path", type=str, default='output/er_magellan/Structured/Walmart-Amazon/Walmart-Amazon/')
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/')
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--intents_num", type=int, default=1)
    parser.add_argument("--iter_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--training_type", type=str, default="active_learning")
    parser.add_argument("--with_intents", type=int, default=1)
    parser.add_argument("--trained_model", type=str, default=None)
    parser.add_argument("--mode", type=str, default="battleships_no_ws_b_alpha=0.27")

    hp = parser.parse_args()

    main_task = hp.task
    iteration = hp.iter_num
    training_type = hp.training_type
    seed = hp.seed
    trained_model = hp.trained_model
    output_path = hp.output_path + str(hp.mode) + '/'

    if training_type == "active_learning":
        file_types = ['train', 'test']
    else:
        file_types = ['test']
    for intent in range(hp.intents_num):
        if "dummy" in hp.mode:
            break
        for file_type in file_types:
            task = main_task + str(intent)
            task_name = task.split('/')[1]
            # load the models
            config, model = load_model(task, hp.checkpoint_path,
                                       hp.lm, hp.use_gpu, seed,
                                       intent, hp.fp16, trained_model)

            if file_type == 'train':
                if training_type == 'active_learning':
                    input_path_available_pool = hp.input_path + '/available_pool.txt'
                    input_path_current_train = hp.input_path + '/current_train.txt'
                    input_path_files = [input_path_available_pool, input_path_current_train]
                else:
                    input_path_files = [hp.input_path + '/train.txt']
            else:
                input_path_files = [config['testset']]

            if training_type == 'active_learning':
                if file_type == 'train':
                    output_path_available_pool = output_path + task_name[:-1] + \
                                                 '_available_pool' + str(intent) + \
                                                 '_iter' + str(iteration) + '_' + file_type + '_output_seed' + \
                                                 str(seed) + '.txt'

                    output_path_current_train = output_path + task_name[:-1] + \
                                                '_current_train' + str(intent) + '_iter' + \
                                                str(iteration) + '_' + file_type + '_output_seed' + \
                                                str(seed) + '.txt'
                    output_path_files = [output_path_available_pool, output_path_current_train]
                else:
                    output_path_files = [output_path + task_name[:-1] +
                                         str(intent) + '_iter' + str(iteration) +
                                         '_' + file_type + '_output_seed' +
                                         str(seed) + '.txt']
            else:
                output_path_files = [output_path + task_name[:-1] +
                                     '_full' + str(intent) + '_' + file_type +
                                     '_output_seed' + str(seed) + '.txt']

            summarizer = dk_injector = None
            if hp.summarize:
                summarizer = Summarizer(config, hp.lm)

            if hp.dk is not None:
                if 'product' in hp.dk:
                    dk_injector = ProductDKInjector(config, hp.dk)
                else:
                    dk_injector = GeneralDKInjector(config, hp.dk)

            # run prediction
            for input_path, output_path_file in zip(input_path_files, output_path_files):
                predict(input_path, output_path, output_path_file, config, model, file_type, seed, intent,
                        summarizer=summarizer,
                        max_len=hp.max_len,
                        lm=hp.lm,
                        dk_injector=dk_injector,
                        trained_model=trained_model)
