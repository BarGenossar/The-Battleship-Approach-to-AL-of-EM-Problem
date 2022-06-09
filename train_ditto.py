import os
import argparse
import json
import sys

sys.path.insert(0, "Snippext_public")

from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Amazon-Google")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--training_type", type=str, default="active_learning")
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--balance", dest="balance", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--intents_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--with_intents", type=int, default=1)
    parser.add_argument("--mode", type=str, default=None)



    hp = parser.parse_args()

    # only a single task for baseline
    main_task = hp.task
    intent = hp.intent
    training_type = hp.training_type
    seed = hp.seed


    print("=======================start training=======================")
    for intent in range(hp.intents_num):
        if "dummy" in hp.mode:
            break
        # create the tag of the run
        if hp.with_intents == 1:
            task = main_task + str(intent)
        else:
            task = main_task
        run_tag = 'checkpoints/' + main_task + '/'

        # load task configuration
        configs = json.load(open('configs.json'))
        configs = {conf['name']: conf for conf in configs}
        config = configs[task]

        if training_type == "active_learning":
            trainset = config['current_train']
        else:
            trainset = config['trainset']

        orig_trainset = config['trainset']

        validset = config['validset']
        testset = config['testset']

        task_type = config['task_type']
        vocab = config['vocab']
        tasknames = [task]

        # summarize the sequences up to the max sequence length
        if hp.summarize:
            summarizer = Summarizer(config, lm=hp.lm)
            trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
            validset = summarizer.transform_file(validset, max_len=hp.max_len)
            testset = summarizer.transform_file(testset, max_len=hp.max_len)

        if hp.dk is not None:
            if hp.dk == 'product':
                injector = ProductDKInjector(config, hp.dk)
            else:
                injector = GeneralDKInjector(config, hp.dk)

            trainset = injector.transform_file(trainset)
            validset = injector.transform_file(validset)
            testset = injector.transform_file(testset)

        # load train/dev/test sets
        train_dataset = DittoDataset(trainset, vocab, task, seed,
                                       lm=hp.lm,
                                       max_len=hp.max_len,
                                       size=hp.size,
                                       balance=hp.balance)
        valid_dataset = DittoDataset(validset, vocab, task, seed, lm=hp.lm)
        test_dataset = DittoDataset(testset, vocab, task, seed, lm=hp.lm)

        if hp.da is None:
            from snippext.baseline import initialize_and_train
            initialize_and_train(config,
                                 train_dataset,
                                 valid_dataset,
                                 test_dataset,
                                 hp,
                                 run_tag,
                                 task.split('/')[1],
                                 seed)
        else:
            from snippext.mixda import initialize_and_train
            augment_dataset = DittoDataset(trainset, vocab, task, seed,
                                          lm=hp.lm,
                                          max_len=hp.max_len,
                                          augment_op=hp.da,
                                          size=hp.size,
                                          balance=hp.balance)
            initialize_and_train(config,
                                 train_dataset,
                                 augment_dataset,
                                 valid_dataset,
                                 test_dataset,
                                 hp,
                                 run_tag,
                                 task.split('/')[1],
                                 seed)

    print("=======================end train=======================")
