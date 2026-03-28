import gc
import warnings
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

from layer import ADC_GCN, GCN
from utils import accuracy
from utils import macro_f1
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import parameter_parser
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed

th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    target_dict = {} # Stores the label for each specific node ID
    with read_file(target_fn, mode="r") as fin:
        for item in fin:
            parts = item.strip().split("\t")
            if len(parts) < 3:
                continue
            node_id = int(parts[0])
            split = parts[1]
            label = int(parts[2])
            
            target_dict[node_id] = label
            if split in {"train", "training"}:
                train_lst.append(node_id)
            else:
                test_lst.append(node_id)

    return train_lst, test_lst, target_dict

class PrepareData:
    def __init__(self, args):
        print("prepare data")
        self.graph_path = "data/graph"
        self.args = args

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
        print_graph_detail(graph)
        
        # Compatible with NetworkX 3.x and NumPy 1.24+
        adj = nx.to_scipy_sparse_array(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # features
        self.nfeat_dim = graph.number_of_nodes()
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = th.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(value)
        shape = th.Size(shape)

        self.features = th.sparse.FloatTensor(indices, values, shape)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target
        target_fn = f"data/text_dataset/{self.args.dataset}.txt"
        
        # Call the updated function to get IDs and label mapping
        self.train_lst, self.test_lst, target_dict = get_train_test(target_fn)

        # Initialize a full target array of size 44,110 with zeros
        self.target = np.zeros(self.nfeat_dim, dtype=int)
        
        # Populate labels ONLY for the nodes mentioned in malware.txt
        for node_id, label in target_dict.items():
            self.target[node_id] = label
            
        self.nclass = len(set(target_dict.values()))

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split
        # train_lst and test_lst are now populated with the correct Node IDs


class TextGCNTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device

        self.max_epoch = self.args.max_epoch
        self.set_seed()

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)

    def set_seed(self):
        th.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        print(self.model.parameters)
        self.model = self.model.to(self.device)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # Option A: Weighted Cross-Entropy Loss
        # Assuming class 0 = Benign (97%) and class 1 = Malware (3%)
        # We give Malware a much higher weight (35.0) to force the model to learn it.
        weights = th.tensor([1.0, 35.0]).to(self.device)
        self.criterion = th.nn.CrossEntropyLoss(weight=weights)

        self.model_param = sum(param.numel() for param in self.model.parameters())
        print('# model parameters:', self.model_param)
        self.convert_tensor()

        start = time()
        self.train()
        self.train_time = time() - start

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        print(string)

    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.target = self.predata.target
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.target = th.tensor(self.target).long().to(self.device)
        self.train_lst = th.tensor(self.train_lst).long().to(self.device)
        self.val_lst = th.tensor(self.val_lst).long().to(self.device)

    def train(self):
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_lst)

            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"], model=self.model):
                break

    @th.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        with th.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            f1, precision, recall = macro_f1(logits[x],
                                             self.target[x],
                                             num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc"           : acc,
                "macro_f1"      : f1,
                "precision"     : precision,
                "recall"        : recall,
            }
        return desc

    @th.no_grad()
    def test(self):
        self.test_lst = th.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(self.test_lst, prefix="test")
        test_desc["train_time"] = self.train_time
        test_desc["model_param"] = self.model_param
        return test_desc


def main(dataset, times):
    args = parameter_parser()
    args.dataset = dataset

    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.nhid = 200
    args.max_epoch = 300
    args.dropout = 0.5
    args.val_ratio = 0.002
    args.early_stopping = 30
    args.lr = 0.01
    model = ADC_GCN

    print(args)

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    for ind, seed in enumerate(return_seed(times)):
        print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        if th.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(framework.test())

        del framework
        gc.collect()

        if th.cuda.is_available():
            th.cuda.empty_cache()

    print("==> seed set:")
    print(seed_lst)
    record.show_str()


if __name__ == '__main__':
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for d in ["mr", "ohsumed", "R52", "R8", "20ng"]:
    #     main(d)
    # Change from main("R52", 1) to:
    main("malware", 1)
    # main("ohsumed")
    # main("R8", 1)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
