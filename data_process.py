import torch
import numpy as np
import scipy as sp
import networkx as nx
from datetime import datetime
import pickle

import pdb

def load_data(args):
    with open(args.data_path+args.dataset+'/doc_date_input.pkl', 'rb') as f:
        doc_date_input = pickle.load(f)
    with open(args.data_path+args.dataset+'/graph_input.pkl', 'rb') as f:
        graph_input = pickle.load(f)
    with open(args.data_path+args.dataset+'/indicator_day_input.pkl', 'rb') as f:
        ind_day_input = pickle.load(f)
    with open(args.data_path+args.dataset+'/indicator_quarter_input.pkl', 'rb') as f:
        ind_quar_input = pickle.load(f)
    with open(args.data_path+args.dataset+'/trend_daily_label.pkl', 'rb') as f:
        trend_daily_label = pickle.load(f)
    return [doc_date_input, ind_day_input, ind_quar_input, graph_input], trend_daily_label

def build_input(predict_date, data, idxs, op):
    data_set = []
    for idx in idxs:
        com_tmp = []
        for stamp in data[idx]:
            if op == 'lt':
                if datetime.strptime(stamp[1], '%Y-%m-%d') <= predict_date:
                    com_tmp.append(stamp)
            elif op == 'gt':
                if datetime.strptime(stamp[1], '%Y-%m-%d') > predict_date:
                    com_tmp.append(stamp)
        data_set.append(com_tmp)
    return data_set

def time_aligner(data_set):
    time_dict_data = [] # [source, entity, timestamp:value]
    for source in data_set:
        entity_tmp = []
        for entity in source:
            time_dict = {}
            for timestamp in entity:
                time_dict[timestamp[1]]=timestamp[0]
            entity_tmp.append(time_dict)
        time_dict_data.append(entity_tmp)

    timestamps = list(time_dict_data[1][0].keys())

    aligned_data = [] # [entity, timestamp, source, [value]]
    for idx in range(len(data_set[1])):
        time_tmp = []
        for stamp in timestamps:
            source_tmp = []
            for source in time_dict_data:
                if stamp in source[idx]:
                    source_tmp.append(source[idx][stamp])
                else:
                    source_tmp.append(None)
            time_tmp.append(source_tmp)
        aligned_data.append(time_tmp)

    return aligned_data

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def graph_process(args, input_graph):
    features = None
    adj_list = []
    for snapshot in input_graph:
        node_list = list(snapshot.nodes())
        if features == None:
            # initial embeddings can be replaced with language model output from description
            features = [torch.rand(args.input_graph_dim) for node in node_list]
            features = torch.stack(features)
        adj_sp_tensor = sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(snapshot))
        adj_list.append(adj_sp_tensor)
    return {'features': features, 'adj_list': adj_list}
