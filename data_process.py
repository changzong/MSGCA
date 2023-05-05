import os
import torch
import numpy as np
import scipy as sp
import networkx as nx
from datetime import datetime
import pickle
import jieba
from transformers import AutoTokenizer, AutoModel

import pdb

def load_data(args, sources):
    input_data = []
    if 'document' in sources:
        with open(args.data_path+args.dataset+'/doc_date_input.pkl', 'rb') as f:
            doc_date_input = pickle.load(f)
        input_data.append(doc_date_input)
    if 'indicator-daily' in sources:
        with open(args.data_path+args.dataset+'/indicator_day_input.pkl', 'rb') as f:
            ind_day_input = pickle.load(f)
        input_data.append(ind_day_input)
    if 'indicator-quarter' in sources:
        with open(args.data_path+args.dataset+'/indicator_quarter_input.pkl', 'rb') as f:
           ind_quar_input = pickle.load(f)
        input_data.append(ind_quar_input)
    if 'graph' in sources:
        with open(args.data_path+args.dataset+'/graph_input.pkl', 'rb') as f:
            graph_input = pickle.load(f)
        input_data.append(graph_input)

    with open(args.data_path+args.dataset+'/trend_daily_label.pkl', 'rb') as f:
        trend_daily_label = pickle.load(f)
    return input_data, trend_daily_label

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

def time_aligner(data_set, word2vec, args):
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
    for idx in range(len(data_set[1])): # stock price/citation count should be exist for every stock/publication 
        time_tmp = []
        for stamp in timestamps:
            source_tmp = []
            for source in time_dict_data:
                if stamp in source[idx]:
                    source_tmp.append(source[idx][stamp][0]) # just get the first value for each list for now, for docs, we can combine them; for indicators, we need to test each of them
                else:
                    source_tmp.append(None)
            time_tmp.append(source_tmp)
        aligned_data.append(time_tmp)

    # initializing input: entity * source * timestamp * dim
    aligned_data = vector_initialize(aligned_data, word2vec, args.input_doc_dim)
    return aligned_data

def vector_initialize(data_set, wordvec, doc_vec_dim):
    final_input = []
    for item in data_set:
        new_timestamp = []
        for timestamp in item:
            source_vec = []
            if timestamp[0]:
                doc_vecs = []
                words = jieba.cut(timestamp[0])
                for word in words:
                    if word in wordvec:
                        doc_vecs.append(wordvec[word])
                doc_vec = torch.mean(torch.FloatTensor(doc_vecs), 0)
            else:
                doc_vec = torch.zeros(doc_vec_dim)
            source_vec.append(doc_vec)
            for ind_value in timestamp[1:]:
                if ind_value:
                    ind_vec = torch.FloatTensor([ind_value])
                else:
                    ind_vec = torch.FloatTensor([0.0])
                source_vec.append(ind_vec)
            new_timestamp.append(source_vec)
        final_input.append(new_timestamp)
    # switch dim 2 and dim 1 for faster process in gpu
    final_input_2 = []
    for item in final_input:
        item_tmp = []
        for i in range(len(item[0])):
            source_tmp = []
            for j in range(len(item)):
                source_tmp.append(item[j][i])
            item_tmp.append(torch.stack(source_tmp))
        final_input_2.append(item_tmp)
    return final_input_2
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def graph_process_static(args, input_graph, time_span):
    features = None
    adj_list = []
    for snapshot in input_graph[:2]:
        if features is None:
            node_list = list(snapshot.nodes())
            features = []
            # randomly initialize node embeddings
            if not args.node_init_lm:
                features = [torch.rand(args.input_graph_dim) for node in node_list]
                features = torch.stack(features)
            # initialize company node embeddings with language model
            else:
                node_init_emb_file = args.data_path + args.dataset + '/node_init_emb.pkl'
                # embeddings from lm already stored in file, load it
                if os.path.isfile(node_init_emb_file):
                    with open(node_init_emb_file, 'rb') as f:
                        node_init_emb = pickle.load(f)
                    for node in node_list:
                        if node in node_init_emb:
                            features.append(node_init_emb[node])
                        else:
                            features.append(torch.rand(args.input_graph_dim))
                # generate embeddings of each name from lm, and save to file
                else:
                    id_2_com_name = input_graph[2]
                    id_2_emb = {}
                    print("Generating node embeddings from LM: " + args.lm_name)
                    tokenizer = AutoTokenizer.from_pretrained(args.lm_name, trust_remote_code=True)
                    model = AutoModel.from_pretrained(args.lm_name, trust_remote_code=True)
                    model = model.eval()
                    for node in node_list:
                        if node in id_2_com_name:
                            node_emb = get_emb_from_lm(model, tokenizer, id_2_com_name[node])
                            features.append(node_emb)
                            id_2_emb[node] = node_emb
                        else:
                            features.append(torch.rand(args.input_graph_dim))
                    with open(node_init_emb_file, 'wb') as f:
                        pickle.dump(id_2_emb, f)
                features = torch.stack(features)
        adj_sp_tensor = sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(snapshot))
        adj_list.append(adj_sp_tensor)
    output = [{'features': features, 'adj_list': adj_list}] * time_span # timestamp * dict
    return output


def get_emb_from_lm(model, tokenizer, input):
    input_ids = tokenizer.encode(input)
    inputs_tensor = torch.tensor([input_ids])
    last_hidden_states = model(inputs_tensor)[0] # 1 * seq_len * emb_size
    emb = torch.squeeze(torch.mean(last_hidden_states, 1), 0)
    return emb
