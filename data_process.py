import os
import torch
import numpy as np
import scipy as sp
import networkx as nx
from datetime import datetime
import pickle
from transformers import AutoTokenizer, AutoModel, BertTokenizer, ErnieModel
from ts_to_prompt import *
import openai

import pdb

openai.api_key = "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"
llm_name = 'text-embedding-ada-002'

def load_data(args, sources):
    input_data = []
    if 'document' in sources:
        with open(args.data_path+args.dataset+'/doc_input.pkl', 'rb') as f:
            doc_date_input = pickle.load(f)
        input_data.append(doc_date_input)
    if 'indicator' in sources:
        with open(args.data_path+args.dataset+'/indicator_input.pkl', 'rb') as f:
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

    with open(args.data_path+args.dataset+'/trend_label.pkl', 'rb') as f:
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

def time_aligner(data_set, args, date, mode):
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
    for idx in range(len(data_set[1])): # stock price should be exist for every stock/publication 
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
    doc_ind_init_emb_file = args.data_path + args.dataset + '/cache/init_emb_' + date.strftime('%Y%m%d') + '_' + mode + '.pkl'
    model = None
    tokenizer = None
    if args.dataset == 'inno_stock':
        tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-nano-zh")
        model = ErnieModel.from_pretrained("nghuyong/ernie-3.0-nano-zh")
    elif args.dataset == 'bd22_stock':
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    else:
        raise Exception('Incorrect dataset!')
    aligned_data = vector_initialize(doc_ind_init_emb_file, aligned_data, args.input_doc_dim, tokenizer, model)
    return aligned_data

def vector_initialize(file_name, data_set, doc_vec_dim, tokenizer, model):
    final_input = None
    # embeddings from lm already stored in file, load it
    if os.path.isfile(file_name):
        print('Initial embeddings already exist, load it.')
        with open(file_name, 'rb') as f:
            final_input = pickle.load(f)
    else:
        print('Get initial embeddings from scratch.')
        final_input = []
        for item in data_set:
            new_timestamp = []
            for timestamp in item:
                source_vec = []
                if timestamp[0]:
                    doc_vec = get_emb_from_lm(model, tokenizer, timestamp[0]) # 128
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
            print('.', end=' ', flush=True)
        # dump to init_emb.pkl file for reuse
        with open(file_name, 'wb') as f:
            pickle.dump(final_input, f)
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

def graph_process_static(args, input_graph):
    features = None
    adj_list = []
    for snapshot in input_graph[:2]:
        if features is None:
            node_list = list(snapshot.nodes())
            features = []
            # randomly initialize node embeddings
            features = [torch.rand(args.input_graph_dim) for node in node_list]
            features = torch.stack(features)
        adj_sp_tensor = sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(snapshot))
        adj_list.append(adj_sp_tensor)
    output = [{'features': features, 'adj_list': adj_list}] # 1 * dict
    return output

def get_emb_from_lm(model, tokenizer, input):
    input_ids = tokenizer.encode(input)
    inputs_tensor = torch.tensor([input_ids])
    last_hidden_states = model(inputs_tensor)[0] # 1 * seq_len * emb_size
    emb = torch.squeeze(torch.mean(last_hidden_states, 1), 0)
    return emb

def get_emb_from_llm(args, time_length, idxs, mode):
    embeddings = None
    llm_emb_file = args.data_path + args.dataset + '/cache/llm_emb_' + mode + '.pkl'
    # embeddings from llm already stored in file, load it
    if os.path.isfile(llm_emb_file):
        print('LLM embeddings already exist, load it.')
        with open(llm_emb_file, 'rb') as f:
            embeddings = pickle.load(f)
        embeddings = embeddings * (args.date_move_steps+1) # repeat to be same length as training and test set
    else:
        print('Get LLM embeddings from scratch.')
        embeddings = []
        prompts = indicator_ts_to_text_prompt(args.data_path+args.dataset+'/stock_price.pkl', time_length, idxs)
        for text in prompts:
            text = text.replace("\n", " ")
            emb = openai.Embedding.create(input = [text], model=llm_name)['data'][0]['embedding'] # 1536 dimension
            embeddings.append(emb)
            print('.', end=' ', flush=True)
        with open(llm_emb_file, 'wb') as f:
            pickle.dump(embeddings, f)
        embeddings = embeddings * (args.date_move_steps+1) # repeat to be same length as training and test set
    print('\n')
    return torch.FloatTensor(embeddings) # entity * 1536
