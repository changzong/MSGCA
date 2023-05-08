import argparse
import torch
import torch.utils.data as data
import random
import datetime as dt
from data_process import *
from model import Model

import pdb

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--dataset", type=str, choices=['inno_stock','dblp_paper'], default='dblp_paper')
    parser.add_argument("--predict_date", type=str, choices=['2022-10-10','2008-01-01'], default='2008-01-01')
    parser.add_argument("--word2vec", type=str, default='wordvec_dict.pkl')
    parser.add_argument("--lm_name", type=str, choices=['bert-base-chinese','bert-base-uncased'], default='bert-base-uncased')
    parser.add_argument("--sample_ratio", type=float, default=0.8)
    parser.add_argument("--date_move_steps", type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--epoch_num", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=300)
    parser.add_argument("--test_batch_size", type=int, default=76)
    parser.add_argument("--input_ind_dim", type=int, default=1)
    parser.add_argument("--input_doc_dim", type=int, default=300)
    parser.add_argument("--input_graph_dim", type=int, default=768) # same with LM hidden size
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--node_init_lm", action='store_true', default=True)
    parser.add_argument("--direction_type", type=str, choices=['st', 'ts', 'bi'], default='ts')
    parser.add_argument("--source_fusion_type", type=str, choices=['cat','trans','expert'], default='cat')
    parser.add_argument("--ts_fusion_type", type=str, choices=['trans', 'rnn'], default='trans')
    parser.add_argument("--all_fusion_type", type=str, choices=['mlp', 'trans_mlp', 'rgcn_mlp'], default='mlp')
    parser.add_argument("--fusion_dim", type=int, default=64)
    parser.add_argument("--score_dim", type=int, default=3)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--use_spike_predictor', action='store_true', default=False)
    args = parser.parse_args()
    return args

def train_model(args, input_data, label, sources):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    print("Sampling training and test data...")
    idxs = list(range(len(input_data[0]))) # sample ids
    train_idxs = random.sample(idxs, int(len(input_data[0])*args.sample_ratio))
    test_idxs = list(set(idxs)^set(train_idxs))
    
    # Use multiple predict dates for generating more samples
    predict_date = dt.datetime.strptime(args.predict_date, '%Y-%m-%d')
    predict_dates = [predict_date]
    for i in range(args.date_move_steps):
        predict_date += dt.timedelta(days=10)
        predict_dates.append(predict_date)
    # Generating samples
    train_set = []
    test_set = []
    train_label = []
    test_label = []

    print("Loading word vectors...")
    with open(args.data_path + args.word2vec, 'rb') as f:
        wordvec_dict = pickle.load(f)

    for date in predict_dates:
        train_subset = [build_input(date, input_data[i], train_idxs, 'lt') for i in range(len(input_data[:-1]))]
        train_set.extend(time_aligner(train_subset, wordvec_dict, args))
        test_subset = [build_input(date, input_data[i], test_idxs, 'lt') for i in range(len(input_data[:-1]))]
        test_set.extend(time_aligner(test_subset, wordvec_dict, args))
        train_label.extend(build_input(date, label, train_idxs, 'gt'))
        test_label.extend(build_input(date, label, test_idxs, 'gt'))

    # make each sample have the same time length
    time_span_train = len(train_set[0][0]) # entity * source * <timestamp> * dim
    tmp_train = []
    tmp_test = []
    for source in train_set:
        tmp1 = []
        for item in source:
            tmp1.append(item[-time_span_train:])
        tmp_train.append(tmp1)
    for source in test_set:
        tmp2 = []
        for item in source:
            tmp2.append(item[-time_span_train:])
        tmp_test.append(tmp2)
    train_set = tmp_train
    test_set = tmp_test
    
    # prepare graph structure input and node indexs
    print("Processing graph data...")
    graph_input = None
    if args.dataset == 'inno_stock':
        graph_input = graph_process_static(args, input_data[-1], time_span_train)
    elif args.dataset == 'dblp_paper':
        graph_input = graph_process_dynamic(args, input_data[-1], time_span_train)

    node_train_idxs = train_idxs * (args.date_move_steps+1)
    node_test_idxs = test_idxs * (args.date_move_steps+1)

    # load model and train
    model = Model(args, sources, device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(args.epoch_num):
        train_loss = 0.0
        test_loss = 0.0
        batch_num = int(len(train_set) / args.train_batch_size)
        model.train()
        print("Epoch:{}".format(epoch))
        for i in range(batch_num):
            start = i * args.train_batch_size
            end = (i+1) * args.train_batch_size
            if end > len(train_set):
                end = len(train_set)
            loss = model(train_set[start: end], graph_input, node_train_idxs[start: end], train_label[start: end], 'train')
            print('Epoch: %s Batch %s Training loss: %s' % (str(epoch), str(i), str(loss.item())))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / batch_num
        print("Epoch: {}, Training loss:{}".format(epoch, train_loss))

        print("Evaluating for epoch: {}".format(epoch))
        with torch.no_grad():
            model.eval()
            acc, mcc = model(test_set, graph_input, node_test_idxs, test_label, 'test')
            print('Test Accuracy: %s' % str(acc))
            print('Test Matthews Correlation Coefficient: %s' % str(mcc))


args = set_args()
# doc should be the first, then indicator types, graph should be the last
sources = ['document', 'indicator', 'graph']
input_data, label = load_data(args, sources)
train_model(args, input_data, label, sources)
