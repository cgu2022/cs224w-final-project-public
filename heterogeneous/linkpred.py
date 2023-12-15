import argparse
import copy
import time

import random
import pickle
import numpy as np
import networkx as nx 
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn

from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from deepsnap.hetero_gnn import (
    HeteroSAGEConv,
    HeteroConv,
    forward_op
)

random.seed(0)

def generate_2convs_link_pred_layers(hete, conv, hidden_size):
    convs1 = {}
    convs2 = {}
    for message_type in hete.message_types:
        n_type = message_type[0]
        s_type = message_type[2]
        n_feat_dim = hete.num_node_features(n_type)
        s_feat_dim = hete.num_node_features(s_type)
        convs1[message_type] = conv(n_feat_dim, hidden_size, s_feat_dim)
        convs2[message_type] = conv(hidden_size, hidden_size, hidden_size)
    return convs1, convs2

def arg_parse():
    parser = argparse.ArgumentParser(description='Link pred arguments.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--data_path', type=str,
                        help='Path to wordnet nx gpickle file.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--mode', type=str,
                        help='Link prediction mode. Disjoint or all.')
    parser.add_argument('--edge_message_ratio', type=float,
                        help='Ratio of edges used for message-passing (only in disjoint mode).')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--lr', type=float,
                        help='The learning rate.')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay.')
    parser.add_argument('--node_embedding_dim', type=int,
                        help='Dimension of node embeddings')

    parser.set_defaults(
            device='cuda:0',
            data_path='data/oncourse.pkl',
            epochs=200, # originally 200
            mode='disjoint',
            edge_message_ratio=0.8,
            hidden_dim=32,
            lr=0.01,  # originally 0.01
            weight_decay=1e-4,
            node_embedding_dim=32,  # originally 5
    )
    return parser.parse_args()


def WN_transform(G, num_edge_types, input_dim):
    H = nx.MultiDiGraph()
    
    for node in G.nodes():
        H.add_node(node, node_type=G.nodes[node]['node_type'], node_feature=torch.ones(input_dim))
    
    for u, v, edge_key in G.edges:
        l = G[u][v][edge_key]['e_label']
        e_feat = torch.zeros(num_edge_types*2)
        e_feat[l] = 1.
        H.add_edge(u, v, edge_feature=e_feat, edge_type=str(l.item()))
        
        # add an edge going the opposite direction
        e_feat_ = torch.zeros(num_edge_types*2)
        e_feat_[l+num_edge_types]
        H.add_edge(v, u, edge_feature=e_feat_, edge_type=str((l+num_edge_types).item()))
    return H


class HeteroGNN(torch.nn.Module):
    def __init__(self, conv1, conv2, hetero, hidden_size, node_embed_size=32):
        super(HeteroGNN, self).__init__()
        
        self.convs1 = HeteroConv(conv1) # Wrap the heterogeneous GNN layers
        self.convs2 = HeteroConv(conv2)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        # self.relus2 = nn.ModuleDict()
        # self.post_mps = nn.ModuleDict()
        self.edge_mlp = nn.ModuleDict()

        for node_type in hetero.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(hidden_size)
            self.bns2[node_type] = torch.nn.BatchNorm1d(hidden_size)
            self.relus1[node_type] = nn.LeakyReLU()
            # self.relus2[node_type] = nn.LeakyReLU()
        
        
        for edge_type in hetero.edge_types:
            self.edge_mlp[edge_type] = nn.Linear(node_embed_size * 2, 1)

    def forward(self, data):
        '''
        node_feature: dictionary of {node type (e.g. 'user') : tensor of all embeddings of that node type (# num nodes, 32)}
        edge_index: dictionary of {edge type (e.g. ('user', '0', 'course')) : tensor of all edges (2, # edges)}
        edge_label_index: same structure as edge_index. this indicates the edges to be evaluated on
        '''
        if type(data) == Batch:
            x = data.node_feature
            edge_index = data.edge_index
            edge_label_index = data.edge_label_index  # dictionary 
        elif type(data) == dict:
            x = data['node_feature']
            edge_index = data['edge_index']
            edge_label_index = data['edge_label_index']
        else:
            raise NotImplementedError("Unknown data format")
        
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)

        pred = {}
        for message_type in edge_label_index:
            nodes_first = torch.index_select(x[message_type[0]], 0, edge_label_index[message_type][0,:].long())  # shape [10134, 32]
            nodes_second = torch.index_select(x[message_type[2]], 0, edge_label_index[message_type][1,:].long())
            pred[message_type] = self.edge_mlp[message_type[1]](torch.cat((nodes_first, nodes_second), dim=-1)).reshape(-1) # shape [10134]
        return pred

    def loss(self, pred, y):
        loss = 0
        for key in pred:
            p = torch.sigmoid(pred[key])
            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
        return loss
    
    def bpr_loss(self, pred, y):
        loss = 0
        epsilon = 1e-7
        for key in pred:
            dim = pred[key].shape[0]
            assert dim % 2 == 0
            assert torch.equal(y[key][:dim//2], torch.ones_like(y[key][:dim//2]))
            assert torch.equal(y[key][dim//2:], torch.zeros_like(y[key][dim//2:]))
            p = pred[key]
            positive = p[:dim//2]
            negative = p[dim//2:]
            loss += -torch.sum(torch.log(torch.sigmoid(positive - negative) + epsilon))
        return loss

def add_negative_samples(batch):
    positive_edges = {}
    for message_type in batch.edge_label_index:
        positive_edges[message_type] = set()
        for edge in batch.edge_index[message_type].T:
            positive_edges[message_type].add((edge[0].item(), edge[1].item()))
        for edge in batch.edge_label_index[message_type].T:
            positive_edges[message_type].add((edge[0].item(), edge[1].item()))
            
    new_edge_label_index = {}
    new_labels = {}
    for message_type in batch.edge_label_index:
        num_users = batch.node_feature[message_type[0]].shape[0]
        num_courses = batch.node_feature[message_type[2]].shape[1]
        
        positive_indices = torch.nonzero(batch.edge_label[message_type]).squeeze()
        num_positive_samples = positive_indices.shape[0]
        positive_edge_index = batch.edge_label_index[message_type][:, positive_indices]
        
        new_edge_label_index[message_type] = torch.cat((positive_edge_index, torch.zeros_like(positive_edge_index)), dim=-1)
        for i in range(num_positive_samples, 2*num_positive_samples):  # use the same number of negative edges as positive edges
            user, random_course = positive_edge_index[0][i-num_positive_samples], -1
            while True:
                random_course = random.randrange(0, num_courses)
                if (user, random_course) not in positive_edges[message_type]:
                    break
            new_edge_label_index[message_type][0][i] = user
            new_edge_label_index[message_type][1][i] = random_course
        new_labels[message_type] = torch.cat((batch.edge_label[message_type][positive_indices], torch.zeros_like(batch.edge_label[message_type][positive_indices])))
    
    new_batch = {
        'node_feature': batch.node_feature,
        'edge_index': batch.edge_index,
        'edge_label_index': new_edge_label_index
    }
    
    return new_batch, new_labels

def train(model, dataloaders, optimizer, args):
    val_max = 0
    best_model = model
    t_accu = []
    v_accu = []
    e_accu = []
    for epoch in range(1, args.epochs + 1):
        for iter_i, batch in enumerate(dataloaders['train']):
            batch.to(args.device)
            new_batch, new_labels = add_negative_samples(batch)
            model.train()
            optimizer.zero_grad()
            pred = model(new_batch)
            loss = model.bpr_loss(pred, new_labels)
            loss.backward()
            optimizer.step()

            log = 'Epoch: {:03d}, Train loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            accs = test_metrics(model, dataloaders, args)
            t_accu.append(accs['train'])
            v_accu.append(accs['val'])
            e_accu.append(accs['test'])

            print(log.format(epoch, loss.item(), accs['train'], accs['val'], accs['test']))
            if val_max < accs['val']:
                val_max = accs['val']
                best_model = copy.deepcopy(model)

    log = 'Best: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    accs = test(best_model, dataloaders, args)
    print(log.format(accs['train'], accs['val'], accs['test']))
    torch.save(best_model, "model/hetero.pt")

    return t_accu, v_accu, e_accu


def test(model, dataloaders, args):
    model.eval()
    accs = {}
    for mode, dataloader in dataloaders.items():
        acc = 0
        for i, batch in enumerate(dataloader):
            num = 0
            batch.to(args.device)
            pred = model(batch)
            for key in pred:
                p = torch.sigmoid(pred[key]).cpu().detach().numpy()
                pred_label = np.zeros_like(p, dtype=np.int64)
                pred_label[np.where(p > 0.5)[0]] = 1
                pred_label[np.where(p <= 0.5)[0]] = 0
                acc += np.sum(pred_label == batch.edge_label[key].cpu().numpy())
                num += len(pred_label)
        accs[mode] = acc / num
    return accs


def test_metrics(model, dataloaders, args):
    model.eval()
    recalls = {}
    RECALL_K = 20
    for mode, dataloader in dataloaders.items():
        recall = 0
        for i, batch in enumerate(dataloader):
            assert i == 0  # only works if there's only one batch
            batch.to(args.device)
            num_users = batch.node_feature['user'].shape[0]
            num_effective_users = num_users
            num_courses = batch.node_feature['course'].shape[0]
            all_pairs_index = torch.Tensor(2, num_users * num_courses).to(args.device)
            
            positive_edges = set()
            num_positive_edges = torch.argmin(batch.edge_label[('user', '0', 'course')]).item()
            user_num_edges = [0 for _ in range(num_users)]
            
            for j in range(num_positive_edges):
                user_ind = batch.edge_label_index[('user', '0', 'course')][0, j].item()
                course_ind = batch.edge_label_index[('user', '0', 'course')][1, j].item()
                if ((user_ind, course_ind)) not in positive_edges:
                    positive_edges.add((user_ind, course_ind))
                    user_num_edges[user_ind] += 1
                
            user_indices, course_indices = torch.meshgrid(
                torch.arange(num_users, device=args.device), 
                torch.arange(num_courses, device=args.device)
            )
            all_pairs_index = torch.stack([user_indices.flatten(), course_indices.flatten()]).to(args.device)
            
            all_pairs = {
                'node_feature': batch.node_feature,
                'edge_index': batch.edge_index,
                'edge_label_index': {
                    ('user', '0', 'course'): all_pairs_index
                }
            }
            
            pred = model(all_pairs)
            for j, key in enumerate(pred):
                assert j == 0  # should only be one type of edge, namely, ('user', '0', 'course')
                p = torch.sigmoid(pred[key]).cpu().detach()
                
                ## convert to matrix
                p_matrix = p.reshape((num_users, num_courses))
                    
                ## exclude edges already in the edge_index
                for edge in batch.edge_index[('user', '0', 'course')].T:
                    if (edge[0].item(), edge[1].item()) in positive_edges: continue
                    p_matrix[edge[0].item(), edge[1].item()] = -(1 << 10)  # set to large negative value
                
                ### compute recall@K for each user
                _, top_k_indices = torch.topk(p_matrix, k=RECALL_K)
                for user in range(num_users):
                    if user_num_edges[user] == 0:
                        num_effective_users -= 1
                        continue
                    num_correct = 0
                    for rec in top_k_indices[user]:
                        if (user, rec.item()) in positive_edges: num_correct += 1
                    recall_user = num_correct / user_num_edges[user]
                    assert recall_user <= 1
                    recall += recall_user
        recalls[mode] = recall / num_effective_users
    return recalls

def main():
    args = arg_parse()

    edge_train_mode = args.mode
    print('edge train mode: {}'.format(edge_train_mode))

    with open(args.data_path, 'rb') as f:
        G = pickle.load(f)
    print(f'dataset: {args.data_path}')
    print(G.number_of_nodes(), G.number_of_edges())

    # find num edge types
    max_label = 0
    labels = []
    for u, v, edge_key in G.edges:
        l = G[u][v][edge_key]['e_label']
        if not l in labels:
            labels.append(l)
    # labels are consecutive (0-17)
    num_edge_types = len(labels)

    H = WN_transform(G, num_edge_types, args.node_embedding_dim)
    hetero = HeteroGraph(H)

    if edge_train_mode == "disjoint":
        dataset = GraphDataset(
            [hetero],
            task='link_pred',
            edge_train_mode=edge_train_mode,
            edge_message_ratio=args.edge_message_ratio
        )
    else:
        dataset = GraphDataset(
            [hetero],
            task='link_pred',
            edge_train_mode=edge_train_mode,
        )

    dataset_train, dataset_val, dataset_test = dataset.split(
        transductive=True, split_ratio=[0.8, 0.1, 0.1]
    )
    train_loader = DataLoader(
        dataset_train, collate_fn=Batch.collate(), batch_size=1
    )
    val_loader = DataLoader(
        dataset_val, collate_fn=Batch.collate(), batch_size=1
    )
    test_loader = DataLoader(
        dataset_test, collate_fn=Batch.collate(), batch_size=1
    )
    dataloaders = {
        'train': train_loader, 'val': val_loader, 'test': test_loader
    }

    hidden_size = args.hidden_dim
    conv1, conv2 = generate_2convs_link_pred_layers(hetero, HeteroSAGEConv, hidden_size)
    model = HeteroGNN(conv1, conv2, hetero, hidden_size, args.node_embedding_dim).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    t_accu, v_accu, e_accu = train(model, dataloaders, optimizer, args)


if __name__ == '__main__':
    main()
