# -*- coding:utf-8 –*-
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.nn import init
from MSGG.chemical_rdkit import mol2node, edgeVec, ATOM_FDIM, BOND_FDIM, EDGE_FDIM, len_ELEM_LIST

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:i for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def seq_cat(prot):
    # print(len(prot))
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return torch.from_numpy(x)

class GRU_channel(nn.Module):
    def __init__(self,  vocabLabel, hidden_size, embedding=None):
        super(GRU_channel, self).__init__()
        self.hidden_size = hidden_size
        self.vocabLabel_size = vocabLabel.size()
        self.vocabLabel = vocabLabel
        self.atom_in_dim = ATOM_FDIM
        self.bond_in_dim = 2 * ATOM_FDIM + BOND_FDIM
        self.link_in_dim = len_ELEM_LIST
        self.atom_out_dim = int(hidden_size / 4)
        self.bond_out_dim = int(hidden_size / 4)
        self.link_out_dim = int(hidden_size / 4)
        self.edge_out_dim = int(hidden_size / 4)
        self.node_in_dim = self.atom_out_dim + self.bond_out_dim + self.link_out_dim
        self.node_out_dim =self.atom_out_dim+self.bond_out_dim+self.link_out_dim
        self.nodeRnn_in_dim = self.node_out_dim
        self.nodeRnn_out_dim = int(hidden_size / 4)
        self.node_embedding_out = int(hidden_size / 4)
        self.nodeRnn_in_dim = self.node_out_dim
        self.edgeRnn_in_dim = 2 * self.node_embedding_out+ self.edge_out_dim
        self.edgeRnn_out_dim = int(hidden_size / 4)
        self.treeRnn_in_dim = self.node_out_dim + self.node_out_dim + self.edge_out_dim
        self.treeRnn_out_dim = int(hidden_size / 4)

        self.attention_node_in_dim=2*self.nodeRnn_out_dim
        self.attention_node_out_dim=int(hidden_size / 2)
        self.attention_tree_in_dim=2*self.treeRnn_out_dim
        self.attention_tree_out_dim=int(hidden_size / 2)
        self.attention_edge_in_dim=2*self.edgeRnn_out_dim
        self.attention_edge_out_dim=int(hidden_size / 2)

        if embedding is None:
            self.embedding = nn.Embedding(vocabLabel.size(), self.node_embedding_out)
            init.uniform_(self.embedding.weight)
        else:
            self.embedding = embedding
        self.att_W_node=nn.Linear(self.attention_node_in_dim,self.attention_node_out_dim)
        self.att_W_tree=nn.Linear(self.attention_tree_in_dim,self.attention_tree_out_dim)
        self.att_W_edge=nn.Linear(self.attention_edge_in_dim,self.attention_edge_out_dim)
        self.attention_a_node=nn.Linear(2*self.attention_node_out_dim,1)
        self.attention_a_tree=nn.Linear(2*self.attention_tree_out_dim,1)
        self.attention_a_edge = nn.Linear(2 * self.attention_tree_out_dim, 1)
        self.atomW_i = nn.Linear(self.atom_in_dim, self.atom_out_dim, bias=True)
        self.bondW_i = nn.Linear(self.bond_in_dim, self.bond_out_dim, bias=True)
        self.linkPositionW_i = nn.Linear(self.link_in_dim, self.link_out_dim, bias=True)

        self.edgeW_i = nn.Linear(EDGE_FDIM, self.edge_out_dim, bias=True)
        self.edgeRnn=nn.GRU(self.edgeRnn_in_dim,self.edgeRnn_out_dim,num_layers =2,bidirectional=True, bias=True)
        self.nodeRnn = nn.GRU(self.nodeRnn_in_dim, self.nodeRnn_out_dim,  num_layers=2, bidirectional=True,bias=True)
        self.treeVecRnn = nn.GRU(self.treeRnn_in_dim, self.treeRnn_out_dim, num_layers=2, bidirectional=True,bias=True)

        self.num_features_xt=25
        self.output_dim=128
        self.embed_dim=128
        self.n_filters=32
        self.embedding_xt = nn.Embedding(self.num_features_xt + 1, self.embed_dim)
        self.ba = nn.BatchNorm1d(32)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=self.n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(self.n_filters * 121,self.output_dim)
        self.ac_Fun = nn.ReLU()
    def forward(self, SGraph_batch):
        TreeVec_batch = []
        node_vec_tree_batch = []
        root_vec_batch = []
        edgeList_batch = []
        label_batch=[]
        root_batch = []
        pro_list_batch=[]
        for sgraph in SGraph_batch:
            try:
                root_batch.append(sgraph.nodes[0])
                label_batch.append([sgraph.label])
                new_pro=seq_cat(sgraph.pro_seq).cuda()
                pro_list_batch.append(new_pro)
            except Exception as e:
                print(e)

        orders = []
        NodeOrders = []
        for root in root_batch:
            order, NodeOrder = get_bfs_order(root)
            orders.append(order)
            NodeOrders.append(NodeOrder)
            root_vec = Node2Vec(self, root)
            root_vec_batch.append(root_vec)

        for batch_id in range(len(NodeOrders)):
            node_order = NodeOrders[batch_id]
            sgraph = SGraph_batch[batch_id]
            cur_root = torch.tensor([sgraph.nodes[0].widFlatten]).cuda()
            cur_root_embedding = self.embedding(cur_root)
            edgeList = sgraph.edgeList
            edgeList_vec_Tree = []
            node_vec_tree = []
            tree_vec = []
            for node in node_order:
                cur_node_vec = Node2Vec(self, node)
                node_vec_tree.append(cur_node_vec)
                cur_node_id = node.idx
                nei_mess = []
                if not node.neighbors:
                    nei_mess.append(torch.zeros(1, (self.node_out_dim + self.edge_out_dim)).cuda())
                for node_neighbor in node.neighbors:
                    node_nei_vec = Node2Vec(self, node_neighbor)
                    node_nei_id = node_neighbor.idx
                    edgeType = FindEdge(cur_node_id, node_nei_id, sgraph.edgeNoseq)[0]
                    edgecode = edgeVec(edgeType).cuda()
                    edgeFeature = self.ac_Fun(self.edgeW_i(edgecode))
                    edge_mess = torch.cat([node_nei_vec, edgeFeature], 1)
                    nei_mess.append(edge_mess)
                nei_mess = torch.stack(nei_mess)
                nei_mess = nei_mess.sum(0)
                node_edge_node_vec = torch.cat([cur_node_vec, nei_mess], 1)
                tree_vec.append(node_edge_node_vec)
            tree_vec = torch.stack(tree_vec, 0)
            tree_vec_out = self.treeVecRnn(tree_vec)
            tree_vec_out = tree_vec_out[0].squeeze(1)
            tree_vec_out = mol_attention(self,tree_vec_out,'tree')
            TreeVec_batch.append(tree_vec_out)
            node_vec_tree = torch.stack(node_vec_tree, 0)
            node_vec_out = self.nodeRnn(node_vec_tree)
            node_vec_out = node_vec_out[0].squeeze(1)
            node_vec_out = mol_attention(self,node_vec_out,'node')
            node_vec_tree_batch.append(node_vec_out)
            if not edgeList:
                edgeList_vec_Tree.append(torch.cat([cur_root_embedding,cur_root_embedding,torch.zeros(1,self.edge_out_dim).cuda()],1))
            for edge in edgeList:
                edgeNo = edge[2][0]
                nodeStart=torch.tensor([edge[0]]).cuda()
                nodeEnd=torch.tensor([edge[1]]).cuda()
                nodeStartVec=self.embedding(nodeStart)
                nodeEndVec=self.embedding(nodeEnd)
                edgecode=edgeVec(edgeNo).cuda()
                edgeFeature=self.ac_Fun(self.edgeW_i(edgecode))
                edgeListFeature=torch.cat([nodeStartVec,nodeEndVec,edgeFeature],dim=1)
                edgeList_vec_Tree.append(edgeListFeature)
            edgeList_vec_Tree = torch.stack(edgeList_vec_Tree, 0)
            edgeList_vec_out = self.edgeRnn(edgeList_vec_Tree)
            edgeList_vec_out = edgeList_vec_out[0].squeeze(1)
            edgeList_vec_out = mol_attention(self, edgeList_vec_out, 'edge')
            edgeList_batch.append(edgeList_vec_out)
        node_vec_tree_batch = torch.stack(node_vec_tree_batch, dim=0)
        node_vec_tree_batch = torch.squeeze(node_vec_tree_batch)
        edgeList_batch = torch.stack(edgeList_batch, dim=0)
        edgeList_batch = torch.squeeze(edgeList_batch)
        TreeVec_batch = torch.stack(TreeVec_batch, dim=0)
        TreeVec_batch = torch.squeeze(TreeVec_batch)
        label_batch = torch.FloatTensor(label_batch).cuda()
        graph_batch_vec = torch.cat([TreeVec_batch, node_vec_tree_batch, edgeList_batch], dim=1)

        pro_batch = torch.stack(pro_list_batch, dim=0)
        embedded_xt = self.embedding_xt(pro_batch.long())
        conv_xt =self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        xc_batch = torch.cat((graph_batch_vec, xt), 1)
        return label_batch,xc_batch


"""
Helper functions
"""
def mol_attention(self,vec_tensor,state):
    super_ave=torch.mean(vec_tensor,0).unsqueeze(0)
    if state=='node':
        W_node_vec=self.att_W_node(vec_tensor)
        W_super_ave=self.att_W_node(super_ave).expand(vec_tensor.shape[0],-1)
        W_cat_vec=torch.cat([W_super_ave,W_node_vec],dim=1)
        alpha_ac=nn.LeakyReLU()(self.attention_a_node(W_cat_vec))
        alpha=torch.exp(alpha_ac)
        alpha_sum=torch.sum(alpha)
        alpha_final=alpha/alpha_sum
        alpha_final=torch.transpose(alpha_final,1,0)
        node_att_vec=self.ac_Fun(alpha_final.mm(W_node_vec))
        return node_att_vec
    if state=='tree':
        W_tree_vec=self.att_W_tree(vec_tensor)
        W_super_ave=self.att_W_tree(super_ave).expand(vec_tensor.shape[0],-1)
        W_cat_vec=torch.cat([W_super_ave,W_tree_vec],dim=1)
        alpha_ac=nn.LeakyReLU()(self.attention_a_tree(W_cat_vec))
        alpha=torch.exp(alpha_ac)
        alpha_sum=torch.sum(alpha)
        alpha_final=alpha/alpha_sum
        alpha_final=torch.transpose(alpha_final,1,0)
        tree_att_vec=self.ac_Fun(alpha_final.mm(W_tree_vec))
        return tree_att_vec
    if state == 'edge':
        W_edge_vec = self.att_W_edge(vec_tensor)
        W_super_ave = self.att_W_edge(super_ave).expand(vec_tensor.shape[0],-1)
        W_cat_vec = torch.cat([W_super_ave, W_edge_vec], dim=1)
        alpha_ac = nn.LeakyReLU()(self.attention_a_edge(W_cat_vec))
        alpha = torch.exp(alpha_ac)
        alpha_sum = torch.sum(alpha)
        alpha_final = alpha / alpha_sum
        alpha_final = torch.transpose(alpha_final, 1, 0)
        tree_att_vec = self.ac_Fun(alpha_final.mm(W_edge_vec))
        return tree_att_vec


def FindEdge(atomId, neiId, edgeNoseq):
    edge_atom_seq = []
    for edge in edgeNoseq:
        edge_atom_seq.append([edge[0], edge[1]])
    if [atomId, neiId] in edge_atom_seq:
        edgeNo = edge_atom_seq.index([atomId, neiId])
    else:
        edgeNo = edge_atom_seq.index([neiId, atomId])
    edgeType = edgeNoseq[edgeNo][2]
    return edgeType

def Node2Vec(self, node):
    fatoms, fbonds, linkpositions = mol2node(node)
    fatomsVec = self.atomW_i(fatoms)
    fatomsVec = fatomsVec.sum(dim=0)
    fatomsVec = torch.unsqueeze(fatomsVec, 0)
    fatomsVec = self.ac_Fun(fatomsVec)
    fbondsVec = self.bondW_i(fbonds)
    fbondsVec = fbondsVec.sum(dim=0)
    fbondsVec = torch.unsqueeze(fbondsVec, 0)
    fbondsVec=self.ac_Fun(fbondsVec)
    linkpositionsVec = self.linkPositionW_i(linkpositions)
    linkpositionsVec = linkpositionsVec.sum(dim=0)
    linkpositionsVec = torch.unsqueeze(linkpositionsVec, 0)
    linkpositionsVec = self.ac_Fun(linkpositionsVec)
    nodevec = torch.cat([fatomsVec, fbondsVec, linkpositionsVec], dim=1)  # 100
    return nodevec
def by_scort(t):
    return t[1]

def get_bfs_order(root):
    queue = deque([root])
    visited = set([root.idx])
    visitedAtoms = [];
    visitedAtoms.append(root)
    root.depth = 0
    order1 = []
    while len(queue) > 0:
        x = queue.popleft()
        x_nei = []
        for y in x.neighbors:
            x_nei_item=(y,y.widFlatten)
            x_nei.append(x_nei_item)
            x_nei=sorted(x_nei,key=by_scort)
        for x_nei_sort_item in x_nei:
            y_nei,_=x_nei_sort_item
            if y_nei.idx not in visited:
                queue.append(y_nei)
                visited.add(y_nei.idx)
                visitedAtoms.append(y_nei)
                y_nei.depth = x.depth + 1
                if y_nei.depth > len(order1):
                    order1.append([])
                order1[y_nei.depth - 1].append((x, y_nei))
    return order1, visitedAtoms






