import pandas as pd
from MSGG.SGraph import SGraph
from torch.autograd import Variable
import rdkit,pickle

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def get_data(dataset_smiles, dataset_y):
    smile_final = [smile for smile in dataset_smiles]
    y_final = [y for y in dataset_y]
    return smile_final, y_final

def save_pickle(path, result_final):
    with open(path, 'wb') as f:
        pickle.dump(result_final,f)
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor, requires_grad=requires_grad).cuda()
def load_SGraph(filename,type):
    sgraph_data_dele=[]
    sgraph_dataset=read_pickle(filename + type+'.pkl')
    for sgraph in sgraph_dataset:
        if  sgraph.nodes:
            sgraph_data_dele.append(sgraph)
    return sgraph_data_dele
#convert molecule into s-graph
def convert_SGraph(smiles,label,vocab,vocabFlatten):
    sgraph_dataset=[]
    for i in range(len(smiles)):
        sgraph = SGraph(smiles[i], label[i])
        sgraph.recover(vocab,vocabFlatten)
        sgraph_dataset.append(sgraph)
    return sgraph_dataset

def convert_pro_SGraph(csv_file,vocab,vocabFlatten):
    sgraph_dataset = []
    df = pd.read_csv( csv_file)
    mol_smiles, pro_target, affinity = list(df['smiles']), list(df['pro']), list(df['affinity'])
    for i in range(len(mol_smiles)):
        sgraph=SGraph(smiles=mol_smiles[i],label=affinity[i],pro_seq=pro_target[i])
        sgraph.recover(vocab,vocabFlatten)
        sgraph_dataset.append(sgraph)
    return sgraph_dataset




