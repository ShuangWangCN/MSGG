from MSGG.chemical_rdkit import get_clique_mol,get_smiles,get_mol,tree_decomp
from MSGG.datautils import save_pickle,read_pickle

filename='ESOL'
type=['train','valid','test']
data_root='../data/'+filename+'/'
smile_np=[]

for type_item in type:
    smile=read_pickle(data_root+type_item+'_smiles.pkl')
    print(type_item + str(len(smile)))
    smile_np+=smile

edgeTotal=[]
vocabTotal=[]
for smiles in smile_np:
    mol = get_mol(smiles)
    clique, _, _, _, _, edge_item = tree_decomp(mol,True)
    edge_item2 = sorted(set(edge_item), key=edge_item.index)
    edgeTotal.extend(edge_item2)
    for c in clique:
        cmol = get_clique_mol(mol, c)
        cmol_smile = get_smiles(cmol)
        vocabTotal.append(cmol_smile)
edgeTotal2 = sorted(set(edgeTotal), key=edgeTotal.index)
vocabTotal2 = sorted(set(vocabTotal), key=vocabTotal.index)
save_pickle(data_root+'edgeTotal_' + filename + '.pkl', edgeTotal2)
print(edgeTotal2)
n_file=open(data_root+'vocab_'+filename+'.txt','w')
for vocab_smile in vocabTotal2:
    n_file.write(vocab_smile)
    n_file.write('\n')
n_file.close()


