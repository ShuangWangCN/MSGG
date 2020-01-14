from MSGG.SGraph import SGraph
from MSGG.SGraph import Vocab
from MSGG.datautils import read_pickle,save_pickle

filename='ESOL'
data_root= '../data/' + filename + '/'
vocab = [x.strip("\r\n ") for x in open(data_root+'vocab_'+filename+'.txt')]
vocab = Vocab(vocab)
VocabLabels = [[] for i in range(vocab.size())]
iter=0
edgeAddelement=[]
smile_np=[]
label_np=[]
type=['train','valid','test']

for type_item in type:
    smile=read_pickle(data_root+type_item+'_smiles.pkl')
    smile_np+=smile
    label=read_pickle(data_root+ type_item + '_y.pkl')
    label_np+=label

for i in range(len(smile_np)):
    smiles=smile_np[i]
    label=label_np[i]
    iter=iter+1
    print(iter, smiles)
    sgrah = SGraph(smiles,label)
    sgrah.recover_vocab(vocab)
    for tNode in sgrah.nodes:
        wid=tNode.wid
        seqence=tNode.sequence
        seqLabel=tNode.seqLabel
        nodeItem={'sequence':seqence,'seqLabel':seqLabel}
        if nodeItem not in VocabLabels[wid]:
            VocabLabels[wid].append(nodeItem)
save_pickle(data_root+filename+'_VocabAll.pkl',VocabLabels)
