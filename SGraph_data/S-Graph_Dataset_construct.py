from MSGG.SGraph import Vocab, VocabLabel
from MSGG.datautils import convert_SGraph,save_pickle,read_pickle

filename='ESOL'
data_root= '../data/' + filename + '/'
vocab = [x.strip("\r\n ") for x in open(data_root+'vocab_'+filename+'.txt')]
vocab = Vocab(vocab)
vocabFlatten=read_pickle(data_root+filename+'_VocabAll.pkl')
vocabFlatten=VocabLabel(vocabFlatten)
type=['train','valid','test']
for type_item in type:
    smile=read_pickle(data_root+type_item+'_smiles.pkl')
    y=read_pickle(data_root+type_item+'_y.pkl')
    S_Graph_dataset = convert_SGraph(smile, y, vocab, vocabFlatten)
    save_pickle(data_root+filename+type_item+'.pkl',S_Graph_dataset)
    print('convert sgraph done:',type_item)