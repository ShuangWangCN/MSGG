from MSGG.SGraph import Vocab, VocabLabel
from MSGG.datautils import convert_pro_SGraph,save_pickle
import pickle,rdkit
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

filename='pdb_full'
data_root= '../data/' + filename + '/'
vocab = [x.strip("\r\n ") for x in open(data_root+'vocab_'+filename+'.txt')]
vocab = Vocab(vocab)
vocabFlattenFile=open(data_root+filename+'_VocabAll.pkl','rb')
vocabFlatten=pickle.load(vocabFlattenFile)
vocabFlattenFile.close()
vocabFlatten=VocabLabel(vocabFlatten)
types=['train','valid','test']
for type_item in types:
    type_file=data_root+type_item+'_sort.csv'
    pro_Graph_dataset = convert_pro_SGraph(type_file,vocab, vocabFlatten)
    save_pickle(data_root +'pro_'+ filename + type_item + '.pkl', pro_Graph_dataset)
    print(type_item+'  done')
