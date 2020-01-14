from MSGG.pro_SG_pre import SG_pre
from MSGG.SGraph import VocabLabel,Vocab
from MSGG.datautils import read_pickle,load_SGraph
import torch.nn as nn
import torch

filename='pdb_full'
task='regression'
label_std = 1.7216773344326362
label_mean = 6.243830971659918
data_root= 'data/' + filename + '/'
model_root='premodel/'+filename+'/'
vocab = [x.strip("\r\n ") for x in open(data_root + 'vocab_' + filename + '.txt')]
vocab = Vocab(vocab)
vocabFlatten=read_pickle(data_root + filename + '_VocabAll.pkl')
vocabFlatten=VocabLabel(vocabFlatten)
print(filename)
ave_test_result=0
for k in range(1,4):
    model_path=model_root+filename+'_MSGG_model'+str(k)
    sgraph_dataset_test=load_SGraph(data_root + filename, 'test')
    if model_path is not None:
        a=torch.load(model_path)
        hidden_size=a['layer1.0.weight'].shape[0]*2
        model = SG_pre(vocabFlatten, hidden_size)
        model = model.cuda()
        model.load_state_dict(a)
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)

    model.eval()
    if task=='regression':
        _,test_rmse=model(sgraph_dataset_test,label_mean,label_std)
        print(str(k)+'  test_rmse:'+str(test_rmse))
        ave_test_result+=test_rmse

print('ave_results:',ave_test_result/3)
