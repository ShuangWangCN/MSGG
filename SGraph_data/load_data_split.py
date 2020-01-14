import deepchem as dc
from MSGG.datautils import get_data,save_pickle
filename='ESOL'

data_root= '../data/' + filename + '/'
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv',split='random')
train_dataset, valid_dataset, test_dataset = delaney_datasets

train_final, train_y = get_data(train_dataset.ids, train_dataset.y)
valid_final, valid_y = get_data(valid_dataset.ids, valid_dataset.y)
test_final, test_y = get_data(test_dataset.ids, test_dataset.y)

save_pickle(data_root+'train_smiles.pkl', train_final)
save_pickle(data_root+'train_y.pkl', train_y)

save_pickle(data_root+'valid_smiles.pkl', valid_final)
save_pickle(data_root+'valid_y.pkl', valid_y)

save_pickle(data_root+'test_smiles.pkl', test_final)
save_pickle(data_root+'test_y.pkl', test_y)