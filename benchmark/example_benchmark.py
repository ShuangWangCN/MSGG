#This example is obtained from deepchem
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import numpy as np
from deepchem.models import GraphConvModel

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from deepchem.molnet import load_delaney
delaney_tasks, delaney_datasets, transformers = load_delaney(
    featurizer='GraphConv', split='random')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

n_feat = 75
# Batch size of models
batch_size = 128
model = GraphConvModel( len(delaney_tasks), batch_size=batch_size, mode='regression')

# Fit trained model
model.fit(train_dataset, nb_epoch=20)

print("Evaluating model")
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Test scores")
print(test_scores)
