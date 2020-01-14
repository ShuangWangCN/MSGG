# MSGG
A pytorch implementation of "Molecular Property Prediction Based on
a Multichannel Substructure Graph"
![](https://github.com/ShuangWangCN/MSGG/blob/master/MSGG.png)

## Requirements
```
    python            3.6.7
    pytorch           1.2.0
    scipy             1.3.1
    numpy             1.17.2
    pandas            0.25.1
    deepchem          2.2.1
    pickle            0.7.5
    rdkit             2019.03.4.0
    sklearn           0.0.0
```

## Getting started
* data/   contains the prepared data including original splitting data (train/valid/test) and the corresponding transformed S-Graphs <br>
* MSGG/   contains the implementation of MSGG <br>
* premodel/ contains the well-trained model <br>
* benchmark/  containds the command for benchmark models and datasets <br>

## Examples

SGraph/<br>
 load train/valid/test data from deepchem.
```
python load_data_split.py
```
Build the vocabulary for dataset.
```
python NodeEdgeBuild.py
python vocabFlatten_construct.py
```
Transform each molecule into SGraph.
```
python S-Graph_Dataset_construct.py
 python pdb_dataset_construct.py                #pdb
```

Load the well-trained model and predict property on test dataset.
```
python eva_model.py
 python eva_model_pdb.py                         #pdb
```

The datasets and benchmark models are obtained from [deepchem](https://github.com/deepchem/deepchem).
