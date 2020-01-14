#The command to load datasets is following deepchem to ensure consistency with benchmarks.
import deepchem
#ESOL
deepchem.molnet.load_delaney(featurizer="GraphConv",split="random")
#FreeSolv
deepchem.molnet.load_sampl(featurizer='GraphConv',split="random")
#Lipophilicity
deepchem.molnet.load_lipo(featurizer='GraphConv',split="random")
#PDBbind
deepchem.molnet.load_pdbbind_grid(featurizer="GraphConv", split="time", subset="full")
#BBBP
deepchem.molnet.load_bbbp(featurizer="GraphConv",split="Scaffold")
#BACE
deepchem.molnet.load_bace_classification(featurizer="GraphConv",split="Scaffold")


