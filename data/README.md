## Dataset setting
This file records the differences which need to be modified when run the eva_model.py for different datasets.<br>
EdgeElement exists in MSGG/chemical_rdkit.py<br>
filename,task, label_std and label_mean exist in eva_model.py<br>
pos_weight which is utilized for classification exists in MSGG.SG_pre.py<br>

### Regression
#### ESOL:
```
filename='ESOL'
task='regression'
EdgeElement=['C', 'O', 'c', 'N', 'cc', 'CC', 'P', 'n', 'S', '[N+]', 'cn', 'CCC', 'CC.CC', '[n+]', 'NN', 'c-c']
label_std = 2.0955117304559443
label_mean = -3.050101950354609
```
#### Lipophilicity:
```
filename='Lip'
task='regression'
EdgeElement=['n', 'C', 'c', 'N', 'cc', 'O', 'S', 'B', 'cn', 'CC', '[N+]', '[S+]', 'CN', '[n+]', '[Si]', 'c-c', 'P', 'NN']
label_std = 1.2028604901336206
label_mean = 2.1863357142857263
```
#### PDBbind:
```
filename='pdb_full'
task='regression'
EdgeElement=['C', 'N', '[NH2+]', 'O', 'c', 'S', '[NH+]', 'n', 'cc', 'P', '[S+]', 'CN', 'cn', '[As+]', '[N+]', '[Se]', 'C[S+]', 'CC', 'B', '[n+]', 'c-c', 'C[Cl+2]', '[Cl+2][Cl+2]', 'c[n+]', 'C=C', '[N+2]', 'C[NH+]', '[SH+]', '[P+]', '[C]', '[V+9]', '[C][C]', 'N[OH+][Fe+7]', '[CH2]', '[Fe+7]', '[Fe+6]', 'C[N+]', 'nn', '[B-2]', 'c1ccccc1', 'cccC', 'ccc', '[B-]', 'N[NH+]', '[Si]', '[PH]', '[FeH2+7]', '[Fe+5]', '[Ru+9]', 'C[Ru+9]', '[Cu+5]', '[N+][Cu+5]', 'C1CNCCN1', '[Br+3]', 'CCNC', '[Ru+7]', 'C[Fe+10]', '[Fe+10]', 'CCC', 'C=[N+]', '[Re+9]', 'C[Re+9]', 'C[Ru+10]', '[C][Ru+10]', '[Ru+10]', '[B+]', '[SH]', 'CC.CC']
label_std = 1.7216773344326362
label_mean = 6.243830971659918
```
#### FreeSolv:
```
filename='SAMPL'
task='regression'
EdgeElement=['N', 'C', 'c', 'O', 'S', 'P', 'cc', 'n', '[N+]', '[S+2]', 'CC']
label_std =  3.8448222046029543
label_mean = -3.8030062305295975
```

### Classification
#### BACE:
```
filename='bace'
task='classification'
EdgeElement=['c', 'C', 'N', 'O', 'S', '[NH2+]', 'cc', 'n', 'CN', '[NH+]', '[N+]', 'CC', '[n+]', 'cn']
posWeight=1.1895803183791607
```
#### BBBP:
```
filename='bbbp'
task='classification'
EdgeElement=['C', 'N', 'O', 'c', 'cc', 'CN', 'cn', 'n', 'CC', 'S', '[N+]', 'P', '[NH+]', '[NH2+]', 'C=C', '[CH-]', 'B', '[S+]', '[N-]', '[n+]', 'c-c', 'ccc', 'c1ccccc1', 'CNCC']
posWeight=0.30705128205128207
```
