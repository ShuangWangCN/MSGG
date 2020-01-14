#This section about how to deal with molecules using Rdkit is following JunctionTree
import torch
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']#23
EdgeElement=['C', 'N', '[NH2+]', 'O', 'c', 'S', '[NH+]', 'n', 'cc', 'P', '[S+]', 'CN', 'cn', '[As+]', '[N+]', '[Se]', 'C[S+]', 'CC', 'B', '[n+]', 'c-c', 'C[Cl+2]', '[Cl+2][Cl+2]', 'c[n+]', 'C=C', '[N+2]', 'C[NH+]', '[SH+]', '[P+]', '[C]', '[V+9]', '[C][C]', 'N[OH+][Fe+7]', '[CH2]', '[Fe+7]', '[Fe+6]', 'C[N+]', 'nn', '[B-2]', 'c1ccccc1', 'cccC', 'ccc', '[B-]', 'N[NH+]', '[Si]', '[PH]', '[FeH2+7]', '[Fe+5]', '[Ru+9]', 'C[Ru+9]', '[Cu+5]', '[N+][Cu+5]', 'C1CNCCN1', '[Br+3]', 'CCNC', '[Ru+7]', 'C[Fe+10]', '[Fe+10]', 'CCC', 'C=[N+]', '[Re+9]', 'C[Re+9]', 'C[Ru+10]', '[C][Ru+10]', '[Ru+10]', '[B+]', '[SH]', 'CC.CC']
MST_MAX_WEIGHT = 100
MAX_NCAND = 2000
EDGE_FDIM=len(EdgeElement)
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
len_ELEM_LIST=len(ELEM_LIST)
edgeNum=len(EdgeElement)

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)#atom has three attributes:  symbol ,get formalcharge   getatommapnum
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()#smile fragement convert into mol reedit mol attribute
    new_mol = sanitize(new_mol)
    return new_mol
#decomp molecules into cliques
def tree_decomp(mol,build):
    edgeElement=EdgeElement
    if build:
        edgeElement=[]
    n_atoms = mol.GetNumAtoms()
    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    #Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = sorted(set(cliques[i]) & set(cliques[j]), key=cliques[i].index)
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = sorted(set(cliques[i]), key=cliques[i].index)
                    cliques[j] = []
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)# save the atom belonging to which clique
    #Build edges and add singleton cliques
    edges = defaultdict(int)
    edgeElements=[]
    edgeNoseq=[]
    EDGElist = []
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]  # c1 the first clique    c2 the second clique
                    inter = sorted(set(cliques[c1]) & set(cliques[c2]),key=cliques[c1].index)  # inter indicates the atom  id   belong to two cliques
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)


    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges,nei_list,EDGElist,edgeNoseq,edgeElements
    #Compute Maximum Spanning Tree
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    for edge in edges:
        n1 = edge[0]
        n2 = edge[1]
        inter = sorted(set(cliques[n1]) & set(cliques[n2]), key=cliques[n1].index)
        interSmile = Chem.MolFragmentToSmiles(mol, atomsToUse=list(inter))
        try:
            edgeNo = [i for i, x in enumerate(edgeElement) if x == interSmile]
            if not edgeNo:
                edgeElements.append(interSmile)
        except ValueError  as e:
            edgeElements.append(interSmile)
        edgeItem = [n1, n2, edgeNo]
        EDGElist.append(edgeItem)
        edgeNoseq.append(edgeItem)

    return (cliques, edges,nei_list,EDGElist,edgeNoseq,edgeElements)

def atom_equal(a1, a2):
  return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

#Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])
#one-hot encoding
def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
#basic features for atom
def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])
#basic features for bond
def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)
#find the linking position
def seqPosition(seq, seqLabel):
    position = []
    for idx in range(len(seqLabel)):
        if seqLabel[idx] == 1:
            seqItem = seq[idx]
            position.append(torch.Tensor(onek_encoding_unk(seqItem, ELEM_LIST)))
    return position

#Obtain the basic features for each node
def mol2node(node):
    fatoms, fbonds = [], []
    mol = get_mol(node.smiles)
    for atom in mol.GetAtoms():
        fatoms.append(atom_features(atom))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        bond_feature = bond_features(bond)
        a1AtomFeature = atom_features(a1)
        a2AtomFeature = atom_features(a2)
        fbondsItem = torch.cat([a1AtomFeature, bond_feature, a2AtomFeature], 0)
        fbonds.append(fbondsItem)
    if fbonds == []:
        fbonds.append(torch.zeros(2 * ATOM_FDIM + BOND_FDIM))
    fatoms = torch.stack(fatoms, 0).cuda()
    fbonds = torch.stack(fbonds, 0).cuda()
    linkpositions = seqPosition(node.sequence, node.seqLabel)
    if linkpositions == []:
        linkpositions.append(torch.zeros(len(ELEM_LIST)))
    linkpositions = torch.stack(linkpositions, 0).cuda()
    return fatoms, fbonds, linkpositions
#one-hot encoding for edge
def edgeVec(edgeNo):
    return torch.Tensor([onek_encoding_unk(EdgeElement[edgeNo], EdgeElement)])


