from MSGG.chemical_rdkit import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap
import rdkit.Chem as Chem
import operator

def change_two_idx(root, node, x, y):
    a, b = x, y
    if x == root.idx:
        a = node.idx
    if x == node.idx:
        a = root.idx
    if y == root.idx:
        b = node.idx
    if y == node.idx:
        b = root.idx
    return a, b
#set the root node id
def change_root_id(root, node, sequence):
    for i in range(len(sequence)):
        item = sequence[i]
        if len(item) == 2:
            x, y = item
            sequence[i] = change_two_idx(root, node, x, y)
        if len(item) == 3:
            x, y, edge = item
            sequence[i] = list(change_two_idx(root, node, x, y))
            sequence[i].append(edge)
    return sequence

class Vocab(object):
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
    def get_index(self, smiles):
        return self.vmap[smiles]
    def get_smiles(self, idx):
        return self.vocab[idx]
    def size(self):
        return len(self.vocab)

class VocabLabel(object):
    def __init__(self, vocabLabel):
        self.vocabLabel = vocabLabel
        self.vocabflatten, self.flattenNum = self.flatten()
    def flatten(self):
        vocabLabel = self.vocabLabel
        flattened_vocab_label = []
        account_number_for_primitive_site = []
        for list in vocabLabel:
            child_list_number = 0
            for child_list in list:
                child_list_number += 1
                flattened_vocab_label.append(child_list)
            account_number_for_primitive_site.append(child_list_number)
        return flattened_vocab_label, account_number_for_primitive_site

    def get_flaten_id(self, pri_id, pri_child_id):
        vocabLabel = self.vocabLabel
        try:
            vocabLabel[pri_id][pri_child_id]
        except:
            raise Exception('pri_id:' + str(pri_id) + ' pri_child_id:' + str(pri_child_id) + 'not exist')
        account_number_for_primitive_site = self.flattenNum
        flaten_id = 0
        for i in range(pri_id):
            flaten_id += account_number_for_primitive_site[i]
        flaten_id += pri_child_id
        return flaten_id

    def get_child_id(self, item, pri_id):
        vocabLabel = self.vocabLabel
        vocab_class = vocabLabel[pri_id]
        for i in range(len(vocab_class)):
            child_vocab_item = vocab_class[i]
            find = operator.eq(item, child_vocab_item)
            if find:
                return i
    def size(self):
        return len(self.vocabflatten)

class SGraphNode(object):
    def __init__(self, smiles, clique):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)
        self.clique = [x for x in clique]
        self.neighbors = []
        self.sequence = []
        self.seqLabel = []
        self.seqIdx = []
        self.cliqueLabel = 0

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def set_sequence(self, original_mol, nei_list):
        clique = self.clique
        curAtomIdx = min(clique)
        atomNum = len(clique)
        # initialize the begin atom
        curAtom = original_mol.GetAtomWithIdx(curAtomIdx)
        self.sequence.append(curAtom.GetSymbol())
        self.seqIdx.append(curAtomIdx)
        if len(nei_list[curAtomIdx]) == 1:
            self.seqLabel.append(0)
        else:
            self.seqLabel.append(1)
        while len(self.sequence) < atomNum:
            curAtom = original_mol.GetAtomWithIdx(curAtomIdx)
            nextAtoms = curAtom.GetNeighbors()
            nextAtomNum = len(nextAtoms)
            examLen = len(self.sequence)
            examNum = 0
            for nextAtom in nextAtoms:
                examNum = examNum + 1
                nextAtomIdx = nextAtom.GetIdx()
                if nextAtomIdx in clique and nextAtomIdx not in self.seqIdx:
                    curAtomIdx = nextAtomIdx
                    self.sequence.append(nextAtom.GetSymbol())
                    self.seqIdx.append(nextAtomIdx)
                    if len(nei_list[nextAtomIdx]) == 1:
                        self.seqLabel.append(0)
                    else:
                        self.seqLabel.append(1)

            if examNum == nextAtomNum and examLen == len(self.sequence):
                remainSeq = sorted(set(clique).difference(set(self.seqIdx)), key=clique.index)
                if len(remainSeq) == 1:
                    remainAtom = original_mol.GetAtomWithIdx(remainSeq[0])
                    self.sequence.append(remainAtom.GetSymbol())
                    self.seqIdx.append(remainSeq[0])
                    if len(nei_list[remainSeq[0]]) == 1:
                        self.seqLabel.append(0)
                    else:
                        self.seqLabel.append(1)
                curAtomIdx = min(remainSeq)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)
        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:
                continue
            for cidx in nei_node.clique:
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)
        clique = sorted(set(clique), key=clique.index)
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)
        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)
        return self.label




class SGraph(object):

    def __init__(self, smiles, label,pro_seq=None):
        self.smiles = smiles
        self.label = label
        self.mol = get_mol(smiles)
        self.pro_seq=pro_seq
        cliques, edges, nei_list, edgesList, edgeNoseq, edgeElements = tree_decomp(self.mol,build=False)
        self.nodes = []
        self.edgeList = edgesList
        self.edgeNoseq = edgeNoseq
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = SGraphNode(get_smiles(cmol), c)
            node.set_sequence(self.mol, nei_list)
            node.idx = i
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]
            self.nodes[0].idx, self.nodes[root].idx = self.nodes[root].idx, self.nodes[0].idx
            self.edgeList = change_root_id(self.nodes[root], self.nodes[0], self.edgeList)
            self.edgeNoseq = change_root_id(self.nodes[root], self.nodes[0], self.edgeNoseq)
        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self, vocab, vocabLabel):
        edgeNum = 0
        for node in self.nodes:
            node.recover(self.mol)
            node.wid = vocab.get_index(node.smiles)
            item = {'seqLabel': node.seqLabel, 'sequence': node.sequence}
            child_id = vocabLabel.get_child_id(item, node.wid)
            node.widFlatten = vocabLabel.get_flaten_id(node.wid, child_id)

        for edgeItem in self.edgeList:
            node1 = edgeItem[0]
            node2 = edgeItem[1]
            edgeType = edgeItem[2]
            edgeItemNew = [self.nodes[node1].widFlatten, self.nodes[node2].widFlatten, edgeType]
            self.edgeList[edgeNum] = edgeItemNew
            edgeNum += 1

    def recover_vocab(self, vocab):
        for node in self.nodes:
            node.recover(self.mol)
            node.wid = vocab.get_index(node.smiles)



