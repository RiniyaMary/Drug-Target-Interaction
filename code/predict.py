import torch
import dgl
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from models import BINDTI
from utils import integer_label_protein
from mol2vec.features import mol2alt_sentence
from rdkit import Chem


# =========================
# BUILD GRAPH (MATCH TRAINING)
# =========================
def build_graph(smiles, max_nodes=290):
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)

    fc = partial(smiles_to_bigraph, add_self_loop=True)

    v_d = fc(
        smiles=smiles,
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer
    )

    # ---- same as dataloader ----
    actual_node_feats = v_d.ndata.pop('h')
    num_actual_nodes = actual_node_feats.shape[0]

    # truncate
    if num_actual_nodes > max_nodes:
        actual_node_feats = actual_node_feats[:max_nodes]
        v_d = dgl.node_subgraph(v_d, list(range(max_nodes)), store_ids=False)
        num_actual_nodes = max_nodes

    # add virtual node bit
    virtual_node_bit = torch.zeros((num_actual_nodes, 1))
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), dim=1)
    v_d.ndata['h'] = actual_node_feats

    # padding nodes
    num_virtual_nodes = max_nodes - num_actual_nodes
    if num_virtual_nodes > 0:
        virtual_node_feat = torch.cat(
            (torch.zeros((num_virtual_nodes, 74)),
             torch.ones((num_virtual_nodes, 1))),
            dim=1
        )
        v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})

    v_d = v_d.add_self_loop()

    return v_d


# =========================
# LOAD MODEL
# =========================
def load_model(model_path, cfg, device):
    model = BINDTI(device=device, **cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# =========================
# MOL2VEC EMBEDDING
# =========================
def get_mol2vec_embedding(smiles, mol2vec_model, max_len=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(max_len, 300)

    sentence = mol2alt_sentence(mol, 1)

    embeddings = []
    for token in sentence[:max_len]:
        if token in mol2vec_model.wv:
            embeddings.append(mol2vec_model.wv[token])
        else:
            embeddings.append(np.zeros(mol2vec_model.vector_size))

    while len(embeddings) < max_len:
        embeddings.append(np.zeros(mol2vec_model.vector_size))

    return torch.tensor(np.array(embeddings), dtype=torch.float)


# =========================
# PREDICT FUNCTION
# =========================
def predict_dti(smiles, protein_seq, model, mol2vec_model, device):

    # -------- Validate SMILES --------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("❌ Invalid SMILES string")

    # -------- Graph --------
    graph = build_graph(smiles)
    graph = dgl.batch([graph]).to(device)

    # -------- Protein --------
    protein = integer_label_protein(protein_seq)
    protein = torch.tensor(np.array([protein])).to(device)

    # -------- Mol2Vec --------
    smiles_emb = get_mol2vec_embedding(smiles, mol2vec_model)
    smiles_emb = smiles_emb.unsqueeze(0).to(device)

    # -------- Prediction --------
    with torch.no_grad():
        _, _, _, score = model(graph, protein, smiles_emb)
        prob = torch.sigmoid(score).item()

    return prob


# =========================
# MAIN (CLI DEMO)
# =========================
if __name__ == "__main__":
    from gensim.models import Word2Vec
    from configs import get_cfg_defaults

    device = torch.device("cpu")

    # Load config
    cfg = get_cfg_defaults()

    # Load model
    model = load_model("../output/checkpoints/best_model.pth", cfg, device)

    # Load Mol2Vec
    mol2vec_model = Word2Vec.load("../models/mol2vec_trained.model")

    print("\n🔬 DTI Prediction System (Final Version)")

    while True:
        try:
            smiles = input("\nEnter SMILES: ")
            protein = input("Enter Protein Sequence: ")

            prob = predict_dti(smiles, protein, model, mol2vec_model, device)

            print(f"\nInteraction Probability: {prob:.4f}")

            if prob > 0.5:
                print("✅ Interaction Likely")
            else:
                print("❌ No Interaction")

        except Exception as e:
            print(f"\n⚠️ Error: {e}")

        cont = input("\nContinue? (y/n): ")
        if cont.lower() != "y":
            break