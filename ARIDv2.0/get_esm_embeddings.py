from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, SamplingConfig
# from esm.utils.constants.models import ESM3_OPEN_SMALL
import torch
import numpy as np

# Method 1: Get embeddings directly from the embedding layer
def get_initial_embeddings(model, encoded_protein):
    """Get the initial token embeddings before transformer processing."""
    with torch.no_grad():
        # Get the sequence tensor
        sequence = encoded_protein.sequence
        
        # Get embeddings from the embedding layer
        embedded = model.embed(sequence)  # Shape: [seq_len, hidden_dim]
        
        return embedded


def fold_embedding(target_shape,em):
    if em.shape[1] % target_shape != 0:
        raise ValueError('Cannot fold irregularly')
    size_window = int(em.shape[1]/target_shape)

    aggregated_ids = np.zeros((em.shape[0],target_shape),dtype=np.float64)
    for ri in range(em.shape[0]):
        for i in range(target_shape):
            start_agg = i*size_window
            end_agg = start_agg + size_window
            aggregated_ids[ri][i] = np.mean(em[ri,start_agg:end_agg])
    return aggregated_ids

def get_folded_embeddings(target_system_path,target_shape_residue,target_shape_residue_large):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Load model
    client = ESMC.from_pretrained("esmc_300m").to(device)

    # Load chain A and chain B separately
    protein_A = ESMProtein.from_pdb(target_system_path, chain_id="A")
    protein_B = ESMProtein.from_pdb(target_system_path, chain_id="B")

    # Encode
    encoded_A = client.encode(protein_A).to(device)
    encoded_B = client.encode(protein_B).to(device)
    eA = get_initial_embeddings(client, encoded_A)[1:-1].cpu().float().numpy()
    eB = get_initial_embeddings(client, encoded_B)[1:-1].cpu().float().numpy()

    em = np.concatenate([eA,eB])

    # folding residue identifiers
    residue_embedding_identifiers_small = fold_embedding(target_shape_residue,em)

    residue_embedding_identifiers_large = fold_embedding(target_shape_residue_large,em)

    em_a = np.mean(residue_embedding_identifiers_large[:eA.shape[0]],axis=0)
    em_b = np.mean(residue_embedding_identifiers_large[eA.shape[0]:],axis=0)

    return residue_embedding_identifiers_small,residue_embedding_identifiers_large,em_a,em_b,eA.shape[0],eB.shape[0]