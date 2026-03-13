# imports
import sys,os
import time
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
import create_interface_features_v1
import models
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(path_weights, device, cap_length=75):

    if not os.path.exists(path_weights):
        raise FileNotFoundError(f"Model weights not found: {path_weights}")

    print(f"Loading model from: {path_weights}")
    model = models.ProteinTransformerRegressorV2(
        n_input=4253,
        d_model=256,
        nhead=8,
        num_layers=2,
        n_outputs=4,
        max_length=cap_length,
        dropout=[0.0, 0.0],
    ).to(device)

    state_dict = torch.load(path_weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _process_single_pdb_bundle(bundle):
    """
    Worker for a single PDB — safe call to feature computation.
    Expects bundle = (pdb_path, p)
    Returns (features, model_id) or None on error.
    """
    pdb_path, p = bundle
    try:
        features, _, model_id = create_interface_features_v1.create_features_interface(pdb_path, p)
        return features, model_id
    except Exception as e:
        print(f"Feature extraction failed for {pdb_path}: {e}")
        return None
    

def _build_parameters_for_system(target_system_path, params_kwargs=None):
    """
    Build and initialize a Parameters instance for the target system (one per scoring run).
    params_kwargs is an optional dict to override defaults.
    Returns an initialized Parameters object.
    """
    if params_kwargs is None:
        params_kwargs = {}

    # Create default Parameters, you can expand the kwargs if needed
    p = create_interface_features_v1.Parameters(
        n_neighbors=params_kwargs.get("n_neighbors", 20),
        cutoff_interface=params_kwargs.get("cutoff_interface", 0.5),
        cutoff_potential=params_kwargs.get("cutoff_potential", 0.85),
        r_on=params_kwargs.get("r_on", 0.65),
        grid_size=params_kwargs.get("grid_size", 3),
        voxel_size=params_kwargs.get("voxel_size", 1.0),
        target_shape_residue=params_kwargs.get("target_shape_residue", 40),
        target_shape_residue_large=params_kwargs.get("target_shape_residue_large", 96),
    )

    # Call intialize_system with the chosen representative path
    p = create_interface_features_v1.intialize_system(target_system_path, p)
    return p


def compute_features(list_pdbs, n_workers=4, params_kwargs=None):
    """
    Compute interface features for multiple PDBs in parallel.
    - Builds a Parameters object 'p' from the first pdb in list_pdbs (as in your original code).
    - Uses Pool to compute features in parallel, passing (pdb_path, p) to each worker.
    Returns:
        features_list: list of np.ndarray (n_interface_residues x n_features)
        model_ids: list of model_id (same ordering as features_list)
    """
    if len(list_pdbs) == 0:
        return [], []

    # Use the first pdb as the representative to initialize system parameters (same behavior as original)
    target_system_path = list_pdbs[0]
    print(f"Initializing Parameters using: {target_system_path}")
    p = _build_parameters_for_system(target_system_path, params_kwargs=params_kwargs)

    # Prepare bundles of (pdb_path, p) so each worker gets the same Parameters instance
    bundles = [(pdb_path, p) for pdb_path in list_pdbs]

    print(f"Computing features for {len(list_pdbs)} PDBs with {n_workers} workers...")
    t0 = time.time()

    # Use multiprocessing Pool; each worker calls create_features_interface with the provided p
    with Pool(n_workers) as pool:
        results = pool.map(_process_single_pdb_bundle, bundles)

    # Filter out None results and unpack
    valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        raise RuntimeError("No valid PDBs were processed successfully.")

    features_list, model_ids = zip(*valid_results)
    features_list = list(features_list)
    model_ids = list(model_ids)

    print(f"Features extracted for {len(features_list)} PDBs in {time.time() - t0:.2f}s")
    return features_list, model_ids


def prepare_inputs(features_list, cap_length=75, feature_cap=1000):
    """
    Pad/truncate each residue-level feature matrix to cap_length.
    Returns:
        X_tensor (torch.FloatTensor) : shape (n_models, cap_length, n_features)
        lengths_tensor (torch.LongTensor) : true residue counts
    """
    n_features = features_list[0].shape[1]
    X_padded = []
    lengths = []

    for f in features_list:
        f = np.nan_to_num(f, nan=0.0)
        f = np.clip(f, -feature_cap, feature_cap)
        n_res = f.shape[0]
        lengths.append(min(n_res, cap_length))

        if n_res > cap_length:
            f = f[:cap_length]
        elif n_res < cap_length:
            pad = np.zeros((cap_length - n_res, n_features), dtype=np.float32)
            f = np.vstack([f, pad])

        X_padded.append(f.astype(np.float32))

    X_tensor = torch.tensor(np.stack(X_padded), dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return X_tensor, lengths_tensor


@torch.no_grad()
def predict_scores(model, X_tensor, lengths_tensor, model_ids, device, batch_size=32):
    """
    Run predictions in batches through the model.
    Returns a DataFrame with Model and 4 predictions.
    """
    model.eval()
    X_tensor = X_tensor.to(device)
    lengths_tensor = lengths_tensor.to(device)

    preds = []
    for i in range(0, len(X_tensor), batch_size):
        batch_X = X_tensor[i:i + batch_size]
        batch_len = lengths_tensor[i:i + batch_size]

        batch_y = model(batch_X, batch_len)
        preds.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    df = pd.DataFrame(preds, columns=["pDockQ", "pFNAT_env", "pJACA", "pJACB"])
    df.insert(0, "Model", model_ids)
    return df


def run_scorer(
    list_pdbs,
    path_weights,
    cap_length=75,
    batch_size=32,
    n_workers=4,
    feature_cap=1000,
    params_kwargs=None,
    device=None,
):
    """
    Main scorer pipeline:
        - computes features (multiprocessing) using a single initialized Parameters 'p'
        - loads model
        - prepares inputs
        - runs batched predictions
        - returns DataFrame of scores (one row per model)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running scorer on device: {device}")

    model = load_model(path_weights, device, cap_length=cap_length)

    features_list, model_ids = compute_features(list_pdbs, n_workers=n_workers, params_kwargs=params_kwargs)

    X_tensor, lengths_tensor = prepare_inputs(features_list, cap_length=cap_length, feature_cap=feature_cap)

    df_preds = predict_scores(model, X_tensor, lengths_tensor, model_ids, device, batch_size=batch_size)

    print(f"Scoring completed for {len(df_preds)} models.")
    return df_preds
