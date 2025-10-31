import os, requests, zipfile, random
import copy

import torch
import numpy as np
# from nnsight.models.UnifiedTransformer import UnifiedTransformer
import clip
import nnsight

import overcomplete
from overcomplete.metrics import r2_score
import losses
import train

import datasets
from pycocotools.coco import COCO

from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go

import hashlib

import importlib

from src.utils.data import get_coco, get_laion
import argparse

parser = argparse.ArgumentParser(description="Run R2 vs Std experiment.")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., 'cuda:0', 'cpu').")
parser.add_argument("--alignment_metric", type=str, default="cosim", choices=["l1", "l2", "cosim"],
                    help="Alignment metric to use ('l1', 'l2', or 'cosim').")

args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
alignment_metric = args.alignment_metric

##########
# Load dataset
##########

DATASET = "laion"
batch_size = 512

if DATASET == "coco":
    train_loader = get_coco(batch_size=batch_size, device=device)
elif DATASET == "laion":
    train_loader = get_laion(batch_size=batch_size, device=device)

##########
# train model for different L0s
##########

top_ks = torch.arange(1, 129, step=2.54).long()

R2s = torch.zeros_like(top_ks).float()
modality_stds = torch.zeros_like(top_ks).float()
modality_stds_corrected = torch.zeros_like(top_ks).float()

print(top_ks, top_ks.shape)
print(R2s, R2s.shape)

for i, top_k in enumerate(top_ks):
    print(f"\n\n##########\n# k: {top_k:.2e}\n##########\n")
    ##########
    # Define SAE
    ##########


    d_model = 512 # image_features.shape[-1] # 512 for CLIP ViT-B/32
    expansion_factor = 8
    nb_concepts = d_model * expansion_factor
    lr = 5e-4
    epochs = 10

    UseTOPK = False

    if UseTOPK:
        top_k_individual = top_k * 2
        sae = overcomplete.sae.BatchTopKSAE(d_model, nb_concepts=nb_concepts, top_k=top_k_individual*batch_size, device=device)
        alpha = 0.1  # Dead features loss penalty
        criterion_1 = lambda *args, **kwargs: overcomplete.sae.losses.top_k_auxiliary_loss(*args, **kwargs, penalty=alpha)
    else:
        sae = overcomplete.sae.MpSAE(d_model, nb_concepts=nb_concepts, k=top_k, device=device)
        alpha = 0.0 # L1 penalty
        criterion_1 = lambda *args, **kwargs: overcomplete.sae.losses.mse_l1(*args, **kwargs, penalty=alpha)

    sae.to(device)
    sae.train()

    beta = 5e-4  # Alignment penalty

    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

    criterion_2 = lambda *args, **kwargs: losses.alignment_penalty(*args, **kwargs, penalty=beta, alignment_metric='cosim')
    criterion = lambda *args, **kwargs: criterion_1(*args, **kwargs) + criterion_2(*args, **kwargs)

    scheduler=None
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch,
    )

    alpha_name = str(alpha).replace('.', '')
    beta_name = str(beta).replace('.', '')
    CENTER_DATASET = False if 'CENTER_DATASET' not in locals() else CENTER_DATASET
    sae_name = f"{DATASET}_{"batchtopk" if UseTOPK else "MP"}_centered_{CENTER_DATASET}_{expansion_factor}_L0_{top_k}_alpha" + alpha_name + "beta" + beta_name + ".pt"

    try:
        check = torch.load(f"./checkpoints/{sae_name}")
        logs = check['logs']
        sae_state_dict = check['model_state_dict']
        sae.load_state_dict(sae_state_dict)
    except FileNotFoundError:
        logs = train.train_multimodal_sae(
            sae, train_loader, criterion, optimizer, scheduler=scheduler, nb_epochs=epochs, device=device,
            monitoring=1, verbose=False,
            checkpoint_path="./checkpoints/", checkpoint_interval=5, checkpoint_name=sae_name,
        )

    r2 = logs["r2"][-1]
    print(f"R2 score: {r2:.4f}")
    R2s[i] = r2

    dead_features = logs["dead_features"][-1]
    if (dead_features > 0.999):
        modality_stds[i] = 0
        modality_stds_corrected[i] = 0
        break


    ##########
    # Measure concept energies
    ##########



    # Measure of energy per concept : average activation per concept

    @torch.no_grad()
    def measure_energy_per_concept(sae, data_loader, device):
        sae.eval()
        energy_per_concept = torch.zeros(sae.nb_concepts).to(device)
        count_per_concept = torch.zeros(sae.nb_concepts).to(device)

        n_inputs = 0

        for image_features, text_features in tqdm(data_loader):
            n_inputs += image_features.shape[0]
            n_inputs += text_features.shape[0]

            image_features = image_features.to(device)
            text_features = text_features.to(device)

            # Forward pass through the model
            _, z_image, _ = sae(image_features)
            _, z_text, _ = sae(text_features)

            energy_per_concept += z_image.sum(dim=0) + z_text.sum(dim=0)
            count_per_concept += (z_image != 0).sum(dim=0) + (z_text != 0).sum(dim=0)

        energy_per_concept /= n_inputs
        frequency_per_concept = count_per_concept / n_inputs
        return energy_per_concept, frequency_per_concept

    energy_per_concept, frequency_per_concept = measure_energy_per_concept(sae, train_loader, device)



    ##########
    # Measure concept modality
    ##########



    # Modality score : proportion of energy in vision vs in language

    @torch.no_grad()
    def measure_modality_score(sae, data_loader, device):
        sae.eval()
        image_energy = torch.zeros(sae.nb_concepts).to(device)
        text_energy = torch.zeros(sae.nb_concepts).to(device)

        n_images = 0
        n_texts = 0

        for image_features, text_features in tqdm(data_loader):
            n_images += image_features.shape[0]
            n_texts += text_features.shape[0]

            image_features = image_features.to(device)
            text_features = text_features.to(device)

            # Forward pass through the model
            _, z_image, _ = sae(image_features)
            _, z_text, _ = sae(text_features)

            image_energy += z_image.sum(dim=0)
            text_energy += z_text.sum(dim=0)

        image_energy /= n_images
        text_energy /= n_texts

        total_energy = image_energy + text_energy
        dead_idx = (total_energy == 0).nonzero(as_tuple=True)[0]

        modality_score = image_energy / (image_energy + text_energy)
        image_energy[dead_idx] = 1000
        text_energy[dead_idx] = 1000
        modality_score[dead_idx] = -1
        return modality_score, image_energy, text_energy

    modality_score, image_energy, text_energy = measure_modality_score(sae, train_loader, device)



    ##########
    # Measure modality std arround 0.5
    ##########


    modality_temp = modality_score.cpu().clone().numpy()
    modality_temp = modality_temp[modality_score.cpu() != -1]
    energy_per_concept_temp = energy_per_concept.cpu().clone().numpy()
    energy_per_concept_temp = energy_per_concept_temp[modality_score.cpu() != -1]
    freq_temp = frequency_per_concept.cpu().clone().numpy()
    freq_temp = freq_temp[modality_score.cpu() != -1]

    non_stupid_mask = freq_temp < 0.6
    modality_temp_corrected = modality_temp[non_stupid_mask]
    energy_per_concept_temp_corrected = energy_per_concept_temp[non_stupid_mask]

    std_around_05 = np.sqrt(
        np.average(
            (modality_temp - 0.5)**2, weights=energy_per_concept_temp
        )
    )
    modality_stds[i] = std_around_05.item()

    std_around_05 = np.sqrt(
        np.average(
            (modality_temp_corrected - 0.5)**2, weights=energy_per_concept_temp_corrected
        )
    )
    print(f"Modality std: {std_around_05:.4f}")
    modality_stds_corrected[i] = std_around_05.item()


##########
# Plot results
# - r2 vs top_k
# - std vs top_k
# - std vs r2
##########

# if all features were dead and for loop was broken, cut the arrays to have the same size

k = len(modality_stds[modality_stds != 0])
k = min(k + 1, len(modality_stds))
top_ks = top_ks[:k]
R2s = R2s[:k]
modality_stds = modality_stds[:k]
modality_stds_corrected = modality_stds_corrected[:k]

print(f"Number of ks used: {k}")
print(f"Top ks: {top_ks}")
print(f"R2s: {R2s}")
print(f"Modality stds: {modality_stds}")
print(f"Modality stds (corrected): {modality_stds_corrected}")

output_dir = f"./figures/r2vsstd/topk/beta{beta_name}/"
os.makedirs(output_dir, exist_ok=True)

top_ks_np = top_ks.cpu().numpy()
r2_np = R2s.cpu().numpy()
stds_np = modality_stds.cpu().numpy()
stds_corrected_np = modality_stds_corrected.cpu().numpy()

# Plot 1: R2 vs top_k
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=top_ks_np, y=r2_np, mode='markers+lines', name='R2 vs k'))
fig1.update_layout(
    # xaxis_type="log",
    title="R2 Score vs Alignment Penalty",
    xaxis_title="Alignment Penalty",
    yaxis_title="R2 Score"
)
fig1.write_image(f"{output_dir}/r2_vs_k.png")
fig1.write_image(f"{output_dir}/r2_vs_k.pdf")
fig1.write_json(f"{output_dir}/r2_vs_k.json")

# Plot 2: Std vs top_k
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=top_ks_np, y=stds_corrected_np, mode='markers+lines', name='Std vs top_k', line=dict(color='green')))
fig2.update_layout(
    # xaxis_type="log",
    title="Modality Std vs Alignment Penalty",
    xaxis_title="Alignment Penalty",
    yaxis_title="Modality Std"
)
fig2.write_image(f"{output_dir}/std_vs_k.png")
fig2.write_image(f"{output_dir}/std_vs_k.pdf")
fig2.write_json(f"{output_dir}/std_vs_k.json")

# Plot 3: Std vs R2
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=r2_np, y=stds_corrected_np, mode='markers+lines', name='Std vs R2', line=dict(color='orange')))
fig3.update_layout(
    title="Modality Std vs R2",
    xaxis_title="R2 Score",
    yaxis_title="Modality Std"
)
fig3.write_image(f"{output_dir}/std_vs_r2.png")
fig3.write_image(f"{output_dir}/std_vs_r2.pdf")
fig3.write_json(f"{output_dir}/std_vs_r2.json")

# Plot 4: R2 vs Std
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=stds_corrected_np, y=r2_np, mode='markers+lines', name='R2 vs Std', line=dict(color='purple')))
fig4.update_layout(
    title="R2 Score vs Modality Std",
    xaxis_title="Modality Std",
    yaxis_title="R2 Score"
)
fig4.write_image(f"{output_dir}/r2_vs_std.png")
fig4.write_image(f"{output_dir}/r2_vs_std.pdf")
fig4.write_json(f"{output_dir}/r2_vs_std.json")

# Plot 5: R2 and Std vs top_k (dual y-axis)
fig5 = go.Figure()

# Create first y-axis (R2 vs top_k)
fig5.add_trace(go.Scatter(x=top_ks_np, y=r2_np, mode='markers+lines', name='R2 Score', line=dict(color='blue')))

# Create second y-axis (Std vs top_k)
fig5.add_trace(go.Scatter(x=top_ks_np, y=stds_np, mode='markers+lines', name='Modality Std', line=dict(color='rgb(119, 74, 175)'), yaxis="y2"))
fig5.add_trace(go.Scatter(x=top_ks_np, y=stds_corrected_np, mode='markers+lines', name='Modality Std (corrected)', line=dict(color='rgb(80, 20, 66)'), yaxis="y2"))

print(f"Modality std: {modality_stds}")
print(f"Modality std (corrected): {modality_stds_corrected}")

fig5.update_layout(
    title="R2 and Modality Std vs Alignment Penalty",
    xaxis_title="Alignment Penalty",
    yaxis_title="R2 Score",
    yaxis2=dict(
        title="Modality Std",
        overlaying="y",
        side="right"
    ),
    # xaxis_type="log",
    plot_bgcolor="white",
)

fig5.write_image(f"{output_dir}/r2_and_std_vs_k.png")
fig5.write_image(f"{output_dir}/r2_and_std_vs_k.pdf")
fig5.write_json(f"{output_dir}/r2_and_std_vs_k.json")