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
import pandas as pd
from pycocotools.coco import COCO

from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import PathPatch, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import colorsys

import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.utils.data import get_coco, get_laion
import metrics

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATASET = "laion"
batch_size = 512

if DATASET == "coco":
    train_loader = get_coco(batch_size=batch_size, device=device)
elif DATASET == "laion":
    train_loader = get_laion(batch_size=batch_size, device=device)

d_model = 512 # image_features.shape[-1] # 512 for CLIP ViT-B/32

expansion_factor = 8
nb_concepts = d_model * expansion_factor

lr = 5e-4
epochs = 10

top_k = 20

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

beta = 5e-4 # Alignment penalty (0 or 5e-4)

optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

criterion_2 = lambda *args, **kwargs: losses.alignment_penalty(*args, **kwargs, penalty=beta, alignment_metric='cosim')
criterion = lambda *args, **kwargs: criterion_1(*args, **kwargs) + criterion_2(*args, **kwargs)

scheduler=None
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch,
)

@torch.no_grad()
def measure_everything(sae, data_loader, device, return_sqr=False):
    sae.eval()
    count_per_concept = torch.zeros(sae.nb_concepts).to(device)
    E_i = torch.zeros(sae.nb_concepts).to(device)
    E_i_sqr = torch.zeros(sae.nb_concepts).to(device)
    E_t = torch.zeros(sae.nb_concepts).to(device)
    E_t_sqr = torch.zeros(sae.nb_concepts).to(device)

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

        E_i += z_image.sum(dim=0)
        E_i_sqr += (z_image ** 2).sum(dim=0)
        E_t += z_text.sum(dim=0)
        E_t_sqr += (z_text ** 2).sum(dim=0)
        count_per_concept += (z_image != 0).sum(dim=0) + (z_text != 0).sum(dim=0)

    n_inputs = n_images + n_texts
    E = E_i + E_t
    E /= n_inputs
    E_sqr = E_i_sqr + E_t_sqr
    E_sqr /= n_inputs
    E_i /= n_images
    E_i_sqr /= n_images
    E_t /= n_texts
    E_t_sqr /= n_texts
    frequency_per_concept = count_per_concept / n_inputs
    modality_score = E_i / (E * 2)
    modality_sqr = E_i_sqr / (E_sqr * 2)

    dead_idx = (E == 0).nonzero(as_tuple=True)[0]
    E_i[dead_idx] = 1000
    E_i_sqr[dead_idx] = 1000
    E_t[dead_idx] = 1000
    E_t_sqr[dead_idx] = 1000
    modality_score[dead_idx] = -1
    modality_sqr[dead_idx] = -1

    if return_sqr:
        return E_sqr, E_i_sqr, E_t_sqr, frequency_per_concept, modality_sqr
    return E, E_i, E_t, frequency_per_concept, modality_score

# Stability :

T = 10
wasss = []
wasss_null = []
for i in range(T):
    if UseTOPK:
        top_k_individual = top_k * 2
        sae1 = overcomplete.sae.BatchTopKSAE(d_model, nb_concepts=nb_concepts, top_k=top_k_individual*batch_size, device=device)
        sae2 = overcomplete.sae.BatchTopKSAE(d_model, nb_concepts=nb_concepts, top_k=top_k_individual*batch_size, device=device)
    else:
        sae1 = overcomplete.sae.MpSAE(d_model, nb_concepts=nb_concepts, k=top_k, device=device)
        sae2 = overcomplete.sae.MpSAE(d_model, nb_concepts=nb_concepts, k=top_k, device=device)
    sae1.to(device)
    sae2.to(device)
    sae1.train()
    sae2.train()
    optimizer1 = torch.optim.Adam(sae1.parameters(), lr=5e-4)
    optimizer2 = torch.optim.Adam(sae2.parameters(), lr=5e-4)

    scheduler=None
    steps_per_epoch = len(train_loader)
    scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer1, max_lr=lr, total_steps=epochs * steps_per_epoch,
    )
    scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer2, max_lr=lr, total_steps=epochs * steps_per_epoch,
    )
    logs1 = train.train_multimodal_sae(
        sae1, train_loader, criterion, optimizer1, scheduler=scheduler1, nb_epochs=epochs, device=device,
        monitoring=1, verbose=False,
        checkpoint_path=None,
    )
    logs2 = train.train_multimodal_sae(
        sae2, train_loader, criterion, optimizer2, scheduler=scheduler2, nb_epochs=epochs, device=device,
        monitoring=1, verbose=False,
        checkpoint_path=None,
    )
    energy_per_concept_sqr_1, _, _, _, _ = measure_everything(sae1, train_loader, device, return_sqr=True)
    energy_per_concept_sqr_2, _, _, _, _ = measure_everything(sae2, train_loader, device, return_sqr=True)
    a = energy_per_concept_sqr_1.cpu().numpy() / np.sum(energy_per_concept_sqr_1.cpu().numpy())
    b = energy_per_concept_sqr_2.cpu().numpy() / np.sum(energy_per_concept_sqr_2.cpu().numpy())

    D1 = sae1.dictionary._fused_dictionary
    D2 = sae2.dictionary._fused_dictionary

    wasserstein = metrics.Wasserstein(D1, D2, a, b, metric="cosim")
    wasserstein_null = metrics.Wasserstein(D1, D2, a, b[np.random.permutation(len(b))], metric="cosim")
    wasss.append(wasserstein)
    wasss_null.append(wasserstein_null)
    print("Wasserstein distance: ", wasserstein)
    print("Wasserstein distance null: ", wasserstein_null)

print("Wasserstein distance: ", np.mean(wasss))
print("Wasserstein distance: ", (wasss))
print("Wasserstein distance null: ", np.mean(wasss_null))
print("Wasserstein distance null: ", (wasss_null))