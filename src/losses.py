"""
Module containing loss functions for the Sparse Autoencoder (SAE) model.

In the Overcomplete library, the loss functions are defined as standalone functions.
They all share the same signature:
    - x: torch.Tensor
        Input tensor.
    - x_hat: torch.Tensor
        Reconstructed tensor.
    - pre_codes: torch.Tensor
        Encoded tensor before activation function.
    - codes: torch.Tensor
        Encoded tensor.
    - dictionary: torch.Tensor
        Dictionary tensor.

Additional arguments can be passed as keyword arguments.
"""

from functools import partial

import torch
from overcomplete.metrics import hoyer, kappa_4, lp, l1, dead_codes

# disable W0613 (unused-argument) to keep the same signature for all loss functions
# pylint: disable=W0613


def _mse_with_penalty(x, x_hat, pre_codes, codes, dictionary, penalty=1.0, penalty_fn=None):
    """
    Compute the Mean Squared Error (MSE) loss with a given penalty function.

    Loss = ||x - x_hat||^2 + penalty * penalty_fn(z)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float
        Penalty coefficient.
    penalty_fn : callable
        Penalty function.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    mse = (x - x_hat).square().mean()

    if penalty_fn is None:
        reg = 0.0
    else:
        reg = torch.mean(penalty_fn(codes)) * penalty

    return mse + reg


# l1 should be dependent on the codes dimension, hoyer and kappa are not
mse_l1 = partial(_mse_with_penalty, penalty_fn=l1)
mse_hoyer = partial(_mse_with_penalty, penalty_fn=hoyer)
mse_kappa_4 = partial(_mse_with_penalty, penalty_fn=kappa_4)


def mse_elastic(x, x_hat, pre_codes, codes, dictionary, alpha=0.5):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes.

    Loss = ||x - x_hat||^2 + (1 - alpha) * ||z||_1 + alpha * ||D||^2

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    alpha : float, optional
        Alpha coefficient in the Elastic-net loss, control the ratio of l1 vs l2.
        alpha=0 means l1 only, alpha=1 means l2 only.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    assert 0.0 <= alpha <= 1.0

    mse = (x - x_hat).square().mean()

    l1_loss = codes.abs().mean()
    l2_loss = dictionary.square().mean()

    loss = mse + (1.0 - alpha) * l1_loss + alpha * l2_loss

    return loss


def top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary, penalty=0.1):
    """
    The Top-K Auxiliary loss (AuxK).

    The loss is defined in the original Top-K SAE paper:
        "Scaling and evaluating sparse autoencoders"
        by Gao et al. (2024).

    Similar to Ghost-grads, it consist in trying to "revive" the dead codes
    by trying the predict the residual using the 50% of the top non choosen codes.

    Loss = ||x - x_hat||^2 + penalty * ||x - (x_hat D * top_half(z_pre - z)||^2

    @tfel the order actually matter here! residual is x - x_hat and
    should be in this specific order.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    """
    # select the 50% of non choosen codes and predict the residual
    # using those non choosen codes
    # the code choosen are the non-zero element of codes

    residual = x - x_hat
    mse = residual.square().mean()

    pre_codes = torch.relu(pre_codes)
    pre_codes = pre_codes - codes  # removing the choosen codes

    auxiliary_topk = torch.topk(pre_codes, k=pre_codes.shape[1] // 2, dim=1)
    pre_codes = torch.zeros_like(codes).scatter(-1, auxiliary_topk.indices,
                                                auxiliary_topk.values)

    residual_hat = pre_codes @ dictionary
    auxilary_mse = (residual - residual_hat).square().mean()

    loss = mse + penalty * auxilary_mse

    return loss


def reanimation_regularizer(x, x_hat, pre_codes, codes, dictionary, penalty=0.1):
    """
    Additional term to the loss function that tries to revive dead codes.

    The idea is to increase the value of the `pre_codes` to allow the dead_codes
    to fire again by pushing the `pre_codes` towards the positive orthant.

        reg = -1/(sum_i dead_mask_i) * sum_i(pre_codes_i * dead_mask_i).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    """
    # codes that havent fired in this batch
    dead_mask = dead_codes(codes)

    # push the `pre_codes` towards the positive orthant
    reg = -(pre_codes * dead_mask).sum() / (dead_mask.sum() + 1e-6)
    reg = reg * penalty

    return reg


def alignment_penalty(x, x_hat, pre_codes, codes, dictionary, penalty=0.03, alignment_metric="cosim"):
    """
    Additional term to the loss function that tries to the first and second half of the codes.
    In multimodal cases (or when joint training a pair of dictionaries),
    one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    alignment_metric : str, optional
        Metric to use for the alignment penalty. Can be "l1" or "l2".
        Default is "l1".
    """

    # split the codes into two halves
    assert codes.shape[0] % 2 == 0, "batch size must be even"
    n = codes.shape[0] // 2
    
    # codes_1 = codes[:n] # shape: (n, d)
    # codes_2 = codes[n:] # shape: (n, d)

    # arange = torch.arange(n, device=codes.device)

    # mask_1 = torch.ones_like(codes_1)
    # mask_2 = torch.ones_like(codes_2)
    
    # k = 2
    # top_k_idxs_1 = torch.topk(codes_1, k=k, dim=1).indices # shape: (n, k)
    # top_k_idxs_2 = torch.topk(codes_2, k=k, dim=1).indices # shape: (n, k)
    # for i in range(k):
    #     top_i_idxs_1 = top_k_idxs_1[:, i]  # shape: (n,)
    #     top_i_idxs_2 = top_k_idxs_2[:, i]  # shape: (n,)
    #     mask_1[arange, top_i_idxs_1] = 0.0
    #     mask_1[arange, top_i_idxs_2] = 0.0
    #     mask_2[arange, top_i_idxs_1] = 0.0
    #     mask_2[arange, top_i_idxs_2] = 0.0
    
    codes_1 = codes[:n] # shape: (n, d)
    codes_2 = codes[n:] # shape: (n, d)

    top_k_idxs_1 = torch.topk(codes_1, k=1, dim=1).indices.squeeze(1) # shape: (n,)
    top_k_idxs_2 = torch.topk(codes_2, k=1, dim=1).indices.squeeze(1) # shape: (n,)

    arange = torch.arange(n, device=codes.device)

    mask_1 = torch.ones_like(codes_1)
    mask_2 = torch.ones_like(codes_2)
    mask_1[arange, top_k_idxs_1] = 0.0
    mask_1[arange, top_k_idxs_2] = 0.0
    mask_2[arange, top_k_idxs_1] = 0.0
    mask_2[arange, top_k_idxs_2] = 0.0

    masked_1 = codes_1 * mask_1
    masked_2 = codes_2 * mask_2

    # compute the alignment penalty
    if alignment_metric == "l1":
        misalignment_score = (masked_1 - masked_2).abs().mean()
    elif alignment_metric == "l2":
        misalignment_score = (masked_1 - masked_2).square().mean()
    elif alignment_metric == "cosim" or "cosine" or "cosine_similarity":
        misalignment_score = - torch.nn.functional.cosine_similarity(masked_1, masked_2, dim=1).mean()
    else:
        raise ValueError(f"Alignment metric must be 'l1' or 'l2', got {alignment_metric}")
    
    # print(f"Alignment penalty: {misalignment_score.item()}")
    
    return misalignment_score * penalty

def alignment_penalty_prime(x, x_hat, pre_codes, codes, dictionary, penalty=0.03, alignment_metric="cosim"):
    """
    Additional term to the loss function that tries to the first and second half of the codes.
    In multimodal cases (or when joint training a pair of dictionaries),
    one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    alignment_metric : str, optional
        Metric to use for the alignment penalty. Can be "l1" or "l2".
        Default is "l1".
    """

    # split the codes into two halves
    assert codes.shape[0] % 2 == 0, "batch size must be even"
    n = codes.shape[0] // 2
    
    codes_1 = codes[:n] # shape: (n, d)
    codes_2 = codes[n:] # shape: (n, d)

    # compute the alignment penalty
    if alignment_metric == "l1":
        misalignment_score = (codes_1 - codes_2).abs().mean()
    elif alignment_metric == "l2":
        misalignment_score = (codes_1 - codes_2).square().mean()
    elif alignment_metric == "cosim" or "cosine" or "cosine_similarity":
        misalignment_score = - torch.nn.functional.cosine_similarity(codes_1, codes_2, dim=1).mean()
    else:
        raise ValueError(f"Alignment metric must be 'l1' or 'l2', got {alignment_metric}")
    
    # print(f"Alignment penalty: {misalignment_score.item()}")
    
    return misalignment_score * penalty

def alignment_penalty_distribution(x, x_hat, pre_codes, codes, dictionary, penalty=0.03):
    """
    Additional term to the loss function that tries to the first and second half of the codes.
    In multimodal cases (or when joint training a pair of dictionaries),
    one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    alignment_metric : str, optional
        Metric to use for the alignment penalty. Can be "l1" or "l2".
        Default is "l1".
    """
    # split the codes into two halves
    assert codes.shape[0] % 2 == 0, "batch size must be even"
    n = codes.shape[0] // 2
    
    codes_1 = codes[:n] # shape: (n, d)
    codes_2 = codes[n:] # shape: (n, d)

    # compute the alignment penalty
    codes_1_E = (codes_1 ** 2).mean(dim=0)  # mean of squares for each code. shape: (d,)
    codes_2_E = (codes_2 ** 2).mean(dim=0)
    
    misalignment_score = (((codes_1_E - codes_2_E) / (codes_1_E + codes_2_E + 1e-5))).mean()
    
    # print(f"Misalignment score: {misalignment_score.item()}, penalty: {penalty}, misalignment_score * penalty: {(misalignment_score * penalty).item()}")
    
    return misalignment_score * penalty

def alignment_penalty_distribution_prime(x, x_hat, pre_codes, codes, dictionary, penalty=0.03):
    """
    Additional term to the loss function that tries to the first and second half of the codes.
    In multimodal cases (or when joint training a pair of dictionaries),
    one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    alignment_metric : str, optional
        Metric to use for the alignment penalty. Can be "l1" or "l2".
        Default is "l1".
    """

    # split the codes into two halves
    assert codes.shape[0] % 2 == 0, "batch size must be even"
    n = codes.shape[0] // 2
    
    codes_1 = codes[:n] # shape: (n, d)
    codes_2 = codes[n:] # shape: (n, d)

    # compute the alignment penalty
    codes_1_E = (codes_1 ** 2).mean(dim=0)  # mean of squares for each code. shape: (d,)
    codes_2_E = (codes_2 ** 2).mean(dim=0)
    
    E_tot = (codes_1_E + codes_2_E) / 2
    E_tot = E_tot.sum().item()
    misalignment_score = (((codes_1_E - codes_2_E) / E_tot)).mean()
    
    # print(f"Alignment penalty: {misalignment_score.item()}, penalty: {penalty}, misalignment_score * penalty: {(misalignment_score * penalty).item()}")
    
    return misalignment_score * penalty
