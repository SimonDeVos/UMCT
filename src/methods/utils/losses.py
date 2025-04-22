# LOAD MODULES
# Standard library

# Third party
import torch

def mmd_rbf_nclass(
    cluster: torch.Tensor,
    num_cluster: int,
    hidden: torch.Tensor,
    sigma: float = 0.1,
    comp_with_all: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Calculates an approximation of the Maximum Mean Discrepancy (MMD) with a Radial Basis Function (RBF) kernel when multiple classes are in the data.

    Parameters:
        cluster (torch.Tensor): The clusters. A 1D tensor with the cluster for each observation.
        num_cluster (int): The number of clusters.
        hidden (torch.Tensor): The hidden states. A 2D tensor where each row is a hidden state and each column is a feature.
        sigma (float, optional): The sigma parameter for the RBF kernel. Defaults to 0.1.
        comp_with_all (bool, optional): Whether to compute the MMD with all clusters or only the current one. Defaults to True.
        kwargs: Dummy to improve compatibility with other functions.

    Returns:
        torch.Tensor: The MMD.
    """
    # ini mmd
    mmd = 0.0

    for i in range(num_cluster):
        # Catch error if no (or only one) observation for head i, and assign no loss in that case
        if hidden[cluster == i].shape[0] <= 1 or hidden[cluster != i].shape[0] <= 1:
            mmd = mmd
        else:
            # Get subset
            subset = hidden[cluster == i]
            num_subset = subset.shape[0]

            # Get other obs
            if comp_with_all:
                other = hidden
            else:
                other = hidden[cluster != i]
            num_other = other.shape[0]

            # Calc kernel representations
            K_tt = torch.exp(-torch.cdist(subset, subset) / (sigma**2))
            K_tc = torch.exp(-torch.cdist(subset, other) / (sigma**2))
            K_cc = torch.exp(-torch.cdist(other, other) / (sigma**2))

            # Take average of kernels - Do not add diag. for K_tt and K_cc
            avg_K_tt = (1 / (num_subset * (num_subset - 1))) * (torch.sum(K_tt) - num_subset)
            avg_K_tc = (1 / (num_other * num_subset)) * torch.sum(K_tc)
            avg_K_cc = (1 / (num_other * (num_other - 1))) * (torch.sum(K_cc) - num_other)

            # Add to joint mmd
            mmd = mmd + (avg_K_tt - 2 * avg_K_tc + avg_K_cc)

    # Get average
    mmd = mmd / num_cluster

    return mmd

def mmd_lin_nclass(
    cluster: torch.Tensor,
    num_cluster: int,
    hidden: torch.Tensor,
    comp_with_all: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Calculates an approximation of the Maximum Mean Discrepancy (MMD) with a linear kernel when multiple classes are in the data.

    Parameters:
        cluster (torch.Tensor): The clusters. A 2D tensor where each row is a cluster and each column is a feature.
        num_cluster (int): The number of clusters.
        hidden (torch.Tensor): The hidden states. A 2D tensor where each row is a hidden state and each column is a feature.
        comp_with_all (bool, optional): Whether to compute the MMD with all clusters or only the current one. Defaults to True.
        kwargs: Dummy to improve compatibility with other functions.

    Returns:
        torch.Tensor: The MMD.
    """
    # ini mmd
    mmd = 0.0

    # Calculate means and divergence
    for i in range(num_cluster):
        # Catch error if no observation for head i, and assign no loss in that case
        if hidden[cluster == i].shape[0] <= 1 or hidden[cluster != i].shape[0] <= 1:
            mmd = mmd
        else:
            # Subset
            subset = hidden[cluster == i]
            # Other
            if comp_with_all:
                other = hidden
            else:
                other = hidden[cluster != i]

            # Means of obs with head i
            means_subset = torch.mean(subset, 0)
            # Get means of complements
            means_other = torch.mean(other, 0)
            # Increment mdd by sum of sqaured divergences multiplied by weight of cluster
            mmd = mmd + torch.sqrt(torch.sum(torch.square(means_subset - means_other)))

    # Calculate average
    mmd = mmd / num_cluster

    return mmd

def mmd_none(**kwargs):
    """
    A dummy function that returns 0.

    Returns:
        torch.Tensor: A tensor with value 0.
    """
    return torch.tensor(0)