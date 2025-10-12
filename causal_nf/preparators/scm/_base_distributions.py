import torch
import torch.distributions as distr

from causal_nf.distributions.heterogeneous import Heterogeneous
from torch.distributions import Independent, Normal, Uniform, Laplace

pu_dict = {}

def base_distribution_3_nodes(version=1):
    if version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(3),
                torch.ones(3),
            ),
            1,
        )
    elif version == 2:
        p_u = Independent(
            Laplace(
                torch.zeros(3),
                torch.ones(3),
            ),
            1,
        )
    elif version == 3:
        p_u = Independent(
            Uniform(
                torch.zeros(3),
                torch.ones(3),
            ),
            1,
        )

    elif version == 4:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=1.0)

        mix_u2 = distr.Categorical(torch.ones(1, 2))
        comp_u2 = distr.Normal(
            loc=torch.tensor([[0.0, 1.0]]), scale=torch.tensor([0.2])
        )
        p_u2 = distr.MixtureSameFamily(
            mixture_distribution=mix_u2, component_distribution=comp_u2
        )

        p_u3 = distr.Uniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3]), 1)
    else:
        raise NotImplementedError(f"Version {version} of p_u not implemented.")
    return p_u


def base_distribution_4_nodes(version=1):
    if version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(4),
                torch.ones(4),
            ),
            1,
        )
    elif version == 2:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u2 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u3 = distr.Laplace(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u4 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3, p_u4]), 1)
    elif version == 3:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u2 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u3 = distr.LogNormal(loc=torch.tensor([0.0]), scale=torch.tensor([0.5]))
        p_u4 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        
        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3, p_u4]), 1)
    elif version == 4:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u2 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u3 = distr.LogNormal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))  # scale=2.0 for variance=4
        p_u4 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        
        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3, p_u4]), 1)

    return p_u


def base_distribution_5_nodes(version=1):
    if version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )
    elif version == 2:
        p_u = Independent(
            Laplace(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )
    elif version == 3:
        p_u = Independent(
            Uniform(
                torch.zeros(5),
                torch.ones(5),
            ),
            1,
        )

    return p_u


def base_distribution_9_nodes(version=1):
    if version == 1:
        p_u = Independent(
            Uniform(
                1e-6,
                torch.ones(9),
            ),
            1,
        )
    elif version == 2:
        raise NotImplementedError(f"Version {version} of p_u not implemented.")

    return p_u

def base_distribution_10_nodes(version=1):
    if version == 1:
        p_u = Independent(
            Normal(
                torch.zeros(10),
                torch.ones(10),
            ),
            1,
        )
    elif version == 2:
        p_u1 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u2 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u3 = distr.Laplace(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u4 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u5 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u6 = distr.Laplace(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u7 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u8 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u9 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        p_u10 = distr.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        
        p_u = Independent(Heterogeneous(distr_list=[p_u1, p_u2, p_u3, p_u4, p_u5, p_u6, p_u7, p_u8, p_u9, p_u10]), 1)

    return p_u

pu_dict[3] = base_distribution_3_nodes
pu_dict[4] = base_distribution_4_nodes
pu_dict[5] = base_distribution_5_nodes
pu_dict[9] = base_distribution_9_nodes
pu_dict[10] = base_distribution_10_nodes
