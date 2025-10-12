import torch

from causal_nf.sem_equations.sem_base import SEM

class Fork4(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 + u2,  # x2 = x1 + u2
                lambda x1, x2, u3: 0.25 * x2 - 1.5 * x1 + 0.5 * u3,
                lambda _, __, x3, u4: 1.0 * x3 + 0.25 * u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1,  # u2 = x2 - x1
                lambda x1, x2, x3: (x3 - 0.25 * x2 + 1.5 * x1) / 0.5,
                lambda _, __, x3, x4: (x4 - 1.0 * x3) / 0.25,
            ]
        elif sem_name == "non-linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 + u2,  # x2 = x1 + u2
                # x3 = softplus(x1*sin(x1²) + 4cos(2x2²-3x2) + u3)
                lambda x1, x2, u3: torch.nn.functional.softplus(
                    x1 * torch.sin(x1**2) + 4 * torch.cos(2 * x2**2 - 3 * x2) + u3
                ),
                # x4 = softplus(sin(x3²) + u4)
                lambda _, __, x3, u4: torch.nn.functional.softplus(
                    torch.sin(x3**2) + u4
                ),
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1,  # u2 = x2 - x1
                # Inverse: u3 = ln(exp(x3) - 1) - x1*sin(x1²) - 4cos(2x2²-3x2)
                lambda x1, x2, x3: torch.log(torch.exp(x3) - 1) - x1 * torch.sin(x1**2) - 4 * torch.cos(2 * x2**2 - 3 * x2),
                # Inverse: u4 = ln(exp(x4) - 1) - sin(x3²)
                lambda _, __, x3, x4: torch.log(torch.exp(x4) - 1) - torch.sin(x3**2),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0])  # x2 depends on x1
        adj[2, :] = torch.tensor([1, 1, 0, 0])
        adj[3, :] = torch.tensor([0, 0, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [1, 2]