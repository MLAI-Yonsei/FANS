import torch

from causal_nf.sem_equations.sem_base import SEM

class Node10(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                # X1 = tanh(u1) (no parents)
                lambda u1: torch.tanh(u1),
                
                # X2 = tanh(sin(X1^2) + u2) (parent: X1)  
                lambda x1, u2: torch.tanh(torch.sin(x1**2) + u2),
                
                # X3 = tanh(u3) (no parents, but receives x1, x2)
                lambda _, __, u3: torch.tanh(u3),
                
                # X4 = tanh(sin(X1^2) + u4) (parent: X1, but receives x1, x2, x3)
                lambda x1, _, __, u4: torch.tanh(torch.sin(x1**2) + u4),
                
                # X5 = tanh(u5) (no parents, but receives x1, x2, x3, x4)
                lambda _, __, ___, ____, u5: torch.tanh(u5),
                
                # X6 = tanh(sin(X2^2) + u6) (parent: X2, receives x1, x2, x3, x4, x5)
                lambda _, x2, __, ___, ____, u6: torch.tanh(torch.sin(x2**2) + u6),
                
                # X7 = tanh(sin(X1^2) + u7) (parent: X1, receives x1...x6)
                lambda x1, _, __, ___, ____, _____, u7: torch.tanh(torch.sin(x1**2) + u7),
                
                # X8 = tanh(sin(X1^2) + u8) (parent: X1, receives x1...x7)
                lambda x1, _, __, ___, ____, _____, ______, u8: torch.tanh(torch.sin(x1**2) + u8),
                
                # X9 = tanh(sin(X1^2) + sin(X5^2) + u9) (parents: X1, X5, receives x1...x8)
                lambda x1, _, __, ___, x5, ____, _____, ______, u9: torch.tanh(torch.sin(x1**2) + torch.sin(x5**2) + u9),
                
                # X10 = tanh(sin(X4^2) + sin(X6^2) + sin(X7^2) + sin(X8^2) + u10) (parents: X4, X6, X7, X8)
                lambda _, __, ___, x4, ____, x6, x7, x8, ______, u10: torch.tanh(torch.sin(x4**2) + torch.sin(x6**2) + torch.sin(x7**2) + torch.sin(x8**2) + u10),
            ]
            
            inverses = [
                # u1 = atanh(X1)
                lambda x1: torch.atanh(torch.clamp(x1, -0.999, 0.999)),
                
                # u2 = atanh(X2) - sin(X1^2)
                lambda x1, x2: torch.atanh(torch.clamp(x2, -0.999, 0.999)) - torch.sin(x1**2),
                
                # u3 = atanh(X3)
                lambda _, __, x3: torch.atanh(torch.clamp(x3, -0.999, 0.999)),
                
                # u4 = atanh(X4) - sin(X1^2)
                lambda x1, _, __, x4: torch.atanh(torch.clamp(x4, -0.999, 0.999)) - torch.sin(x1**2),
                
                # u5 = atanh(X5)
                lambda _, __, ___, ____, x5: torch.atanh(torch.clamp(x5, -0.999, 0.999)),
                
                # u6 = atanh(X6) - sin(X2^2)
                lambda _, x2, __, ___, ____, x6: torch.atanh(torch.clamp(x6, -0.999, 0.999)) - torch.sin(x2**2),
                
                # u7 = atanh(X7) - sin(X1^2)
                lambda x1, _, __, ___, ____, _____, x7: torch.atanh(torch.clamp(x7, -0.999, 0.999)) - torch.sin(x1**2),
                
                # u8 = atanh(X8) - sin(X1^2)
                lambda x1, _, __, ___, ____, _____, ______, x8: torch.atanh(torch.clamp(x8, -0.999, 0.999)) - torch.sin(x1**2),
                
                # u9 = atanh(X9) - sin(X1^2) - sin(X5^2)
                lambda x1, _, __, ___, x5, ____, _____, ______, x9: torch.atanh(torch.clamp(x9, -0.999, 0.999)) - torch.sin(x1**2) - torch.sin(x5**2),
                
                # u10 = atanh(X10) - sin(X4^2) - sin(X6^2) - sin(X7^2) - sin(X8^2)
                lambda _, __, ___, x4, ____, x6, x7, x8, ______, x10: torch.atanh(torch.clamp(x10, -0.999, 0.999)) - torch.sin(x4**2) - torch.sin(x6**2) - torch.sin(x7**2) - torch.sin(x8**2),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((10, 10))
        
        # adj[child, :] format: each row represents the parents of the corresponding node
        adj[0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X1: no parents
        adj[1, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X2: X1 is parent
        adj[2, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X3: no parents
        adj[3, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X4: X1 is parent
        adj[4, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X5: no parents
        adj[5, :] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # X6: X2 is parent
        adj[6, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X7: X1 is parent
        adj[7, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # X8: X1 is parent
        adj[8, :] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # X9: X1, X5 are parents
        adj[9, :] = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1, 0, 0])  # X10: X4, X6, X7, X8 are parents

        if add_diag:
            adj += torch.eye(10)
        return adj

    def intervention_index_list(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]