import torch
import torch_geometric as pyg
import ogb.graphproppred
import ogb.graphproppred.mol_encoder
from tqdm import tqdm
import torch_scatter
import sklearn

if torch.cuda.is_available(): # NVIDIA
    device = torch.device('cuda')
elif torch.backends.mps.is_available(): # apple M1/M2
    device = torch.device('mps')
else:
    device = torch.device('cpu')
device

# Loaders for the dataset

dataset = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func")
peptides_train = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func", split="train")
peptides_val   = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func", split="val")
peptides_test  = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func", split="test")

batch_size = 32
train_loader = pyg.loader.DataLoader(peptides_train, batch_size = batch_size, shuffle = True)
val_loader = pyg.loader.DataLoader(peptides_val, batch_size = batch_size, shuffle = True)
test_loader = pyg.loader.DataLoader(peptides_test, batch_size = batch_size, shuffle = True)





### Tasks:


### 1. Implement virtual nodes
### 2. Implement GINE (GIN+Edge features) based on the sparse implementation from Exercise 2
### 3. Test everything (especially the effects of virtual nodes and edge features) on peptides-func.
### 4. Draw the molecule peptides_train[0]




#Layers for the GCN
#Standard GCN Layer
class GCNLayer(torch.nn.Module):  #Standard GCN Layer
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayer, self).__init__()
        self.activation = activation
        self.W: torch.Tensor = torch.nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.kaiming_normal_(self.W)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
        out = H.clone()
        new_H = torch_scatter.scatter_add(H[edge_index[0]], edge_index[1], dim=0, dim_size=out.shape[0])
        out = out + new_H
        out = out.matmul(self.W)
        if self.activation:
            out = self.activation(out)
        return out

#GCN Layer with Edge Features
class GCNLayerWithVirtualNode(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayerWithVirtualNode, self).__init__()
        self.activation = activation
        self.W = torch.nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.kaiming_normal_(self.W)

        # Virtual node parameter
        self.virtual_node = torch.nn.Parameter(torch.zeros(1, in_features))
        torch.nn.init.zeros_(self.virtual_node)  # Initialize virtual node to zeros

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
        # Step 1: Standard message passing with GCN
        out = H.clone()
        new_H = torch_scatter.scatter_add(H[edge_index[0]], edge_index[1], dim=0, dim_size=out.shape[0])
        out = out + new_H

        # Step 2: Aggregate information into virtual node
        virtual_node_msg = torch.mean(out, dim=0, keepdim=True)  # Aggregate mean of all nodes
        self.virtual_node = self.virtual_node + virtual_node_msg  # Update virtual node features

        # Step 3: Distribute virtual node information back to all nodes
        out = out + self.virtual_node

        # Step 4: Apply transformation and activation
        out = out.matmul(self.W)
        if self.activation:
            out = self.activation(out)
        return out




