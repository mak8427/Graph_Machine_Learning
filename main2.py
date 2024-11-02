import torch
import torch_geometric as pyg
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
import torch_scatter
import copy
import time
import random
from tqdm import tqdm

# Find device
if torch.cuda.is_available():  # NVIDIA
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # Apple M1/M2
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device:", device)

# Define GCN Layer
class GCNLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = pyg.utils.add_self_loops(edge_index, num_nodes=H.size(0))
        row, col = edge_index
        deg = pyg.utils.degree(row, H.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        H = self.linear(H)
        H = H[col] * norm.unsqueeze(-1)
        H = torch_scatter.scatter_add(H, row, dim=0)

        if self.activation is not None:
            H = self.activation(H)
        return H

# Modify GraphNet to include an embedding layer
class GraphNet(torch.nn.Module):
    def __init__(self, num_node_types: int, out_features: int, hidden_features: int, activation=torch.nn.functional.relu, dropout=0.1):
        super(GraphNet, self).__init__()
        self.embedding = torch.nn.Embedding(num_node_types, hidden_features)
        self.gcn1 = GCNLayer(hidden_features, hidden_features, activation)
        self.gcn2 = GCNLayer(hidden_features, hidden_features, activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_features, out_features)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor, batch=None):
        H = self.embedding(H.squeeze(-1))
        H = self.gcn1(H, edge_index)
        H = self.gcn2(H, edge_index)
        H = self.dropout(H)
        if batch is not None:
            H = pyg.nn.global_mean_pool(H, batch)
        out = self.linear(H)
        return out.squeeze()

def get_accuracy(model, graph, mask):
    model.eval()
    with torch.no_grad():
        outputs = model(graph.x, graph.edge_index)
    correct = (outputs[mask].argmax(-1) == graph.y[mask]).sum()
    return int(correct) / int(mask.sum())

if __name__ == '__main__':
    # Load Cora dataset
    print("Loading Cora dataset...")
    cora = pyg.datasets.Planetoid(root="dataset/cora", name="Cora")
    cora_graph = cora[0].to(device)
    print("Cora dataset loaded.")

    # Model parameters for Cora
    in_features = cora_graph.num_node_features
    hidden_features = 16
    out_features = cora.num_classes
    learning_rate = 0.01
    weight_decay = 5e-4
    num_epochs = 200

    # Define GraphNetCora for Cora dataset
    class GraphNetCora(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, hidden_features: int, activation=torch.nn.functional.relu, dropout=0.1):
            super(GraphNetCora, self).__init__()
            self.gcn1 = GCNLayer(in_features, hidden_features, activation)
            self.dropout = torch.nn.Dropout(dropout)
            self.gcn2 = GCNLayer(hidden_features, out_features, activation=None)

        def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
            H = self.gcn1(H, edge_index)
            H = self.dropout(H)
            H = self.gcn2(H, edge_index)
            return H

    # Initialize model, loss function, and optimizer for Cora
    model = GraphNetCora(in_features, out_features, hidden_features, dropout=0.5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_mask = cora_graph.train_mask
    val_mask = cora_graph.val_mask
    test_mask = cora_graph.test_mask

    print("\nStarting training on Cora dataset...")
    # Training loop for Cora
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(cora_graph.x, cora_graph.edge_index)
        loss = criterion(out[train_mask], cora_graph.y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = get_accuracy(model, cora_graph, val_mask)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')

    print("Finished training on Cora dataset.")

    print("\nStarting testing on Cora dataset...")
    test_acc = get_accuracy(model, cora_graph, test_mask)
    print(f'Test Accuracy on Cora dataset: {test_acc:.4f}')
    print("Finished testing on Cora dataset.")

    # Load ZINC dataset
    print("\nLoading ZINC dataset...")
    dataset = pyg.datasets.ZINC(root='dataset/ZINC', split='train', subset=True)
    dataset_val = pyg.datasets.ZINC(root='dataset/ZINC', split='val', subset=True)
    dataset_test = pyg.datasets.ZINC(root='dataset/ZINC', split='test', subset=True)
    print("ZINC dataset loaded.")

    batch_size = 128
    num_workers = 0  # Set to 0 for compatibility

    train_loader = pyg.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = pyg.loader.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = pyg.loader.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Determine the number of node types in the ZINC dataset
    def get_num_node_types(dataset):
        node_types = set()
        for data in dataset:
            node_types.update(data.x.squeeze(-1).tolist())
        return max(node_types) + 1  # Assuming node types start from 0

    num_node_types = get_num_node_types(dataset)

    # Model parameters for ZINC
    hidden_features = 128
    out_features = 1
    learning_rate = 0.001
    num_epochs = 100

    # Initialize model, loss function, and optimizer for ZINC
    model = GraphNet(num_node_types, out_features, hidden_features, dropout=0.1).to(device)
    criterion = torch.nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting training on ZINC dataset...")
    # Training loop for ZINC
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        total_loss /= len(train_loader.dataset)
        print(f'Epoch: {epoch+1}, Training Loss: {total_loss:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
            val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')
    print("Finished training on ZINC dataset.")

    print("\nStarting testing on ZINC dataset...")
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            test_loss += loss.item() * data.num_graphs
        test_loss /= len(test_loader.dataset)
    print(f'Test Loss on ZINC dataset: {test_loss:.4f}')
    print("Finished testing on ZINC dataset.")

#%%
