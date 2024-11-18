import torch
import torch_geometric as pyg
from tqdm import tqdm
import torch_cluster
import sklearn
from torch_cluster import random_walk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class PQWalkDataset(Dataset):
    def __init__(self, walks_tensor):
        """
        Custom dataset for PQ-walks.

        Args:
            walks_tensor (torch.Tensor): Precomputed walks. Shape [num_walks, walk_length + 1].
        """
        self.walks_tensor = walks_tensor

    def __len__(self):
        return self.walks_tensor.size(0)

    def __getitem__(self, idx):
        walk = self.walks_tensor[idx]
        return walk

class Node2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        """
        Node2Vec embedding model.

        Args:
            num_nodes (int): Total number of nodes in the graph.
            embedding_dim (int): Dimension of the embedding space.
        """
        super(Node2Vec, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, nodes):
        # Return embeddings for given nodes
        return self.node_embeddings(nodes)

def generate_negative_samples(batch_size, num_nodes, num_neg_samples):
    """
    Generate negative samples for Node2Vec training.

    Args:
        batch_size (int): Number of positive samples.
        num_nodes (int): Total number of nodes.
        num_neg_samples (int): Number of negative samples per positive sample.

    Returns:
        torch.Tensor: Negative samples of shape [batch_size, num_neg_samples].
    """
    neg_samples = torch.randint(0, num_nodes, (batch_size, num_neg_samples))
    return neg_samples

def node2vec_loss(model, pos_u, pos_v, neg_v):
    """
    Compute the Node2Vec loss with negative sampling.

    Args:
        model (Node2Vec): The Node2Vec model.
        pos_u (torch.Tensor): Positive source nodes.
        pos_v (torch.Tensor): Positive target nodes.
        neg_v (torch.Tensor): Negative target nodes.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Positive samples
    u_emb = model(pos_u)  # shape [batch_size, embedding_dim]
    v_emb = model(pos_v)  # shape [batch_size, embedding_dim]
    pos_score = torch.mul(u_emb, v_emb).sum(dim=1)  # shape [batch_size]
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

    # Negative samples
    neg_emb = model(neg_v.view(-1))  # shape [batch_size * num_neg_samples, embedding_dim]
    neg_emb = neg_emb.view(neg_v.size(0), neg_v.size(1), -1)  # [batch_size, num_neg_samples, embedding_dim]
    u_emb_expanded = u_emb.unsqueeze(1)  # shape [batch_size, 1, embedding_dim]
    neg_score = torch.bmm(neg_emb, u_emb_expanded.transpose(1, 2)).squeeze()  # [batch_size, num_neg_samples]
    neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-15).mean()

    return pos_loss + neg_loss

def get_accuracy(model, embeddings, y, mask):
    out = model(embeddings[mask])
    pred = out.argmax(dim=1)
    acc = sklearn.metrics.accuracy_score(y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    return acc

if __name__ == "__main__":
    # Find device
    if torch.cuda.is_available():  # NVIDIA
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():  # Apple M1/M2
        device = torch.device('mps')
        print("Using MPS GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print("Loading dataset...")
    dataset = pyg.datasets.Planetoid(root='./dataset/cora', name='Cora')
    cora = dataset[0]
    print("Dataset loaded.")

    # Extract edge_index and nodes
    edge_index = cora.edge_index  # shape [2, num_edges]
    num_nodes = cora.x.size(0)  # number of nodes, e.g., 2708 for Cora

    # Define parameters for random walks
    num_walks_per_node = 10  # Number of walks per node
    walk_length = 5  # Length of each walk
    context_size = 2  # Context window size

    print("Generating random walks...")
    # Generate PQ-walks for each node in the graph
    start = torch.arange(num_nodes).repeat_interleave(num_walks_per_node)
    walks = random_walk(edge_index[0], edge_index[1], start, walk_length)
    all_walks_tensor = walks  # Shape: [num_walks * (walk_length + 1)]

    # Reshape to [num_walks, walk_length + 1]
    all_walks_tensor = all_walks_tensor.view(-1, walk_length + 1)
    print("Random walks generated.")

    # Create the PQWalkDataset
    walk_dataset = PQWalkDataset(all_walks_tensor)

    # Define DataLoader with batch size and shuffling
    batch_size = 64
    train_loader = DataLoader(walk_dataset, batch_size=batch_size, shuffle=True)

    # Parameters
    embedding_dim = 128
    num_epochs = 10
    num_neg_samples = 5  # Negative samples per positive pair
    learning_rate = 0.01

    # Instantiate the Node2Vec model
    embedding = Node2Vec(num_nodes=num_nodes, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(embedding.parameters(), lr=learning_rate)

    print("Starting Node2Vec training...")
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        # Wrap the DataLoader with tqdm for a progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for walks in train_loader:
                walks = walks.to(device)  # [batch_size, walk_length + 1]
                batch_size_walks, walk_length_plus_one = walks.size()

                # Generate positive pairs and negative samples
                batch_loss = 0
                for walk in walks:
                    nodes = walk
                    for i in range(len(nodes)):
                        context_indices = list(range(max(0, i - context_size), i)) + \
                                          list(range(i + 1, min(len(nodes), i + context_size + 1)))
                        if len(context_indices) == 0:
                            continue  # Skip if context is empty
                        pos_v = nodes[context_indices]
                        pos_u = nodes[i].repeat(len(pos_v))
                        # Generate negative samples
                        neg_v = generate_negative_samples(pos_u.size(0), num_nodes, num_neg_samples).to(device)
                        # Compute loss
                        loss = node2vec_loss(embedding, pos_u, pos_v, neg_v)
                        batch_loss += loss.item()
                        # Backpropagation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total_loss += batch_loss
                pbar.update(1)
                pbar.set_postfix({'Batch Loss': batch_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.4f}")
    print("Node2Vec training completed.")

    print("Visualizing Node2Vec embeddings using t-SNE...")
    # Visualize embeddings using t-SNE
    with torch.no_grad():
        embeddings = embedding.node_embeddings.weight.cpu().numpy()
        y = cora.y.cpu().numpy()

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y, cmap='tab10')
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title('Node2Vec Embeddings Visualized with t-SNE')
        plt.savefig('node2vec_embeddings.png')
        plt.show()
    print("Node2Vec embeddings visualization saved as 'node2vec_embeddings.png'.")

    print("Starting MLP training for node classification...")
    # Define the MLP model for node classification
    model = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 256),  # Input layer
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # Hidden layer
        torch.nn.ReLU(),
        torch.nn.Linear(128, dataset.num_classes),  # Output layer
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define an optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Define loss function

    cora = cora.to(device)

    print("Training MLP model...")
    for epoch in range(100):  # 100 epochs
        model.train()
        optimizer.zero_grad()
        out = model(embedding.node_embeddings.weight[cora.train_mask])  # Forward pass
        loss = criterion(out, cora.y[cora.train_mask])
        loss.backward()
        optimizer.step()

        # Print out loss info
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.3e}")
    print("MLP training completed.")

    # Compute accuracy
    print("Evaluating model performance...")
    model.eval()
    with torch.no_grad():
        train_acc = get_accuracy(model, embedding.node_embeddings.weight, cora.y, cora.train_mask)
        val_acc = get_accuracy(model, embedding.node_embeddings.weight, cora.y, cora.val_mask)
        test_acc = get_accuracy(model, embedding.node_embeddings.weight, cora.y, cora.test_mask)

    print(f"Node classification accuracy for Cora: {test_acc:.2f} (train: {train_acc:.2f}, val: {val_acc:.2f})")

    print("Visualizing MLP output embeddings using t-SNE...")
    # Visualize the MLP output embeddings using t-SNE
    with torch.no_grad():
        output_embeddings = model[0](embedding.node_embeddings.weight).cpu().numpy()
        embeddings_2d = tsne.fit_transform(output_embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y, cmap='tab10')
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title('MLP Output Embeddings Visualized with t-SNE')
        plt.savefig('mlp_output_embeddings.png')
        plt.show()
    print("MLP output embeddings visualization saved as 'mlp_output_embeddings.png'.")
