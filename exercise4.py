#%load_ext autoreload
#%autoreload 2
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







class PQWalkDataset(Dataset):
    def __init__(self, walks_tensor, features, labels):
        """
        Custom dataset for PQ-walks with node features and labels.

        Args:
            walks_tensor (torch.Tensor): Precomputed walks. Shape [num_walks, walk_length + 1].
            features (torch.Tensor): Node feature matrix from `data.x`. Shape [num_nodes, num_features].
            labels (torch.Tensor): Node labels from `data.y`. Shape [num_nodes].
        """
        self.walks_tensor = walks_tensor
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.walks_tensor.size(0)

    def __getitem__(self, idx):
        walk = self.walks_tensor[idx]

        # Get features and labels for each node in the walk
        walk_features = self.features[walk]
        walk_labels = self.labels[walk]

        return walk_features, walk_labels









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
    u_emb = model(pos_u)
    v_emb = model(pos_v)
    pos_score = torch.mul(u_emb, v_emb).sum(dim=1)  # dot product
    pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()

    # Negative samples
    neg_emb = model(neg_v)
    neg_score = torch.bmm(neg_emb, u_emb.unsqueeze(2)).squeeze()
    neg_loss = -torch.log(torch.sigmoid(-neg_score)).mean()

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
    elif torch.backends.mps.is_available():  # Apple M1/M2
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    dataset = pyg.datasets.Planetoid(root='./dataset/cora', name='Cora')
    cora = dataset[0]
    dataset = pyg.datasets.PPI(root='./dataset/ppi')
    ppi = dataset[0]

    # for link prediction
    link_splitter = pyg.transforms.RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = link_splitter(cora)


    # Extract edge_index and nodes
    edge_index = cora.edge_index  # shape [2, num_edges]
    num_nodes = cora.x.size(0)  # number of nodes, e.g., 2708 for Cora

    # Define parameters for random walks
    num_walks_per_node = 10  # Number of walks per node
    walk_length = 5  # Length of each walk

    # Generate PQ-walks for each node in the graph
    all_walks = []
    for node in range(num_nodes):
        # Perform random walks starting from the current node
        walks = random_walk(edge_index[0], edge_index[1], node, walk_length, num_walks_per_node)
        all_walks.append(walks)

    # Combine all walks into a single tensor
    all_walks_tensor = torch.cat(all_walks, dim=0)

    # Create the PQWalkDataset
    walk_dataset = PQWalkDataset(all_walks_tensor, cora.x, cora.y)

    # Define DataLoader with batch size and shuffling
    batch_size = 64
    train_loader = DataLoader(walk_dataset, batch_size=batch_size, shuffle=True)

    # Parameters
    embedding_dim = 128
    num_epochs = 10
    num_neg_samples = 5  # Negative samples per positive pair
    learning_rate = 0.01

    # Instantiate the Node2Vec model
    embedding = Node2Vec(num_nodes=num_nodes, embedding_dim=embedding_dim)
    optimizer = optim.Adam(embedding.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for walk_features, _ in train_loader:
            batch_size, walk_length, _ = walk_features.size()

            # Flatten walk features to work with node IDs
            walk_features = walk_features.view(-1)

            # Generate positive pairs within the context window
            for i in range(walk_length):
                start = max(0, i - 2)  # Context window size of 2
                end = min(walk_length, i + 2 + 1)
                pos_u = walk_features[i].repeat(end - start - 1)
                pos_v = torch.cat([walk_features[j] for j in range(start, end) if j != i])

                # Generate negative samples
                neg_v = generate_negative_samples(pos_u.size(0), num_nodes, num_neg_samples)

                # Compute loss
                loss = node2vec_loss(embedding, pos_u, pos_v, neg_v)
                total_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

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

    node2vec_embeddings = embedding.to(device)
    cora = cora.to(device)

    for epoch in range(100):  # 100 epochs
        model.train()
        optimizer.zero_grad()
        out = model(node2vec_embeddings[cora.train_mask])  # Forward pass
        loss = criterion(out, cora.y[cora.train_mask])
        loss.backward()
        optimizer.step()

        # Print out loss info
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.3e}")

    # Compute accuracy
    train_acc = get_accuracy(model, node2vec_embeddings, cora.y, cora.train_mask)
    val_acc = get_accuracy(model, node2vec_embeddings, cora.y, cora.val_mask)
    test_acc = get_accuracy(model, node2vec_embeddings, cora.y, cora.test_mask)

    print(f"Node classification accuracy for Cora: {test_acc:.2f} (train: {train_acc:.2f}, val: {val_acc:.2f})")





    #Link prediction




    # Train embeddings on the training graph (train_data)
    edge_index = train_data.edge_index
    num_nodes = train_data.x.size(0)

    # Define parameters for random walks
    num_walks_per_node = 10
    walk_length = 5

    # Generate PQ-walks for each node in the training graph
    all_walks = []
    for node in range(num_nodes):
        walks = random_walk(edge_index[0], edge_index[1], node, walk_length, num_walks_per_node)
        all_walks.append(walks)

    # Combine all walks into a single tensor
    all_walks_tensor = torch.cat(all_walks, dim=0)

    # Create the PQWalkDataset
    walk_dataset = PQWalkDataset(all_walks_tensor, train_data.x, train_data.y)

    # Define DataLoader with batch size and shuffling
    batch_size = 64
    train_loader = DataLoader(walk_dataset, batch_size=batch_size, shuffle=True)

    # Parameters
    embedding_dim = 128
    num_epochs = 10
    num_neg_samples = 5
    learning_rate = 0.01

    # Instantiate the Node2Vec model
    embedding = Node2Vec(num_nodes=num_nodes, embedding_dim=embedding_dim)
    optimizer = optim.Adam(embedding.parameters(), lr=learning_rate)

    # Training loop for Node2Vec
    for epoch in range(num_epochs):
        total_loss = 0
        for walk_features, _ in train_loader:
            batch_size, walk_length, _ = walk_features.size()
            walk_features = walk_features.view(-1)

            # Generate positive and negative samples for node pairs
            for i in range(walk_length):
                start = max(0, i - 2)  # Context window size of 2
                end = min(walk_length, i + 2 + 1)
                pos_u = walk_features[i].repeat(end - start - 1)
                pos_v = torch.cat([walk_features[j] for j in range(start, end) if j != i])

                neg_v = generate_negative_samples(pos_u.size(0), num_nodes, num_neg_samples)
                loss = node2vec_loss(embedding, pos_u, pos_v, neg_v)
                total_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Perform link prediction
    node2vec_embeddings = embedding.to(device)


    def link_prediction(embedding, edge_label_index, edge_label):
        """
        Perform link prediction by computing similarity scores between node pairs.
        """
        node_emb = embedding.weight  # Node embeddings
        edge_emb_u = node_emb[edge_label_index[0]]  # Embedding for source nodes
        edge_emb_v = node_emb[edge_label_index[1]]  # Embedding for target nodes

        # Compute dot product similarity
        scores = (edge_emb_u * edge_emb_v).sum(dim=1)
        preds = torch.sigmoid(scores)  # Map scores to [0, 1] using sigmoid

        # Compute accuracy for link prediction
        pred_labels = (preds > 0.5).long()
        acc = (pred_labels == edge_label).float().mean().item()
        return acc


    # Evaluate on validation set
    val_acc = link_prediction(node2vec_embeddings, val_data.edge_label_index, val_data.edge_label)
    print(f"Validation Link Prediction Accuracy: {val_acc:.4f}")

    # Evaluate on test set
    test_acc = link_prediction(node2vec_embeddings, test_data.edge_label_index, test_data.edge_label)
    print(f"Test Link Prediction Accuracy: {test_acc:.4f}")