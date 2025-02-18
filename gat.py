import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
import math

# -------------------------------
# Step 1. Convert SMILES to Graph
# -------------------------------
def smiles_to_data(smiles, label):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    Here, each atom is a node with a simple feature (atomic number).
    Edges are added for every bond in both directions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid molecules

    # Create node features: using the atomic number as a feature
    atom_features = []
    for atom in mol.GetAtoms():
        # You can extend this list with additional features as needed.
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    # Create edge_index from bonds
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        # For molecules with no bonds, create an empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create the label tensor (for classification)
    # Adjust dtype if you are doing regression (e.g. torch.float)
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# -----------------------------------------
# Step 2. Load Dataset and Convert to Graphs
# -----------------------------------------
# Assume your CSV has columns: 'smiles' and 'label'
# df = pd.read_csv("smiles_dataset.csv")
df = pd.read_csv("result_with_ATG_AP_1_CIS_up.csv")

data_list = []
for idx, row in df.iterrows():
    try:
        print(row['smiles'], row['label'])
        data_obj = smiles_to_data(row['smiles'], row['label'])
        if data_obj is not None:
            data_list.append(data_obj)
    except  Exception as e:
        print(f"Ignore error")

# (Optional) Split into train and test sets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# -------------------------------
# Step 3. Define the GAT Model
# -------------------------------
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        """
        A GAT model for graph-level classification.
        Two GAT layers are used, followed by global mean pooling and a linear layer.
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        # First GAT layer; note that multiple heads yield hidden_channels * heads output features.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer; using a single head and not concatenating.
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        # Final linear layer for graph-level prediction
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # data.batch is added by the DataLoader
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # Global pooling: aggregates node features to a graph-level representation
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Set hyperparameters (adjust as needed)
in_channels = 1             # We use the atomic number as the sole feature
hidden_channels = 8
num_classes = df['label'].nunique()  # Number of classes in your dataset
heads = 8
dropout = 0.6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(in_channels, hidden_channels, num_classes, heads=heads, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# -------------------------------
# Step 4. Training and Evaluation
# -------------------------------
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

# Training loop
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    loss = train()
    test_acc = test(test_loader)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
