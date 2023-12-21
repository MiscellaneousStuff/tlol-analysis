import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

EMBEDS = {
    "champ": 30,
    "turret": 18,
    "minion": 18,
    "missile": 11,
    "monster": 18
}

EMBED_SETS = {
    "champ": 10,
    "turret": 30,
    "minion": 30,
    "missile": 30,
    "monster": 30
}

def set_all_seeds(seed):
    # Python's built-in random module
    random.seed(seed)

    # Numpy's random number generator
    np.random.seed(seed)

    # PyTorch's random number generator for CPU and GPU
    torch.manual_seed(seed)

    # If you are using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Additionally, you might want to disable benchmark mode for convolution operations
        torch.backends.cudnn.benchmark = False

        # And ensure deterministic operations
        torch.backends.cudnn.deterministic = True

set_all_seeds(42)

class TLoLDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.files = os.listdir(dataset_dir)
        self.files = \
            [fi for fi in self.files
             if ".npy" in fi]
        with open("cols.txt") as f:
            self.cols = f.read().split("\n")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        full_path = os.path.join(self.dataset_dir, self.files[i])
        # print(full_path)
        vals = np.load(full_path)
        replay_df = pd.DataFrame(vals, columns=self.cols)
        # print(replay_df.head())
        # replay_df = pd.read_csv(full_path)
        obs = replay_df.iloc[:, :-17]
        act = replay_df.iloc[:, -17:]
        return obs, act

class ProcessSet(nn.Module):
    def __init__(self, input_size, set_size, output_size):
        # print("process set input_size, set_size, output_size", input_size, set_size, output_size)
        super(ProcessSet, self).__init__()
        self.fc1 = nn.Linear(input_size, set_size)
        self.fc2 = nn.Linear(set_size, output_size)

    def forward(self, x):
        # x shape expected: [batch_size, num_entities, input_size]
        # print("process set input:", x.shape)
        x = F.relu(self.fc1(x))
        # print("process set emb:", x.shape)
        x = F.relu(self.fc2(x))
        # print("process set emb:", x.shape)
        x, _ = torch.max(x, dim=1)  # Apply max-pooling over num_entities dimension
        # print("process set max:", x.shape)
        return x


class ProcessSetModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        
        self.champ_process_set = ProcessSet(
            EMBEDS["champ"],
            EMBED_SETS["champ"],
            hidden_dim)
        
        self.fc1 = nn.Linear(1 + hidden_dim, hidden_dim)
        # self.fc2 = nn.Linear(in_dim // 2, in_dim // 2)
        # self.fc3 = nn.Linear(in_dim // 2, hidden_dim)      

        self.move_x = nn.Linear(hidden_dim, 9)
        self.move_y = nn.Linear(hidden_dim, 9)

    def forward(self, x):
        # Split observation by object type

        # 1. Time (1 per time)
        tm = torch.tensor(x[:, 0]).unsqueeze(1)

        # 2. Allied champions (30 per champ * 10)
        champ_data = x[:, 1:(EMBEDS["champ"] * EMBED_SETS["champ"]) + 1] # [rows, cols - time]
        # print("champ_data.shape:", champ_data.shape, EMBEDS["champ"] * EMBED_SETS["champ"])
        # champ_data = champ_data.view(-1, EMBED_SETS["champ"], EMBEDS["champ"])  # Reshape for ProcessSet
        champ_data = champ_data.reshape(-1, EMBED_SETS["champ"], EMBEDS["champ"])
        # print("champ_data.shape:", champ_data.shape)
        champ_processed = self.champ_process_set(champ_data)
        # x = self.champ_process_set()

        # print("tm.shape, champ_processed.shape:", tm.shape, champ_processed.shape)
        x = torch.concat((tm, champ_processed), axis=1)
        x = self.fc1(x)
        # print("TIME + CHAMPS:", x.shape)

        m_x = self.move_x(x)
        m_y = self.move_y(x)
        return m_x, m_y

class OriginalModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim // 2)
        self.fc2 = nn.Linear(in_dim // 2, in_dim // 2)
        self.fc3 = nn.Linear(in_dim // 2, hidden_dim)
        self.move_x = nn.Linear(hidden_dim, 9)
        self.move_y = nn.Linear(hidden_dim, 9)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        m_x = self.move_x(x)
        m_y = self.move_y(x)
        return m_x, m_y
    
def collate_fn(batch):
    # batch is a list of tuples where each tuple is (observation, action)
    
    # Separate observations and actions
    obs_list, act_list = zip(*batch)

    # Convert lists of observations and actions into tensors
    obs_tensor = torch.cat([torch.tensor(obs.values, dtype=torch.float32) for obs in obs_list])
    x_target_tensor = torch.cat([torch.tensor(act.iloc[:, 0:1].values, dtype=torch.long) + 4 for act in act_list])
    y_target_tensor = torch.cat([torch.tensor(act.iloc[:, 1:2].values, dtype=torch.long) + 4 for act in act_list])

    return obs_tensor, x_target_tensor, y_target_tensor

def train(correct, total, accuracies, optimizer, obs, act, ROW_MAX, COL_MAX, log=False):
    # obs, act = dataset[game]
    optimizer.zero_grad()

    obs_vals = obs.iloc[0:ROW_MAX, :COL_MAX].values
    np.random.shuffle(obs_vals)
    x = torch.tensor(obs_vals, dtype=torch.float32)

    m_x, m_y = model(x)
    
    x_target = torch.tensor(act.iloc[0:ROW_MAX, 0:1].values, dtype=torch.long) + 4
    y_target = torch.tensor(act.iloc[0:ROW_MAX, 1:2].values, dtype=torch.long) + 4
    
    # print("obs_vals.shape, x_target.shape, y_target.shape:", obs_vals.shape, x_target.shape, y_target.shape)

    # Compute the loss for each axis
    loss_x = criterion(m_x, x_target.squeeze(-1))  # Ensure the target is correctly squeezed
    loss_y = criterion(m_y, y_target.squeeze(-1))  # Ensure the target is correctly squeezed
    
    # Total loss
    loss = loss_x + loss_y
    loss.backward()
    optimizer.step()
    if log:
        losses.append(loss.item())

    _, predicted_x = torch.max(m_x.data, 1)
    _, predicted_y = torch.max(m_y.data, 1)
    total += x_target.size(0)
    correct += (predicted_x == x_target.squeeze(-1)).sum().item()
    correct += (predicted_y == y_target.squeeze(-1)).sum().item()

    # Multiply total by 2 as we have two predictions per sample (x and y)
    accuracy = 100 * correct / (2 * total)  
    if log:
        accuracies.append(accuracy)

    return loss, accuracy, accuracies, correct, total

if __name__ == "__main__":
    MAX_IDX = 100 # 992 replays max
    ROW_MAX = 200
    COL_MAX = 301

    dataset = TLoLDataset("/Users/joe/Downloads/NP-2")
    batch_size = 1  # You can adjust this
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    print("dataset")
    obs, act = dataset[0]
    obs = obs.iloc[0:ROW_MAX, 0:COL_MAX]
    # x_target = torch.tensor(act.iloc[:, 0:1].values, dtype=torch.long) + 4
    # y_target = torch.tensor(act.iloc[:, 1:2].values, dtype=torch.long) + 4

    original_model = OriginalModel(in_dim=obs.shape[1], hidden_dim=512)
    processet_model = ProcessSetModel(in_dim=obs.shape[1], hidden_dim=512) 
    # lr := 1e-3, dim=128, acc=72.50%

    colors = ['r', 'b']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    models = [original_model, processet_model]
    model_names = ["Original", "ProcessSet"]
    for i in range(len(models)):
        model = models[i]
        model_name = model_names[i]
        color = colors[i]

        epochs = 100

        lr = 1e-4
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []
        accuracies = []

        for epoch in range(epochs):
            correct = 0
            total = 0
            i = 1
            for game in range(MAX_IDX):
            # for obs, act in dataloader:
                # print(f"Game {i}/{MAX_IDX}")
                
                obs, act = dataset[game]
                loss, accuracy, accuracies, correct, total = \
                    train(
                        correct,
                        total,
                        accuracies,
                        optimizer,
                        obs,
                        act,
                        ROW_MAX,
                        COL_MAX,
                        log=(game==0))

                i += 1

            if epoch:
                print(f"Model: {model_name}, Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%")

        # Initial plot in subplot 1
        ax1.plot(losses, f'{color}-', label=f'{model_name} Loss')
        ax1.set_title('Original vs Process Set Loss')
        ax1.legend()

        # Initial plot in subplot 2
        ax2.plot(accuracies, f'{color}-', label=f'{model_name} Acc')
        ax2.set_title('Original vs Process Set Acc')
        ax2.legend()

    plt.show()