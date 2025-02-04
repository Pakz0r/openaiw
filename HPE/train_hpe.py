from model import HPEnet, HPERnet, HPECnet, Hopenet
from dataset import PandoraDataset, AIWatchDataset, AFLW2000Dataset
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import os
import utils
import numpy as np

def combined_loss(cross_entropy, rotation_matrix, class_label, predictions, alfa=0.2, beta=0.5):
    # Regression loss
    regression_loss = torch.nn.MSELoss()(predictions[0], rotation_matrix[:, 0]) + torch.nn.MSELoss()(predictions[1], rotation_matrix[:, 1]) + torch.nn.MSELoss()(predictions[2], rotation_matrix[:, 2])
    regression_loss = regression_loss.mean()

    # # Orthogonality loss
    # predicted_rotation_matrix = torch.stack((predictions[0], predictions[1], predictions[2]), dim=1)
    # identity_matrix = torch.eye(3, device=predicted_rotation_matrix.device).unsqueeze(0).expand_as(predicted_rotation_matrix)
    # product_matrices = torch.bmm(predicted_rotation_matrix, torch.transpose(predicted_rotation_matrix, 1, 2))
    # orthogonality_loss = torch.norm(product_matrices - identity_matrix, p='fro') / predicted_rotation_matrix.size(0)

    # # Classification loss
    # classification_loss = cross_entropy(predictions[3], class_label)

    # # Total loss
    # loss = regression_loss.mean() + alfa * orthogonality_loss + beta * classification_loss
    return regression_loss.mean()


# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
generator = torch.Generator()
generator.manual_seed(42)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{timestamp}'
writer = SummaryWriter(log_dir)
parser = argparse.ArgumentParser()
parser.add_argument("--continue_training", action="store_true")
args = parser.parse_args()

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 20

# Data loading
pandora_dataset = PandoraDataset(cached=True)
aflw2000_dataset = AFLW2000Dataset(cached=True)
aiwatch_dataset = AIWatchDataset(cached=True)
dataset = torch.utils.data.ConcatDataset([pandora_dataset, aflw2000_dataset, aiwatch_dataset])

# Retrieve class weights
class_weights = utils.calculate_class_weights(dataset, cached=True)
class_weights = class_weights.to(device=device, dtype=torch.float32)

# Divide the dataset into train, validation and test sets
num_samples = len(dataset)
num_train_samples = int(0.9 * num_samples)
num_val_samples = int(0.05 * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

print(f"Total number of samples: {num_samples}")
print(f"Number of training samples: {num_train_samples}")
print(f"Number of validation samples: {num_val_samples}")
print(f"Number of test samples: {num_test_samples}")

# Split the dataset and create the dataloaders
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train_samples, num_val_samples, num_test_samples], generator)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)

# Training parameters
model = HPEnet().to(device)
if args.continue_training:
    # take the model with lower validation loss
    models_path = "/home/spoleto/hpe/models/"
    models = os.listdir(models_path)
    models.sort()
    model.load_state_dict(torch.load(models_path + models[0]))
    print("Loading weights from: " + models_path + models[0])

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training loop
best_val_loss = float('inf')

print("Starting training...")
epoch_iterator = tqdm(range(num_epochs), desc="Epochs")
for epoch in epoch_iterator:
    # Training step 
    model.train()
    train_loss = 0.0
    train_dataloader_iterator = tqdm(train_dataloader, desc="Training iterations", leave=False)
    for batch_idx, (face, rotation_matrix, class_label) in enumerate(train_dataloader_iterator):
        face, rotation_matrix, class_label = face.to(device=device), rotation_matrix.to(device=device, dtype=torch.float32), class_label.to(device=device, dtype=torch.int64)

        # Forward pass
        predictions = model(face)

        # Compute loss
        loss = combined_loss(cross_entropy, rotation_matrix, class_label, predictions)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dataloader_iterator.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
        writer.add_scalar('Training Batch Loss', loss.item(), batch_idx + len(train_dataloader)*epoch)

    train_loss /= len(train_dataloader)
    writer.add_scalar('Training Loss', train_loss, epoch)

    # Validation step
    model.eval()
    val_loss = 0.0
    val_dataloader_iterator = tqdm(val_dataloader, desc="Validation iterations", leave=False)
    with torch.no_grad():
        for batch_idx, (face, rotation_matrix, class_label) in enumerate(val_dataloader_iterator):
            face, rotation_matrix, class_label = face.to(device=device), rotation_matrix.to(device=device, dtype=torch.float32), class_label.to(device=device, dtype=torch.int64)

            # Forward pass
            predictions = model(face)

            # Compute loss
            loss = combined_loss(cross_entropy, rotation_matrix, class_label, predictions)

            val_loss += loss.item()
            val_dataloader_iterator.set_postfix(loss=f"{loss.item():.4f}", refresh=True)

    val_loss /= len(val_dataloader)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"/home/spoleto/hpe/models/val_{val_loss:.4f}_train_{train_loss:.4f}.pt")
    
    # Save the model
    epoch_iterator.set_description(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")