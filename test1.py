import numpy as np
import pandas as pd
import os
import warnings
import librosa
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load data
directory = 'train'
csv_file = "train_gt.csv"
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"File {csv_file} not found")

# Update file paths
data['filename'] = data['filename'].apply(lambda x: os.path.join(directory, x))

# Initialize the Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=3)

# Prepare dataset and dataloaders
data['label'] = data['label'].astype(int)
X_temp, X_test, y_temp, y_test = train_test_split(data, data['label'], test_size=0.2, random_state=101)
X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=0.10, random_state=101)

# Function to preprocess audio files
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio_input = processor(audio, return_tensors='pt', sampling_rate=16000,
                            padding=True, truncation=True, max_length=16000)
    return audio_input.input_values[0]

# Preprocess datasets
X_train_processed = np.array(
    [preprocess_audio(file) for file in tqdm(X_train['filename'], desc="Processing train files")])
X_validation_processed = np.array(
    [preprocess_audio(file) for file in tqdm(X_validation['filename'], desc="Processing validation files")])
X_test_processed = np.array([preprocess_audio(file) for file in tqdm(X_test['filename'], desc="Processing test files")])

# Convert processed data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed)
y_train_tensor = torch.tensor(y_train.values)
X_validation_tensor = torch.tensor(X_validation_processed)
y_validation_tensor = torch.tensor(y_validation.values)
X_test_tensor = torch.tensor(X_test_processed)

# Create DataLoader for training and validation
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()

# Early stopping parameters
patience = 5  # Number of epochs to wait before stopping
min_val_loss = float('inf')  # Initialize the minimum validation loss
counter = 0  # Counter to track how long validation loss has not improved
best_model_path = 'best_model.pth'  # Path to save the best model

# Training the model with early stopping and model saving
epochs = 50
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        audio_inputs, labels = batch
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(audio_inputs).logits
        loss = loss_fn(outputs, labels)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    print(f"Training loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validating"):
            audio_inputs, labels = batch
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)

            outputs = model(audio_inputs).logits
            val_loss += loss_fn(outputs, labels).item()

    val_loss /= len(validation_loader)
    print(f"Validation loss: {val_loss:.4f}")

    # Check early stopping condition
    if val_loss < min_val_loss:
        print(f"Validation loss decreased from {min_val_loss:.4f} to {val_loss:.4f}. Saving model...")
        torch.save(model.state_dict(), best_model_path)
        min_val_loss = val_loss
        counter = 0  # Reset the counter
    else:
        counter += 1
        print(f"No improvement in validation loss for {counter} epoch(s).")
        if counter >= patience:
            print(f"Early stopping triggered after {patience} epochs of no improvement.")
            break

# Load the best model after training is complete
model.load_state_dict(torch.load(best_model_path))
print(f"Best model loaded from {best_model_path}.")

# Make predictions on test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).logits
    y_pred_classes = np.argmax(y_pred.cpu().numpy(), axis=1)

# Evaluate the model
print(classification_report(y_test, y_pred_classes))
