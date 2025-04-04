import torch
import torch.optim as optim
import copy

from torch import nn

from chord import *
from network import AudioCNN
import pickle

from data_preprocesser import preprocess

from torch.utils.data import Dataset, DataLoader
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Add a channel dimension to the feature
        feature = feature.unsqueeze(0)  # New shape will be [1, 128, 44]

        return feature, label

def save_label_encoder(label_encoder, file_name='label_encoder.pkl'):
    with open(file_name, 'wb') as file:
        pickle.dump(label_encoder, file)

def save_num_unique_chords(num_unique_chords, file_name='num_unique_chords.pkl'):
    with open(file_name, 'wb') as file:
        pickle.dump(num_unique_chords, file)

def train(batch_size, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Create dataset instances
    x_train, y_train, x_val, y_val, x_test, y_test, num_unique_chords, label_encoder = preprocess()
    save_label_encoder(label_encoder)
    save_num_unique_chords(num_unique_chords)
    train_dataset = AudioDataset(x_train, y_train)
    val_dataset = AudioDataset(x_val, y_val)
    test_dataset = AudioDataset(x_test, y_test)

    # Create DataLoader instances
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = AudioCNN(num_unique_chords)
    model.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    # Training Loop
    num_epochs = epochs
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())  # To save the best model weights
    patience = 10  # For early stopping
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):  # Assuming you have a DataLoader
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_model_checkpoint.pth')
            patience_counter = 0  # Reset the patience counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping!')
                break

        scheduler.step(val_loss)  # Adjust the learning rate

    # Load best model weights
    model.load_state_dict(torch.load('best_model_checkpoint.pth'))

    # Set model to evaluation mode
    model.eval()

    # Testing Loop with NC Handling
    correct = 0
    total = 0
    num_classes = num_unique_chords

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            for i in range(len(labels)):
                expected_label = label_encoder.inverse_transform([labels[i].cpu().numpy()])[0]
                predicted_label = label_encoder.inverse_transform([predicted[i].cpu().numpy()])[0]

                correct_prediction = False

                # Handle 'NC' cases
                if expected_label == 'NC' or predicted_label == 'NC':
                    correct_prediction = (expected_label == predicted_label)
                else:
                    expected_chord = parse_chord_string(expected_label)
                    predicted_chord = parse_chord_string(predicted_label)

                    # Check compatibility
                    correct_prediction = isCompatible(predicted_chord, expected_chord)

                correct += correct_prediction

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

