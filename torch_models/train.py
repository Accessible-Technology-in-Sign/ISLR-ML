import torch
import torch.nn as nn
import torch.optim as optim
from cnn_2d import SimpleCNN2D, ComplexCNN2D, WhisperInspiredClassifier
from lstm import SimpleLSTM, DoubleLSTM, ScaledLSTM, TransformerClassifier, SpatioTemporalTransformerClassifier, ComplexTransformerClassifier, TransformerCNNClassifier, ProjectedLSTM, WorldTestLSTM
from gnn import SimpleGNN
from data import SingleHandH5Dataset_MobileLastN, SingleHandH5Dataset_MobileLastN_VarLen
import sys
import os
# import ai_edge_torch
# os.environ['PJRT_DEVICE'] = 'CPU'

from tqdm import tqdm



# Assuming these are imported or defined elsewhere:
# from model import SimpleCNN2D
# from data import SingleHandH5Dataset_MobileLastN

# Step 1: Load the data
if __name__ == '__main__':
    print("Running train.py")
    model_name = sys.argv[1]
    print(f"Training {model_name}")

    model_type = SpatioTemporalTransformerClassifier
    is_lstm = True
    is_whisper = False
    is_gnn = False
    padding="default"
    num_epochs = 50
    window_length = 90
    batch_size = 2 ** 8
    num_coords = 2
    loss = nn.CrossEntropyLoss()
    # loss = nn.NLLLoss()
    # dataset = SingleHandH5Dataset_MobileLastN(root_dir, "/meta/563_sign_list.txt")

    # Step 2: Split the dataset into train, validation, and test sets
    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.15 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size, test_size])

    # Step 3: Create DataLoaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    train_loader = torch.utils.data.DataLoader(
        SingleHandH5Dataset_MobileLastN("/data/train", "/meta/563_sign_list.txt", padding=padding, n=window_length, is_gnn=is_gnn, is_lstm=is_lstm, is_whisper=is_whisper, debug=False),
        batch_size=batch_size, shuffle=True, 
        num_workers=16
    )
    val_loader = torch.utils.data.DataLoader(
        SingleHandH5Dataset_MobileLastN("/data/validation", "/meta/563_sign_list.txt", padding=padding, n=window_length, is_gnn=is_gnn, is_lstm=is_lstm, is_whisper=is_whisper, debug=False),
        batch_size=batch_size, shuffle=False,
        num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        SingleHandH5Dataset_MobileLastN("/data/test", "/meta/563_sign_list.txt", padding=padding, n=window_length, is_gnn=is_gnn, is_lstm=is_lstm, is_whisper=is_whisper, debug=False),
        batch_size=batch_size, shuffle=False,
        num_workers=16
    )


    # train_loader = torch.utils.data.DataLoader(
    #     SingleHandH5Dataset_MobileLastN_VarLen("/data/train", "/meta/563_sign_list.txt", n=window_length, debug=False),
    #     batch_size=batch_size, shuffle=True, 
    #     collate_fn= lambda batch: (torch.nested.to_padded_tensor(torch.nested.nested_tensor([i[0] for i in batch]), padding=0.0), torch.stack([i[1] for i in batch])),
    #     num_workers=16
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     SingleHandH5Dataset_MobileLastN_VarLen("/data/validation", "/meta/563_sign_list.txt", n=window_length, debug=False),
    #     batch_size=batch_size, shuffle=False,
    #     collate_fn= lambda batch: (torch.nested.to_padded_tensor(torch.nested.nested_tensor([i[0] for i in batch]), padding=0.0), torch.stack([i[1] for i in batch])),
    #     num_workers=16
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     SingleHandH5Dataset_MobileLastN_VarLen("/data/test", "/meta/563_sign_list.txt", n=window_length, debug=False),
    #     batch_size=batch_size, shuffle=False,
    #     collate_fn= lambda batch: (torch.nested.to_padded_tensor(torch.nested.nested_tensor([i[0] for i in batch]), padding=0.0), torch.stack([i[1] for i in batch])),
    #     num_workers=16
    # )


    # Step 4: Initialize the model
    model = model_type(window_length, 21, num_coords, 563)
    print(model)

    # Step 5: Set up the loss function and optimizer
    criterion = loss
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 6: Define training and evaluation functions
    def train_one_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        print("----------TRAIN----------")
        for inputs, labels in tqdm(dataloader):
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float for BCEWithLogitsLoss
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate the running loss
            running_loss += loss.item() * inputs.size(0)
            
        return running_loss / len(dataloader.dataset)

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        new_correct = 0
        total = 0
        with torch.no_grad():
            print("----------EVAL----------")
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels.argmax(dim=1))
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy: convert outputs to probabilities and round them
                predicted = (torch.softmax(outputs, dim=1))  # Threshold at 0.5 for binary decision per class
                new_correct += (predicted.argmax(dim=1) == labels.argmax(dim=1)).sum().item()  # Count correct multi-class matches
                correct += ((predicted  > 0.5).float()==labels).all(dim=1).sum().item()
                total += labels.size(0)
                
                
        accuracy = correct / total
        new_accuracy = new_correct / total
        return running_loss / len(dataloader.dataset), accuracy, new_accuracy, correct, new_correct, total

    # Step 7: Train and evaluate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========RUNNING ON {device}==========")
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"---------Running Epoch {epoch}----------")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_new_accuracy, correct, new_correct, total = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} OR {val_new_accuracy:.4f} | correct, new correct, total | {correct, new_correct, total}")

        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"/saves/{model_name}.pth")
            print(f"saved at: /saves/{model_name}.pth")

    # Step 8: Evaluate on the test set
    model.load_state_dict(torch.load(f"/saves/{model_name}.pth"))
    model.eval()
    test_loss, test_accuracy, test_new_accuracy, correct, new_correct, total = evaluate(model, test_loader, criterion, device)
    scripted_model = torch.jit.script(model)
    scripted_model.save(f"/models/{model_name}.pt")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} OR {test_new_accuracy:.4f} | correct, new correct, total | {correct, new_correct, total}")
