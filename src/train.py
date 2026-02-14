from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def train(model, train_dataloader, criterion, optimizer, device):

    model.train()
    train_loss = 0.0
    acc = 0.0
    total_train = 0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_train += batch_size

        _, predicted = torch.max(outputs, 1)
        train_loss += loss.item() * batch_size
        acc += (predicted == labels).sum().item() 

    avg_loss = train_loss/total_train
    avg_accuracy = 100*acc/total_train

    return avg_loss, avg_accuracy

def test(model, test_dataloader, criterion, device):
    model.eval()

    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)
            total_test += batch_size
            correct_test += (preds == labels).sum().item()

    avg_loss = test_loss / total_test
    accuracy = 100.0 * correct_test / total_test

    return avg_loss, accuracy

def plot_train_result(train_accs, train_losses, val_accs, val_losses, img_path):
    sns.set_style("whitegrid")
    epochs = range(1, len(train_accs) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(epochs, train_accs, label = "Training", marker = 'o')
    axs[0].plot(epochs, val_accs, label = "Validation", marker = 'o')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy over Epochs")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, train_losses, label = "Training", marker = 'o')
    axs[1].plot(epochs, val_losses, label = "Validation", marker = 'o')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Losses")
    axs[1].set_title("Losses over Epochs")
    axs[1].legend()
    axs[1].grid(True)

    plt.savefig(img_path)

    plt.tight_layout()
    plt.show()

