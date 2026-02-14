from src.data import data_transform, preview_image
from torchvision import datasets
from models.CNN import CNN
from models.ANN import ANN
from src.train import train, test, plot_train_result
from src.test import preview_test

import torch
from torch import nn, optim
from torchinfo import summary

import time
import os

ROOT = "./data"

class Trainer:
    def __init__(self, type = "CNN"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.dataset_full = datasets.FashionMNIST(root=ROOT, train=True, download=True)
        self.dataset_test = datasets.FashionMNIST(root=ROOT, train=False, download=True)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = data_transform(self.dataset_full, self.dataset_test, batch_size=64)

        self.input_size = (1, 1, 28, 28)
        self.num_classes = 10

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.time = []
        self.lr = []

        self.test_acc = None
        self.test_loss = None

        if type == "ANN":
            self.model = ANN(self.input_size, self.num_classes).to(self.device)
        else:
            self.model = CNN(self.input_size, self.num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

    def preview_data(self, cols, rows):
        preview_image(cols, rows, self.dataset_full)
    
    def model_summary(self):
        a = summary(self.model, input_size=self.input_size)
        return a
    
    def train_model(self, epochs, save_path):
        
        best_evals = float("inf")

        os.makedirs(save_path, exist_ok=True)

        for epoch in range(epochs):
            time_start = time.time()

            train_loss, train_acc = train(self.model, 
                                          self.train_dataloader, 
                                          self.criterion, self.optimizer, 
                                          self.device)
            
            val_loss, val_acc = test(self.model, 
                                     self.val_dataloader, 
                                     self.criterion, 
                                     self.device)

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

            time_end = time.time()

            self.time.append(time_end - time_start)

            print(f"\n==== Epoch          {epoch+1}/{epochs}  \n"
                  f"==== Train Loss:    {train_loss:.4f}  \n"
                  f"==== Train Acc:     {train_acc:.2f}%  \n"
                  f"==== Val Loss:      {val_loss:.4f}  \n"
                  f"==== Val Acc:       {val_acc:.2f}%  \n"
                  f"==== Time:          {time_end - time_start:.2f}s  \n"
                  f"==== LR:            {lr:.6f}  \n")
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.lr.append(lr)
        
            if val_loss < best_evals:
                best_evals = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_fashion_model.pth"))
                print("Best model save in path: ", os.path.join(save_path, "best_fashion_model.pth"))

        torch.save(self.model.state_dict(), os.path.join(save_path, "final_fashion_model.pth"))
   
        return self.train_losses, self.train_accs, self.val_losses, self.val_accs, self.time, self.lr
    def plot_training_results(self, plot_path):
        os.makedirs(plot_path, exist_ok=True)
        plot_train_result(self.train_accs, 
                          self.train_losses, 
                          self.val_accs, 
                          self.val_losses,
                          os.path.join(plot_path, "training_validation_results.png"))
    
    def test_model(self, model_path):
        
        if isinstance(self.model, ANN):
            model = ANN(self.input_size, self.num_classes).to(self.device)
        else:
            model = CNN(self.input_size, self.num_classes).to(self.device)

        model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.test_loss, self.test_acc = test(model, 
                                self.test_dataloader, 
                                self.criterion,
                                self.device)
        
        print(f"\n==== Test Loss: {self.test_loss:.4f} ===="
              f"\n==== Test Acc: {self.test_acc:.2f}% ====\n")
        
        preview_test(cols=4, rows=4, 
                     model=model, 
                     dataset_test=self.dataset_test, 
                     device=self.device)
        
        return self.test_loss, self.test_acc
    

