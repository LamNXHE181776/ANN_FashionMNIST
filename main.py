import os
from train_cnn import Trainer
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

def main():
    cnn_trainer = Trainer(type="CNN")
    summary = cnn_trainer.model_summary()

    with open("results/CNN/cnn_model_summary.txt", "w") as f:
        f.write(str(summary))

    cnn_trainer.preview_data(cols=4, rows=4)
    cnn_trainer.train_model(epochs=30, save_path="./saved_models/CNN")
    cnn_trainer.plot_training_results(plot_path="./results/CNN")
    cnn_trainer.test_model(model_path="./saved_models/CNN/best_fashion_model.pth")
    

    ann_trainer = Trainer(type="ANN")

    with open("results/ANN/ann_model_summary.txt", "w") as f:
        f.write(str(ann_trainer.model_summary()))

    ann_trainer.preview_data(cols=4, rows=4)
    ann_trainer.train_model(epochs=30, save_path="./saved_models/ANN")
    ann_trainer.plot_training_results(plot_path="./results/ANN")  
    ann_trainer.test_model(model_path="./saved_models/ANN/best_fashion_model.pth")

    os.makedirs("results", exist_ok=True)

    comparison_data = {
        "Epoch": list(range(1, 31)),
        "CNN_Train_Acc": cnn_trainer.train_accs,
        "CNN_Val_Acc": cnn_trainer.val_accs,
        "CNN_Train_Loss": cnn_trainer.train_losses,
        "CNN_Val_Loss": cnn_trainer.val_losses,
        "CNN_Time": cnn_trainer.time,
        "ANN_Train_Acc": ann_trainer.train_accs,
        "ANN_Val_Acc": ann_trainer.val_accs,
        "ANN_Train_Loss": ann_trainer.train_losses,
        "ANN_Val_Loss": ann_trainer.val_losses,
        "ANN_Time": ann_trainer.time
    }

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv("results/comparison_data.csv", index=False)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    axs[0,0].set_title("CNN vs ANN Training and Validation Accuracy")
    axs[0,0].plot(cnn_trainer.train_accs, label="CNN Train Accuracy")
    axs[0,0].plot(cnn_trainer.val_accs, label="CNN Validation Accuracy")
    axs[0,0].plot(ann_trainer.train_accs, label="ANN Train Accuracy")
    axs[0,0].plot(ann_trainer.val_accs, label="ANN Validation Accuracy")

    axs[0,0].set_xlabel("Epochs")
    axs[0,0].set_ylabel("Accuracy (%)")
    axs[0,0].legend()
    axs[0,0].grid()

    axs[0,1].set_title("CNN vs ANN Training and Validation Loss")
    axs[0,1].plot(ann_trainer.train_losses, label="ANN Train Loss")
    axs[0,1].plot(ann_trainer.val_losses, label="ANN Validation Loss")
    axs[0,1].plot(cnn_trainer.train_losses, label="CNN Train Loss")
    axs[0,1].plot(cnn_trainer.val_losses, label="CNN Validation Loss")
    axs[0,1].set_xlabel("Epochs")
    axs[0,1].set_ylabel("Losses")
    axs[0,1].legend()
    axs[0,1].grid()

    axs[0,2].set_title("ANN vs CNN Learning Rate")
    axs[0,2].plot(ann_trainer.lr, label="ANN LR")
    axs[0,2].plot(cnn_trainer.lr, label="CNN LR")
    axs[0,2].set_xlabel("Epochs")
    axs[0,2].set_ylabel("Learning Rate")
    axs[0,2].legend()
    axs[0,2].grid()

    axs[1,0].set_title("Test Accuracy Comparison")
    axs[1,0].bar(["CNN", "ANN"], [cnn_trainer.test_acc, ann_trainer.test_acc], color=['blue', 'orange'], width=0.3)
    axs[1,0].set_ylim(87, 92)
    axs[1,0].set_ylabel("Accuracy (%)")
    axs[1,0].grid()

    axs[1,1].set_title("Test Loss Comparison")
    axs[1,1].bar(["CNN", "ANN"], [cnn_trainer.test_loss, ann_trainer.test_loss], color=['blue', 'orange'], width=0.3)
    axs[1,1].set_ylabel("Loss")
    axs[1,1].grid()

    axs[1,2].set_title("Training Time Comparison")
    axs[1,2].bar(["CNN", "ANN"], [sum(cnn_trainer.time), sum(ann_trainer.time)], color=['blue', 'orange'], width=0.3)
    axs[1,2].set_ylim(200, max(sum(cnn_trainer.time), sum(ann_trainer.time)) * 1.2)
    axs[1,2].set_ylabel("Time (s)")
    axs[1,2].grid()

    plt.tight_layout()
    plt.savefig("results/comparison_cnn_ann.png")
    plt.show()


if __name__ == "__main__":
    main()    