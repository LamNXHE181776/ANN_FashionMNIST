import torch
import matplotlib.pyplot as plt
import seaborn as sns

def preview_test(cols, rows, model, dataset_test, device):
    model.eval()

    fig, axs = plt.subplots(rows, cols, figsize=(7, 7))
    axs = axs.flatten()

    class_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    with torch.no_grad():
        for ax in axs:
            sample_idx = torch.randint(len(dataset_test), (1,)).item()
            img, label = dataset_test[sample_idx]

            img = img.to(device)
            logits = model(img.unsqueeze(0))
            pred_idx = logits.argmax(dim=1).item()

            predicted = class_names[pred_idx]
            actual = class_names[int(label)]

            ax.axis("off")
            ax.set_title(f"Actual: {actual}\nPred: {predicted}")

            ax.imshow(img.squeeze().cpu(), cmap="gray")

    plt.tight_layout()
    plt.show()

