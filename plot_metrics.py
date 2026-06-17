import matplotlib.pyplot as plt
import re
import os

log_file = "training_log.txt"
if not os.path.exists(log_file):
    print("Log file not found.")
    exit()

with open(log_file, "r") as f:
    lines = f.readlines()

train_loss = []
train_acc = []
val_acc = []

# Pattern to extract: Epoch [01/25] Train Loss: 1.2170 Train Acc: 44.0% Val Acc: 25.4%
# Note: sometimes Val Acc is missing if no val loader
pattern = re.compile(r"Train Loss: ([\d.]+) Train Acc: ([\d.]+)%(?: Val Acc: ([\d.]+)%)?")

for line in lines:
    if "Train Loss:" in line:
        match = pattern.search(line)
        if match:
            train_loss.append(float(match.group(1)))
            train_acc.append(float(match.group(2)))
            if match.group(3):
                val_acc.append(float(match.group(3)))
            else:
                val_acc.append(None)

epochs = list(range(1, len(train_loss) + 1))

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'r-', label='Train Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
if len([v for v in val_acc if v is not None]) > 0:
    # Filter out None values for val_acc
    val_epochs = [e for e, v in zip(epochs, val_acc) if v is not None]
    val_acc_values = [v for v in val_acc if v is not None]
    plt.plot(val_epochs, val_acc_values, 'g-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_loss_graph.png", dpi=300)
print("Saved accuracy_loss_graph.png")
