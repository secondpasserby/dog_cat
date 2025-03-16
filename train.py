import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.models as models
import os
import importlib

# preprocessing
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

# load data
data_dir = '/media/zhangsc/Little_Disk/ntu/ee6483/datasets'
train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# load model
model_dir = "models"
available_models = [f[:-3] for f in os.listdir(model_dir) if f.endswith(".py")]

if not available_models:
    raise FileNotFoundError("No model found, please add model in file model")

print("avail models: ")
for i, model_name in enumerate(available_models):
    print(f"{i + 1}. {model_name}")

choice = int(input("input the train model index: ")) - 1
model_name = available_models[choice]
print(f"select model: {model_name}")

model_module = importlib.import_module(f"models.{model_name}")
ModelClass = getattr(model_module, model_name)
print(f"successful load {model_name} ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)
model = ModelClass(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []

# train
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item()) 

    avg_loss = running_loss/len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

model_path = os.path.join('weights', f"{model_name}_epoch{epoch+1}.pth")
torch.save(model.state_dict(), model_path)
print("model saved as model.pth")

# print loss
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()
