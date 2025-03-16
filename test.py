import os 
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm

model_dir = "models"
available_models = [f[:-3] for f in os.listdir(model_dir) if f.endswith(".py") and f != "__init__"]

if not available_models:
    raise FileNotFoundError("No available model")

print("available models: ")
for i, model_name in enumerate(available_models):
    print(f"{i+1}.{model_name}")

choice = int(input("Please input the index: ")) - 1
model_path = os.path.join(model_dir, available_models[choice])
print(f"select: {model_path}")

model_module = importlib.import_module(f"models.{model_name}")
ModelClass = getattr(model_module, model_name)
print(f"successfully load: {model_name} ")

weight_dir = os.path.join(os.getcwd(), "weights")
available_weights = [f for f in os.listdir(weight_dir) if f.startswith(model_name) and f.endswith(".pth")]

if not available_weights:
    raise FileNotFoundError(f"No available weights")

print("\n available weights:")
for i, weight_name in enumerate(available_weights):
    print(f"{i + 1}. {weight_name}")

choice = int(input("Please input the index: ")) - 1
weight_path = os.path.join(weight_dir, available_weights[choice])
print(f"successfully load: {weight_path}")

transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

data_dir = "./datasets"
test_dataset = ImageFolder(root=f"{data_dir}/val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(test_dataset.classes)
model = ModelClass(num_classes).to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()
print("model loaded!")

correct = 0
total = 0
with torch.no_grad():
    progress_bar = tqdm(test_loader, desc=f"testing>>")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"accuracy: {accuracy:.2f}%")

