import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Data pipeline
transform=transforms.Compose([
 transforms.ToTensor(),
 transforms.Normalize((0.1307,), (0.3081,))
]
)

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

#Architecture Design
model = nn.Sequential(
   
   nn.Flatten(),
   nn.Linear(28*28,128),
   nn.ReLU(),                  
   nn.Linear(128, 64),         
   nn.ReLU(),                  
   nn.Linear(64, 10)



)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

for epoch in range(5):  
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        labels = labels.long()
        loss = criterion(outputs, labels)

       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()

test_loss = 0.0
correct = 0
total = 0        


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        final_accuracy = 100 * correct / total
        print(f"Test Accuracy: {final_accuracy:.2f}%") 

def predict_digit(image_tensor, model):
 
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  
        
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    confidence_scores = probabilities.squeeze(0).tolist()
    return predicted_class, confidence_scores


sample_img, sample_label = test_dataset[0]
predicted_digit, conf_scores = predict_digit(sample_img, model)

print(f"\nGround Truth Label: {sample_label}")
print(f"Predicted Digit: {predicted_digit}")
print("Confidence Scores (0-9):")
for digit, prob in enumerate(conf_scores):
    print(f"  Digit {digit}: {prob * 100:.2f}%")