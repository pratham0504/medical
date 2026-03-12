import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MNIST_CNN
from data_loader import get_available_class_names

def predict_single_image(image_path):
    device = torch.device("cpu")

    checkpoint = torch.load("models/global_fedavg.pth", map_location=device)
    num_classes = checkpoint["fc2.weight"].shape[0]
    model = MNIST_CNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: 
        print("Image not found.")
        return
    
    image = Image.fromarray(image)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    names = get_available_class_names()
    if prediction >= len(names):
        print(f"PREDICTION CLASS INDEX: {prediction}")
        return
    print(f"PREDICTION: {names[prediction]}")

if __name__ == "__main__":
    path = input("Enter image path: ")
    predict_single_image(path)