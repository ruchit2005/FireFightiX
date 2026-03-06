import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")

# Load model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("fire_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# CHANGE IMAGE PATH
img_path = "test_fire.jpg"

img = Image.open(img_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

fire_prob = probs[0][1].item()

print("Fire probability:", fire_prob)

if fire_prob > 0.6:
    print("🔥 FIRE DETECTED")
else:
    print("SAFE")