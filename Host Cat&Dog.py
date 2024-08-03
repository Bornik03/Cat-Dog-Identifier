from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)  # Adjusted size based on training input size
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load your PyTorch model state_dict
model = SimpleCNN()
model.load_state_dict(torch.load("C:\\Users\\borni\\OneDrive\\Desktop\\Jupyter\\cat_dog_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the input size used during training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('html2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['image']

        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type, allowed types are png, jpg, jpeg'}), 400

        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        labels = ["Cat", "Dog"]
        predicted_label = labels[predicted.item()]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
