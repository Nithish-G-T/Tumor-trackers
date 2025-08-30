import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io
import os

class TumorPredictor:
    """
    Brain tumor prediction utility class that handles model loading, 
    preprocessing, inference, and Grad-CAM visualization.
    """
    
    def __init__(self, model_path="tumor_model.pth"):
        """
        Initialize the tumor predictor with the trained model.
        
        Args:
            model_path (str): Path to the saved model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self._load_model(model_path)
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _build_model(self):
        """
        Build the EfficientNet-B3 model with the same architecture as trained.
        
        Returns:
            torch.nn.Module: The model architecture
        """
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        # Freeze layers as in training
        freeze_ratio = 0.449
        n_layers = int(len(list(model.features)) * freeze_ratio)
        for i, layer in enumerate(model.features):
            for param in layer.parameters():
                param.requires_grad = i >= n_layers
        
        # Modify classifier for 4 classes
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4626),
            nn.Linear(model.classifier[1].in_features, 4)
        )
        
        return model.to(self.device)
    
    def _load_model(self, model_path):
        """
        Load the trained model weights.
        
        Args:
            model_path (str): Path to the saved model weights
        """
        try:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Model loaded successfully from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found. Using pre-trained weights.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using pre-trained weights as fallback.")
    
    def preprocess_image(self, image):
        """
        Preprocess the input image for model inference.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(image).convert('RGB')
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image):
        """
        Perform tumor prediction on the input image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'class_names': self.class_names
        }

    def generate_gradcam(self, image, target_class=None):
        """
        Generate Grad-CAM visualization for the input image.

        Args:
            image: PIL Image or numpy array
            target_class (int): Target class for Grad-CAM (if None, uses predicted class)

        Returns:
            str: Base64 encoded Grad-CAM image
        """
        try:
            image_tensor = self.preprocess_image(image)

            # Hooks to capture feature maps and gradients
            feature_maps = []
            gradients = []

            def forward_hook(module, input, output):
                feature_maps.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            # Register hooks on last conv layer
            target_layer = self.model.features[-1][0]
            f_handle = target_layer.register_forward_hook(forward_hook)
            b_handle = target_layer.register_full_backward_hook(backward_hook)

            # Forward pass
            outputs = self.model(image_tensor)
            if target_class is None:
                target_class = torch.argmax(outputs, dim=1).item()

            # Backward pass
            self.model.zero_grad()
            outputs[0, target_class].backward()

            # Remove hooks
            f_handle.remove()
            b_handle.remove()

            # Extract maps & gradients
            feature_map = feature_maps[0].detach()
            gradient = gradients[0].detach()

            # Compute weights (GAP)
            weights = torch.mean(gradient, dim=(2, 3), keepdim=True)

            # Compute Grad-CAM
            gradcam = torch.sum(weights * feature_map, dim=1, keepdim=True)
            gradcam = F.relu(gradcam)
            gradcam = F.interpolate(gradcam, size=(300, 300), mode='bilinear', align_corners=False)
            heatmap = gradcam.squeeze().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            # Prepare original image
            original_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min())

            # Visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_np)
            plt.title('Original MRI Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(original_np)
            plt.imshow(heatmap, alpha=0.6, cmap='jet')
            plt.title(f'Grad-CAM: {self.class_names[target_class]}')
            plt.axis('off')

            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()

            return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            return self._create_fallback_visualization(image, target_class)

    def _create_fallback_visualization(self, image, target_class=None):
        """
        Create a fallback visualization when Grad-CAM fails.
        """
        try:
            image_tensor = self.preprocess_image(image)
            original_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min())

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_np)
            plt.title('Original MRI Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(original_np)
            plt.title('Analysis Complete')
            plt.axis('off')

            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()

            return base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            print(f"Fallback visualization also failed: {e}")
            return None

    def generate_medical_summary(self, prediction):
        """
        Generate a medical-style summary based on the prediction.
        """
        tumor_type = prediction['class']
        confidence = prediction['confidence']

        tumor_descriptions = {
            'glioma': {
                'description': 'Gliomas are tumors that arise from glial cells in the brain.',
                'severity': 'Gliomas can be either benign or malignant, with glioblastoma being the most aggressive form.',
                'location': 'Commonly found in the cerebral hemispheres.',
                'symptoms': 'May cause headaches, seizures, personality changes, and neurological deficits.'
            },
            'meningioma': {
                'description': 'Meningiomas are tumors that arise from the meninges, the protective membranes covering the brain.',
                'severity': 'Most meningiomas are benign and slow-growing.',
                'location': 'Typically found on the surface of the brain.',
                'symptoms': 'May cause headaches, vision problems, and seizures depending on location.'
            },
            'pituitary': {
                'description': 'Pituitary tumors are growths that develop in the pituitary gland.',
                'severity': 'Most pituitary tumors are benign adenomas.',
                'location': 'Located at the base of the brain in the pituitary gland.',
                'symptoms': 'May cause hormonal imbalances, vision problems, and headaches.'
            },
            'no_tumor': {
                'description': 'No evidence of brain tumor detected in the MRI scan.',
                'severity': 'Normal brain tissue appearance.',
                'location': 'N/A',
                'symptoms': 'No tumor-related symptoms expected.'
            }
        }

        info = tumor_descriptions.get(tumor_type, tumor_descriptions['no_tumor'])

        if confidence >= 0.95:
            confidence_level = "very high"
        elif confidence >= 0.85:
            confidence_level = "high"
        elif confidence >= 0.75:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        if tumor_type == 'glioma':
            malignancy = 'MALIGNANT'
        elif tumor_type in ['meningioma', 'pituitary', 'no_tumor']:
            malignancy = 'BENIGN'
        else:
            malignancy = 'UNKNOWN'

        summary = f"""
MEDICAL ANALYSIS REPORT

DIAGNOSIS: {tumor_type.upper().replace('_', ' ')}
MALIGNANCY: {malignancy}
CONFIDENCE: {confidence:.1%} ({confidence_level} confidence)

CLINICAL FINDINGS:
{info['description']}

SEVERITY ASSESSMENT:
{info['severity']}

ANATOMICAL LOCATION:
{info['location']}

CLINICAL IMPLICATIONS:
{info['symptoms']}

RECOMMENDATIONS:
- Further clinical correlation is recommended
- Consider additional imaging studies if clinically indicated
- Consult with a neurosurgeon or neuro-oncologist for treatment planning
- Regular follow-up imaging may be necessary

NOTE: This analysis is based on AI-assisted image interpretation and should be reviewed by qualified medical professionals.
        """.strip()

        return summary
