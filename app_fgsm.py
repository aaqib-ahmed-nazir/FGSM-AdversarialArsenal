import os
import io
import torch
import uvicorn
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
import torchvision.transforms as transforms
from fgsm import fgsm_attack, get_model_prediction
from fastapi.middleware.cors import CORSMiddleware
from load_n_train_model import SimpleMNISTModel, MODEL_PATH
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

app = FastAPI(
    title="FGSM Adversarial Attack API",
    description="API for running FGSM attacks on images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize model on startup
    """
    global model
    try:
        # Load the model
        model = SimpleMNISTModel()
        
        # Check if we have a pretrained model
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            print(f"Successfully loaded pretrained model from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Pretrained model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load pretrained model: {str(e)}")

def process_image(image_data: bytes) -> torch.Tensor:
    """
    Process image data into a tensor suitable for the model
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        Processed image tensor
    """

    img = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    
    # Apply transformations similar to MNIST preprocessing
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)  
    return img_tensor

@app.post("/attack", response_class=JSONResponse)
async def attack(
    file: UploadFile = File(...),
    label: int = Form(...),
    epsilon: float = Form(0.1)
):
    """
    Apply FGSM attack to an image
    
    Args:
        file: Uploaded image file
        label: True label of the image (0-9)
        epsilon: Attack strength parameter (0.0 to 1.0)
        
    Returns:
        JSON response with attack results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    if not 0 <= label <= 9:
        raise HTTPException(status_code=400, detail="Label must be between 0 and 9")
    
    if not 0 <= epsilon <= 1:
        raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")
    
    try:
        # Process uploaded file
        image_data = await file.read()
        img_tensor = process_image(image_data)
        img_tensor = img_tensor.to(device)
        
        # Create label tensor
        label_tensor = torch.tensor([label], device=device)
        
        # Get original prediction
        orig_pred, orig_probs = get_model_prediction(model, img_tensor)
        orig_pred = orig_pred.item()
        orig_conf = orig_probs[0, orig_pred].item()
        
        # Perform FGSM attack
        adv_img = fgsm_attack(model, img_tensor, label_tensor, epsilon)
        
        # Get adversarial prediction
        adv_pred, adv_probs = get_model_prediction(model, adv_img)
        adv_pred = adv_pred.item()
        adv_conf = adv_probs[0, adv_pred].item()
        
        # commented off as need to add screenshots that can fit the while result 
        # orig_probs_list = orig_probs[0].detach().cpu().numpy().tolist()
        # adv_probs_list = adv_probs[0].detach().cpu().numpy().tolist()
        
        response = {
            "original_prediction": {
                "class": int(orig_pred),
                "confidence": float(orig_conf)
            },
            "adversarial_prediction": {
                "class": int(adv_pred),
                "confidence": float(adv_conf)
            },
            "true_label": int(label),
            "epsilon": float(epsilon),
            "success": orig_pred != adv_pred
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("app_fgsm:app", host="0.0.0.0", port=8000, reload=True) 