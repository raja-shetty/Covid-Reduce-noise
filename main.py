from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
import os
from typing import Dict, Any
import logging
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Denoising API (ONNX)", version="1.0.0")

# Configure CORS - Allow ALL origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# Alternative more explicit configuration (choose one or the other)
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000", 
        "http://127.0.0.1:8080",
        "https://localhost:3000",
        "https://localhost:8000",
        "https://localhost:8080",
        "https://*.onrender.com",
        "https://*.netlify.app",
        "https://*.vercel.app",
        "https://*.github.io"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)
"""

# Global variables
MODEL_PATH = "denoiser.onnx"
IMG_SIZE = 512
ort_session = None

# ---------- Metrics ----------
def iqi_metric(y_true, y_pred, eps=1e-8):
    """Image Quality Index (IQI) metric"""
    x = y_true.astype(np.float32)
    y = y_pred.astype(np.float32)
    x_mu, y_mu = x.mean(), y.mean()
    x_var, y_var = x.var(), y.var()
    xy_cov = ((x - x_mu) * (y - y_mu)).mean()
    num = (2 * x_mu * y_mu + eps) * (2 * xy_cov + eps)
    den = (x_mu**2 + y_mu**2 + eps) * (x_var + y_var + eps)
    return float(num / (den + eps))

# ---------- Model Loading ----------
def load_onnx_model():
    """Load ONNX model"""
    global ort_session
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"ONNX model file not found at {MODEL_PATH}")
            return False
        
        logger.info(f"Loading ONNX model from {MODEL_PATH}...")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        
        # Try to use GPU if available
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
        
        ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
        
        # Get model info
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()[0]
        
        logger.info("✅ Successfully loaded ONNX model")
        logger.info(f"Input: {input_info.name} - {input_info.shape} - {input_info.type}")
        logger.info(f"Output: {output_info.name} - {output_info.shape} - {output_info.type}")
        logger.info(f"Providers: {ort_session.get_providers()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        return False

# ---------- Utils ----------
def preprocess_image(image_bytes: bytes) -> (np.ndarray, tuple):
    """Preprocess image for ONNX model input"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        original_size = image.size
        image_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.clip(img_array, 0.0, 1.0)
        return img_array, original_size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def resize_back(img_array: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image back to original size"""
    img_8bit = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_8bit, mode="L").resize(size, Image.LANCZOS)
    return np.array(img_pil, dtype=np.float32) / 255.0

def add_gaussian_noise(img: np.ndarray, sigma_range=(10, 35)) -> np.ndarray:
    """Add Gaussian noise to image"""
    sigma = np.random.uniform(*sigma_range) / 255.0
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)

def array_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    img_8bit = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_8bit, mode="L")
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def run_onnx_inference(image_array: np.ndarray) -> np.ndarray:
    """Run inference using ONNX model"""
    try:
        # ONNX expects NHWC format: (batch_size, height, width, channels)
        model_input = image_array[None, ..., None].astype(np.float32)
        
        # Get input name
        input_name = ort_session.get_inputs()[0].name
        
        # Run inference
        inputs = {input_name: model_input}
        outputs = ort_session.run(None, inputs)
        
        # Extract output and clip values
        denoised_output = outputs[0]
        denoised_image = np.clip(denoised_output[0, ..., 0], 0.0, 1.0)
        
        return denoised_image
        
    except Exception as e:
        logger.error(f"ONNX inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# ---------- Events ----------
@app.on_event("startup")
async def startup_event():
    """Load ONNX model on startup"""
    logger.info("🚀 Starting FastAPI Image Denoising API (ONNX)...")
    logger.info(f"ONNX Runtime version: {ort.__version__}")
    logger.info("🌐 CORS configured to allow ALL origins")
    
    success = load_onnx_model()
    if not success:
        logger.error("❌ Failed to load ONNX model")
    else:
        logger.info("🎉 ONNX model loaded successfully! API ready for denoising.")

# ---------- CORS Test Endpoint ----------
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle preflight OPTIONS requests for CORS"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ---------- Endpoints ----------
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎯 Image Denoising API (ONNX)",
        "status": "running",
        "cors_enabled": "✅ All origins allowed",
        "model_loaded": ort_session is not None,
        "model_type": "ONNX",
        "onnx_version": ort.__version__,
        "providers": ort_session.get_providers() if ort_session else [],
        "endpoints": {
            "denoise": "POST /denoise",
            "denoise_existing": "POST /denoise-existing", 
            "health": "GET /health",
            "model_info": "GET /model-info",
        },
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "🟢 healthy" if ort_session is not None else "🟡 degraded",
        "model_loaded": ort_session is not None,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "cors_status": "✅ All origins allowed",
        "onnx_providers": ort_session.get_providers() if ort_session else [],
    }

@app.post("/denoise")
async def denoise_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Denoise an uploaded image with added noise simulation"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        logger.info(f"📷 Processing: {file.filename}")
        image_bytes = await file.read()
        clean_image, original_size = preprocess_image(image_bytes)
        
        # Add noise for simulation
        noisy_image = add_gaussian_noise(clean_image)
        
        # Run ONNX inference
        logger.info("🧠 Running ONNX inference...")
        denoised_image = run_onnx_inference(noisy_image)
        
        # Resize back to original size
        clean_resized = resize_back(clean_image, original_size)
        noisy_resized = resize_back(noisy_image, original_size)
        denoised_resized = resize_back(denoised_image, original_size)
        
        # Calculate metrics using scikit-image
        logger.info("📊 Calculating metrics...")
        psnr_value = peak_signal_noise_ratio(clean_image, denoised_image, data_range=1.0)
        ssim_value = structural_similarity(clean_image, denoised_image, data_range=1.0)
        iqi_value = iqi_metric(clean_image, denoised_image)
        
        return {
            "status": "✅ success",
            "message": "Image successfully denoised with ONNX model",
            "images": {
                "original": f"data:image/png;base64,{array_to_base64(clean_resized)}",
                "noisy": f"data:image/png;base64,{array_to_base64(noisy_resized)}",
                "denoised": f"data:image/png;base64,{array_to_base64(denoised_resized)}",
            },
            "metrics": {
                "psnr": round(psnr_value, 4),
                "ssim": round(ssim_value, 4),
                "iqi": round(iqi_value, 4),
            },
            "processing_info": {
                "filename": file.filename,
                "input_size": list(original_size),
                "processed_size": [IMG_SIZE, IMG_SIZE],
                "noise_added": "Gaussian (σ=10-35)",
                "model_type": "ONNX",
                "providers": ort_session.get_providers()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/denoise-existing")
async def denoise_existing_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Denoise an already noisy image without adding more noise"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        logger.info(f"📷 Denoising existing noisy image: {file.filename}")
        image_bytes = await file.read()
        noisy_image, original_size = preprocess_image(image_bytes)
        
        # Run ONNX inference directly
        denoised_image = run_onnx_inference(noisy_image)
        
        # Resize back to original size
        noisy_resized = resize_back(noisy_image, original_size)
        denoised_resized = resize_back(denoised_image, original_size)
        
        return {
            "status": "✅ success",
            "message": "Existing noisy image denoised with ONNX model",
            "images": {
                "input_noisy": f"data:image/png;base64,{array_to_base64(noisy_resized)}",
                "denoised": f"data:image/png;base64,{array_to_base64(denoised_resized)}",
            },
            "processing_info": {
                "filename": file.filename,
                "note": "No additional noise added",
                "model_type": "ONNX",
                "providers": ort_session.get_providers()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get detailed ONNX model information"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded")
    
    try:
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()[0]
        
        return {
            "model_name": "Dual-Branch Denoiser (ONNX)",
            "model_type": "ONNX Runtime",
            "onnx_version": ort.__version__,
            "input_info": {
                "name": input_info.name,
                "shape": input_info.shape,
                "type": input_info.type
            },
            "output_info": {
                "name": output_info.name,
                "shape": output_info.shape,
                "type": output_info.type
            },
            "providers": ort_session.get_providers(),
            "file_info": {
                "path": MODEL_PATH,
                "size_mb": round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2) if os.path.exists(MODEL_PATH) else "N/A"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve model info: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
