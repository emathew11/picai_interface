# FastAPI backend for PI-CAI prostate cancer segmentation

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import shutil
from pathlib import Path
from typing import List
import logging

from inference import run_picai_inference

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PiCAI Inference API",
    description="API for prostate cancer segmentation using nnUNet",
    version="1.0.0"
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "PiCAI Inference API", "status": "running"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

"""
Run inference on uploaded MRI volumes.

Parameters:
    t2w: T2-weighted MRI file
    adc: ADC map file
    hbv: High b-value DWI file

Returns:
    JSON with paths to prediction and input volumes
"""
@app.post("/predict")
async def predict(
    t2w: UploadFile = File(..., description="T2-weighted MRI"),
    adc: UploadFile = File(..., description="ADC map"),
    hbv: UploadFile = File(..., description="High b-value DWI")
):
    temp_dir = None
    try:
        # create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # validate file extensions
        valid_extensions = ['.mha', '.nii.gz', '.nii']
        for file in [t2w, adc, hbv]:
            if not any(file.filename.endswith(ext) for ext in valid_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file format for {file.filename}. Use .mha or .nii.gz"
                )
        
        # save uploaded files
        files = {'t2w': t2w, 'adc': adc, 'hbv': hbv}
        file_paths = {}
        
        for key, file in files.items():
            file_path = temp_path / file.filename
            with open(file_path, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths[key] = str(file_path)
            logger.info(f"Saved {key} to {file_path}")
        
        # run inference
        logger.info("Starting inference...")
        result = run_picai_inference(
            t2w_path=file_paths['t2w'],
            adc_path=file_paths['adc'],
            hbv_path=file_paths['hbv'],
            output_dir=str(temp_path)
        )
        
        logger.info("Inference completed successfully")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup is handled by the caller
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)