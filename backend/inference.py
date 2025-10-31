# Inference wrapper for PiCAI nnUNet Docker container. Handles Docker execution and file I/O.

import os
import subprocess
import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)

# Docker image name
DOCKER_IMAGE = os.getenv(
    "PICAI_DOCKER_IMAGE", 
    "picai_nnunet_gc_algorithm:latest"
)

# check if Docker is available.
def check_docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

"""
Run PiCAI nnUNet inference using Docker container.

The PiCAI algorithm expects files in a specific format:
- Images in /input/images/
- Named: <case_id>_<modality>.mha

Parameters:
    t2w_path: Path to T2-weighted MRI
    adc_path: Path to ADC map
    hbv_path: Path to high b-value DWI
    output_dir: Directory for outputs

Returns:
    Dictionary with paths to results
"""
def run_picai_inference(
    t2w_path: str,
    adc_path: str,
    hbv_path: str,
    output_dir: str
) -> Dict:
    if not check_docker_available():
        raise RuntimeError("Docker is not available. Please install Docker.")
    
    # create directory structure expected by the algorithm
    input_dir = Path(output_dir) / "input" / "images"
    output_path = Path(output_dir) / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # pre-create output subdirectories that the container expects
    (output_path / "images" / "cspca-detection-map").mkdir(parents=True, exist_ok=True)

    # set permissions to ensure Docker container can write
    import stat
    os.chmod(output_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(output_path / "images", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # copy files to expected location with correct naming
    # PI-CAI expects: <case_id>_<modality>.mha
    case_id = "case_001"
    
    logger.info("Preparing input files...")
    input_files = {
        't2w': input_dir / f"{case_id}_t2w.mha",
        'adc': input_dir / f"{case_id}_adc.mha",
        'hbv': input_dir / f"{case_id}_hbv.mha"
    }
    
    # load and save in correct format
    for modality, src_path in [('t2w', t2w_path), ('adc', adc_path), ('hbv', hbv_path)]:
        img = sitk.ReadImage(src_path)
        sitk.WriteImage(img, str(input_files[modality]))
        logger.info(f"Prepared {modality}: {input_files[modality]}")
    
    # run Docker container
    logger.info(f"Running Docker container: {DOCKER_IMAGE}")
    
    try:
        # mount volumes and run inference
        cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{input_dir.parent.parent}:/input",
            "-v", f"{output_path}:/output",
            DOCKER_IMAGE
        ]
        
        logger.info(f"Docker command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Docker stderr: {result.stderr}")
            raise RuntimeError(f"Docker inference failed: {result.stderr}")
        
        logger.info(f"Docker stdout: {result.stdout}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker inference timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Error running Docker: {str(e)}")
    
    # find output file
    prediction_files = list(output_path.glob("*.mha")) + list(output_path.glob("*.nii.gz"))
    
    if not prediction_files:
        raise RuntimeError("No prediction file found in output directory")
    
    prediction_path = prediction_files[0]
    logger.info(f"Prediction saved to: {prediction_path}")
    
    # load prediction for statistics
    pred_img = sitk.ReadImage(str(prediction_path))
    pred_array = sitk.GetArrayFromImage(pred_img)
    
    # calculate basic statistics
    stats = {
        "mean_probability": float(np.mean(pred_array)),
        "max_probability": float(np.max(pred_array)),
        "positive_voxels": int(np.sum(pred_array > 0.5)),
        "total_voxels": int(pred_array.size)
    }
    
    return {
        "status": "success",
        "prediction_path": str(prediction_path),
        "t2w_path": str(input_files['t2w']),
        "adc_path": str(input_files['adc']),
        "hbv_path": str(input_files['hbv']),
        "statistics": stats
    }

"""
Load MRI volume from file.

Parameters:
    file_path: Path to MRI file
    
Returns:
    NumPy array of volume data
"""
def load_mri_volume(file_path: str) -> np.ndarray:
    
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)
