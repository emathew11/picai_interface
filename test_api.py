# Simple test script for the PiCAI API.
# Tests health check and provides example for prediction endpoint.

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

# test health endpoint
def test_health():
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

# test prediction endpoint with sample files
def test_prediction(t2w_path: str, adc_path: str, hbv_path: str):
    
    print("\nTesting prediction endpoint...")
    
    # check files exist
    for name, path in [("T2W", t2w_path), ("ADC", adc_path), ("HBV", hbv_path)]:
        if not Path(path).exists():
            print(f"File not found: {path}")
            return False
    
    try:
        # Prepare files
        files = {
            't2w': open(t2w_path, 'rb'),
            'adc': open(adc_path, 'rb'),
            'hbv': open(hbv_path, 'rb')
        }
        
        print("   Uploading files and running inference...")
        response = requests.post(f"{API_URL}/predict", files=files)
        
        # Close files
        for f in files.values():
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction successful")
            print(f"   Status: {result['status']}")
            print(f"   Statistics:")
            stats = result['statistics']
            print(f"     - Mean probability: {stats['mean_probability']:.4f}")
            print(f"     - Max probability: {stats['max_probability']:.4f}")
            print(f"     - Positive voxels: {stats['positive_voxels']}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False

# run tests
def main():
    
    print("=" * 50)
    print("PI-CAI API Test Suite")
    print("=" * 50)
    
    # test health
    if not test_health():
        print("\nAPI is not responding. Is the backend running?")
        print("   Start it with: cd backend && uvicorn main:app --reload")
        sys.exit(1)
    
    # check if sample files are provided
    if len(sys.argv) == 4:
        t2w_path, adc_path, hbv_path = sys.argv[1], sys.argv[2], sys.argv[3]
        test_prediction(t2w_path, adc_path, hbv_path)
    else:
        print("\nTo test prediction, provide paths to MRI files:")
        print("   python test_api.py <t2w_path> <adc_path> <hbv_path>")
    
    print("\n" + "=" * 50)
    print("Tests complete!")


if __name__ == "__main__":
    main()