import requests
import base64
from PIL import Image
import io
import numpy as np
import json

def test_sign_to_speech_api():
    """Test the sign language processing endpoint"""
    url = 'http://localhost:8000/sign-to-speech/'  # Adjust URL as needed
    
    # Test 1: Send base64 encoded image
    def test_base64_image():
        # Create a sample image (red square)
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode to base64
        base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
        
        payload = {
            'image_base64': base64_encoded,
            'convert_to_speech': True,
            'target_language': 'eng'
        }
        
        response = requests.post(url, json=payload)
        print(f"Base64 Image Test Status Code: {response.status_code}")
        print(f"Response: {response.json() if response.status_code == 200 else response.text}")
        return response.status_code == 200

    # Test 2: Send raw frame data
    def test_raw_frame():
        # Create a sample frame (numpy array)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red channel
        
        payload = {
            'frame': frame.tolist(),  # Convert numpy array to list
            'convert_to_speech': False
        }
        
        response = requests.post(url, json=payload)
        print(f"Raw Frame Test Status Code: {response.status_code}")
        print(f"Response: {response.json() if response.status_code == 200 else response.text}")
        return response.status_code == 200

    # Run tests
    print("Testing API endpoints...")
    base64_test = test_base64_image()
    frame_test = test_raw_frame()
    
    return base64_test, frame_test

if __name__ == "__main__":
    base64_test, frame_test = test_sign_to_speech_api()
    print("\nTest Results:")
    print(f"Base64 Image Test: {'✓ Passed' if base64_test else '✗ Failed'}")
    print(f"Raw Frame Test: {'✓ Passed' if frame_test else '✗ Failed'}")