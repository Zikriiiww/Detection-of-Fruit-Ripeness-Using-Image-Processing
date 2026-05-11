import cv2 
import numpy as np
import os

def ripeness_index(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the hue of ripe bananas
    lower_bound = np.array([20, 100, 100])
    upper_bound = np.array([30, 255, 255])
    
    # Create a mask using the defined bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Calculate the percentage of ripe pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    ripe_pixels = cv2.countNonZero(mask)
    
    ripeness_percentage = (ripe_pixels / total_pixels) * 100
    
    return ripeness_percentage
def upload_image():
    # Simulate image upload (in a real application, this would involve file handling)
    image_path = 'input_image.jpg'  # Replace with the actual path to the uploaded image
    if os.path.exists(image_path):
        return cv2.imread(image_path)
    else:
        print("Error: Image file does not exist.")
        return None
def display_results(ripeness_percentage):
    print(f"Ripeness Index: {ripeness_percentage:.2f}%")
def process_image(image):
    if image is not None:
        ripeness_percentage = ripeness_index(image)
        display_results(ripeness_percentage)
    else:
        print("Error: No image to process.")
def main(): 
    image = upload_image()
    process_image(image)
if __name__ == "__main__":
    main()