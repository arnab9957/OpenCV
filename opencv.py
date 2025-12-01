# -*- coding: utf-8 -*-
"""
OpenCV Tutorial - Comprehensive Image Processing Examples
Adapted from Google Colab notebook for local Windows environment
"""

import cv2
import numpy as np
import sys


# ============================================================================
# HELPER FUNCTION: Display images in a window (works on local system)
# ============================================================================
def show_image(image, window_name: str = "Image"):
    """
    Display an image in a window until a key is pressed.
    
    Args:
        image: The image array to display (BGR or grayscale)
        window_name: Name of the window (default: "Image")
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait indefinitely until any key is pressed
    cv2.destroyAllWindows()  # Close all OpenCV windows


# ============================================================================
# SECTION 1: BASIC IMAGE OPERATIONS
# ============================================================================

# Read an image from file (use raw string to avoid escape sequence issues)
image = cv2.imread(r"D:\Desktop\OpenCV\Arnab.jpg")

# Check if the image was successfully loaded
if image is None:
    print("Error: Image not found. Please provide a valid image path.")
else:
    # Display the original image
    print("Displaying original image...")
    show_image(image, "Original Image")
    
    # Convert the image to grayscale (from BGR to GRAY)
    # cv2.cvtColor() changes the color space of an image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(gray, "Grayscale Image")
    
    # Get image dimensions: height, width, and number of channels
    h, w, c = image.shape
    print(f"Image dimensions - Height: {h}, Width: {w}, Channels: {c}")
    
    # Resize the image to half its original size
    # cv2.resize() changes the spatial dimensions of an image
    image_resized = cv2.resize(image, (w // 2, h // 2))
    show_image(image_resized, "Resized Image (50%)")
    
    # Save the grayscale image to disk
    # cv2.imwrite() writes the image array to a file
    cv2.imwrite("ArnabGray.jpg", gray)
    print("Grayscale image saved as 'ArnabGray.jpg'")



# ============================================================================
# SECTION 2: INTERACTIVE IMAGE PROCESSING
# ============================================================================

# Prompt user for an image path
img_path = input("Enter Image Path (or press Enter to skip): ")

# Only proceed if user provided a path
if img_path.strip():
    image = cv2.imread(img_path)
    
    # Verify the image loaded successfully
    if image is None:
        print(f"Error: Image not found at '{img_path}'. Please provide a valid image path.")
    else:
        # Prompt user for an action to perform on the image
        user_action = input(
            "Choose an action:\n"
            "  's' - Show the image\n"
            "  'g' - Convert to grayscale and show\n"
            "  'l' - Convert to grayscale and save\n"
            "Your choice: "
        )
        
        # Process based on user choice
        while True:
            if user_action == 's':
                # Display the original image
                show_image(image, "User Image")
                break
            elif user_action == 'g':
                # Convert to grayscale and display
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                show_image(gray, "Grayscale User Image")
                break
            elif user_action == 'l':
                # Convert to grayscale and save to disk
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                grayImage_name = input("Enter the name for the grayscale image (e.g., 'my_image'): ")
                
                # Append .jpg extension if no image extension is present
                if not grayImage_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    grayImage_name += '.jpg'
                
                # Write the grayscale image to disk
                cv2.imwrite(grayImage_name, gray)
                print(f"Grayscale image saved as '{grayImage_name}'")
                break
            else:
                # Invalid input - prompt again
                user_action = input("Invalid choice. Please select 's', 'g', or 'l': ")
else:
    print("No image path provided. Skipping interactive section.")


# ============================================================================
# SECTION 3: IMAGE TRANSFORMATIONS
# ============================================================================

# Note: The following sections assume 'image' is loaded. 
# If running this section independently, ensure an image is loaded first.

if 'image' in locals() and image is not None:
    
    # --- Resize Operation ---
    # Resize image to specific dimensions (200x250 pixels)
    resized = cv2.resize(image, (200, 250))
    show_image(resized, "Resized (200x250)")
    
    # --- Crop Operation ---
    # Crop a region from the image using array slicing [startY:endY, startX:endX]
    # This extracts a rectangular region from coordinates (300,415) to (550,600)
    cropped = image[415:600, 300:550]
    show_image(cropped, "Cropped Region")
    
    # --- Rotate Operation ---
    # Get image dimensions (height and width)
    h, w = image.shape[:2]
    
    # Calculate the center point for rotation
    center = (w // 2, h // 2)
    
    # Define rotation parameters
    angle = 90  # Rotate 90 degrees clockwise
    scale = 1   # Keep original size (1 = 100%)
    
    # Get the 2D rotation matrix
    # cv2.getRotationMatrix2D(center, angle, scale) computes the affine transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply the rotation using warpAffine
    # cv2.warpAffine() applies affine transformation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    show_image(rotated_image, "Rotated 90 Degrees")
    
    # --- Flip Operations ---
    print("Original resized image:")
    show_image(resized, "Original")
    
    # Flip horizontally (mirror left-right)
    # cv2.flip(image, flipCode) where flipCode=1 means horizontal flip
    horiz = cv2.flip(resized, 1)
    print("Flipped horizontally:")
    show_image(horiz, "Horizontal Flip")
    
    # Flip vertically (mirror top-bottom)
    # flipCode=0 means vertical flip
    vert = cv2.flip(resized, 0)
    print("Flipped vertically:")
    show_image(vert, "Vertical Flip")
    
    # Flip both horizontally and vertically (180-degree rotation)
    # flipCode=-1 means both horizontal and vertical flip
    flip_both = cv2.flip(resized, -1)
    print("Flipped both directions:")
    show_image(flip_both, "Both Flips")



# ============================================================================
# SECTION 4: DRAWING SHAPES AND TEXT
# ============================================================================

if 'resized' in locals() and resized is not None:
    
    # Get dimensions of the resized image
    h_resized, w_resized = resized.shape[:2]
    
    # --- Draw Lines ---
    # cv2.line(image, start_point, end_point, color_BGR, thickness)
    # Create a copy to avoid modifying the original
    line_img = resized.copy()
    
    # Draw diagonal line from top-left to bottom-right (green)
    cv2.line(line_img, (0, 0), (w_resized, h_resized), (0, 255, 0), 2)
    
    # Draw diagonal line from top-right to bottom-left (custom purple color)
    cv2.line(line_img, (w_resized, 0), (0, h_resized), (255, 66, 120), 2)
    
    show_image(line_img, "Lines Drawn")
    
    # --- Draw Rectangle ---
    # cv2.rectangle(image, top_left_corner, bottom_right_corner, color_BGR, thickness)
    rect_img = resized.copy()
    cv2.rectangle(rect_img, (40, 50), (200, 250), (255, 0, 0), 5)  # Blue rectangle
    show_image(rect_img, "Rectangle Drawn")
    
    # --- Draw Circle ---
    # cv2.circle(image, center_point, radius, color_BGR, thickness)
    # Use the horizontally flipped image if available, otherwise use resized
    if 'horiz' in locals():
        circle_img = horiz.copy()
    else:
        circle_img = resized.copy()
    
    h_circle, w_circle = circle_img.shape[:2]
    center_circle = (w_circle // 2, h_circle // 2)  # Center of the image
    
    # Draw a filled circle (thickness=-1 fills the circle)
    cv2.circle(circle_img, center_circle, 50, (255, 0, 0), 5)  # Blue circle with thick border
    show_image(circle_img, "Circle Drawn")
    
    # --- Add Text ---
    # cv2.putText(image, text, position, font, font_scale, color_BGR, thickness)
    text_img = circle_img.copy()
    
    # Position text near the center
    text_position = (w_circle // 2 - 50, h_circle // 2)
    
    # Add text to the image
    cv2.putText(
        text_img, 
        "Arnab",                          # Text to display
        text_position,                    # Bottom-left corner of text
        cv2.FONT_HERSHEY_SIMPLEX,        # Font type
        1,                                # Font scale (size)
        (0, 255, 0),                     # Color (green in BGR)
        5                                 # Thickness
    )
    show_image(text_img, "Text Added")


# ============================================================================
# SECTION 5: WEBCAM CAPTURE
# ============================================================================

print("\nWebcam section: Press 'q' to quit the webcam feed.")

# Open the default webcam (index 0)
# cv2.VideoCapture(0) creates a video capture object
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    # Continuous loop to capture frames
    while True:
        # Read a frame from the webcam
        # ret: boolean indicating success, frame: the captured image
        ret, frame = cap.read()
        
        # Check if frame was successfully captured
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Display the frame in a window
        cv2.imshow("Webcam", frame)
        
        # Wait 1ms for a key press and check if 'q' was pressed
        # 0xFF masks the key to get the ASCII value
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Release the webcam resource
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()


# ============================================================================
# SECTION 6: IMAGE FILTERING AND BLURRING
# ============================================================================

# Load an image for filtering demonstrations
filter_image = cv2.imread(r"D:\Desktop\OpenCV\monalisha.png")

if filter_image is not None:
    show_image(filter_image, "Original (for filtering)")
    
    # --- Gaussian Blur ---
    # cv2.GaussianBlur(image, kernel_size, sigmaX)
    # kernel_size must be odd numbers (e.g., (3,3), (5,5), (7,7))
    # Larger kernel = more blur. sigmaX controls the Gaussian standard deviation
    gaussian_blur = cv2.GaussianBlur(filter_image, (7, 7), 5)
    show_image(gaussian_blur, "Gaussian Blur")
    
    # --- Median Blur ---
    # cv2.medianBlur(image, kernel_size)
    # Good for removing salt-and-pepper noise
    # kernel_size must be odd and positive
    median_blur = cv2.medianBlur(filter_image, 9)
    show_image(median_blur, "Median Blur")
    
    # --- Sharpen Filter ---
    # Create a custom sharpening kernel (convolution matrix)
    # The center value is high, surrounded by negative values
    sharpen_kernel = np.array([
        [0,  -1,  0],
        [-1,  5, -1],
        [0,  -1,  0]
    ])
    
    # Apply the kernel using filter2D
    # cv2.filter2D(image, depth, kernel) performs 2D convolution
    # depth=-1 means output has same depth as input
    sharpened = cv2.filter2D(filter_image, -1, sharpen_kernel)
    show_image(sharpened, "Sharpened Image")
else:
    print("Warning: Could not load image for filtering section.")



# ============================================================================
# SECTION 7: EDGE DETECTION AND THRESHOLDING
# ============================================================================

# Load an image and convert to grayscale for edge detection
edge_image = cv2.imread(r"D:\Desktop\OpenCV\monalisha.png", cv2.IMREAD_GRAYSCALE)

if edge_image is not None:
    # Resize for consistent viewing
    edge_image = cv2.resize(edge_image, (200, 250))
    show_image(edge_image, "Grayscale (for edge detection)")
    
    # --- Canny Edge Detection ---
    # cv2.Canny(image, threshold1, threshold2)
    # threshold1: lower threshold for edge linking
    # threshold2: upper threshold for edge detection
    # Edges with gradient > threshold2 are sure edges
    # Edges between threshold1 and threshold2 are weak edges (kept if connected to strong edges)
    canny = cv2.Canny(edge_image, 100, 200)
    show_image(canny, "Canny Edge Detection")
    
    # --- Binary Thresholding ---
    # cv2.threshold(image, threshold_value, max_value, threshold_type)
    # Pixels > threshold_value become max_value, others become 0
    # Returns: (threshold_value, thresholded_image)
    _, thresholded = cv2.threshold(edge_image, 90, 255, cv2.THRESH_BINARY)
    show_image(thresholded, "Binary Threshold")
else:
    print("Warning: Could not load image for edge detection section.")


# ============================================================================
# SECTION 8: BITWISE OPERATIONS
# ============================================================================

# Create two blank black images (300x300 pixels, 8-bit unsigned integers)
# np.zeros creates an array filled with zeros
img1 = np.zeros((300, 300), dtype=np.uint8)
img2 = np.zeros((300, 300), dtype=np.uint8)

# Draw a white circle on img1 (filled circle with thickness=-1)
cv2.circle(img1, (150, 150), 100, 255, -1)

# Draw a white rectangle on img2 (filled rectangle with thickness=-1)
cv2.rectangle(img2, (100, 100), (250, 250), 255, -1)

show_image(img1, "Image 1 - Circle")
show_image(img2, "Image 2 - Rectangle")

# --- Bitwise AND ---
# Returns white pixels only where both images have white pixels (intersection)
bitwise_and = cv2.bitwise_and(img1, img2)
show_image(bitwise_and, "Bitwise AND")

# --- Bitwise OR ---
# Returns white pixels where either image has white pixels (union)
bitwise_or = cv2.bitwise_or(img1, img2)
show_image(bitwise_or, "Bitwise OR")

# --- Bitwise XOR ---
# Returns white pixels where exactly one image has white pixels (symmetric difference)
bitwise_xor = cv2.bitwise_xor(img1, img2)
show_image(bitwise_xor, "Bitwise XOR")

# --- Bitwise NOT ---
# Inverts all pixels (white becomes black, black becomes white)
bitwise_not = cv2.bitwise_not(img1)
show_image(bitwise_not, "Bitwise NOT (Circle)")


# ============================================================================
# SECTION 9: CONTOUR DETECTION
# ============================================================================

# Load an image for contour detection
contour_img = cv2.imread(r"D:\Desktop\OpenCV\OIP.jpg")

if contour_img is not None:
    # Resize for consistent display
    contour_img = cv2.resize(contour_img, (200, 250))
    
    # Convert to grayscale (required for thresholding)
    gray_contour = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to create a binary image
    # Pixels > 200 become 255 (white), others become 0 (black)
    _, threshold_contour = cv2.threshold(gray_contour, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    # cv2.findContours(image, retrieval_mode, approximation_method)
    # RETR_TREE: retrieves all contours and creates a full hierarchy
    # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(threshold_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all detected contours on the image
    # cv2.drawContours(image, contours, contour_index, color, thickness)
    # contour_index=-1 means draw all contours
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 4)
    show_image(contour_img, "Contours Detected")
else:
    print("Warning: Could not load image for contour section.")



# ============================================================================
# SECTION 10: SHAPE DETECTION USING CONTOURS
# ============================================================================

# Load an image for shape detection
shape_img = cv2.imread(r"D:\Desktop\OpenCV\OIP.jpg")

if shape_img is not None:
    # Resize for consistent display
    shape_img = cv2.resize(shape_img, (200, 250))
    
    # Convert to grayscale
    gray_shape = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold with Otsu's method for automatic threshold calculation
    # THRESH_BINARY + THRESH_OTSU automatically determines the optimal threshold value
    _, threshold_shape = cv2.threshold(
        gray_shape, 
        200,                                    # Initial threshold (ignored with OTSU)
        255,                                    # Max value
        cv2.THRESH_BINARY + cv2.THRESH_OTSU    # Binary threshold with Otsu's algorithm
    )
    
    # Find all contours
    contours_shape, _ = cv2.findContours(
        threshold_shape, 
        cv2.RETR_TREE,              # Retrieve all contours with full hierarchy
        cv2.CHAIN_APPROX_SIMPLE     # Compress contours to save memory
    )
    
    # Iterate through each detected contour
    for contour in contours_shape:
        
        # Calculate the area of the contour
        # cv2.contourArea() returns the number of pixels enclosed
        area = cv2.contourArea(contour)
        
        # Skip very small contours (likely noise)
        # Adjust this threshold based on your image size
        if area < 300:
            continue
        
        # Approximate the contour to a simpler polygon
        # cv2.approxPolyDP(contour, epsilon, closed)
        # epsilon: maximum distance between original contour and approximation
        # 0.02 * perimeter is a common heuristic
        approx = cv2.approxPolyDP(
            contour, 
            0.02 * cv2.arcLength(contour, True),  # Epsilon based on perimeter
            True                                   # Contour is closed
        )
        
        # Count the number of vertices (edges) in the approximated shape
        num_vertices = len(approx)
        
        # Classify shape based on number of vertices
        if num_vertices == 3:
            shape = "Triangle"
        
        elif num_vertices == 4:
            # Distinguish between square and rectangle using aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # If aspect ratio is close to 1, it's a square
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        
        elif num_vertices == 5:
            shape = "Pentagon"
        
        elif num_vertices == 6:
            shape = "Hexagon"
        
        else:
            # For shapes with many vertices, check if it's a circle
            # Calculate the minimum enclosing circle
            (x_center, y_center), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            
            # If the contour area is close to the circle area, it's circular
            if 0.7 <= area / circle_area <= 1.2:
                shape = "Circle"
            else:
                shape = "Other"
        
        # Draw the approximated contour in green
        cv2.drawContours(shape_img, [approx], -1, (0, 255, 0), 2)
        
        # Add text label for the detected shape
        # Get the first point of the approximated contour for text placement
        x_text = approx.ravel()[0]
        y_text = approx.ravel()[1] - 10  # Place text slightly above the shape
        
        cv2.putText(
            shape_img, 
            shape,                          # Text to display
            (x_text, y_text),              # Position (above the shape)
            cv2.FONT_HERSHEY_SIMPLEX,      # Font
            0.5,                           # Font scale
            (255, 0, 0),                   # Color (blue in BGR)
            2                               # Thickness
        )
    
    # Display the image with detected shapes labeled
    show_image(shape_img, "Shape Detection")
else:
    print("Warning: Could not load image for shape detection section.")


# ============================================================================
# END OF OPENCV TUTORIAL
# ============================================================================

print("\n" + "="*60)
print("OpenCV Tutorial Complete!")
print("="*60)
print("All sections have been executed successfully.")
print("Note: Update image paths to match your local file structure.")


# ============================================================================
# SECTION 11: REAL-TIME FACE, EYE, AND SMILE DETECTION
# ============================================================================

print("\n" + "="*60)
print("BONUS: Real-Time Face, Eye & Smile Detection")
print("="*60)
print("Press 'q' to quit the detection window.")

# Import required modules (already imported at top, but shown for clarity)
# import cv2
# import os
# import sys

# --- Load Haar Cascade Classifiers ---
# Haar Cascades are pre-trained XML classifiers for object detection
# They use machine learning to detect faces, eyes, smiles, etc.

# Load face detection cascade (frontal cat face classifier as example)
# Note: For human faces, use 'haarcascade_frontalface_default.xml'
face_Cascade = cv2.CascadeClassifier(r"D:\Desktop\OpenCV\haarcascade_frontalcatface.xml")

# Load eye detection cascade
eye_Cascade = cv2.CascadeClassifier(r"D:\Desktop\OpenCV\eye.xml")

# Load smile detection cascade
smile_Cascade = cv2.CascadeClassifier(r"D:\Desktop\OpenCV\haarcascade_smile.xml")

# Optional: Verify that cascades loaded successfully
# Uncomment the following lines to enable cascade validation:
# if face_Cascade.empty() or eye_Cascade.empty() or smile_Cascade.empty():
#     print("Error: Failed to load one or more cascade classifiers.")
#     print("Make sure the XML files exist and paths are correct.")
#     sys.exit(1)

# --- Initialize Webcam ---
# Open the default camera (index 0)
camera = cv2.VideoCapture(0)

# Verify that the camera opened successfully
if not camera or not camera.isOpened():
    print("Error: Unable to open camera index 0")
    print("Make sure your webcam is connected and not being used by another application.")
    sys.exit(1)

# --- Main Detection Loop ---
while True:
    # Capture frame-by-frame from the webcam
    # ret: boolean indicating if frame was read successfully
    # frame: the actual video frame (BGR image)
    ret, frame = camera.read()
    
    # Check if frame was captured successfully
    if not ret or frame is None:
        print("Failed to read frame from camera. Exiting.")
        break
    
    # Convert the frame to grayscale for better detection performance
    # Haar Cascades work better with grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- Detect Faces ---
    # detectMultiScale() detects objects at different scales
    # Parameters:
    #   - gray: input grayscale image
    #   - scaleFactor=1.1: how much the image size is reduced at each scale (1.1 = 10% reduction)
    #   - minNeighbors=5: how many neighbors each candidate rectangle should have to retain it
    # Returns: list of rectangles [(x, y, w, h), ...] where face was detected
    faces = face_Cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the detected face
        # Parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3, lineType=2)
        
        # --- Define Region of Interest (ROI) ---
        # Extract only the face region for detecting eyes and smile
        # This improves accuracy and performance by limiting the search area
        region_of_interest_gray = gray[y: y + h, x: x + w]    # Grayscale ROI
        region_of_interest_color = frame[y: y + h, x: x + w]  # Color ROI (for future use)
        
        # --- Detect Eyes within Face Region ---
        # Search for eyes only within the detected face area
        eyes = eye_Cascade.detectMultiScale(region_of_interest_gray, scaleFactor=1.1, minNeighbors=5)
        
        # If one or more eyes are detected, display "Eyes" label
        if len(eyes) > 0:
            # cv2.putText() adds text overlay on the image
            # Position: (x, y - 30) places text above the face rectangle
            cv2.putText(
                frame,                          # Image to draw on
                "Eyes",                         # Text to display
                (x, y - 30),                   # Position (above the face)
                cv2.FONT_HERSHEY_SIMPLEX,      # Font type
                0.7,                           # Font scale (size)
                (255, 255, 255),               # Color (white in BGR)
                2                               # Thickness
            )
        
        # --- Detect Smile within Face Region ---
        # Search for smile only within the detected face area
        smiles = smile_Cascade.detectMultiScale(region_of_interest_gray, scaleFactor=1.1, minNeighbors=5)
        
        # If one or more smiles are detected, display "Smiling" label
        if len(smiles) > 0:
            # Position: (x, y - 10) places text closer to the face than "Eyes" label
            cv2.putText(
                frame,                          # Image to draw on
                "Smiling",                      # Text to display
                (x, y - 10),                   # Position (above the face, below "Eyes" if present)
                cv2.FONT_HERSHEY_SIMPLEX,      # Font type
                0.7,                           # Font scale (size)
                (255, 255, 255),               # Color (white in BGR)
                2                               # Thickness
            )
    
    # Display the resulting frame with detections
    cv2.imshow("Face, Eye & Smile Detection", frame)
    
    # Wait 1ms for a key press
    # If 'q' is pressed, break the loop and exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
# Release the camera resource
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Face Detection Session Ended")
print("="*60)

