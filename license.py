import cv2
import pytesseract

print("Real Time License plate Recognition")
print("Using:\nobject detection techniques\nOptical Character Recognition(OCR)")

# Set Tesseract path (adjust for your OS; Windows example)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the Haar cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device. Check camera connection or index.")
    exit()

# Minimum area threshold to filter small detections
min_area = 500

# List to store detected plate texts
detected_plates = []

while True:
    # Read frame from video stream
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break
    
    # Convert frame to grayscale for detection
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates using the cascade classifier
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)
    
    # Clear previous detections for this frame
    current_frame_plates = []
    
    for (x, y, w, h) in plates:
        # Calculate area to filter noise
        area = w * h
        if area > min_area:
            # Draw rectangle around detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "License Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            # Crop the detected plate region
            img_roi = frame[y:y + h, x:x + w]
            
            # Preprocess the cropped region for better OCR accuracy
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            _, thresh_roi = cv2.threshold(blur_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR to extract text
            plate_text = pytesseract.image_to_string(thresh_roi, config='--psm 8')
            plate_text = ''.join(char for char in plate_text if char.isalnum())
            
            # Display the recognized text on the frame if valid
            if plate_text:
                cv2.putText(frame, plate_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                current_frame_plates.append(plate_text)
    
    # Update the list of detected plates if new text is found
    if current_frame_plates:
        detected_plates.extend(current_frame_plates)
        # Remove duplicates and print unique plates
        unique_plates = list(dict.fromkeys(detected_plates))
        print("Detected License Plates:", unique_plates)
    
    # Display the processed frame
    cv2.imshow("License Plate Recognition", frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows
