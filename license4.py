import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
import json
import os

class LicensePlateRecognizer:
    def __init__(self, min_area=500, tesseract_config=None):
        """
        Initialize the License Plate Recognizer
        
        Args:
            min_area: Minimum area for license plate detection
            tesseract_config: Custom tesseract configuration
        """
        self.min_area = min_area
        self.tesseract_config = tesseract_config or '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # License plate patterns (can be customized for different regions)
        self.plate_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # Indian format: XX00XX0000
            r'^[A-Z]{3}\d{4}$',               # US format: XXX0000
            r'^[A-Z]{2}\d{2}[A-Z]{3}$',       # UK format: XX00XXX
            r'^[A-Z0-9]{6,8}$'                # Generic alphanumeric
        ]
        
        # Create output directory for detected plates
        self.output_dir = "detected_plates"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def preprocess_image(self, image):
        """
        Preprocess the image for better license plate detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        bilateral_filter = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges using Canny edge detection
        edges = cv2.Canny(bilateral_filter, 30, 200)
        
        return gray, bilateral_filter, edges
    
    def detect_license_plates(self, image):
        """
        Detect potential license plate regions in the image
        """
        gray, filtered, edges = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_candidates = []
        
        for contour in contours:
            # Get the area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip small contours
            if area < self.min_area:
                continue
            
            # Approximate the contour
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            # Look for rectangular contours (license plates are typically rectangular)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (license plates have specific width/height ratios)
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 5.0:  # Typical license plate aspect ratio
                    # Extract the region of interest
                    roi = gray[y:y+h, x:x+w]
                    plate_candidates.append({
                        'roi': roi,
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'area': area
                    })
        
        return plate_candidates
    
    def enhance_plate_image(self, plate_roi):
        """
        Enhance the license plate image for better OCR
        """
        # Resize the image
        height, width = plate_roi.shape
        if width < 200:
            scale_factor = 200 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            plate_roi = cv2.resize(plate_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Blackhat operation to highlight dark text on light background
        blackhat = cv2.morphologyEx(plate_roi, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine with original
        enhanced = cv2.add(plate_roi, blackhat)
        
        # Apply Gaussian blur to smooth the image
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_text(self, plate_image):
        """
        Extract text from the license plate image using OCR
        """
        # Enhance the plate image
        enhanced_plate = self.enhance_plate_image(plate_image)
        
        try:
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(enhanced_plate, config=self.tesseract_config)
            text = text.strip().replace(" ", "").replace("\n", "")
            
            # Clean the text
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def validate_plate_text(self, text):
        """
        Validate if the extracted text matches license plate patterns
        """
        if len(text) < 4:
            return False
        
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return True
        
        # Additional validation: check if it contains both letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        return has_letters and has_numbers and 4 <= len(text) <= 10
    
    def process_frame(self, frame):
        """
        Process a single frame to detect and recognize license plates
        """
        results = []
        
        # Detect license plate candidates
        plate_candidates = self.detect_license_plates(frame)
        
        for i, candidate in enumerate(plate_candidates):
            roi = candidate['roi']
            bbox = candidate['bbox']
            
            # Extract text from the plate
            plate_text = self.extract_text(roi)
            
            # Validate the extracted text
            if self.validate_plate_text(plate_text):
                results.append({
                    'text': plate_text,
                    'bbox': bbox,
                    'confidence': len(plate_text) / 10.0,  # Simple confidence measure
                    'timestamp': datetime.now().isoformat()
                })
                
                # Save detected plate image
                plate_filename = f"{self.output_dir}/plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                cv2.imwrite(plate_filename, roi)
        
        return results
    
    def draw_results(self, frame, results):
        """
        Draw detection results on the frame
        """
        for result in results:
            x, y, w, h = result['bbox']
            text = result['text']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - 30), (x + text_size[0], y), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def process_video_stream(self, source=0, save_video=False):
        """
        Process video stream in real-time
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            save_video: Whether to save the output video
        """
        cap = cv2.VideoCapture(source)
        
        # Set video properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup video writer if saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('license_plate_detection.avi', fourcc, 20.0, (640, 480))
        
        detected_plates = []
        
        print("Starting license plate detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            results = self.process_frame(frame)
            
            # Draw results on frame
            frame_with_results = self.draw_results(frame.copy(), results)
            
            # Store detected plates
            for result in results:
                if result['text'] not in [p['text'] for p in detected_plates]:
                    detected_plates.append(result)
                    print(f"Detected license plate: {result['text']} at {result['timestamp']}")
            
            # Display the frame
            cv2.imshow('License Plate Recognition', frame_with_results)
            
            # Save frame if required
            if save_video:
                out.write(frame_with_results)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        # Save detection results
        self.save_results(detected_plates)
        
        return detected_plates
    
    def save_results(self, results):
        """
        Save detection results to JSON file
        """
        results_file = f"{self.output_dir}/detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")


def main():
    """
    Main function to run the license plate recognition system
    """
    # Initialize the recognizer
    recognizer = LicensePlateRecognizer(min_area=1000)
    
    print("License Plate Recognition System")
    print("1. Process webcam feed")
    print("2. Process video file")
    print("3. Process image file")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Process webcam feed
        detected_plates = recognizer.process_video_stream(source=0, save_video=True)
        
    elif choice == '2':
        # Process video file
        video_path = input("Enter video file path: ")
        detected_plates = recognizer.process_video_stream(source=video_path, save_video=True)
        
    elif choice == '3':
        # Process single image
        image_path = input("Enter image file path: ")
        image = cv2.imread(image_path)
        
        if image is not None:
            results = recognizer.process_frame(image)
            image_with_results = recognizer.draw_results(image, results)
            
            # Display result
            cv2.imshow('License Plate Detection', image_with_results)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save result
            cv2.imwrite('result_image.jpg', image_with_results)
            print(f"Detected plates: {[r['text'] for r in results]}")
        else:
            print("Could not load image file.")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
