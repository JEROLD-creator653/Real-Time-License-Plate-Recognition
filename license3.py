import cv2
import pytesseract
from collections import deque, Counter

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def stable_text(results, window_size=5):
    if not results:
        return ""
    most_common = Counter(results).most_common(1)[0][0]
    return most_common

cap = cv2.VideoCapture(0)

print("Press 's' to select a license plate region, 'q' to quit.")

roi = None
ocr_buffer = deque(maxlen=5)  # store last N OCR results

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if roi is not None:
        x, y, w, h = roi
        plate_img = frame[y:y+h, x:x+w]
        if plate_img.size > 0:
            frame_count += 1
            if frame_count % 5 == 0:  # OCR every 5th frame
                processed = preprocess_plate(plate_img)
                config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                text = pytesseract.image_to_string(processed, config=config).strip()
                if text:
                    ocr_buffer.append(text)
            stable_plate = stable_text(ocr_buffer)
            cv2.putText(display_frame, stable_plate, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("License Plate OCR", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        r = cv2.selectROI("License Plate OCR", frame, fromCenter=False, showCrosshair=True)
        if r != (0, 0, 0, 0):
            roi = tuple(map(int, r))
            ocr_buffer.clear()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()