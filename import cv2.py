import cv2
import numpy as np

cap cv2.VideoCapture(0)   

while True
    ret, frame ) cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 50, 250)

    cv2.imshow("Bordes en vivo", bordes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
    print("end")