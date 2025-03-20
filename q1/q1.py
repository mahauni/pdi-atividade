import cv2
import numpy as np

cap = cv2.VideoCapture("q1A.mp4")

rect_lower_hsv = np.array([100, 150, 0])
rect_upper_hsv = np.array([140, 255, 255])

square_lower_hsv1 = np.array([0, 120, 70])
square_upper_hsv1 = np.array([10, 255, 255])
square_lower_hsv2 = np.array([170, 120, 70])
square_upper_hsv2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # Seu código aqui....... 
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    rect_mask_hsv = cv2.inRange(frame_hsv, rect_lower_hsv, rect_upper_hsv)

    square_mask_hsv1 = cv2.inRange(frame_hsv, square_lower_hsv1, square_upper_hsv1)
    square_mask_hsv2 = cv2.inRange(frame_hsv, square_lower_hsv2, square_upper_hsv2)

    square_mask_hsv = square_mask_hsv1 + square_mask_hsv2

    rect_frame = cv2.bitwise_and(frame, frame, mask=rect_mask_hsv)
    square_frame = cv2.bitwise_and(frame, frame, mask=square_mask_hsv)

    contornos, _ = cv2.findContours(square_mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    mask_rgb = cv2.cvtColor(square_mask_hsv, cv2.COLOR_GRAY2RGB) 
    contornos_img = mask_rgb.copy()

    cv2.drawContours(square_frame, contornos, -1, [0, 255, 0], 4)

    result_frame = square_frame + rect_frame

    text = ""
    font = cv2.FONT_HERSHEY_SIMPLEX
    origem = (1000, 50)

    cnt = contornos[0]
    _, _, w, h = cv2.boundingRect(cnt)
    M = cv2.moments(cnt)

    half_width = w // 2
    half_height = h // 2

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    size = 20
    color = (128,128,0)


    # utilize the frame em si para verificar qual e o pixel
    # Linha horizontal baixa
    cv2.line(result_frame, (cx - half_width, cy + half_height), (cx + half_width, cy + half_height), color, 5)
    # Linha horizontal alta
    cv2.line(result_frame, (cx - half_width, cy - half_height), (cx + half_width, cy - half_height), color, 5)

    # Linha vertical direita
    cv2.line(result_frame, (cx + half_width, cy - half_height), (cx + half_width, cy + half_height), color, 5)
    # Linha vertical esquerda
    cv2.line(result_frame, (cx - half_width, cy - half_height), (cx - half_width, cy + half_height), color, 5)


    cv2.putText(result_frame, str(text), origem, font, 1, (200, 50, 0), 2, cv2.LINE_AA)

    # Exibe resultado
    cv2.imshow("Feed", result_frame)
    # cv2.imshow("Feed", square_frame)
    # cv2.imshow("Feed", rect_frame)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()