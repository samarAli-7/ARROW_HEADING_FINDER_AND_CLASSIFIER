import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
arrow_type = None
text = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    true_frame = frame.copy()
    g = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = []
            for p in cnt:
                x, y = p[0]
                d = math.hypot(x - cx, y - cy)
                dist.append((d, (x, y)))
            dist.sort(key=lambda x: x[0], reverse=True)
            farthest_point1 = dist[0][1]
            for d, p in dist[1:]:
                if math.hypot(p[0] - farthest_point1[0], p[1] - farthest_point1[1]) > 30:
                    farthest_point2 = p
                    break
            else:
                farthest_point2 = dist[1][1]
            middle_point = (int((farthest_point1[0] + farthest_point2[0]) / 2), int((farthest_point1[1] + farthest_point2[1]) / 2))
            dx = middle_point[0] - cx
            dy = cy - middle_point[1]
            angle = math.degrees(math.atan2(-dx, -dy))
            if angle < 0:
                angle += 360
            # cv2.circle(frame, middle_point, 4, (0, 0, 255), -1)
            # cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"{int(angle)} deg", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            hull = cv2.convexHull(cnt)
            cnt_area=1
            hull_area=1
            cnt_area = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(hull)
            percentage = (cnt_area/hull_area)*100
            # print(f"percentage{percentage}")
            if percentage>=80 and percentage<=90:
                arrow_type = 3
                text = "ARROW_TYPE3"
            elif percentage>=65 and percentage<=75:
                arrow_type = 1
                text = "ARROW_TYPE1"
            elif percentage>=40 and percentage<=55:
                arrow_type = 2
                text = "ARROW_TYPE2"
            else:
                arrow_type = "unknown"
                text = "ARROW_TYPE_UNKNOWN"
            try:
                # print(arrow_type)
                pass
            except Exception as e:
                pass
            cv2.putText(frame, text, (cx+50,cy+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            # cv2.drawContours(frame, [hull], 0, (0, 0, 0), 3)
    cv2.imshow("Detection", frame)
    cv2.imshow("Live Video", true_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()