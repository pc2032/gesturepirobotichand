import cv2
import mediapipe as mp
import math
videoCapture = cv2.VideoCapture(0)

handDetector = mp.solutions.hands.Hands()
drawingUtils = mp.solutions.drawing_utils

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while True:
    success, frame = videoCapture.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frameHeight, frameWidth, _ = frame.shape
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = handDetector.process(rgbFrame)
    thumb = index = middle = ring = pinky = False

    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawingUtils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            landmark_coords = [(int(landmark.x * frameWidth), int(landmark.y * frameHeight)) for landmark in landmarks]

            if calculate_distance(*landmark_coords[8], *landmark_coords[5]) < 30:
                index = True
            if calculate_distance(*landmark_coords[12], *landmark_coords[9]) < 30:
                middle = True
            if calculate_distance(*landmark_coords[16], *landmark_coords[13]) < 30:
                ring = True
            if calculate_distance(*landmark_coords[20], *landmark_coords[17]) < 30:
                pinky = True
            if calculate_distance(*landmark_coords[4], *landmark_coords[9]) < 30:
                thumb = True
        print(f'Thumb: {thumb}, Index: {index}, Middle: {middle}, Ring: {ring}, Pinky: {pinky}')
    cv2.imshow('Hand Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

videoCapture.release()
cv2.destroyAllWindows()
