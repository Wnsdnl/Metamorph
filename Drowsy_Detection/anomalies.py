import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
from model import predict_drowsiness
import os
import time
import math


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# MediaPipe 기준 눈 인덱스
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

# 임계값
EAR_THRESHOLD = 0.23
HEAD_PITCH_THRESHOLD = -12
CONSEC_FRAMES = 30 # 지속 프레임 수 기준
COUNT = 0 # 졸음 횟수

frame_count = 0

save_dir = "captured"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(1)
ear_buffer = deque(maxlen=3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            size = frame.shape
            image_points = np.array([
                [face_landmarks.landmark[1].x * size[1], face_landmarks.landmark[1].y * size[0]],     # Nose tip
                [face_landmarks.landmark[152].x * size[1], face_landmarks.landmark[152].y * size[0]], # Chin
                [face_landmarks.landmark[263].x * size[1], face_landmarks.landmark[263].y * size[0]], # Right eye right corner
                [face_landmarks.landmark[33].x * size[1], face_landmarks.landmark[33].y * size[0]],   # Left eye left corner
                [face_landmarks.landmark[287].x * size[1], face_landmarks.landmark[287].y * size[0]], # Right mouth corner
                [face_landmarks.landmark[57].x * size[1], face_landmarks.landmark[57].y * size[0]]    # Left mouth corner
            ], dtype="double")


            # 입꼬리 웃음 판단
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]
            upper_lip_center = face_landmarks.landmark[13]

            left_smile = (upper_lip_center.y - mouth_left.y) * size[0]
            right_smile = (upper_lip_center.y - mouth_right.y) * size[0]
            smile_score = (left_smile + right_smile) / 2

            SMILE_THRESHOLD = 2.5  # 픽셀 기준
            is_smiling = smile_score > SMILE_THRESHOLD

            # 3D 모델 좌표 (고정된 평균 인체 얼굴 모델)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -63.6, -12.5),         # Chin
                (-43.3, 32.7, -26.0),        # Left eye left corner
                (43.3, 32.7, -26.0),         # Right eye right corner
                (-28.9, -28.9, -24.1),       # Left mouth corner
                (28.9, -28.9, -24.1)         # Right mouth corner
            ])

            # 카메라 내부 파라미터 설정
            size = frame.shape
            focal_length = size[1]
            center = (size[1] // 2, size[0] // 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4,1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            # pitch 계산
            rmat, _ = cv2.Rodrigues(rotation_vector)
            pitch = math.degrees(math.asin(-rmat[2][1]))

            # EAR 계산
            leftEye = np.array([
                [face_landmarks.landmark[i].x * size[1], face_landmarks.landmark[i].y * size[0]]
                for i in LEFT_EYE_INDEXES
            ])
            rightEye = np.array([
                [face_landmarks.landmark[i].x * size[1], face_landmarks.landmark[i].y * size[0]]
                for i in RIGHT_EYE_INDEXES
            ])
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ear_buffer.append(ear)
            avg_ear = sum(ear_buffer) / len(ear_buffer)

            # EAR 또는 고개 숙임 지속 시 졸음 감지
            if not is_smiling and (
                (avg_ear < EAR_THRESHOLD)
                or
                (pitch < HEAD_PITCH_THRESHOLD and avg_ear < EAR_THRESHOLD + 0.02)
            ):              
                frame_count += 1
                if frame_count >= CONSEC_FRAMES:
                    filename = os.path.join(save_dir, f"anomaliy_{int(time.time())}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"[이상징후 감지] 이미지 저장: {filename}")
                    
                    # 모델 추론 호출
                    result = predict_drowsiness(frame)
                
                    if result == 'drowsy':
                        COUNT += 1
                    
                    if COUNT == 2:
                        #LLM 대화 시도 & 백엔드에 전송
                        print('졸리세요?')
                        COUNT = 0
                    
                    frame_count = 0
                    time.sleep(0.5)
            else:
                frame_count = 0

            
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Smile Score: {smile_score:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.circle(frame, (int(image_points[0][0]), int(image_points[0][1])), 3, (0, 0, 255), -1)
            cv2.circle(frame, (int(image_points[1][0]), int(image_points[1][1])), 3, (0, 0, 255), -1)
            cv2.line(frame,
                    (int(image_points[0][0]), int(image_points[0][1])),
                    (int(image_points[1][0]), int(image_points[1][1])),
                    (0, 0, 255), 2)

            for (x, y) in leftEye:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)
            for (x, y) in rightEye:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)

    cv2.imshow("Sleepiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()