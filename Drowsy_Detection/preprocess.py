# preprocess_data.py
import os
import cv2
import mediapipe as mp
from tqdm import tqdm

# 스크립트 파일 위치를 기준으로 디렉터리 설정
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, 'train_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessed_data')

IMG_SIZE   = 145

# MediaPipe & Haarcascade 초기화
mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
face_mesh   = mp_facemesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 클래스별 폴더 순회
for cls in ['drowsy', 'notdrowsy']:
    in_dir  = os.path.join(INPUT_DIR, cls)
    out_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(os.listdir(in_dir), desc=f'Preprocessing {cls}'):
        src_path = os.path.join(in_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        # 1) 얼굴 검출 & 크롭
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = img[y:y+h, x:x+w]
        else:
            roi = img

        # 2) MediaPipe FaceMesh 테셀레이션만 오버레이
        overlay = roi.copy()
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=fl,
                    connections=mp_facemesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1, circle_radius=1, color=(255,255,255))
                )

        # 3) 리사이즈 및 저장
        out_img = cv2.resize(overlay, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(out_dir, fname), out_img)