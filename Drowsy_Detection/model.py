import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# Aliases for drawing and face mesh
mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils

# Haar Cascade for full-face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Model input size
IMG_SIZE = 145

# Eye landmark indices from MediaPipe FaceMesh
LEFT_EYE_INDEXES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
# Initialize FaceMesh once
_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                             max_num_faces=1,
                                             refine_landmarks=False,
                                             min_detection_confidence=0.5)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")
 
# CNN 모델 정의
class DrowsyEyeNet(nn.Module):
    def __init__(self, img_size=145):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=10, padding=5)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=12, padding=6)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2)
        self.drop  = nn.Dropout(0.1)
        fc_input = 128 * (img_size // 16) * (img_size // 16)
        self.fc1   = nn.Linear(fc_input, 128)
        self.drop2 = nn.Dropout(0.25)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool(x); x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool(x); x = self.drop(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool(x); x = self.drop(x)
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = self.drop2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 모델 가중치 파일명
MODEL_PATH = "model.pth"

# 모델 로드 함수
def load_model():
    model = DrowsyEyeNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        #print(f"모델 가중치 '{MODEL_PATH}' 로드 완료")
    else:
        print(f"모델 가중치 '{MODEL_PATH}'가 없습니다. 새 모델 생성 후 학습 필요")
    model.eval()
    return model

# 예측 함수: anomalies.py에서 호출할 함수
def predict_drowsiness(frame: np.ndarray) -> str:
    """
    frame: anomalies.py에서 캡처된 frame (numpy array, BGR)
    반환: "drowsy" 또는 "notdrowsy"
    """
    # 1) Detect full face and crop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
    else:
        face_roi = frame

    # 2) Overlay face mesh tessellation and eye landmarks
    overlay = face_roi.copy()
    rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb_roi)
    if results.multi_face_landmarks:
        for fl in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=fl,
                connections=mp_facemesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255,255,255))
            )
            for li in LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES:
                lm = fl.landmark[li]
                px = int(lm.x * overlay.shape[1])
                py = int(lm.y * overlay.shape[0])
                cv2.circle(overlay, (px, py), 2, (255,255,255), -1)

    # 3) Resize to IMG_SIZE
    resized = cv2.resize(overlay, (IMG_SIZE, IMG_SIZE))

    # 4) Convert to tensor [0,1] only
    tensor_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])(resized)
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = load_model()(tensor_img)
        probability = torch.sigmoid(output).item()
    
    result = "notdrowsy" if probability > 0.5 else "drowsy"
    print(f"[모델 예측] 결과: {result}")
    return result

# 테스트 코드 (직접 실행 시 샘플 이미지로 모델 추론 테스트)
if __name__ == "__main__":
    # 테스트용: captured 폴더에서 임의의 이미지를 불러와서 추론
    test_img_path = os.path.join("captured", os.listdir("captured")[0])
    print(f"테스트 이미지: {test_img_path}")
    test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)  # 원래 anomalies.py에서는 이미지를 저장
    if test_img is None:
        print("이미지를 불러올 수 없습니다.")
    else:
        predict_drowsiness(test_img)