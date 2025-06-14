import pvporcupine
from pvrecorder import PvRecorder
from TTS import speak_text
import os
import json
import requests
import time
import speech_recognition as sr


SECRET_ID = 'dLbO1Nxecz86TA77Qd_B'
SECRET_PW = 'c9Z74tjKh2ZxF5UJxLcA-HtC_ASoGeyHYiNO4kXt'


# OAuth2 토큰 요청
oauth2_url = 'https://openapi.vito.ai/v1/authenticate'
token_data = {
    'grant_type': 'client_credentials',
    'client_id': SECRET_ID,
    'client_secret': SECRET_PW
}


# OAuth2 토큰 요청
token_resp = requests.post(oauth2_url, data=token_data)
token_resp.raise_for_status()
access_token = token_resp.json().get('access_token')

if not access_token:
    raise ValueError("액세스 토큰을 받지 못했습니다.")

porcupine = pvporcupine.create(
    access_key = 'ffLn9yII4vDE2qpG0AnVEscPG8r0cO1/oIQQ6tyOK2m6341Im76ftg==',
    keyword_paths=[os.getcwd()+"/길동아_ko_mac_v3_0_0.ppn"], # 여러 개 가능
    model_path=os.getcwd()+"/porcupine_params_ko.pv",
)

recorder = PvRecorder(frame_length=512, device_index=1)
recorder.start()
print("[준비 완료] 마이크 입력이 준비되었습니다")

recognizer = sr.Recognizer()

while True:
    pcm = recorder.read()
    keyword_index = porcupine.process(pcm)
    if keyword_index >= 0:
        speak_text('네, 듣고있어요.')
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.pause_threshold = 1.7
            print("말씀해주세요...")
            audio_data = recognizer.listen(source)
        audio_file = "sample.wav"
        with open(audio_file, "wb") as f:
            f.write(audio_data.get_wav_data())
        break


# STT API 호출 설정
config = {
  "use_itn": True,
  "use_disfluency_filter": True,
  "use_profanity_filter": False,
  "use_paragraph_splitter": False,
  "keywords": ["엉뜨", "하이패스", "차량", "김서림", "비보호 좌회전"]
}

transcribe_url = 'https://openapi.vito.ai/v1/transcribe'
headers = {'Authorization': f'Bearer {access_token}'}


# 음성 파일을 읽어 STT API에 요청 전송
with open('sample.wav', 'rb') as audio_file:
    files = {'file': audio_file}
    data = {'config': json.dumps(config)}
    transcribe_resp = requests.post(transcribe_url, headers=headers, data=data, files=files)

transcribe_resp.raise_for_status()
TRANSCRIBE_ID = str(transcribe_resp.json()['id'])


resp = requests.get(
    'https://openapi.vito.ai/v1/transcribe/'+ TRANSCRIBE_ID,
    headers=headers
)

# 응답 받기
start = time.time()
while(resp.json()['status'] != 'completed'):
    resp = requests.get(
    'https://openapi.vito.ai/v1/transcribe/'+ TRANSCRIBE_ID,
    headers=headers
    )
    resp.raise_for_status()
end = time.time()


print(resp.json()['results']['utterances'][0]['msg'])
print(f"실행 시간: {end - start:.4f}초")
