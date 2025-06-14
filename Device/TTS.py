import pyttsx3

def speak_text(text):
    engine = pyttsx3.init()
    # 음성 속도를 조절합니다. (기본값보다 느리게 설정, 필요시 값을 조정하세요.)
    engine.setProperty('rate', 165)
    engine.say(text)
    engine.runAndWait()


if __name__ == '__main__':
    api_text = '차량 열선 키는 법은 보통 기어 변속기 근처에 있어요. 졸음 운전은 위험해요!'
    speak_text(api_text)