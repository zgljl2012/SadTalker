
import base64
import requests
import json

def aiy_sadtalker_test(image: str, audio: str, image_ext: str, audio_ext: str):
    with open(image, 'rb') as fi:
        image_encoded = base64.b64encode(fi.read()).decode('utf-8')
    with open(audio, 'rb') as fi:
        audio_encoded = base64.b64encode(fi.read()).decode('utf-8')
    print('Start post...')
    res = requests.post('http://127.0.0.1:5001/sadtalker', json={
        'image': image_encoded,
        'audio': audio_encoded,
        'image_ext': image_ext,
        'audio_ext': audio_ext
    })
    print('Get response...')
    print(res.status_code)
    print(res.text)

if __name__ == '__main__':
    # image = './examples/source_image/full_body_1.png'
    image = '/mnt/d/workspace/Aiy/小和尚说禅/00002-1230880017.png'
    image_ext = 'png'
    audio = '/mnt/d/workspace/Aiy/outputs/d2d2bcd3-8f59-4ffe-8fca-51d8a662b53e.wav'
    audio_ext = 'wav'
    aiy_sadtalker_test(image, audio, image_ext, audio_ext)
