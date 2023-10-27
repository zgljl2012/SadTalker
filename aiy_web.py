from flask import Flask, abort, request, send_file
import os
from flask_cors import CORS
import base64
from aiy_main import SadTalkerTask
from aiy_scheduler import Scheduler

app = Flask(__name__)
CORS(app, origins="*")

sched = Scheduler()
sched.async_run()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sadtalker", methods=['POST'])
def sad_talker():
    data = request.json
    # Base64
    image = data['image']
    image_ext = data['image_ext']
    audio = data['audio']
    audio_ext = data['audio_ext']
    try:
        # create temp directory
        tmp_dir = 'tmp-1'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        # decode image to the temp directory
        img_path = f'{tmp_dir}/image.{image_ext}'
        with open(img_path, 'wb') as fw:
            fw.write(base64.b64decode(image))
        # decode audio to the temp directory
        audio_path = f'{tmp_dir}/audio.{audio_ext}'
        with open(audio_path, 'wb') as fw:
            fw.write(base64.b64decode(audio))
        req = sched.submit_task(SadTalkerTask(img_path, audio_path))
        return {
            'id': req.req_id
        }
    except Exception as e:
        print(e)
        return {
            'error': e
        }
    finally:
        # TODO Remove tmp directory
        pass

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)
