# README

```bash

source venv/bin/activate

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

# tts
# 因为 pip 下载太慢，所以根据 tts 所需版本，直接在 https://download.pytorch.org/whl/torch/ 下载
# wget  wget https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.1.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl#sha256=88f1ee550c6291af8d0417871fb7af84b86527d18bc02ac4249f07dcd84dda56
# pip install wheel
# pip install torch-2.1.0+cpu.cxx11.abi-cp310-cp310-linux_x86_64.whl

pip install tts

```

在 checkpoints 文件夹中准备好预训练模型。

```bash

python app_sadtalker.py

```
