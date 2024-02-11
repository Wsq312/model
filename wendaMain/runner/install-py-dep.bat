"%CD%/py310/python.exe" ./backend-python/get-pip.py -i https://pypi.tuna.tsinghua.edu.cn/simple
"%CD%/py310/python.exe" -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
"%CD%/py310/python.exe" -m pip install -r ./backend-python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
exit