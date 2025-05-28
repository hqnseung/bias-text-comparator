import os
import shutil
import subprocess

for folder in ['model/LEFT_model', 'model/RIGHT_model']:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
print("모델 폴더 초기화 완료")

open('data/train_LEFT.txt', 'w').close()
open('data/train_RIGHT.txt', 'w').close()
print("train_LEFT.txt, train_RIGHT.txt 초기화 완료")

try:
    subprocess.run(['python', 'src/preprocess.py'], check=True)
    print("preprocess.py 실행 성공")
except subprocess.CalledProcessError:
    print("preprocess.py 실행 실패")

try:
    subprocess.run(['python', 'src/train_model.py'], check=True)
    print("train_model.py 실행 성공")
except subprocess.CalledProcessError:
    print("train_model.py 실행 실패")

try:
    subprocess.run(['python', 'src/generate.py'], check=True)
    print("generate.py 실행 성공")
except subprocess.CalledProcessError:
    print("generate.py 실행 실패")