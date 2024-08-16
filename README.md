# Overview

본 코드는 MNIST dataset을 이용하여 ResNet-20을 learning rate 또는 weight offeset을 다양하게 주어 여러 경우의 모델을 학습시킬 수 있다. 이를 통해 동일한 epochs만큼 학습하고 나면, 각 경우의 모델이 loss가 제대로 수렴하는지 안하는지 관찰할 수 있다. 우리는 이를 loss의 수렴경계라고 지칭하고, 이 수렴경계가 프렉탈 형상을 띄는 것을 관찰할 수 있었다. 

This project is demonstration that the convergence boundary of training loss is fractal, and proves that the layer of skip-connection could shrink the fractal dimension.

# Method

###

# How to use

### 처음 내려받을 때...

**1. 본인 로컬에 anaconda를 설치한다.**
   
**2. Terminal 또는 CMD를 열고, conda를 통해 가상환경을 설치한다. 아래 코드를 참고!** \
   `conda init` \
   `conda create -n (가상환경이름) python=3.10` \
   `conda activate (가상환경이름)`
   
**3. git을 설치하고, github ID와 email을 등록한다.** \
   `sudo apt-get install git`    # Linux는 terminal에서 이걸 입력 \
   https://git-scm.com/download/win    # Windows는 여기서 다운받아 설치 \
   `git --version`    # version check; 버전나오면 설치OK \
   `git config --global user.name (githubID)` \
   `git config --global user.email (email)` \
   
**4. 이제 파일을 내려받는다! 그리고 packages를 설치한다.** \
   `git init`
   `git clone https://github.com/KY-HDC/fractal.git` \
   `pip install -r requirements.txt` \

**5. 세팅 완료!**


### 코드수정하고 올리고 싶을 때...(각자 branch로)

**1. git clone해서 받은 코드를 필요에 맞게 수정한다.**

**2. 내 branch를 생성하고, 해당 branch로 이동한다.** \
   `git checkout -b vX.XO`    # 현재 최신버전이 v4.6이므로, 수정 후 v4.7A, v4.7B라고 명명한다. A, B는 작성자 이니셜임. \
   `git status`         # "On branch vX.XO"라고 나오면 OK \

**4. 내가 작성 또는 수정한 파일들을 올린다.** \
   `git add .` \
   `git commit -m "변경사항을 간략하게 적어줌"` \
   `git remote add origin https://github.com/KY-HDC/fractal.git` \
   `git push origin vX.XO` \

**5. Github에서 본인 branch에 파일이 들어왔는지 확인한다.** 

### 본 코드를 돌리는 방법...

**1. 위에서 설치한 가상환경으로 준비한다.** \
   `conda activate (가상환경이름)` \
   `cd (본 코드가 위치한 디렉토리)`

**2. 자신이 설정하고 싶은 hyperparameter를 설정하여 main.py를 구동한다.** \
   `python main.py --num_epochs 300 --resolution 256` on gpu
   `python main.py --num_epochs 5 --resolution 4` on cpu if cannot use gpu

**3. 다른 hyperparameter를 보고싶다면 아래 코드를 참고!** \
   `python main.py -h`


# Hardware Requirements

Nvidia A100 80GB x4

# Dependency

CUDA TOOLKIT 12.5 \
CUDNN 9.3