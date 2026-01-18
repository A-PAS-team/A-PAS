# A-PAS (AI-based Pedestrian Alert System)
### 보행자 중심의 엣지 AI 스마트 횡단보도 경고 시스템

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A)
![TensorFlow Lite](https://img.shields.io/badge/AI-TFLite%20%26%20Coral%20TPU-orange)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)

<br>

## 프로젝트 개요 (Project Overview)
**A-PAS**는 사각지대 및 보행자 부주의(스몸비)로 인한 횡단보도 사고를 예방하기 위한 시스템입니다. 
기존 시스템과 달리 **인프라(횡단보도)**가 스스로 위험을 판단하고, 보행자에게 직관적인 **시각/청각 경고**를 제공하여 사고를 능동적으로 방지합니다.

> **핵심 목표 (Core Goal)**
> * **One Scenario Perfection**: 주간 환경에서의 차량 접근 탐지 및 단계별 위험 경고 완벽 구현
> * **Edge AI**: 클라우드 없이 독립적으로 작동하는 고성능 엣지 컴퓨팅 구현

**개발 기간**: 2026.01.07 ~ 2026.07 (약 7개월)

<br>

## 시스템 아키텍처 (System Architecture)

### Hardware Configuration
| 구분 | 장비명 | 용도 |
| --- | --- | --- |
| **Main Controller** | Raspberry Pi 5 (8GB) | 전체 시스템 제어 및 연산 |
| **AI Accelerator** | Google Coral USB TPU | 딥러닝 추론 가속 (Real-time) |
| **Vision Sensor** | CCTV (IP Camera / RTSP) | 횡단보도 실시간 영상 수집 |
| **Alert System** | LED Strip (WS2812B), Active Buzzer | 위험 상황 알림 (시각/청각) |
| **Simulation Env** | Virtual Simulator (Unity/CARLA) | 가상 차량 접근 시나리오 검증 |

### Technology Stack
* **AI Model**
    * **Detection**: YOLOv8n (Optimized for Coral TPU)
    * **Prediction**: LSTM / 1D-CNN (Time-Series Risk Analysis)
* **Software**
    * Python 3.x, OpenCV, TensorFlow Lite, PyTorch
* **OS**
    * Raspberry Pi OS (Bookworm 64-bit)

<br>

## 폴더 구조 (Directory Structure)
```text
A-PAS/
├── 📂 raw_data/               # [NEW] 원본 영상 모아두는 곳 (학습의 시작점)
├── 📂 ai_model/
│   └── yolo_train/
│       ├── 📂 data/           # [NEW] 영상에서 추출한 CSV 데이터 (자동 생성)
│       ├── 🐍 video_to_csv.py # [NEW] 영상 -> CSV 변환 코드
│       ├── 🐍 tr_trajectory.py # [NEW] LSTM 학습 코드
│       ├── 🐍 train.py        # (기존) YOLO 파인튜닝이 필요하면 사용
│       └── 🧠 trajectory_model.pth # (결과물) 최종 LSTM 모델
├── 📂 embedded/               # [KEEP] 라즈베리파이용 하드웨어 제어 코드
│   ├── 🐍 camera.py           # 카메라 영상 수집 모듈
│   └── 🐍 alert.py            # LED/부저/스피커 제어 모듈
├── 📂 dataset/                # [OPTIONAL] YOLO 학습용 이미지 데이터셋 (필요시 사용)
├── 🐍 main.py                 # 전체 시스템 실행 파일
├── 📄 requirements.txt        # PC 환경 설정
├── 📄 requirements-pi.txt     # 라즈베리파이 환경 설정
└── 📄 README.md               # 설명서
```
<br>

## 시작하기 (Getting Started)
이 프로젝트는 **학습용 PC(Windows/Linux)**와 **실행용 라즈베리 파이(Embedded)**의 환경 설정 방법이 다릅니다.

### 1. 가상환경 생성 (공통)
프로젝트 루트 경로에서 가상환경을 생성합니다.

```bash
## 가상환경 생성
python -m venv a-pas-env

# 가상환경 실행 (Windows)
.\a-pas-env\Scripts\activate

# 가상환경 실행 (Mac/Linux/Raspberry Pi)
source a-pas-env/bin/activate
```
<br>

## 2. 라이브러리 설치 (Environment Setup)
Case A: 학습용 PC / 노트북 (Windows) GPU 학습 및 데이터 분석을 위한 라이브러리(PyTorch, Ultralytics 등)를 설치합니다.

```bash
pip install -r requirements.txt
```
Case B: 라즈베리 파이 5 (Run-time) 실시간 추론을 위한 경량화 라이브러리(TFLite, GPIO 등)를 설치합니다.
```bash
pip install -r requirements-pi.txt
```

<br>

## 사용방법(Usage)
### 1. AI 모델 학습 (Training on PC)
데이터셋이 준비되면 PC에서 아래 명령어로 학습을 진행합니다.
```bash
python ai_model/yolo_train/train.py
```
### 2. 메인 시스템 구동 (Running on Pi)
라즈베리 파이에서 시스템을 시작합니다. (카메라 및 센서 연결 필수)
```bash
python main.py
```

<br>

### 본 프로젝트는 한국공학대학교 전자공학부 졸업작품으로 진행됐습니다.