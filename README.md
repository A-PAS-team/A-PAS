# A-PAS (AI-based Pedestrian Alert System)
### 보행자 중심의 엣지 AI 스마트 횡단보도 경고 시스템

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A)
![Hailo-8](https://img.shields.io/badge/AI-Hailo--8%20AI%20HAT%2B%2026TOPS-brightgreen)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)

<br>

## 프로젝트 개요 (Project Overview)
**A-PAS**는 사각지대 및 보행자 부주의(스몸비)로 인한 횡단보도 사고를 예방하기 위한 시스템입니다.  
기존 시스템과 달리 **인프라(횡단보도)**가 스스로 위험을 판단하고, 보행자에게 직관적인 **시각/청각 경고**를 제공하여 사고를 능동적으로 방지합니다.

> **핵심 목표 (Core Goal)**
> * **Vision-only Solution**: 라이다 없이 영상과 AI만으로 정밀 충돌 예측
> * **Edge AI**: 클라우드 없이 독립적으로 작동하는 고성능 엣지 컴퓨팅 구현

**개발 기간**: 2026.01.07 ~ 2026.07 (약 7개월)

<br>

## 시스템 아키텍처 (System Architecture)

### Hardware Configuration
| 구분 | 장비명 | 사양 | 용도 |
| --- | --- | --- | --- |
| **Main Controller** | Raspberry Pi 5 | 16GB LPDDR4X | 전체 시스템 제어 및 LSTM 추론 |
| **AI Accelerator** | Hailo-8 AI HAT+ | 26 TOPS (PCIe M.2) | YOLOv8 실시간 추론 (430+ FPS) |
| **Vision Input** | HDMI-to-CSI 캡처 보드 | 22핀 MIPI CSI | CARLA 시뮬레이션 영상 수신 |
| **Display** | 포터블 모니터 | 15인치 Micro HDMI | Risk Score 및 예측 경로 시각화 |
| **Alert System** | LED Strip + Active Buzzer | WS2812B | 위험 상황 알림 (시각/청각) |
| **Simulation Env** | CARLA Simulator | Logitech G29 연동 | 가상 차량 접근 시나리오 검증 |
| **Storage** | MicroSD | 128GB V30 이상 | OS + 코드 + 모델 가중치 저장 |

### AI 추론 역할 분담
| 모델 | 실행 위치 | 이유 |
| --- | --- | --- |
| **YOLOv8n** (객체 탐지) | Hailo-8 AI HAT+ | 이미지 병렬 연산 → TPU 최적화 |
| **Trajectory LSTM** (경로 예측) | Raspberry Pi 5 CPU | 순차적 시계열 연산 → CPU 적합 |

### Technology Stack
* **AI Model**
    * **Detection**: YOLOv8n (Optimized for Hailo-8, int8 양자화)
    * **Prediction**: Trajectory LSTM (PyTorch) - 17개 피처 기반 이동 경로 예측 및 충돌 감지
* **Software**
    * Python 3.x, OpenCV, PyTorch, hailo-platform
* **OS**
    * Raspberry Pi OS (Bookworm 64-bit)

### 시스템 처리 흐름
```
CARLA Simulator (PC)
        ↓ HDMI
HDMI-to-CSI 캡처 보드
        ↓ CSI
  Raspberry Pi 5
        ├── Hailo-8 AI HAT+: YOLOv8 추론 → 객체 좌표 추출 (430+ FPS)
        │           ↓
        └── RPi CPU: LSTM 추론 → TTC(충돌 예상 시간) 계산
                    ↓ 위험 감지
            LED / Buzzer 경고 출력
                    ↓
            모니터 시각화 (Risk Score + 예측 경로 오버레이)
```

<br>

## 폴더 구조 (Directory Structure)
```text
A-PAS/
├── 📂 raw_data/                  # 원본 영상 모아두는 곳 (학습의 시작점)
├── 📂 ai_model/
│   └── yolo_train/
│       ├── 📂 data/              # 추출된 NPY 데이터 저장 폴더
│       │   ├── Training/         # 학습용 NPY 파일
│       │   └── Validation/       # 검증용 NPY 파일
│       ├── 🐍 video_to_csv.py    # 영상 → CSV 변환 (YOLOv8 tracking 기반)
│       ├── 🐍 jpg_to_csv.py      # JPG 이미지 → NPY 변환 (17개 피처 추출)
│       ├── 🐍 merge_npy.py       # NPY 파일 통합 및 9:1 분할
│       ├── 🐍 tr_trajectory.py   # LSTM 학습 코드
│       └── 🐍 train.py           # YOLO 파인튜닝 (필요시 사용)
├── 📂 models/                    # 학습 결과물 저장
│   ├── 🧠 best_trajectory_model.pth  # 최고 성능 LSTM 모델 가중치
│   └── 📊 training_report.png        # 학습 결과 그래프 (Loss / ADE / FDE)
├── 📂 embedded/                  # 라즈베리파이용 하드웨어 제어 코드
│   ├── 🐍 camera.py              # 카메라 영상 수집 모듈
│   └── 🐍 alert.py               # LED / 부저 제어 모듈
├── 🐍 main.py                    # 전체 시스템 실행 파일 (Raspberry Pi)
├── 🐍 optimize_model.py          # LSTM 모델 경량화 (동적 양자화)
├── 📄 requirements.txt           # PC 환경 설정
├── 📄 requirements-pi.txt        # 라즈베리파이 환경 설정
└── 📄 README.md                  # 설명서
```

<br>

## 시작하기 (Getting Started)
이 프로젝트는 **학습용 PC (Windows)** 와 **실행용 라즈베리 파이 (Embedded)** 의 환경 설정 방법이 다릅니다.

### 1. 가상환경 생성 (공통)
프로젝트 루트 경로에서 가상환경을 생성합니다.

```bash
# 가상환경 생성
python -m venv a-pas-env

# 가상환경 실행 (Windows)
.\a-pas-env\Scripts\activate

# 가상환경 실행 (Mac/Linux/Raspberry Pi)
source a-pas-env/bin/activate
```

### 2. 라이브러리 설치 (Environment Setup)

**Case A: 학습용 PC / 노트북 (Windows)**  
GPU 학습 및 데이터 전처리를 위한 라이브러리(PyTorch, Ultralytics 등)를 설치합니다.
```bash
pip install -r requirements.txt
```

**Case B: 라즈베리 파이 5 (Run-time)**  
실시간 추론을 위한 경량화 라이브러리(hailo-platform, GPIO 등)를 설치합니다.
```bash
pip install -r requirements-pi.txt
```

<br>

## 💻 사용방법 (Usage)

### Step 1. 데이터 전처리 (PC)
원본 영상(`.mp4`) 또는 JPG 이미지에서 AI 학습용 피처 데이터를 추출합니다.

```bash
# 학습 코드가 있는 폴더로 이동
cd ai_model/yolo_train

# [방법 A] 영상 → CSV 변환 (video_to_csv.py)
python video_to_csv.py
# 결과: data/ 폴더에 normal_data_*.csv 파일 생성

# [방법 B] JPG 이미지 → NPY 변환 (17개 피처, jpg_to_csv.py)
python jpg_to_csv.py
# 결과: X_Training_17f.npy, y_Training_17f.npy 등 생성
```

### Step 2. 데이터 통합 및 분할 (PC)
수집된 NPY 파일들을 하나로 합치고 9:1 비율로 Train / Validation 분할합니다.

```bash
python merge_npy.py
# 결과: data/X_train_final.npy, data/y_train_final.npy
#       data/X_val_final.npy,   data/y_val_final.npy
```

### Step 3. LSTM 충돌 예측 모델 학습 (PC)
17개 피처 기반 Trajectory LSTM 모델을 학습합니다.

```bash
python tr_trajectory.py
# 결과: models/best_trajectory_model.pth (최고 성능 모델 가중치)
#       models/training_report.png       (Loss / ADE / FDE 그래프)
```

> **평가지표 설명**
> | 지표 | 설명 | 목표 |
> | --- | --- | --- |
> | **MSE Loss** | 학습 손실값 (낮을수록 좋음) | - |
> | **ADE (px)** | 예측 경로 전체의 평균 오차 | 20px 이하 |
> | **FDE (px)** | 1초 후 최종 도착 지점의 오차 | 30px 이하 |

### Step 4. 메인 시스템 구동 (Raspberry Pi)
학습된 모델을 라즈베리 파이로 옮긴 후 실시간 경고 시스템을 실행합니다.

```bash
# 프로젝트 최상위 폴더로 이동
cd ../..

# 메인 시스템 실행
python main.py
```

<br>

## 📊 학습 데이터 규격
| 항목 | 값 |
| --- | --- |
| **총 시퀀스 수** | 124,492개 (Train 112,042 / Val 12,450) |
| **입력 피처 수** | 17개 (위치, 속도, 가속도, 방향, 크기, 클래스) |
| **관찰 길이** | 20 프레임 (2.0초 @ 10FPS) |
| **예측 길이** | 10 프레임 (1.0초 @ 10FPS) |
| **데이터 소스** | CCTV 영상 (video_to_csv) + JPG 이미지 (jpg_to_csv) |

<br>

## ⚠️ 주의사항
* 전원 어댑터는 반드시 **30W (5V 6A) 이상 PD 어댑터**를 사용하세요. 27W 이하는 Hailo-8 피크 전력 부족으로 불안정할 수 있습니다.
* MicroSD는 **V30 등급 이상** (SanDisk Extreme 또는 공식 Raspberry Pi 카드)을 권장합니다.
* `models/` 폴더 내 `.pth` 파일은 `.gitignore`에 의해 Git에서 제외됩니다. 라즈베리 파이로 직접 전송하세요.

<br>

---
### 본 프로젝트는 한국공학대학교 전자공학부 졸업작품으로 진행됐습니다.