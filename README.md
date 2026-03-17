# 🚶‍♂️ A-PAS (AI-based Pedestrian Alert System)
### 보행자 중심의 엣지 AI 스마트 횡단보도 경고 시스템
> 한국공학대학교(TUKOREA) 전자공학부 졸업작품 (2026.01 ~ 2026.07, 진행 중)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A)
![TensorFlow Lite](https://img.shields.io/badge/AI-TFLite%20%26%20Coral%20TPU-orange)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)

<br>

## 📌 프로젝트 개요 (Project Overview)
**A-PAS**는 사각지대 및 보행자 부주의(스몸비)로 인한 횡단보도 사고를 예방하기 위한 시스템입니다.
기존 시스템과 달리 **인프라(횡단보도)**가 스스로 위험을 판단하고, 보행자에게 직관적인 **시각/청각 경고**를 제공하여 사고를 능동적으로 방지합니다.

> **핵심 목표 (Core Goal)**
> * **Vision-only Solution**: 라이다 없이 영상과 AI만으로 정밀 충돌 예측
> * **Edge AI**: 클라우드 없이 독립적으로 작동하는 고성능 엣지 컴퓨팅 구현

* **개발 기간**: 2026.01.07 ~ 2026.07 (약 7개월, 진행 중)
* **팀 구성 및 역할**: 4인 팀 / **Team Leader (System Integration & Edge HW Pipeline 최적화 담당)**

<br>

## 🚀 핵심 문제 해결 및 시스템 최적화 (Key Engineering Achievements)
자원과 환경이 제한된 엣지 디바이스 위에서 실시간 AI 파이프라인을 구축하며 다음과 같은 시스템/데이터 구조 최적화를 수행했습니다.

**1. 고품질 학습 데이터 파이프라인 전면 재구축 (Stitching & Optimization)**
* **이슈**: AI 허브 공개 데이터가 10프레임 단위로 분절되어 있어, 설계한 30프레임 규격(과거 20프레임 입력 + 미래 10프레임 예측)의 LSTM 학습 파이프라인을 통과하지 못하고 전량 스킵되는 구조적 문제 발생.
* **해결**:
  * 폴더 이름의 연속성(C0189, C0190 등)을 이용해 끊어진 조각들을 하나의 긴 시퀀스로 이어 붙이는 **스티칭(Stitching) 로직** 직접 설계.
  * 도시도로교통공사 CCTV 데이터(CSV)를 엣지 환경에 맞게 10fps로 변환 및 NPY 포맷으로 가공하여 피처 규격을 17개로 통일, 단일 학습 데이터셋으로 통합.
* **결과**: 버려질 뻔한 데이터를 고품질 시퀀스로 복원하여, 검증 데이터 1.2만 개 및 훈련 데이터 12만 개 규모로 학습 인프라 정상화.

**2. 엣지 환경(Raspberry Pi + TPU) 실시간 추론 파이프라인 최적화**
* 제한된 메모리와 연산 능력을 가진 Raspberry Pi 5 환경에서 YOLOv8 객체 탐지와 LSTM 경로 예측이 지연 없이 맞물려 돌아가도록 시스템 아키텍처 및 데이터 전달 구조 최적화 중. (2026.07 완성 목표)

<br>

## 🏗 시스템 아키텍처 (System Architecture)

### Hardware Configuration
| 구분 | 장비명 | 용도 |
| --- | --- | --- |
| **Main Controller** | Raspberry Pi 5 | 전체 시스템 제어 및 연산 |
| **AI Accelerator** | Google Coral USB TPU | 딥러닝 추론 가속 (Real-time) |
| **Vision Input** | 사전 녹화 영상 (CCTV 수집 데이터) | 환경 변수 통제 및 모델 성능 검증용 입력 |
| **Alert System** | LED Strip (WS2812B), Active Buzzer | 위험 상황 알림 (시각/청각) |
| **Simulation Env** | Virtual Simulator (Unity/CARLA) | 가상 차량 접근 시나리오 검증 |

> 📝 **입력 방식 선택 근거**: 실시간 CCTV 연결 방식은 네트워크 지연, 하드웨어 인터페이스 오류 등 통제 불가능한 변수가 존재합니다. 사전 녹화 영상 방식은 조명, 날씨, 혼잡도 등 환경 변수를 직접 통제할 수 있어 AI 파이프라인의 정확도와 안정성을 일관되게 검증할 수 있다는 판단 하에 채택했습니다.

### Technology Stack
* **AI Model**
    * **Detection**: YOLOv8n (Optimized for Coral TPU)
    * **Prediction**: Trajectory LSTM (PyTorch) - 이동 경로 예측 및 충돌 감지
* **Software**
    * Python 3.x, OpenCV, TensorFlow Lite, PyTorch
* **OS**
    * Raspberry Pi OS (Bookworm 64-bit)

<br>

## 📂 폴더 구조 (Directory Structure)
```text
A-PAS/
├── 📂 raw_data/               # 원본 영상 모아두는 곳 (학습의 시작점)
├── 📂 ai_model/
│   └── yolo_train/
│       ├── 📂 data/           # 영상에서 추출한 CSV 데이터 (스티칭 및 전처리 완료)
│       ├── 🐍 video_to_csv.py # 영상 -> CSV 변환 및 시퀀스 스티칭 코드
│       ├── 🐍 tr_trajectory.py # LSTM 학습 코드
│       ├── 🐍 train.py        # YOLO 파인튜닝용 스크립트
│       └── 🧠 trajectory_model.pth # 최종 LSTM 모델 가중치
├── 📂 embedded/               # 라즈베리파이용 하드웨어 제어 코드
│   ├── 🐍 camera.py           # 카메라 영상 수집 모듈
│   └── 🐍 alert.py            # LED/부저/스피커 제어 모듈
├── 📂 dataset/                # YOLO 학습용 이미지 데이터셋
├── 🐍 main.py                 # 전체 시스템 실행 파일
├── 📄 requirements.txt        # PC 환경 설정
├── 📄 requirements-pi.txt     # 라즈베리파이 환경 설정
└── 📄 README.md               # 설명서
```

<br>

## 🏁 시작하기 (Getting Started)
이 프로젝트는 **학습용 PC(Windows/Linux)**와 **실행용 라즈베리 파이(Embedded)**의 환경 설정 방법이 다릅니다.

**1. 가상환경 생성 (공통)**
```bash
# 가상환경 생성
python -m venv a-pas-env

# 가상환경 실행 (Windows)
.\a-pas-env\Scripts\activate

# 가상환경 실행 (Mac/Linux/Raspberry Pi)
source a-pas-env/bin/activate
```

**2. 라이브러리 설치 (Environment Setup)**

Case A: 학습용 PC / 노트북 (Windows)
```bash
pip install -r requirements.txt
```

Case B: 라즈베리 파이 5 (Run-time)
```bash
pip install -r requirements-pi.txt
```

<br>

## 💻 사용방법 (Usage)

**Step 1. 데이터 준비 및 전처리 (PC)**
```bash
# 1. 학습 코드가 있는 폴더로 이동
cd ai_model/yolo_train

# 2. 영상 데이터를 분석하여 CSV로 변환 (전처리)
python video_to_csv.py

# (결과: data/ 폴더 내에 전처리된 normal_data_*.csv 파일들이 자동 생성됨)
```

**Step 2. LSTM 충돌 예측 모델 학습 (PC)**
```bash
# 3. LSTM 모델 학습 시작
python tr_trajectory.py

# (결과: 학습이 완료되면 같은 폴더에 'trajectory_model.pth' 가중치 파일 생성)
```

**Step 3. 메인 시스템 구동 (Raspberry Pi)**
```bash
# 4. 프로젝트 최상위 폴더(A-PAS)로 이동
cd ../..

# 5. 메인 시스템 실행
python main.py
```
