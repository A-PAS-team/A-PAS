# 🚶‍♂️ A-PAS (AI-based Pedestrian Alert System)
### 보행자 중심의 엣지 AI 스마트 횡단보도 경고 시스템
> 한국공학대학교(TUKOREA) 전자공학부 졸업작품 (2026.01 ~ 2026.07, 진행 중)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A)
![Hailo-8](https://img.shields.io/badge/NPU-Hailo--8%20AI%20HAT%2B%2026TOPS-brightgreen)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)

<br>

## 프로젝트 개요 (Project Overview)
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
| 구분 | 장비명 | 사양 | 용도 |
| --- | --- | --- | --- |
| **Main Controller** | Raspberry Pi 5 | 16GB LPDDR4X | 전체 시스템 제어 및 LSTM 추론 |
| **AI Accelerator** | Hailo-8 AI HAT+ | 26 TOPS NPU (PCIe M.2) | YOLOv8 실시간 추론 (430+ FPS) |
| **Vision Input** | HDMI-to-CSI 캡처 보드 | 22핀 MIPI CSI | CARLA 시뮬레이션 영상 수신 |
| **Display** | 포터블 모니터 | 15인치 Micro HDMI | Risk Score 및 예측 경로 시각화 |
| **Alert System** | LED Strip + Active Buzzer | WS2812B | 위험 상황 알림 (시각/청각) |
| **Simulation Env** | CARLA Simulator | Logitech G29 연동 | 가상 차량 접근 시나리오 검증 |
| **Storage** | MicroSD | 128GB V30 이상 | OS + 코드 + 모델 가중치 저장 |

### AI 추론 역할 분담
| 모델 | 실행 위치 | 이유 |
| --- | --- | --- |
| **YOLOv8n** (객체 탐지) | Hailo-8 NPU | 이미지 병렬 연산 → NPU 최적화 |
| **Trajectory LSTM** (경로 예측) | Raspberry Pi 5 CPU | 순차적 시계열 연산 → CPU 적합 |

> 📝 **입력 방식 선택 근거**: 실시간 CCTV 연결 방식은 네트워크 지연, 하드웨어 인터페이스 오류 등 통제 불가능한 변수가 존재합니다. 사전 녹화 영상 방식은 조명, 날씨, 혼잡도 등 환경 변수를 직접 통제할 수 있어 AI 파이프라인의 정확도와 안정성을 일관되게 검증할 수 있다는 판단 하에 채택했습니다.

### Technology Stack
* **AI Model**
    * **Detection**: YOLOv8n (Optimized for Hailo-8 NPU, int8 양자화)
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
        ├── Hailo-8 NPU: YOLOv8 추론 → 객체 좌표 추출 (430+ FPS)
        │           ↓
        └── RPi CPU: LSTM 추론 → TTC(충돌 예상 시간) 계산
                    ↓ 위험 감지
            LED / Buzzer 경고 출력
                    ↓
            모니터 시각화 (Risk Score + 예측 경로 오버레이)
```

<br>

## 📂 폴더 구조 (Directory Structure)
```text
A-PAS/
├── 📂 raw_data/                        # 원본 영상 모아두는 곳 (학습의 시작점)
├── 📂 ai_model/
│   ├── 📂 npy_processing/              # NPY 데이터 생성 관련 코드
│   │   ├── 🐍 video_to_csv.py          # 영상 → CSV 변환 (YOLOv8 ByteTrack 기반)
│   │   ├── 🐍 jpg_to_csv.py            # JPG 이미지 → NPY 변환
│   │   ├── 🐍 csv_to_npy_10fps.py      # CSV → NPY 변환 (10FPS 샘플링)
│   │   └── 🐍 csv_to_npy_30fps.py      # CSV → NPY 변환 (30FPS 원본)
│   ├── 📂 trajectory/                  # LSTM 모델 학습 관련 코드
│   │   ├── 🐍 tr_trajectory_10fps.py   # LSTM 학습 (10FPS, SEQ=20)
│   │   ├── 🐍 tr_trajectory_30fps.py   # LSTM 학습 (30FPS, SEQ=60)
│   │   └── 🐍 ONNX_convert.py          # PyTorch → ONNX 변환
│   |── 📂 data/                        # 추출된 NPY 데이터 저장 폴더
│   |   ├── Training/                   # 학습용 NPY 파일
│   |   └── Validation/                 # 검증용 NPY 파일
|   └── 🐍 optimize_model.py            # LSTM 모델 경량화 (동적 양자화)
├── 📂 models/                          # 학습 결과물 저장 (모델 구조별 분리)
│   ├── 📂 Self_LSTM/                   # ✅ 기본 LSTM (완료)
│   │   ├── 🧠 best_model_10fps.pth     # 10FPS 모델 가중치
│   │   ├── 🧠 best_model_30fps.pth     # 30FPS 모델 가중치
│   │   └── 📊 training_report_*.png    # 학습 결과 그래프
│   ├── 📂 Bidirectional/               # 🔜 양방향 LSTM (예정)
│   ├── 📂 Attention/                   # 🔜 LSTM + Attention (예정)
│   └── 📂 Residual/                    # 🔜 LSTM + Residual (예정)
├── 📂 embedded/                        # 라즈베리파이용 하드웨어 제어 코드
│   ├── 🐍 camera.py                    # 카메라 영상 수집 모듈
│   └── 🐍 alert.py                     # LED / 부저 제어 모듈
├── 🐍 main.py                          # 전체 시스템 실행 파일 (Raspberry Pi)
├── 📄 requirements.txt                 # PC 환경 설정
├── 📄 requirements-pi.txt              # 라즈베리파이 환경 설정
└── 📄 README.md                        # 설명서
```

<br>

## 시작하기 (Getting Started)
이 프로젝트는 **학습용 PC (Windows)** 와 **실행용 라즈베리 파이 (Embedded)** 의 환경 설정 방법이 다릅니다.

**1. 가상환경 생성 (공통)**
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
```bash
pip install -r requirements.txt
```

**Case B: 라즈베리 파이 5 (Run-time)**
```bash
pip install -r requirements-pi.txt
```

<br>

## 💻 사용방법 (Usage)

### Step 1. 영상 → CSV 변환 (PC)
원본 영상(`.mp4`)에서 YOLO ByteTrack으로 객체를 추적하여 CSV를 생성합니다.

```bash
cd ai_model/npy_processing

python video_to_csv.py
# 결과: ai_model/data/csv/*.csv 생성
```

### Step 2. CSV → NPY 변환 (PC)
CSV 데이터에서 17개 피처를 추출하고 학습용 NPY 파일을 생성합니다.

```bash
cd ai_model/npy_processing
# [방법 A] 10FPS 샘플링 (SEQ=20, PRED=10)
python csv_to_npy_10fps.py
# 결과: data/Training/X_train_10fps.npy, y_train_10fps.npy 등

# [방법 B] 30FPS 원본 (SEQ=60, PRED=30)
python csv_to_npy_30fps.py
# 결과: data/Training/X_train_30fps.npy, y_train_30fps.npy 등
```

### Step 3. LSTM 충돌 예측 모델 학습 (PC)
17개 피처 기반 Trajectory LSTM 모델을 학습합니다.

```bash
cd ai_model/trajectory

# [방법 A] 10FPS 모델 학습
python tr_trajectory_10fps.py
# 결과: models/Self_LSTM/best_model_10fps.pth
#       models/Self_LSTM/training_report_10fps.png

# [방법 B] 30FPS 모델 학습
python tr_trajectory_30fps.py
# 결과: models/Self_LSTM/best_model_30fps.pth
#       models/Self_LSTM/training_report_30fps.png
```

> **평가지표 설명**
> | 지표 | 설명 | 목표 |
> | --- | --- | --- |
> | **MSE Loss** | 학습 손실값 (낮을수록 좋음) | - |
> | **ADE (px)** | 예측 경로 전체의 평균 오차 | 10px 이하 |
> | **FDE (px)** | 1초 후 최종 도착 지점의 오차 | 15px 이하 |

### Step 4. ONNX 변환 (PC)
학습된 모델을 추론용 ONNX 포맷으로 변환합니다.

```bash
cd ai_model/trajectory
python ONNX_convert.py
# 결과: models/Self_LSTM/best_model_10fps.onnx
```

### Step 5. 메인 시스템 구동 (Raspberry Pi)
학습된 모델을 라즈베리 파이로 옮긴 후 실시간 경고 시스템을 실행합니다.

```bash
# 프로젝트 최상위 폴더로 이동
cd ../..

# 메인 시스템 실행
python main.py
```

<br>

## 📊 학습 데이터 규격
| 항목 | 10FPS | 30FPS |
| --- | --- | --- |
| **관찰 길이** | 20 프레임 (2.0초) | 60 프레임 (2.0초) |
| **예측 길이** | 10 프레임 (1.0초) | 30 프레임 (1.0초) |
| **입력 피처 수** | 17개 | 17개 |
| **DT** | 0.1초 | 1/30초 |
| **데이터 소스** | 실제 CCTV 영상 + CARLA 시뮬레이션 | 동일 |

### 17개 피처 구성
| 구분 | 피처 | 개수 |
| --- | --- | --- |
| 위치 | x, y | 2 |
| 속도 | vx, vy | 2 |
| 가속도 | ax, ay | 2 |
| 운동 특성 | speed, sin(heading), cos(heading) | 3 |
| 크기 | w, h, area | 3 |
| 클래스 원-핫 | 사람, 차량, 버스, 트럭, 오토바이 | 5 |

### 모델 구조별 성능 비교
| 모델 구조 | FPS | ADE | FDE | 상태 |
| --- | --- | --- | --- | --- |
| **Self LSTM** | 10FPS | - | - | 🔄 학습 중 |
| Bidirectional LSTM | - | - | - | 🔜 예정 |
| LSTM + Attention | - | - | - | 🔜 예정 |
| LSTM + Residual | - | - | - | 🔜 예정 |

<br>

## ⚠️ 주의사항
* 전원 어댑터는 반드시 **27W (5V 5A) 이상 PD 어댑터**를 사용하세요. 27W 미만은 Hailo-8 NPU 피크 전력 부족으로 불안정할 수 있습니다.
* MicroSD는 **V30 등급 이상** (SanDisk Extreme 또는 공식 Raspberry Pi 카드)을 권장합니다.
* `models/` 폴더 내 `.pth`, `.onnx` 파일은 `.gitignore`에 의해 Git에서 제외됩니다. 라즈베리 파이로 직접 전송하세요.
* `data/Training/`, `data/Validation/` 폴더 내 NPY 파일은 용량이 크므로 Git에서 제외됩니다.

<br>

---
### 본 프로젝트는 한국공학대학교 전자공학부 졸업작품으로 진행됐습니다.
