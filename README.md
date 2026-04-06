# 🚶‍♂️ A-PAS (AI-based Pedestrian Alert System)
### 보행자 중심의 엣지 AI 스마트 횡단보도 경고 시스템
> 한국공학대학교(TUKOREA) 전자공학부 졸업작품 (2026.01 ~ 2026.07, 진행 중)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-C51A4A)
![Hailo-8](https://img.shields.io/badge/NPU-Hailo--8%20AI%20HAT%2B%2026TOPS-brightgreen)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)

<br>

## 프로젝트 개요 (Project Overview)
**A-PAS**는 **불법 주정차로 인한 사각지대**에서 **스몸비(스마트폰 좀비) 보행자**의 횡단보도 사고를 예방하기 위한 인프라 기반 AI 경고 시스템입니다.

기존 스마트 횡단보도가 "신호 위반 감지" 수준에 머무는 반면, A-PAS는 **AI 궤적 예측으로 충돌을 사전에 감지**하여 보행자에게 시각/청각 경고를 제공합니다. 운전자의 사각지대를 **인프라의 시야가 보완**하는 것이 핵심 차별점입니다.

> **핵심 목표 (Core Goal)**
> * **Vision-only Solution**: 라이다 없이 영상과 AI만으로 정밀 충돌 예측
> * **Edge AI**: 클라우드 없이 독립적으로 작동하는 고성능 엣지 컴퓨팅 구현
> * **사전 경고**: 충돌 1~2초 전 예측 기반 선제적 경고 발생

* **개발 기간**: 2026.01.07 ~ 2026.07 (약 7개월, 진행 중)

<br>

## 🏗 시스템 아키텍처 (System Architecture)

### Hardware Configuration
| 구분 | 장비명 | 사양 | 용도 |
| --- | --- | --- | --- |
| **Main Controller** | Raspberry Pi 5 | 16GB LPDDR4X | 전체 시스템 제어 및 LSTM 추론 |
| **AI Accelerator** | Hailo-8 AI HAT+ | 26 TOPS NPU (PCIe M.2) | YOLOv8 실시간 추론 |
| **Training GPU** | RTX 5060 | Blackwell, sm_120 | 모델 학습 (PyTorch nightly + CUDA 12.8) |
| **Vision Input** | HDMI-to-CSI 캡처 보드 | 22핀 MIPI CSI | CARLA 시뮬레이션 영상 수신 |
| **Display** | 포터블 모니터 | 15인치 Micro HDMI | Risk Score 및 예측 경로 시각화 |
| **Alert System** | LED Strip + Active Buzzer | WS2812B + GPIO | 위험 상황 알림 (시각/청각) |
| **Simulation Env** | CARLA Simulator | Logitech G29 연동 | 가상 차량 접근 시나리오 검증 |
| **Storage** | MicroSD | 128GB V30 이상 | OS + 코드 + 모델 가중치 저장 |

### AI 추론 역할 분담
| 모델 | 실행 위치 | 이유 |
| --- | --- | --- |
| **YOLOv8s** (객체 탐지) | Hailo-8 NPU | 이미지 병렬 연산 → NPU 최적화 |
| **Residual LSTM** (경로 예측) | Raspberry Pi 5 CPU | 순차적 시계열 연산 → CPU 적합 |

### Technology Stack
* **AI Model**
    * **Detection**: YOLOv8s + ByteTrack (객체 탐지 및 추적)
    * **Prediction**: Residual LSTM / Attention+Residual LSTM (17개 피처 기반 궤적 예측)
    * **Inference**: ONNX Runtime (엣지 배포용)
* **Software**
    * Python 3.x, OpenCV, PyTorch, ONNX Runtime, hailo-platform
* **OS**
    * Raspberry Pi OS (Bookworm 64-bit)

### 시스템 처리 흐름
```
CARLA Simulator (PC) 또는 실제 CCTV
        ↓ HDMI / CSI
  Raspberry Pi 5
        ├── Hailo-8 NPU: YOLOv8s 추론 → 객체 좌표 추출
        │           ↓
        ├── ByteTrack: 다중 객체 추적 → Track ID 부여
        │           ↓
        ├── 17-Feature 추출: 위치, 속도, 가속도, 크기, 클래스
        │           ↓
        └── RPi CPU: Residual LSTM (ONNX) → 궤적 예측 → TTC 계산
                    ↓ 위험 감지
            LED / Buzzer 경고 출력
                    ↓
            모니터 시각화 (예측 경로 + TTC + Risk Score)
```

<br>

## 📊 모델 구조별 성능 비교

### 2초 관찰 → 1초 예측 (SEQ=60, PRED=30, 30FPS)
| 모델 구조 | ADE (px) | FDE (px) | 파라미터 | 상태 |
| --- | --- | --- | --- | --- |
| Self LSTM | 4.48 | 6.93 | ~800K | ✅ 완료 |
| **Residual LSTM** | **2.08** | **2.93** | ~800K | ✅ **배포 모델** |
| Attention LSTM (sqrt(d_k)) | 7.00 | 10.71 | ~1.3M | ✅ 완료 (실패) |
| **Attention+Residual LSTM** | **1.98** | **2.69** | ~1.3M | ✅ **최고 성능** |

> **배포 전략**: Attention+Residual이 최고 성능(ADE 1.98px)이지만, Residual 단독(ADE 2.08px)과 0.1px 차이로 RPi5 추론 효율을 고려하여 **Residual LSTM을 엣지 배포 모델로 채택**. Attention+Residual은 발표/논문용 최고 성능 모델로 보존.

### 1초 관찰 → 2초 예측 (SEQ=30, PRED=60, 30FPS)
| 모델 구조 | ADE (px) | FDE (px) | 상태 |
| --- | --- | --- | --- |
| **Residual LSTM** | **~2.5** | **~4.5** | ✅ 완료 |
| **Attention+Residual LSTM** | **~2.5** | **~3.5** | ✅ 완료 |

> **예측 구간 변경 근거**: Attention 히트맵 분석 결과 모델이 프레임 30~60(최근 1초)에 가중치를 집중하고 프레임 0~20(2초 전)은 거의 무시 → 관찰 시간을 1초로 줄이고 예측 시간을 2초로 확장하여 경고 시간 확보.
>
> **결과 분석**: 예측 구간을 2배로 늘렸음에도 ADE가 약 0.5px만 상승. 관찰 시간을 절반으로 줄여도 성능 손실이 미미한 것은 Attention 히트맵 분석의 타당성을 실험적으로 입증. 2초 전 경고가 가능해져 스몸비 보행자의 반응 시간 확보.

<br>

## 📂 폴더 구조 (Directory Structure)
```text
A-PAS/
├── 📂 raw_data/                            # 원본 영상 (학습의 시작점)
├── 📂 ai_model/
│   ├── 📂 npy_processing/                  # NPY 데이터 생성 관련 코드
│   │   ├── 🐍 video_to_csv.py              # 영상 → CSV (YOLOv8 + ByteTrack)
│   │   ├── 🐍 jpg_to_csv.py                # JPG → CSV
│   │   ├── 🐍 merge_npy.py                 # NPY 통합 및 재분할
│   │   ├── 🐍 csv_to_npy_10fps.py          # CSV → NPY (10FPS, SEQ=20, PRED=10)
│   │   ├── 🐍 csv_to_npy_30fps.py          # CSV → NPY (30FPS, SEQ=60, PRED=30)
│   │   └── 🐍 csv_to_npy_30fps_pred2s.py   # CSV → NPY (30FPS, SEQ=30, PRED=60)
│   ├── 📂 trajectory/                      # LSTM 모델 학습 코드
│   │   ├── 🐍 tr_trajectory_30fps.py       # Self LSTM (30FPS)
│   │   ├── 🐍 tr_residual_30fps.py         # Residual LSTM (30FPS)
│   │   ├── 🐍 tr_attention_30fps.py        # Attention LSTM (sqrt(d_k))
│   │   ├── 🐍 tr_attention_residual_30fps.py    # Attention+Residual (30FPS)
│   │   ├── 🐍 tr_residual_30fps_pred2s.py       # Residual (1s→2s 예측)
│   │   ├── 🐍 tr_attention_residual_30fps_pred2s.py  # Attn+Res (1s→2s 예측)
│   │   ├── 🐍 ONNX_convert_residual.py     # Residual → ONNX 변환
│   │   └── 🐍 ONNX_convert_attention.py    # Attention 계열 → ONNX 변환
│   ├── 📂 data/                            # NPY 데이터 저장
│   │   └── Training/                       # 학습/검증 NPY 파일
│   ├── 🐍 comprehensive_eval.py            # 종합 성능 평가 (CDF, Miss Rate 포함)
│   ├── 🐍 visualize_trajectory.py          # 궤적 시각화 (Grid/Overlay/클래스별)
│   ├── 🐍 compare_models.py                # 모델 간 비교 그래프
│   └── 🐍 optimize_model.py                # 모델 경량화 (동적 양자화)
├── 📂 models/                              # 학습 결과물 (모델 구조별 분리)
│   ├── 📂 Self_LSTM/                       # ✅ 기본 LSTM (완료)
│   ├── 📂 Residual/                        # ✅ Residual LSTM (완료, 배포 모델)
│   ├── 📂 Attention/                       # ✅ Attention LSTM (완료)
│   └── 📂 Attention_Residual/              # ✅ Attn+Res LSTM (완료, 최고 성능)
├── 📂 embedded/                            # 라즈베리파이용 하드웨어 제어 코드
│   ├── 🐍 camera.py                        # 카메라 영상 수집 모듈
│   └── 🐍 alert.py                         # LED / 부저 제어 모듈
├── 🐍 main.py                              # 전체 시스템 실행 (30FPS Residual LSTM)
├── 📄 requirements.txt                     # PC 환경 설정
├── 📄 requirements-pi.txt                  # 라즈베리파이 환경 설정
└── 📄 README.md
```

<br>

## 시작하기 (Getting Started)

### 1. 가상환경 생성 (공통)
```bash
python -m venv a-pas-env

# Windows
.\a-pas-env\Scripts\activate

# Mac/Linux/Raspberry Pi
source a-pas-env/bin/activate
```

### 2. 라이브러리 설치

**Case A: 학습용 PC (Windows + RTX GPU)**
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
원본 영상에서 YOLOv8 + ByteTrack으로 객체를 추적하여 CSV를 생성합니다.

```bash
cd ai_model/npy_processing
python video_to_csv.py
# 결과: ai_model/data/csv/*.csv
```

### Step 2. CSV → NPY 변환 (PC)
CSV에서 17개 피처를 추출하고 학습용 NPY 파일을 생성합니다.

```bash
cd ai_model/npy_processing

# [A] 2초 관찰 → 1초 예측 (SEQ=60, PRED=30)
python csv_to_npy_30fps.py

# [B] 1초 관찰 → 2초 예측 (SEQ=30, PRED=60) ← 최신
python csv_to_npy_30fps_pred2s.py
```

### Step 3. 모델 학습 (PC)

```bash
cd ai_model/trajectory

# [추천] Residual LSTM (배포용, 가벼움)
python tr_residual_30fps.py

# [최고 성능] Attention+Residual LSTM (발표용)
python tr_attention_residual_30fps.py

# 1초→2초 예측 버전
python tr_residual_30fps_pred2s.py
python tr_attention_residual_30fps_pred2s.py
```

### Step 4. 모델 평가 및 시각화 (PC)

```bash
cd ai_model/trajectory

# 종합 평가 (클래스별, 속도별, CDF, Miss Rate)
python comprehensive_eval.py --model residual --fps 30 --version v2_cctv_carla

# 궤적 시각화
python visualize_trajectory.py --model attention_residual --mode all

# 모델 간 비교
python compare_models.py --fps 30 --version v2_cctv_carla
```

### Step 5. ONNX 변환 (PC)

```bash
cd ai_model/trajectory
python ONNX_convert_residual.py
# 결과: models/Residual/best_model_30fps.onnx
```

### Step 6. 메인 시스템 구동 (PC 테스트 / Raspberry Pi)

```bash
python main.py
# YOLOv8s 탐지 → Residual LSTM 궤적 예측 → TTC 충돌 판정 → 경고 출력
```

<br>

## 📊 학습 데이터 규격

### 데이터 구성
| 항목 | 2s→1s (기존) | 1s→2s (최신) |
| --- | --- | --- |
| **관찰 길이** | 60프레임 (2.0초) | 30프레임 (1.0초) |
| **예측 길이** | 30프레임 (1.0초) | 60프레임 (2.0초) |
| **최소 트랙** | 90프레임 (3.0초) | 90프레임 (3.0초) |
| **FPS** | 30 | 30 |
| **데이터 소스** | 잠실 CCTV + CARLA | 동일 |
| **데이터셋** | v1(CCTV), v2(CCTV+CARLA) | 동일 |

### 17개 피처 구성
| 구분 | 피처 | 개수 |
| --- | --- | --- |
| 위치 | x, y (정규화) | 2 |
| 속도 | vx, vy (클리핑 ±2.0) | 2 |
| 가속도 | ax, ay (클리핑 ±1.0) | 2 |
| 운동 특성 | speed, sin(heading), cos(heading) | 3 |
| 크기 | w, h, area | 3 |
| 클래스 원-핫 | Person, Car, Bus, Truck, Motorcycle | 5 |

### 데이터 품질 관리
* **Global Track ID**: 파일명 + track_id로 크로스 비디오 ID 충돌 방지
* **갭 처리**: 트래킹 끊김(frame_diff > 2) 시 속도/가속도 제로화
* **물리 클리핑**: vx/vy ±2.0, ax/ay ±1.0으로 이상치 억제

<br>

## 🔬 실험 과정 및 주요 발견

### Attention 메커니즘 실험
1. **Attention 단독 + sqrt(d_k)**: ADE 7.00px — MSE loss 환경에서 smooth trajectory에 sharp attention이 불필요하여 실패
2. **Attention+Residual + sqrt(d_k)**: ADE 1.98px — Residual Connection이 gradient highway 역할을 하여 attention 학습 가능하게 만듦
3. **Attention 히트맵 분석**: 8개 샘플 전부 프레임 30~60(최근 1초)에 가중치 집중 → 1초 관찰 → 2초 예측 설계 근거

### FC 구조 실험
* 복잡한 FC (512→256→ReLU→Dropout→60): 좌표 회귀에 ReLU/Dropout이 방해 → ADE 악화
* 단순 Linear FC (512→60): 좌표 회귀에 최적 → 최고 성능 달성

### 변인 통제 원칙
* 모든 실험에서 한 번에 하나의 변수만 변경
* sqrt(d_k) + FC 동시 변경으로 원인 분리 실패한 경험 → 이후 엄격히 준수

<br>

## 📈 평가지표

| 지표 | 설명 | 비고 |
| --- | --- | --- |
| **ADE (px)** | 예측 경로 전체 프레임의 평균 오차 | 전반적 정확도 |
| **FDE (px)** | 최종 예측 시점의 오차 | TTC 계산과 직결 |
| **CDF** | 오차 분포의 누적 확률 | 롱테일/이상치 분석 |
| **Miss Rate** | FDE가 임계값 초과한 비율 | 치명적 실패 비율 |
| **TTC (s)** | 충돌까지 예상 시간 | 경고 발생 기준 |

<br>

## ⚠️ 주의사항
* 전원 어댑터는 반드시 **27W (5V 5A) 이상 PD 어댑터**를 사용하세요.
* MicroSD는 **V30 등급 이상**을 권장합니다.
* `models/` 폴더 내 `.pth`, `.onnx` 파일은 `.gitignore`에 의해 Git에서 제외됩니다.
* `data/Training/` 내 NPY 파일은 용량이 크므로 Git에서 제외됩니다.
* CARLA 시뮬레이션 사용 시 카메라 높이 5~8m, FOV 60~70으로 설정하여 보행자 탐지율을 확보하세요.

<br>

---
### 본 프로젝트는 한국공학대학교 전자공학부 졸업작품으로 진행됐습니다.