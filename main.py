"""
A-PAS Main System - Version F (CTRV/Kalman)
==============================================
main_vE_minimal_bbox.py based version replacing ONNX LSTM + MC Dropout
with CTRV(Constant Turn Rate & Velocity) trajectory prediction.

Core changes:
  - Trajectory prediction: ONNX LSTM + MC Dropout -> CTRV physics model
  - Vehicles(class 1,2,3): curved extrapolation using yaw-rate estimate
  - Pedestrians(class 0): CV(constant velocity) extrapolation
  - Time-growing std model instead of MC uncertainty

The core assess_risk logic, visualization, ROI handling, and ByteTrack flow
remain aligned with vE.
"""

import cv2
import numpy as np
import time
import math
import supervision as sv
from collections import deque, defaultdict, Counter
from ultralytics import YOLO



# ==========================================
# ⚙️ [설정]
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH      = "test_video1.mp4"
YOLO_MODEL_PATH = "best_v7.pt"
YOLO_DEVICE     = "auto"
YOLO_HALF       = True

# --- 해상도 ---
IMG_W, IMG_H = 1920, 1080

# --- 모델 설정 ---
PRED_LENGTH  = 15  # changed for 10fps obs1s pred1.5s
DT           = 0.1

# --- 영상 → 모델 샘플링 ---
VIDEO_FPS       = 30
TARGET_FPS      = 10
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # = 3

# --- 속도 스무딩 ---
SMOOTH_WINDOW = 3

# --- 물리적 클리핑 ---

# --- 충돌 판정 ---
COLLISION_DIST = 45.0  # v8: tighten collision dist
COLLISION_TTC  = 1.5  # changed for 10fps obs1s pred1.5s
TRAJECTORY_RISK_TTC = 0.8  # minimal_v1: 먼 미래 예측은 위험/주의 모두 표시하지 않음
SIGMA_SCALE    = 3.0  # v5: use actual position for clamp, reduce sigma inflation
# --- 동적 임계값 비율 (차량 bbox 대각선 기준) ---
BBOX_DANGER_RATIO  = 0.20   # danger: 차량 대각선의 20%
BBOX_MIN_DANGER    = 20.0   # 최소 danger 거리 (너무 작은 객체 보호)
MOTO_MIN_DIAG = 130.0  # 오토바이(class 2)는 최소 이 대각선으로 취급 (차량급 위험 범위)
FRONT_DANGER_LENGTH_MULT = 4.5  # 전방 danger만 옆폭은 유지하고 길이만 4.5배 연장
FRONT_DANGER_REF_DIAG = 170.0  # 일반적인 차량 bbox 대각선 기준값
_NORMAL_SIDE = max(BBOX_MIN_DANGER, FRONT_DANGER_REF_DIAG * BBOX_DANGER_RATIO)
FRONT_DANGER_SLOW_SCALE = 0.50  # 베이스라인 살짝 상향 (was 0.43)
FRONT_DANGER_SLOW_SPEED = 60.0  # 이전 80과 40의 중간값
TURN_LATERAL_BOOST = 1.4  # 회전 감지 시 front_side 40% 확장
FRONT_DANGER_NORMAL_SPEED = 160.0
FRONT_DANGER_SIZE_SCALE_MIN = 0.75  # bbox 비율 정규화: 작은 차량/원거리 과축소 방지
FRONT_DANGER_SIZE_SCALE_MAX = 1.6   # close camera 압축 완화 (was 1.25)
FRONT_DANGER_SIDE_MIN = 20.0
FRONT_DANGER_SIDE_MAX = 78.0  # close camera 여유 (was 65)
FRONT_DANGER_LENGTH_MIN = 42.0  # 이전 60과 25의 중간값
FRONT_DANGER_LENGTH_MAX = 240.0  # close camera 여유 (was 180)
FRONT_DANGER_CLASS_SCALE = {1: 1.0, 2: 0.9, 3: 1.15}
FRONT_CAUTION_SIDE_MULT = 1.4
FRONT_CAUTION_LENGTH_MULT = 1.4
FRONT_CAUTION_SIDE_MAX = 110.0   # danger 78 x 1.4 (was 75)
FRONT_CAUTION_LENGTH_MAX = 280.0  # danger 240 x ~1.17 비례 (was 200)
# --- CTRV trajectory prediction ---
CTRV_YAW_RATE_MAX = 0.45        # rad/s, conservative clamp for early turn spikes
CTRV_USE_TURN_RATE = {0: False, 1: True, 2: True, 3: True}  # pedestrians use CV
CTRV_STD_BASE = 12.0            # px, uncertainty at first step
CTRV_STD_GROWTH = 4.0           # px/sample for vehicles
CTRV_STD_MAX = 70.0             # px, late horizon cap
CTRV_PED_STD_GROWTH = 6.0       # pedestrians are jerkier
CTRV_YAW_ENTER = 0.18           # enter turn mode above this yaw rate
CTRV_YAW_EXIT = 0.08            # exit turn mode below this yaw rate
CTRV_OMEGA_EMA = 0.25           # yaw-rate smoothing, lower is smoother
CTRV_TRAJ_EMA = 0.4             # output trajectory smoothing, lower is smoother
TURN_HORIZON_SCALE = 1.15       # extend prediction horizon while turning
TURN_HORIZON_RAMP_START = 0.18  # rad/s, weak turns keep horizon near 1.0
TURN_HORIZON_RAMP_FULL = 0.70   # rad/s, strong turns use full horizon scale
OMEGA_DECAY_RATE = 0.9          # rad/s per second decay toward straight motion
CTRV_PSI_EMA = 0.4              # heading direction smoothing
CTRV_SPEED_EMA = 0.4            # speed magnitude smoothing
PED_MAX_SPEED = 150.0           # px/s, pedestrian hard cap against bbox jitter
HEADING_LONG_WINDOW = 8      # 직선 주행 시 사용할 velocity 평균 윈도우 (sample)
HEADING_COHERENCE_THRESH = 0.92  # vector coherence 임계값 (1에 가까울수록 직선)

# --- 횡단보도 ROI ---
# 사용 예시 (1920x1080 해상도 기준, 좌상단 원점):
# CROSSWALK_ROIS = [
#     [(19, 681), (525, 343), (565, 358), (123, 684)],
#     [(633, 339), (1077, 482), (1078, 442), (670, 320)],
# ]
# 빈 리스트면 ROI 필터 비활성화 → 모든 보행자 평가
CROSSWALK_ROIS_BY_VIDEO = {
    1: [
        [(222, 598), (608, 460), (634, 490), (222, 664)],
        [(688, 432), (914, 334), (952, 364), (726, 468)],
        [(1138, 370), (1194, 348), (1710, 592), (1656, 634)],
    ],
    2: [
        [(6, 670), (168, 508), (212, 522), (62, 684)],
        [(234, 438), (374, 302), (422, 310), (288, 454)],
        [(542, 278), (564, 252), (876, 314), (866, 342)],
    ],
    3: [
        [(2, 486), (2, 542), (252, 378), (212, 360)],
        [(288, 312), (422, 222), (464, 238), (338, 332)],
        [(586, 214), (616, 192), (928, 250), (928, 274)],
    ],
    4: [
        [(110, 644), (292, 424), (352, 436), (164, 672)],
        [(346, 356), (434, 252), (490, 260), (392, 370)],
        [(582, 220), (602, 192), (886, 226), (882, 264)],
    ],
    5: [
        [(26, 680), (302, 500), (338, 520), (110, 686)],
        [(378, 450), (516, 354), (548, 368), (412, 472)],
        [(670, 360), (700, 336), (1024, 430), (1024, 462)],
    ],
}


def select_crosswalk_rois(video_path):
    """VIDEO_PATH의 test_video 번호에 맞는 ROI를 선택한다."""
    video_name = str(video_path).lower()
    for video_idx, rois in CROSSWALK_ROIS_BY_VIDEO.items():
        if f"test_video{video_idx}" in video_name:
            return rois
    return []


CROSSWALK_ROIS = select_crosswalk_rois(VIDEO_PATH)
ROI_MARGIN = 20

# --- YOLO 설정 ---
YOLO_IMGSZ   = 1280
USE_YOLO_EVERY_FRAME = False


# --- 예측 시각화 안정화 ---
# --- 런타임 캡 ---
MAX_TTC_PAIRS   = 12  # ROI 통과 pair 모두 커버 (성능 영향 ~0.3ms/sample, 무시 수준)
TTC_PAIR_CURRENT_DIST = 500.0  # v3: stopped vehicle EMA bypass
STOPPED_VEL_THRESH = 40.0  # v6: fix stopped detection threshold

# --- 시각화 설정 ---
DISPLAY_SCALE    = 0.75
PRED_DRAW_STRIDE = 3
MAX_VIS_TRACKS   = 24

# --- 통계 ---
STATS_WARMUP_FRAMES = 3
DANGER_VISUAL_HOLD = 10  # samples (= 1초 at 10 FPS sample rate). 빨강 한 번 뜨면 유지.

# --- Pi 배포용 플래그 ---
ENABLE_TRACK_MERGE = False  # True면 가까운 트랙 ID 머지 (보행자 누락 위험)
DEBUG_DRAW_DANGER_ZONE = False  # True면 danger/caution zone 시각화

# --- 클래스 매핑 ---
VEHICLE_CLASSES = {1, 2, 3}
CONF_THRESHOLD  = {0: 0.5, 1: 0.6, 2: 0.45, 3: 0.35}  # 차량/오토바이 false positive 감소 (나뭇잎/그림자 거름)
CLASS_NAMES     = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}
CLASS_COLORS    = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}
COCO_TO_APAS_CLASS = {
    0: 0,  # person
    2: 1,  # car
    3: 2,  # motorcycle
    5: 3,  # bus -> large vehicle
    7: 3,  # truck -> large vehicle
}

COLORS = {
    "safe":    (0, 220, 0),
    "caution": (0, 140, 255),   # BGR: clear orange / RGB(255,140,0)
    "danger":  (0, 0, 255),
    "text":    (230, 230, 230),
}


# ==========================================
# 📊 [Latency Profiler]
# ==========================================
STAGES = [
    "yolo_inference",
    "detection_filter",
    "tracker",
    "trajectory",
    "ttc",
    "visualization",
    "imshow_waitkey",
]


class StageTimer:
    def __init__(self):
        self._last = time.perf_counter()
        self.times = {name: 0.0 for name in STAGES}

    def mark(self, stage):
        now = time.perf_counter()
        self.times[stage] += (now - self._last) * 1000.0
        self._last = now


class LatencyProfiler:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)

    def print_report(self, skip_first=STATS_WARMUP_FRAMES):
        recs = [r for r in self.records if r["frame"] > skip_first]
        if not recs:
            recs = self.records
        if not recs:
            return

        print(f"\n{'='*72}")
        print("Latency breakdown by frame type (ms)")
        print(f"{'='*72}")

        for frame_type in ("sample", "reuse"):
            subset = [r for r in recs if r["type"] == frame_type]
            if not subset:
                continue
            print(f"\n[{frame_type.upper()}] frames={len(subset)}")
            for stage in ["total", *STAGES]:
                values = [r["total_ms"] if stage == "total"
                          else r["stages"].get(stage, 0.0) for r in subset]
                arr = np.array(values, dtype=np.float32)
                print(f"  {stage:<22} mean={arr.mean():7.2f}  "
                      f"p50={np.percentile(arr,50):7.2f}  "
                      f"p90={np.percentile(arr,90):7.2f}  "
                      f"max={arr.max():7.2f}")

        slow = sorted(recs, key=lambda r: r["total_ms"], reverse=True)[:5]
        print("\nTop slow frames:")
        for r in slow:
            parts = ", ".join(f"{k}={v:.1f}" for k, v in r["stages"].items() if v > 0.05)
            print(f"  frame={r['frame']:>5} type={r['type']:<6} "
                  f"total={r['total_ms']:.2f}ms "
                  f"tracks={r['tracks']} preds={r['preds']} "
                  f"traj={r.get('traj_count', 0)} | {parts}")
        print(f"{'='*72}")


# ==========================================
# 📊 [트래킹 상태]
# ==========================================
prev_boxes           = {}
velocity_history     = {}
heading_vel_history  = {}  # CTRV yaw-rate estimate history.
track_classes        = {}
class_votes          = defaultdict(lambda: deque(maxlen=10))
prev_frames          = {}
prev_omega           = {}  # track_id -> smoothed yaw rate (rad/s)
prev_psi             = {}  # track_id -> smoothed heading (rad)
prev_speed           = {}  # track_id -> smoothed speed (px/s)
prev_trajectory      = {}  # track_id -> smoothed trajectory mean
is_turning_state     = {}  # track_id -> hysteresis turn state
danger_visual_cache  = {}  # (ped_id, veh_id) -> samples remaining


# ==========================================
# 🔧 [피처 추출]
# ==========================================
# CTRV trajectory helpers
# ==========================================
def update_track_velocity(box, prev_box, track_id, class_id):
    """Update velocity histories needed by CTRV prediction."""
    if track_id not in velocity_history:
        velocity_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
    if track_id not in heading_vel_history:
        heading_vel_history[track_id] = deque(maxlen=HEADING_LONG_WINDOW)

    if prev_box is not None:
        raw_vx = (box[0] - prev_box[0]) / DT
        raw_vy = (box[1] - prev_box[1]) / DT
        velocity_history[track_id].append([raw_vx, raw_vy])
        heading_vel_history[track_id].append([raw_vx, raw_vy])

    track_classes[track_id] = class_id


def angle_blend(new_angle, old_angle, alpha):
    """EMA blend for angles with wrap-around handling."""
    delta = new_angle - old_angle
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return old_angle + alpha * delta


def estimate_yaw_rate(track_id):
    """Estimate yaw rate from older vs recent velocity headings."""
    hvh = heading_vel_history.get(track_id)
    if not hvh or len(hvh) < 4:
        return 0.0

    vels = list(hvh)
    half = len(vels) // 2
    older = vels[:half]
    recent = vels[half:]
    psi_older = math.atan2(
        sum(v[1] for v in older) / len(older),
        sum(v[0] for v in older) / len(older),
    )
    psi_recent = math.atan2(
        sum(v[1] for v in recent) / len(recent),
        sum(v[0] for v in recent) / len(recent),
    )

    delta = psi_recent - psi_older
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi

    omega = delta / (max(1, half) * DT)
    return max(-CTRV_YAW_RATE_MAX, min(CTRV_YAW_RATE_MAX, omega))


_roi_mask = None
def _build_roi_mask():
    global _roi_mask
    if not CROSSWALK_ROIS:
        _roi_mask = None
        return
    _roi_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for polygon in CROSSWALK_ROIS:
        if len(polygon) < 3:
            continue
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(_roi_mask, [pts], 255)
    if ROI_MARGIN > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_MARGIN*2+1, ROI_MARGIN*2+1))
        _roi_mask = cv2.dilate(_roi_mask, kernel)


def is_point_in_roi_fast(x, y):
    if _roi_mask is None:
        return True
    ix, iy = int(round(x)), int(round(y))
    if 0 <= ix < IMG_W and 0 <= iy < IMG_H:
        return _roi_mask[iy, ix] > 0
    return False


def _veh_diag(box):
    """차량 bbox의 대각선 길이."""
    return math.hypot(float(box[2]), float(box[3]))


def _front_danger_shape(v_speed, veh_diag, class_id):
    """전방 danger 사각형 크기: 차량 bbox 대각선 비율로 정규화."""
    if class_id == 2:
        veh_diag = max(veh_diag, MOTO_MIN_DIAG)

    class_scale = FRONT_DANGER_CLASS_SCALE.get(class_id, 1.0)
    size_scale = (veh_diag / max(FRONT_DANGER_REF_DIAG, 1e-6)) * class_scale
    size_scale = max(FRONT_DANGER_SIZE_SCALE_MIN, min(FRONT_DANGER_SIZE_SCALE_MAX, size_scale))

    side = _NORMAL_SIDE * size_scale
    side = max(FRONT_DANGER_SIDE_MIN, min(FRONT_DANGER_SIDE_MAX, side))

    if v_speed <= FRONT_DANGER_SLOW_SPEED:
        speed_scale = FRONT_DANGER_SLOW_SCALE
    elif v_speed >= FRONT_DANGER_NORMAL_SPEED:
        speed_scale = 1.0
    else:
        ratio = (v_speed - FRONT_DANGER_SLOW_SPEED) / max(1e-6, FRONT_DANGER_NORMAL_SPEED - FRONT_DANGER_SLOW_SPEED)
        speed_scale = FRONT_DANGER_SLOW_SCALE + ratio * (1.0 - FRONT_DANGER_SLOW_SCALE)

    front_length = side * FRONT_DANGER_LENGTH_MULT * speed_scale
    front_length = max(FRONT_DANGER_LENGTH_MIN, min(FRONT_DANGER_LENGTH_MAX, front_length))
    return side, front_length


def _front_danger_hit(v_box, p_box, hx, hy, side_radius, front_length, ped_radius):
    """전방 사각형 안 + ped_radius 마진 판정.

    Args:
        v_box, p_box: 차량/보행자 bbox (cx, cy, w, h)
        hx, hy: 차량 heading 단위 벡터
        side_radius: 사각형 옆폭
        front_length: 사각형 전방 길이
        ped_radius: 보행자 bbox 반경 (가장자리 마진용)

    Returns: True if 보행자 bbox가 사각형에 닿음 (가장자리 포함).
    """
    if side_radius <= 0.0 or front_length <= 0.0:
        return False

    to_ped_x = float(p_box[0] - v_box[0])
    to_ped_y = float(p_box[1] - v_box[1])
    forward = to_ped_x * hx + to_ped_y * hy

    # forward 범위 + ped_radius 마진
    if forward < -ped_radius or forward > front_length + ped_radius:
        return False

    # lateral 거리 + ped_radius 마진
    lateral_x = to_ped_x - forward * hx
    lateral_y = to_ped_y - forward * hy
    lateral = math.hypot(lateral_x, lateral_y)
    return lateral <= side_radius + ped_radius


def get_trajectory(track_id, current_box):
    """Future trajectory for track_id using CV/CTRV prediction."""
    if current_box is None:
        return None, None

    vel_hist = velocity_history.get(track_id)
    if not vel_hist or len(vel_hist) == 0:
        mean = np.tile(
            np.array([[current_box[0], current_box[1]]], dtype=np.float32),
            (PRED_LENGTH, 1),
        )
        std = np.array(
            [min(CTRV_STD_BASE + t * CTRV_STD_GROWTH, CTRV_STD_MAX)
             for t in range(PRED_LENGTH)],
            dtype=np.float32,
        )
        return mean, std

    n = len(vel_hist)
    vx = sum(v[0] for v in vel_hist) / n
    vy = sum(v[1] for v in vel_hist) / n
    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        mean = np.tile(
            np.array([[current_box[0], current_box[1]]], dtype=np.float32),
            (PRED_LENGTH, 1),
        )
        std = np.array(
            [min(CTRV_STD_BASE + t * CTRV_STD_GROWTH, CTRV_STD_MAX)
             for t in range(PRED_LENGTH)],
            dtype=np.float32,
        )
        return mean, std

    class_id = track_classes.get(track_id)

    speed_raw = speed
    if class_id == 0:
        speed_raw = min(speed_raw, PED_MAX_SPEED)
    speed = (CTRV_SPEED_EMA * speed_raw
             + (1.0 - CTRV_SPEED_EMA) * prev_speed.get(track_id, speed_raw))
    prev_speed[track_id] = speed

    psi_raw = math.atan2(vy, vx)
    psi = angle_blend(psi_raw, prev_psi.get(track_id, psi_raw), CTRV_PSI_EMA)
    prev_psi[track_id] = psi

    if CTRV_USE_TURN_RATE.get(class_id, False):
        omega_raw = estimate_yaw_rate(track_id)
        omega_smooth = (CTRV_OMEGA_EMA * omega_raw
                        + (1.0 - CTRV_OMEGA_EMA) * prev_omega.get(track_id, 0.0))
        prev_omega[track_id] = omega_smooth
    else:
        omega_smooth = 0.0
        prev_omega[track_id] = 0.0

    was_turning = is_turning_state.get(track_id, False)
    if was_turning:
        is_turning = abs(omega_smooth) >= CTRV_YAW_EXIT
    else:
        is_turning = abs(omega_smooth) >= CTRV_YAW_ENTER
    is_turning_state[track_id] = is_turning

    omega = omega_smooth if is_turning else 0.0

    std_growth = CTRV_PED_STD_GROWTH if class_id == 0 else CTRV_STD_GROWTH
    mean = np.zeros((PRED_LENGTH, 2), dtype=np.float32)
    std = np.zeros(PRED_LENGTH, dtype=np.float32)
    x0, y0 = float(current_box[0]), float(current_box[1])

    if omega != 0.0 and class_id != 0:
        omega_abs = abs(omega)
        ramp = (omega_abs - TURN_HORIZON_RAMP_START) / max(
            1e-6,
            TURN_HORIZON_RAMP_FULL - TURN_HORIZON_RAMP_START,
        )
        ramp = max(0.0, min(1.0, ramp))
        horizon_scale = 1.0 + (TURN_HORIZON_SCALE - 1.0) * ramp
        decay_rate = OMEGA_DECAY_RATE
    else:
        horizon_scale = 1.0
        decay_rate = 0.0

    dt_step = DT * horizon_scale
    psi_t = psi
    x_t, y_t = x0, y0

    for t in range(PRED_LENGTH):
        elapsed = t * dt_step
        if omega == 0.0:
            omega_t = 0.0
        else:
            omega_t = omega * math.exp(-decay_rate * elapsed)
            if abs(omega_t) < 1e-3:
                omega_t = 0.0

        if omega_t == 0.0:
            x_t += speed * math.cos(psi_t) * dt_step
            y_t += speed * math.sin(psi_t) * dt_step
        else:
            radius = speed / omega_t
            x_t += radius * (math.sin(psi_t + omega_t * dt_step) - math.sin(psi_t))
            y_t -= radius * (math.cos(psi_t + omega_t * dt_step) - math.cos(psi_t))
            psi_t += omega_t * dt_step

        mean[t, 0] = x_t
        mean[t, 1] = y_t
        std[t] = min(CTRV_STD_BASE + t * std_growth * horizon_scale, CTRV_STD_MAX)

    prev_mean = prev_trajectory.get(track_id)
    if prev_mean is not None and prev_mean.shape == mean.shape:
        mean = (CTRV_TRAJ_EMA * mean + (1.0 - CTRV_TRAJ_EMA) * prev_mean).astype(np.float32)
    prev_trajectory[track_id] = mean.copy()

    return mean, std


def assess_risk(p_id, p_box, v_id, v_box):
    """?? (???, ??) pair ?? ?? ? ?? ?? ??.

    ?? ??: ??? ???? ?? ?? ?? ????,
              ?? ?? ??? ????? ??.

    ?? ??:
      1. ROI filter (???? ???? ?/??)
      2. ?? ?? (?? ??? ?? ??)
      3. forward/lateral ?? (???? ?? ?? ?? ?? + ?? ???)
      4. get_trajectory()? ?? ?? + uncertainty
      5. ?? ?? ?? < safety_margin ? danger / caution

    Returns:
        dict: {ped_id, veh_id, min_dist, ttc_sec, danger, caution,
               point, std_radius, source} or None
    """
    # === 1. ROI filter ===
    if not is_point_in_roi_fast(p_box[0], p_box[1]):
        return None

    # === 2. velocity + heading ===
    veh_vel_hist = velocity_history.get(v_id)
    if not veh_vel_hist or len(veh_vel_hist) == 0:
        return None
    n = len(veh_vel_hist)
    vx = sum(v[0] for v in veh_vel_hist) / n
    vy = sum(v[1] for v in veh_vel_hist) / n
    v_speed = math.hypot(vx, vy)
    if v_speed < STOPPED_VEL_THRESH:
        return None

    # === 2.5. 회전 감지 (heading_vel_history maxlen=5 활용) ===
    is_turning = False
    turn_hx, turn_hy = 0.0, 0.0
    heading_hist = heading_vel_history.get(v_id)
    if heading_hist and len(heading_hist) >= 4:
        hist_list = list(heading_hist)
        rec = hist_list[-2:]
        old = hist_list[:-2]
        rx = (rec[0][0] + rec[1][0]) * 0.5
        ry = (rec[0][1] + rec[1][1]) * 0.5
        ox = sum(v[0] for v in old) / len(old)
        oy = sum(v[1] for v in old) / len(old)
        rs = math.hypot(rx, ry)
        os_ = math.hypot(ox, oy)
        if rs > STOPPED_VEL_THRESH and os_ > STOPPED_VEL_THRESH:
            cos_change = (rx * ox + ry * oy) / (rs * os_)
            if cos_change < HEADING_COHERENCE_THRESH:
                is_turning = True
                turn_hx = rx / rs
                turn_hy = ry / rs

    # === 3. forward 계산 (회전 시 recent heading 사용) ===
    if is_turning:
        hx, hy = turn_hx, turn_hy
    else:
        hx, hy = vx / v_speed, vy / v_speed
    dx = p_box[0] - v_box[0]
    dy = p_box[1] - v_box[1]
    forward = dx * hx + dy * hy

    if forward < 0.0:
        return None

    veh_d = _veh_diag(v_box)

    # === 4. Zone check (?? ?? ??? + ped_radius ??) ===
    front_side, front_length = _front_danger_shape(v_speed, veh_d, track_classes.get(v_id))
    if is_turning:
        front_side = min(front_side * TURN_LATERAL_BOOST, FRONT_DANGER_SIDE_MAX)

    ped_radius = max(float(p_box[2]), float(p_box[3])) * 0.5
    if _front_danger_hit(v_box, p_box, hx, hy, front_side, front_length, ped_radius):
        ttc_est = forward / max(v_speed, 1.0)
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(max(0.0, forward - front_length)) if forward > front_length else 0.0,
            "ttc_sec": float(min(ttc_est, COLLISION_TTC)),
            "danger": True,
            "caution": True,
            "point": np.array(
                [(p_box[0] + v_box[0]) * 0.5, (p_box[1] + v_box[1]) * 0.5],
                dtype=np.float32
            ),
            "std_radius": 10.0,
            "source": "zone",
        }

    caution_side = min(front_side * FRONT_CAUTION_SIDE_MULT, FRONT_CAUTION_SIDE_MAX)
    caution_length = min(front_length * FRONT_CAUTION_LENGTH_MULT, FRONT_CAUTION_LENGTH_MAX)
    if _front_danger_hit(v_box, p_box, hx, hy, caution_side, caution_length, ped_radius):
        ttc_est = forward / max(v_speed, 1.0)
        is_imminent = forward <= ped_radius + 30.0
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(max(0.0, forward - caution_length)) if forward > caution_length else 0.0,
            "ttc_sec": float(min(ttc_est, COLLISION_TTC)),
            "danger": is_imminent,
            "caution": True,
            "point": np.array(
                [(p_box[0] + v_box[0]) * 0.5, (p_box[1] + v_box[1]) * 0.5],
                dtype=np.float32
            ),
            "std_radius": 10.0,
            "source": "zone_imminent" if is_imminent else "zone_caution",
        }

    # === 5. ?? ?? ===
    p_traj, p_std = get_trajectory(p_id, p_box)
    v_traj, v_std = get_trajectory(v_id, v_box)
    if p_traj is None or v_traj is None:
        return None

    # === 5. ?? ?? ?? + uncertainty ===
    min_dist = float('inf')
    min_t = 0
    danger_t = -1
    danger_threshold_step = int(COLLISION_TTC / DT)

    for t in range(PRED_LENGTH):
        dist = math.hypot(
            float(p_traj[t][0] - v_traj[t][0]),
            float(p_traj[t][1] - v_traj[t][1])
        )
        margin = COLLISION_DIST + float(p_std[t]) + float(v_std[t])

        if dist < margin and danger_t < 0 and t < danger_threshold_step:
            danger_t = t

        if dist < min_dist:
            min_dist = dist
            min_t = t

    # === 6. ?? ?? ===
    if danger_t >= 0 and danger_t * DT <= TRAJECTORY_RISK_TTC:
        point = np.array(
            ((p_traj[danger_t][0] + v_traj[danger_t][0]) * 0.5,
             (p_traj[danger_t][1] + v_traj[danger_t][1]) * 0.5),
            dtype=np.float32
        )
        std_r = (float(p_std[danger_t]) + float(v_std[danger_t])) * 0.5
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(min_dist),
            "ttc_sec": float(danger_t * DT),
            "danger": True,
            "caution": True,
            "point": point,
            "std_radius": float(std_r),
            "source": "minimal_v1",
        }

    if min_dist < COLLISION_DIST * 1.5 and min_t * DT <= TRAJECTORY_RISK_TTC:
        point = np.array(
            ((p_traj[min_t][0] + v_traj[min_t][0]) * 0.5,
             (p_traj[min_t][1] + v_traj[min_t][1]) * 0.5),
            dtype=np.float32
        )
        std_r = (float(p_std[min_t]) + float(v_std[min_t])) * 0.5
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(min_dist),
            "ttc_sec": float(min_t * DT),
            "danger": False,
            "caution": True,
            "point": point,
            "std_radius": float(std_r),
            "source": "minimal_v1",
        }

    return None


def scaled_pt(pt):
    return int(pt[0] * DISPLAY_SCALE), int(pt[1] * DISPLAY_SCALE)


def draw_bbox(frame, box, track_id, class_id, conf, level="safe"):
    x, y, w, h = box
    x1, y1 = scaled_pt((x - w/2, y - h/2))
    x2, y2 = scaled_pt((x + w/2, y + h/2))
    color = COLORS[level]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} [{CLASS_NAMES.get(class_id, '?')}] {conf:.2f}"
    cv2.putText(frame, label, (x1, max(14, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_prediction(frame, track_id, level="safe"):
    box = prev_boxes.get(track_id)
    if box is None:
        return
    mean, _ = get_trajectory(track_id, box)
    if mean is None:
        return

    pts = [scaled_pt((float(mean[t, 0]), float(mean[t, 1]))) for t in range(0, PRED_LENGTH, 2)]
    if len(pts) >= 2:
        if level == "safe":
            color = (180, 180, 180)
            thickness = 1
        else:
            color = COLORS[level]
            thickness = 2 if level == "danger" else 1
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False,
                      color, thickness, cv2.LINE_AA)

def draw_risk_marker(frame, result):
    color = COLORS["danger"] if result["danger"] else COLORS["caution"]
    center = scaled_pt(result["point"])
    size = 7 if result["danger"] else 5

    cv2.line(frame, (center[0]-size, center[1]), (center[0]+size, center[1]), color, 2, cv2.LINE_AA)
    cv2.line(frame, (center[0], center[1]-size), (center[0], center[1]+size), color, 2, cv2.LINE_AA)

    radius = int(np.clip(result.get("std_radius", 10) * DISPLAY_SCALE, 8, 65))
    cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)

    src = result.get("source", "-")
    cv2.putText(frame, f"TTC {result['ttc_sec']:.1f}s {src}",
                (center[0]+8, center[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_alert(frame, ttc_sec, min_dist):
    w_scaled = int(500 * DISPLAY_SCALE)
    h_scaled = int(90 * DISPLAY_SCALE)
    cv2.rectangle(frame, (0, 0), (w_scaled, h_scaled), COLORS["danger"], -1)
    cv2.putText(frame, "!! COLLISION WARNING !!",
                (int(10*DISPLAY_SCALE), int(35*DISPLAY_SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85*DISPLAY_SCALE, (255,255,255), 2)
    cv2.putText(frame, f"TTC: {ttc_sec:.1f}s | Dist: {min_dist:.0f}px",
                (int(10*DISPLAY_SCALE), int(70*DISPLAY_SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55*DISPLAY_SCALE, (255,255,255), 1)


def draw_status(frame, record, n_tracks, n_preds):
    fps = 1000.0 / record["total_ms"] if record["total_ms"] > 0 else 0
    text = (f"{record['type']} FPS:{fps:.1f} Lat:{record['total_ms']:.1f}ms "
            f"Traj Tracks:{n_preds} Active:{n_tracks}")
    y = frame.shape[0] - 12
    cv2.rectangle(frame, (4, y-20), (int(700*DISPLAY_SCALE), y+6), (0,0,0), -1)
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5*DISPLAY_SCALE, COLORS["text"], 1, cv2.LINE_AA)


def draw_minimal_danger_zone(frame, v_box, v_id):
    """assess_risk와 같은 zone 계산식으로 danger/caution 영역을 표시."""
    if not DEBUG_DRAW_DANGER_ZONE:
        return

    veh_vel_hist = velocity_history.get(v_id)
    if not veh_vel_hist:
        return
    n = len(veh_vel_hist)
    vx = sum(v[0] for v in veh_vel_hist) / n
    vy = sum(v[1] for v in veh_vel_hist) / n
    v_speed = math.hypot(vx, vy)
    if v_speed < STOPPED_VEL_THRESH:
        return

    hx, hy = vx / v_speed, vy / v_speed
    veh_d = _veh_diag(v_box)
    front_side, front_length = _front_danger_shape(v_speed, veh_d, track_classes.get(v_id))
    if front_side <= 0.0 or front_length <= 0.0:
        return

    caution_side = min(front_side * FRONT_CAUTION_SIDE_MULT, FRONT_CAUTION_SIDE_MAX)
    caution_length = min(front_length * FRONT_CAUTION_LENGTH_MULT, FRONT_CAUTION_LENGTH_MAX)

    cx, cy = scaled_pt(v_box[:2])
    s = DISPLAY_SCALE
    px, py = -hy, hx

    def zone_poly(side, length):
        return np.array([
            (int(cx + px * side * s), int(cy + py * side * s)),
            (int(cx - px * side * s), int(cy - py * side * s)),
            (int(cx + hx * length * s - px * side * s), int(cy + hy * length * s - py * side * s)),
            (int(cx + hx * length * s + px * side * s), int(cy + hy * length * s + py * side * s)),
        ], dtype=np.int32)

    cv2.polylines(frame, [zone_poly(caution_side, caution_length)], True, COLORS["caution"], 1, cv2.LINE_AA)
    cv2.polylines(frame, [zone_poly(front_side, front_length)], True, COLORS["danger"], 1, cv2.LINE_AA)


def draw_roi(frame):
    """횡단보도 ROI가 설정되어 있으면 화면에 다각형 외곽선으로 표시."""
    if not CROSSWALK_ROIS:
        return
    for polygon in CROSSWALK_ROIS:
        if len(polygon) < 3:
            continue
        pts = np.array(
            [(int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE)) for x, y in polygon],
            dtype=np.int32
        )
        cv2.polylines(frame, [pts], True, (80, 255, 120), 2, cv2.LINE_AA)


def make_visual_frame(frame):
    if DISPLAY_SCALE == 1.0:
        return frame
    w = int(frame.shape[1] * DISPLAY_SCALE)
    h = int(frame.shape[0] * DISPLAY_SCALE)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


# ==========================================
# 🧹 [트랙 정리]
# ==========================================
def cleanup_stale_tracks(active_ids, frame_count, max_lost=60):
    stale = [
        tid for tid in prev_frames
        if tid not in active_ids
        and (frame_count - prev_frames.get(tid, 0)) > max_lost
    ]
    for tid in stale:
        for d in [prev_boxes, velocity_history, heading_vel_history,
                  track_classes, prev_frames, prev_omega, prev_psi, prev_speed,
                  prev_trajectory, is_turning_state, class_votes]:
            d.pop(tid, None)


def merge_duplicate_tracks(merge_dist=100):
    ids = list(prev_boxes.keys())
    to_remove = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] in prev_boxes and ids[j] in prev_boxes:
                dist = math.hypot(
                    float(prev_boxes[ids[i]][0] - prev_boxes[ids[j]][0]),
                    float(prev_boxes[ids[i]][1] - prev_boxes[ids[j]][1])
                )
                if dist < merge_dist:
                    to_remove.add(max(ids[i], ids[j]))
    for tid in to_remove:
        for d in [prev_boxes, velocity_history, heading_vel_history,
                  track_classes, prev_frames, prev_omega, prev_psi, prev_speed,
                  prev_trajectory, is_turning_state, class_votes]:
            d.pop(tid, None)


def main():
    yolo_device = "cuda" if YOLO_DEVICE == "auto" else YOLO_DEVICE

    print(f"\n{'='*60}")
    print("A-PAS Version C — Sampled ByteTrack Hybrid")
    print(f"{'='*60}")
    print("  YOLO backend    : Ultralytics (Windows test)")
    print(f"  YOLO device     : {yolo_device}, half={YOLO_HALF}")
    print(f"  YOLO imgsz      : {YOLO_IMGSZ}")
    print(f"  PT path         : {YOLO_MODEL_PATH}")
    print("  Trajectory      : CTRV + Kalman (no LSTM)")
    print(f"  Yaw rate clamp  : +/-{CTRV_YAW_RATE_MAX:.2f} rad/s")
    print("  Pedestrian      : CV (constant velocity)")
    print("  Vehicle         : CTRV (constant turn rate)")
    mode_text = "every frame" if USE_YOLO_EVERY_FRAME else f"sample frames only (every {SAMPLE_INTERVAL} frames)"
    print("  Tracker         : supervision.ByteTrack (same as Pi deploy)")
    print(f"  Tracking        : supervision.ByteTrack, YOLO {mode_text}")
    print(f"  Display scale   : {DISPLAY_SCALE}")
    print(f"  Display rate    : sample frames only (~{TARGET_FPS} FPS)")
    print(f"  Runtime caps    : ttc_pairs={MAX_TTC_PAIRS}")
    print(f"  Warmup skip     : {STATS_WARMUP_FRAMES} frames")
    print(f"{'='*60}\n")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    try:
        yolo_model.fuse()
    except Exception:
        pass
    bytetracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=15,
        minimum_matching_threshold=0.8,
        frame_rate=10,
    )
    _build_roi_mask()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    profiler = LatencyProfiler()
    frame_count = 0
    yolo_runs = 0
    sample_index = 0

    # 비샘플 프레임에서 시각화/경고를 재사용할 캐시
    last_tracked = []  # [(box, track_id, voted_class, conf), ...]
    last_draw_data = last_tracked
    risk_results = []
    last_ttc_sec = PRED_LENGTH * DT
    last_min_dist = float('inf')

    try:
        while cap.isOpened():
            frame_start = time.perf_counter()
            timer = StageTimer()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            is_sample = (frame_count % SAMPLE_INTERVAL == 0)
            trajectory_count = 0
            active_ids = set()

            collision_risk = False
            min_dist_global = last_min_dist
            ttc_sec_global = last_ttc_sec

            # ① YOLO + ByteTrack (Version C: 샘플 프레임만, 필요 시 fallback)
            run_yolo_this_frame = USE_YOLO_EVERY_FRAME or is_sample
            boxes_arr = []
            track_ids_arr = []
            class_ids_arr = []
            confs_arr = []
            if run_yolo_this_frame:
                yolo_runs += 1
                results = yolo_model(
                    frame,
                    imgsz=YOLO_IMGSZ,
                    classes=[0, 1, 2, 3],
                    conf=0.1,
                    iou=0.3,
                    verbose=False,
                    device=yolo_device,
                    half=YOLO_HALF if yolo_device != "cpu" else False,
                )
                sv_dets = sv.Detections.from_ultralytics(results[0])
                tracked = bytetracker.update_with_detections(sv_dets)
                if len(tracked) > 0 and tracked.tracker_id is not None:
                    for i in range(len(tracked)):
                        x1, y1, x2, y2 = tracked.xyxy[i]
                        cx_, cy_ = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        w_, h_ = x2 - x1, y2 - y1
                        boxes_arr.append(np.array([cx_, cy_, w_, h_], dtype=np.float32))
                        track_ids_arr.append(int(tracked.tracker_id[i]))
                        class_ids_arr.append(int(tracked.class_id[i]))
                        confs_arr.append(float(tracked.confidence[i]))
            timer.mark("yolo_inference")

            # 필터링
            filtered_items = []
            for box, track_id, class_id, conf in zip(boxes_arr, track_ids_arr, class_ids_arr, confs_arr):
                active_ids.add(track_id)
                x, y, w, h = box
                if class_id == 0 and w * h < 400:
                    continue
                if conf < CONF_THRESHOLD.get(int(class_id), 0.25):
                    continue
                class_votes[track_id].append(int(class_id))
                voted_class = Counter(class_votes[track_id]).most_common(1)[0][0]
                filtered_items.append((box.astype(np.float32), track_id, voted_class, float(conf)))

            if run_yolo_this_frame and filtered_items:
                last_tracked = filtered_items
                last_draw_data = last_tracked
            elif not run_yolo_this_frame:
                active_ids.update(tid for _, tid, _, _ in last_tracked)

            timer.mark("detection_filter")
            timer.mark("tracker")  # ByteTrack은 YOLO 내부에서 처리됨

            # ? CTRV trajectory state update (sample frames only)
            if is_sample:
                sample_index += 1

                for box, track_id, class_id, conf in filtered_items:
                    if track_id in prev_frames:
                        frame_diff = frame_count - prev_frames[track_id]
                        if frame_diff > 15:
                            if track_id in velocity_history:
                                velocity_history[track_id].clear()
                            if track_id in heading_vel_history:
                                heading_vel_history[track_id].clear()
                            prev_omega.pop(track_id, None)
                            prev_psi.pop(track_id, None)
                            prev_speed.pop(track_id, None)
                            prev_trajectory.pop(track_id, None)
                            is_turning_state.pop(track_id, None)

                    prev_frames[track_id] = frame_count
                    prev_box = prev_boxes.get(track_id)
                    update_track_velocity(box, prev_box, track_id, class_id)
                    prev_boxes[track_id] = box
                    trajectory_count += 1

                if ENABLE_TRACK_MERGE:
                    merge_duplicate_tracks()

            timer.mark("trajectory")


            # ? TTC (?? ????)
            if is_sample:
                risk_results = []

                # === 1. (???, ??) pair ?? ?? + ?? ?? ===
                ped_items = [(b, t) for b, t, c, _ in filtered_items if c == 0]
                veh_items = [(b, t) for b, t, c, _ in filtered_items if c in VEHICLE_CLASSES]

                pair_candidates = []
                for p_box, p_id in ped_items:
                    if not is_point_in_roi_fast(float(p_box[0]), float(p_box[1])):
                        continue
                    for v_box, v_id in veh_items:
                        dist_now = math.hypot(
                            float(p_box[0] - v_box[0]),
                            float(p_box[1] - v_box[1])
                        )
                        if dist_now <= TTC_PAIR_CURRENT_DIST:
                            pair_candidates.append((dist_now, p_id, p_box, v_id, v_box))

                pair_candidates.sort(key=lambda x: x[0])
                pair_candidates = pair_candidates[:MAX_TTC_PAIRS]

                # === 2. assess_risk? ?? ?? ===
                for _, p_id, p_box, v_id, v_box in pair_candidates:
                    result = assess_risk(p_id, p_box, v_id, v_box)
                    if result is not None:
                        risk_results.append(result)


                # === 3. collision_risk ?? (draw_alert?) ===
                for r in risk_results:
                    if r["danger"]:
                        collision_risk = True
                        if r["ttc_sec"] < ttc_sec_global:
                            ttc_sec_global = r["ttc_sec"]
                        if r["min_dist"] < min_dist_global:
                            min_dist_global = r["min_dist"]

                risk_results.sort(key=lambda r: (not r["danger"], r["ttc_sec"]))
                last_ttc_sec = ttc_sec_global
                last_min_dist = min_dist_global

            timer.mark("ttc")

            # ④ 시각화 (sample frame에서만 화면 갱신 — reuse는 redundant 회피)
            key = 0
            display_this_frame = is_sample or USE_YOLO_EVERY_FRAME
            if display_this_frame:
                vis = make_visual_frame(frame)
                draw_roi(vis)

                risk_tids = set()
                danger_tids = set()
                for r in risk_results:
                    if r["danger"]:
                        danger_tids.add(r["ped_id"])
                        danger_tids.add(r["veh_id"])
                        risk_tids.add(r["ped_id"])
                        risk_tids.add(r["veh_id"])
                    elif r.get("caution", False):
                        risk_tids.add(r["ped_id"])
                        risk_tids.add(r["veh_id"])

                # 1초 hold: danger/caution 모두 적용
                for r in risk_results:
                    if r["danger"] or r.get("caution", False):
                        pair_key = (r["ped_id"], r["veh_id"])
                        level = "danger" if r["danger"] else "caution"
                        danger_visual_cache[pair_key] = (DANGER_VISUAL_HOLD, level, r)

                cached_markers = []
                expired_pairs = []
                for pair_key in list(danger_visual_cache.keys()):
                    frames, level, snapshot = danger_visual_cache[pair_key]
                    ped_id, veh_id = pair_key

                    if level == "danger":
                        danger_tids.add(ped_id)
                        danger_tids.add(veh_id)
                        risk_tids.add(ped_id)
                        risk_tids.add(veh_id)
                        cached_markers.append(snapshot)
                    else:
                        risk_tids.add(ped_id)
                        risk_tids.add(veh_id)
                        # caution zone 진입 시 bbox + marker 동시 발화
                        cached_markers.append(snapshot)

                    new_frames = frames - 1
                    if new_frames <= 0:
                        expired_pairs.append(pair_key)
                    else:
                        danger_visual_cache[pair_key] = (new_frames, level, snapshot)

                for pair_key in expired_pairs:
                    del danger_visual_cache[pair_key]

                # 현재 프레임의 detection 사용 (ByteTrack이라 매 프레임 있음)
                draw_data = filtered_items if filtered_items else last_draw_data
                if filtered_items:
                    last_draw_data = filtered_items

                for box, track_id, class_id, conf in draw_data[:MAX_VIS_TRACKS]:
                    if track_id in danger_tids:
                        level = "danger"
                    elif track_id in risk_tids:
                        level = "caution"
                    else:
                        level = "safe"

                    draw_bbox(vis, box, track_id, class_id, conf, level)
                    if class_id in VEHICLE_CLASSES:
                        draw_minimal_danger_zone(vis, box, track_id)
                    draw_prediction(vis, track_id, level)

                for r in cached_markers[:3]:
                    draw_risk_marker(vis, r)

                if collision_risk:
                    draw_alert(vis, ttc_sec_global, min_dist_global)

            timer.mark("visualization")

            # 기록
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            record = {
                "frame": frame_count,
                "type": "sample" if is_sample else "reuse",
                "total_ms": total_ms,
                "stages": timer.times.copy(),
                "tracks": len(active_ids),
                "preds": len(prev_boxes),
                "traj_count": trajectory_count,
            }
            if display_this_frame:
                draw_status(vis, record, len(active_ids), len(prev_boxes))
                cv2.imshow("A-PAS Monitor", vis)
                key = cv2.waitKey(1) & 0xFF
            timer.mark("imshow_waitkey")
            record["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
            record["stages"] = timer.times.copy()
            profiler.add(record)

            if key == ord('q'):
                break

            if frame_count % 30 == 0:
                cleanup_stale_tracks(active_ids, frame_count, max_lost=60)

    finally:
        cap.release()
        cv2.destroyAllWindows()

        recs = profiler.records
        if recs:
            report_recs = [r for r in recs if r["frame"] > STATS_WARMUP_FRAMES]
            if not report_recs:
                report_recs = recs
            total = np.array([r["total_ms"] for r in report_recs], dtype=np.float32)
            sample_recs = [r for r in report_recs if r["type"] == "sample"]
            reuse_recs  = [r for r in report_recs if r["type"] == "reuse"]

            print(f"\n{'='*60}")
            print("📊 A-PAS Version C — 최종 성능 통계")
            print(f"{'='*60}")
            print(f"  처리 프레임        : {frame_count:,}장 (stats skip first {STATS_WARMUP_FRAMES})")
            print(f"  YOLO 실행 횟수     : {yolo_runs:,}회 ({100*yolo_runs/max(1,frame_count):.1f}%)")
            print(f"  전체 평균 Latency  : {total.mean():.2f}ms")
            print(f"  전체 평균 FPS      : {1000/total.mean():.1f}")
            print(f"  Min / Max Latency  : {total.min():.2f}ms / {total.max():.2f}ms")
            if sample_recs:
                arr = np.array([r["total_ms"] for r in sample_recs], dtype=np.float32)
                print(f"  [샘플]  평균 Latency : {arr.mean():.2f}ms")
            if reuse_recs:
                arr = np.array([r["total_ms"] for r in reuse_recs], dtype=np.float32)
                print(f"  [재사용] 평균 Latency: {arr.mean():.2f}ms")
            print(f"  YOLO runs          : {yolo_runs:,} / {frame_count:,} frames")
            print(f"  Config             : imgsz={YOLO_IMGSZ}, scale={DISPLAY_SCALE}, yolo_every_frame={USE_YOLO_EVERY_FRAME}")
            print(f"{'='*60}")

            profiler.print_report()


if __name__ == "__main__":
    main()


