# 🌿 A-PAS Git 협업 가이드

> 팀원 모두가 안전하게 협업할 수 있도록 브랜치 전략과 사용법을 정리한 문서입니다.
> **반드시 읽고 시작해주세요!**

---

## 📌 왜 브랜치를 나눠서 쓰나요?

지금까지는 `main` 브랜치 하나에서 작업했는데, 이런 문제가 생겨요:

- 누군가 코드를 잘못 수정하면 **전체 팀의 작업이 막힘**
- 심사 2주 전에 "어제까지 되던 게 오늘 안 됨" 사태 발생
- 누가 뭘 건드렸는지 **추적 불가능**

그래서 브랜치를 나눠서 쓰는 거예요. 안전장치라고 생각하면 돼요.

---

## 🌲 A-PAS 브랜치 구조

```
main                          ← 심사장에서 쓸 "안정 버전" (함부로 건드리지 말 것!)
  │
  └── develop                 ← 평소 개발용 통합 브랜치
        │
        ├── feature/hailo     ← Hailo 통합 작업
        ├── feature/carla     ← CARLA 시나리오 작업
        ├── feature/model     ← 횡단보도 모형 제작
        └── feature/xxx       ← 각자 맡은 작업
```

### 역할 정리

| 브랜치 | 역할 | 누가 건드릴 수 있나? |
|---|---|---|
| `main` | **심사용 안정 버전**. 건드리면 안 됨 | 팀장만 (리뷰 후 merge) |
| `develop` | 개발 중인 통합 버전. 테스트 끝난 기능만 모임 | 팀장이 merge |
| `feature/xxx` | **내가 맡은 작업 공간**. 자유롭게 커밋 | 각자 |

---

## 🚀 기본 작업 흐름 (이것만 외우면 됨)

### 1단계: 최초 1회 세팅

```bash
# 프로젝트 처음 받는 경우
git clone https://github.com/A-PAS-team/A-PAS.git
cd A-PAS

# develop 브랜치로 이동
git checkout develop
```

### 2단계: 새 작업 시작 — feature 브랜치 생성

**절대 develop이나 main에 직접 작업하지 마세요!** 항상 feature 브랜치를 따서 작업합니다.

```bash
# 작업 시작 전 반드시 최신 develop 받기
git checkout develop
git pull origin develop

# 내 작업용 브랜치 만들기
git checkout -b feature/내작업이름

# 예시:
git checkout -b feature/carla-scenario       # CARLA 시나리오 작업
git checkout -b feature/led-circuit          # LED 회로 작업
git checkout -b feature/model-3d-print       # 3D 프린팅 작업
```

### 3단계: 작업하고 커밋

```bash
# 파일 수정 후...

# 변경사항 확인
git status

# 변경된 파일 추가
git add .

# 커밋 (메시지는 아래 규칙 참고)
git commit -m "feat: CARLA 스몸비 시나리오 추가"

# 원격 저장소에 올리기
git push origin feature/내작업이름
```

### 4단계: 작업 완료 — 팀장에게 알리기

작업이 끝나면 **Pull Request(PR)** 를 올리고 팀장에게 리뷰 요청하세요.

1. GitHub 레포 페이지 접속
2. "Compare & pull request" 버튼 클릭
3. `base: develop` ← `compare: feature/내작업이름` 확인
4. 제목 + 설명 작성 후 PR 생성
5. 팀장(의진)에게 리뷰어 지정

**PR이 merge되면 develop 브랜치로 병합돼요.** 본인이 직접 develop에 merge하지 마세요!

---

## 📝 커밋 메시지 규칙

통일된 포맷으로 쓰면 나중에 뭘 했는지 보기 편해요.

### 형식

```
<타입>: <간단한 설명>
```

### 타입 종류

| 타입 | 언제 쓰나 |
|---|---|
| `feat:` | 새 기능 추가 |
| `fix:` | 버그 수정 |
| `docs:` | 문서 수정 (README 등) |
| `refactor:` | 기능 변화 없이 코드 정리 |
| `test:` | 테스트 추가/수정 |
| `chore:` | 설정 파일, 빌드 등 기타 |

### 예시

```bash
git commit -m "feat: CARLA 스몸비 시나리오 3종 추가"
git commit -m "fix: LED 회로 GPIO 핀 번호 수정"
git commit -m "docs: README 하드웨어 사양 업데이트"
git commit -m "refactor: main.py 함수 분리"
git commit -m "chore: .gitignore에 runs/ 폴더 추가"
```

### ❌ 안 좋은 예시

```bash
git commit -m "수정"                    # 뭘 수정했는지 모름
git commit -m "아 왜 안돼"              # 감정표현은 커밋 메시지 아님
git commit -m "asdf"                   # 안 됨
```

---

## 🎯 실전 시나리오

### 시나리오 1: 처음 작업 시작

```bash
# 월요일 아침, 작업 시작
git checkout develop
git pull origin develop          # 다른 팀원이 업데이트한 게 있으면 받음
git checkout -b feature/carla-scenario

# ... 파일 수정 ...

git add .
git commit -m "feat: 정상 보행자 시나리오 추가"
git push origin feature/carla-scenario
```

### 시나리오 2: 중간에 develop이 업데이트된 경우

내가 작업하는 동안 다른 팀원이 develop에 변경사항을 merge했어요. 최신 상태 반영하려면:

```bash
# feature 브랜치에 있는 상태에서
git checkout feature/내작업
git fetch origin
git merge origin/develop

# 충돌(conflict) 안 나면 그대로 계속 작업
# 충돌 나면 파일 수정 후 다시 add/commit
```

### 시나리오 3: 실수로 잘못된 브랜치에서 작업함

"develop에서 작업해버렸네..."

```bash
# 걱정 마세요. 이렇게 옮길 수 있어요.
git stash                         # 현재 작업 임시 저장
git checkout -b feature/복구용    # 새 브랜치 생성
git stash pop                     # 저장한 작업 복원
git add .
git commit -m "feat: 작업 내용"
```

### 시나리오 4: 중요한 시점에 태그 달기

심사 직전이나 마일스톤 달성 시:

```bash
git checkout main
git tag -a v1.0-hailo-ready -m "Hailo 통합 완료"
git push --tags
```

나중에 그 시점으로 돌아가고 싶으면:

```bash
git checkout v1.0-hailo-ready
```

---

## ⚠️ 절대 하지 말아야 할 것

### 1. ❌ main에 직접 커밋

```bash
# 이러면 안 됨!
git checkout main
git commit -m "..."
```

main은 **심사 당일 시연할 버전**이에요. 항상 develop에서 PR로 merge돼야 해요.

### 2. ❌ force push

```bash
# 절대 쓰지 말 것
git push --force
git push -f
```

다른 사람 작업 날려버릴 수 있어요. 문제 생기면 팀장한테 물어보세요.

### 3. ❌ 큰 파일 커밋

```bash
# .gitignore에 있는 건 커밋하면 안 됨
# *.pt, *.npy, *.mp4, *.jpg 등
```

실수로 커밋했으면 바로 알려주세요. 나중에 지우는 게 훨씬 어려워요.

### 4. ❌ 다른 사람 브랜치에 push

내 feature 브랜치만 push하세요. 남의 브랜치는 절대 건드리지 말 것!

---

## 🆘 자주 발생하는 문제 해결

### 문제 1: "Your branch is behind origin/develop"

다른 팀원이 먼저 push해서 내가 뒤처진 상태예요.

```bash
git pull origin develop
```

### 문제 2: Merge Conflict (충돌) 발생

같은 파일을 두 사람이 수정해서 충돌이 났어요.

```bash
# 충돌 파일 열어보면 이런 식으로 표시됨:
<<<<<<< HEAD
내가 쓴 코드
=======
다른 사람이 쓴 코드
>>>>>>> origin/develop

# 어느 쪽을 쓸지 결정하고 <<<< ==== >>>> 기호 다 지운 뒤 저장
# 그리고 다시 커밋
git add .
git commit -m "fix: merge conflict 해결"
```

해결이 어려우면 팀장한테 연락!

### 문제 3: 커밋 메시지 오타 냈어요

마지막 커밋 메시지 수정:

```bash
git commit --amend -m "올바른 메시지"

# 이미 push했으면 팀장한테 문의
```

### 문제 4: 잘못 커밋한 파일 되돌리기

```bash
# 마지막 커밋 취소 (파일은 유지)
git reset --soft HEAD~1

# 파일 수정 후 다시 커밋
git add .
git commit -m "fix: ..."
```

---

## 💡 유용한 명령어 치트시트

```bash
# 현재 어느 브랜치에 있는지 확인
git branch
git branch -a                    # 원격 포함 전체

# 내 작업 상태 확인
git status

# 최근 커밋 내역 보기
git log --oneline -10

# 특정 파일 변경 내역 보기
git log --follow -- 파일명

# 변경사항 미리보기
git diff                         # 수정했지만 add 안 한 것
git diff --cached                # add 했지만 commit 안 한 것

# 브랜치 이동
git checkout 브랜치이름
git switch 브랜치이름            # 최신 Git에서 권장

# 브랜치 삭제 (작업 완료된 feature 브랜치)
git branch -d feature/완료된작업     # 로컬
git push origin --delete feature/완료된작업   # 원격

# 원격 최신 상태 가져오기
git fetch origin                 # 받기만 (merge X)
git pull origin 브랜치명         # 받아서 merge까지
```

---

## 📋 작업 시작 전 체크리스트

새 작업 시작할 때마다 이거 확인하세요!

- [ ] `git status` 로 현재 브랜치 확인
- [ ] `develop` 브랜치에서 시작
- [ ] `git pull origin develop` 으로 최신 버전 받기
- [ ] `git checkout -b feature/작업이름` 으로 새 브랜치 생성
- [ ] `.gitignore` 확인 (대용량 파일 실수로 커밋 X)

---

## 🙋 도움이 필요할 때

- **Git 관련 문제**: 팀장(의진)에게 연락
- **코드 관련 질문**: Slack이나 카톡
- **심각한 실수** (예: main에 직접 push): **즉시** 팀장에게 알리기

---

## 🎉 한 줄 요약

> **"develop에서 feature 따서 작업하고, PR로 올려서 리뷰받기"**
> 이거 하나만 기억하면 됩니다!

---

**마지막 업데이트**: 2026년 4월
**A-PAS Team**
