# GitHub 업로드 완료 가이드

## ✅ 완료된 작업

1. ✅ `.gitignore` 파일 생성
2. ✅ `requirements.txt` 파일 생성
3. ✅ `README.md` 파일 생성
4. ✅ Git 저장소 초기화 (`git init`)
5. ✅ 첫 커밋 완료 (43개 파일, 10,432줄 추가)

## 다음 단계: GitHub에 업로드하기

### 1단계: GitHub에서 새 저장소 생성

1. **GitHub 웹사이트 접속**
   - https://github.com 접속
   - 로그인

2. **새 저장소 생성**
   - 우측 상단 `+` 버튼 클릭
   - `New repository` 선택

3. **저장소 설정**
   - **Repository name**: 원하는 이름 입력 (예: `moms-imbalanced-learning`)
   - **Description**: 선택사항 (예: "MOMS: Minority Oversampling with Majority Selection")
   - **Public/Private**: 선택
   - ⚠️ **중요**: 다음 항목들은 **체크하지 않기**:
     - ❌ "Initialize this repository with a README"
     - ❌ "Add .gitignore"
     - ❌ "Choose a license" (원하면 나중에 추가 가능)
   - `Create repository` 클릭

4. **GitHub에서 제공하는 명령어 확인**
   - 저장소 생성 후 GitHub에서 보여주는 페이지에 명령어가 표시됩니다
   - 예시:
     ```bash
     git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
     git branch -M main
     git push -u origin main
     ```

### 2단계: 로컬 저장소를 GitHub에 연결

**PowerShell 또는 Command Prompt에서 실행:**

```powershell
# 1. 원격 저장소 추가 (YOUR_USERNAME과 REPOSITORY_NAME을 실제 값으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git

# 2. 브랜치 이름을 main으로 변경 (이미 master인 경우)
git branch -M main

# 3. GitHub에 푸시
git push -u origin main
```

**예시:**
```powershell
git remote add origin https://github.com/smcha/moms-imbalanced-learning.git
git branch -M main
git push -u origin main
```

### 3단계: 인증 (필요한 경우)

GitHub에 푸시할 때 인증이 필요할 수 있습니다:

**옵션 1: Personal Access Token (권장)**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" 클릭
3. 권한 선택: `repo` 체크
4. 토큰 생성 후 복사
5. 푸시할 때 비밀번호 대신 토큰 사용

**옵션 2: GitHub CLI**
```powershell
# GitHub CLI 설치 후
gh auth login
```

**옵션 3: SSH 키 사용**
```powershell
# SSH URL 사용
git remote set-url origin git@github.com:YOUR_USERNAME/REPOSITORY_NAME.git
```

### 4단계: 푸시 확인

푸시가 성공하면:
- GitHub 웹사이트에서 저장소 페이지를 새로고침
- 파일들이 표시되는지 확인
- README.md가 자동으로 표시되는지 확인

## 다른 컴퓨터에서 Clone하기

### 1단계: 저장소 Clone

```bash
git clone https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

### 2단계: 의존성 설치

```bash
pip install -r requirements.txt
```

### 3단계: PyTorch 설치 (필요시)

**CPU 버전:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU 버전 (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4단계: pydpc 설치 (선택사항)

```bash
cd pydpc
pip install -e .
cd ..
```

**참고:** Windows에서 pydpc 설치가 실패할 수 있습니다. 코드에 fallback 메커니즘이 포함되어 있어서 설치하지 않아도 작동합니다.

### 5단계: 환경 변수 설정 (Windows)

**PowerShell:**
```powershell
$env:PYTHONPATH = "$PWD;$PWD\src;$PWD\src\models;$PWD\src\training;$PWD\src\utils;$PWD\pydpc"
```

**Command Prompt:**
```batch
set PYTHONPATH=%CD%;%CD%\src;%CD%\src\models;%CD%\src\training;%CD%\src\utils;%CD%\pydpc
```

### 6단계: 데이터 준비

`data/raw/` 폴더에 데이터 파일을 준비하세요. GitHub에 큰 데이터 파일을 올리지 않았다면 별도로 준비해야 합니다.

### 7단계: 테스트 실행

```bash
python src/experiments/run_ablation_study.py --config experiments/configs/ablation_study/quick_test.yaml
```

## 추가 파일 업로드하기

나중에 변경사항을 업로드하려면:

```bash
# 변경사항 확인
git status

# 변경된 파일 추가
git add .

# 또는 특정 파일만 추가
git add src/models/new_file.py

# 커밋
git commit -m "Add new feature: description"

# GitHub에 푸시
git push
```

## 주의사항

### 큰 파일 처리

`.gitignore`에 다음이 포함되어 있습니다:
- `results/` - 결과 파일 (용량이 크면 제외)
- `data/raw/` - 원본 데이터 (용량이 크면 제외)
- `*.pth`, `*.pt` - 모델 체크포인트

**큰 파일을 업로드하려면:**
1. `.gitignore`에서 해당 라인 제거 또는 주석 처리
2. `git add` 및 `git commit`
3. 또는 Git LFS 사용

### 보안

- API 키, 비밀번호는 절대 커밋하지 마세요
- `.env` 파일은 `.gitignore`에 포함되어 있습니다
- 민감한 정보가 포함된 파일은 제외하세요

## 문제 해결

### "remote origin already exists" 오류

```bash
# 기존 원격 저장소 제거
git remote remove origin

# 새로 추가
git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
```

### "Permission denied" 오류

- Personal Access Token이 필요할 수 있습니다
- SSH 키를 사용하거나 GitHub CLI를 사용하세요

### "Large files" 오류

- Git LFS를 사용하거나
- 큰 파일을 `.gitignore`에 추가하세요

## 완료 확인

✅ GitHub 저장소 생성 완료
✅ 로컬 저장소 연결 완료
✅ 첫 푸시 완료
✅ 다른 컴퓨터에서 clone 테스트 완료

이제 어디서든 프로젝트를 사용할 수 있습니다!

