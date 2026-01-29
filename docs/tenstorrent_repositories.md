# Tenstorrent 레포지토리 구조 및 연결 관계

이 문서는 Tenstorrent AI 추론 시스템을 구성하는 여러 레포지토리들의 역할과 연결 관계를 설명합니다.

## 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Layer                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      tt-inference-server                               │  │
│  │  (추론 서버 통합, 워크플로우, 벤치마킹, 평가)                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Framework Layer                                      │
│  ┌──────────────┐  ┌───────────────────┐  ┌────────────────────────────┐   │
│  │ vLLM (Fork)  │  │   tt-vllm-plugin  │  │     Diffusers/Others       │   │
│  │ (LLM 서빙)   │  │ (vLLM TT 플러그인)│  │  (이미지/오디오 모델)      │   │
│  └──────────────┘  └───────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Model Layer                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           tt-metal                                     │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │ tt_transformers │  │    demos/       │  │   models/           │   │  │
│  │  │ (최적화 모델)   │  │ (데모 구현)     │  │ (모델 아키텍처)     │   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Runtime Layer                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       TTNN (TT Neural Network)                         │  │
│  │              (고성능 신경망 연산 라이브러리)                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         System Software Layer                                │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐     │
│  │ tt-firmware │  │  tt-kmd  │  │  tt-smi  │  │     tt-topology      │     │
│  │ (펌웨어)    │  │ (드라이버)│  │ (관리)   │  │   (토폴로지 설정)    │     │
│  └─────────────┘  └──────────┘  └──────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Hardware Layer                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │          Tenstorrent Accelerators (N150, N300, P100, P150, etc.)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 레포지토리 상세 설명

### 1. tt-inference-server

**GitHub**: https://github.com/tenstorrent/tt-inference-server

**역할**: Tenstorrent 하드웨어에서 AI 모델 추론을 위한 통합 서버 플랫폼

**주요 기능**:
- LLM 및 미디어 모델 추론 서버 제공
- 워크플로우 오케스트레이션 (`workflows/`)
- 벤치마킹 및 성능 평가 (`benchmarking/`, `evals/`)
- Docker 이미지 빌드 및 배포

**디렉토리 구조**:
```
tt-inference-server/
├── tt-vllm-plugin/        # vLLM 플러그인 (독립적)
├── vllm-tt-metal-llama3/  # vLLM fork 기반 서버
├── tt-media-server/       # 미디어 모델 서버 (DiT, SDXL, Whisper 등)
├── workflows/             # 워크플로우 시스템
├── benchmarking/          # 성능 벤치마킹
├── evals/                 # 정확도 평가
└── utils/                 # 유틸리티
```

**의존 관계**:
- `tt-metal`: 모델 실행 런타임
- `vLLM fork`: LLM 서빙 프레임워크
- `Diffusers`: 이미지/비디오 생성 모델

---

### 2. tt-metal

**GitHub**: https://github.com/tenstorrent-metal/tt-metal

**역할**: Tenstorrent 하드웨어를 위한 핵심 런타임 및 모델 라이브러리

**주요 컴포넌트**:

| 컴포넌트 | 경로 | 역할 |
|----------|------|------|
| **TTNN** | `ttnn/` | Tenstorrent Neural Network API - 고수준 신경망 연산 |
| **tt_transformers** | `models/tt_transformers/` | 최적화된 Transformer 모델 구현 |
| **demos** | `models/demos/` | 모델별 데모 및 구현 예제 |

**모델 구현 예시**:
```python
# tt_transformers의 LlamaForCausalLM
# vLLM과 통합을 위해 generator_vllm 모듈 제공
models.tt_transformers.tt.generator_vllm:LlamaForCausalLM
```

**환경 변수**:
```bash
TT_METAL_HOME=/path/to/tt-metal
PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib
```

**빌드 방법**:
```bash
git clone https://github.com/tenstorrent-metal/tt-metal.git
cd tt-metal
git checkout <commit_sha>
bash ./build_metal.sh
bash ./create_venv.sh
```

---

### 3. vLLM (Tenstorrent Fork)

**GitHub**: https://github.com/tenstorrent/vllm (branch: `dev`)

**역할**: vLLM을 Tenstorrent 하드웨어에서 실행할 수 있도록 수정한 포크

**원본과의 차이점**:
```
vllm (Tenstorrent fork)
├── tt_metal/              # TT 백엔드 통합 코드 (추가됨)
├── vllm/
│   ├── platforms/
│   │   └── tt_platform.py # TT 플랫폼 정의 (추가됨)
│   ├── worker/
│   │   └── tt_worker.py   # TT 워커 구현 (추가됨)
│   └── ...
```

**tt-inference-server에서의 사용**:
```dockerfile
# Dockerfile에서 특정 커밋 체크아웃
RUN git clone https://github.com/tenstorrent/vllm.git ${vllm_dir} \
    && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG}
```

---

### 4. tt-vllm-plugin

**위치**: `tt-inference-server/tt-vllm-plugin/`

**역할**: vLLM 플러그인 시스템을 활용한 독립적인 TT 플랫폼 지원

**vs vLLM Fork**:
| 항목 | tt-vllm-plugin | vLLM Fork |
|------|----------------|-----------|
| 설치 | `pip install` | 소스 빌드 |
| vLLM 버전 | 0.10.1.1 고정 | dev 브랜치 |
| 업데이트 | 쉬움 | 복잡 |
| 기능 | 표준 기능 | 확장 기능 |

---

### 5. 시스템 소프트웨어 레포지토리

#### tt-firmware
- **역할**: Tenstorrent 칩 펌웨어
- **문서**: https://docs.tenstorrent.com/getting-started/

#### tt-kmd (Kernel Mode Driver)
- **역할**: Linux 커널 드라이버
- **설치**: Tenstorrent 하드웨어 인식 및 통신

#### tt-smi (System Management Interface)
- **역할**: 시스템 관리 도구
- **기능**: 디바이스 상태 모니터링, 리셋 등

#### tt-topology
- **역할**: 멀티 디바이스 토폴로지 설정
- **기능**: 디바이스 간 연결 구성

## 레포지토리 간 커밋 연결

`tt-inference-server`는 `model_spec.py`에서 각 모델별로 사용할 `tt-metal`과 `vLLM`의 특정 커밋을 지정합니다:

```python
# workflows/model_spec.py 예시
ModelSpec(
    model_name="Llama-3.1-8B-Instruct",
    tt_metal_commit="c180ef7",  # tt-metal 커밋
    vllm_commit="2a8debd",       # vLLM 커밋
    ...
)
```

**Docker 태그 생성 규칙**:
```python
# 예: 0.8.0-c180ef7abcde-2a8debdabcde
tag = f"{version}-{tt_metal_commit[:12]}-{vllm_commit[:12]}"
```

## 데이터 흐름

### LLM 추론 요청 흐름

```
1. Client Request
        │
        ▼
2. tt-inference-server (vllm-tt-metal-llama3 또는 tt-vllm-plugin)
        │
        ├── vLLM API Server (run_vllm_api_server.py)
        │       │
        │       ▼
        ├── vLLM Engine
        │       │
        │       ▼
        └── TTPlatform / TTWorker
                │
                ▼
3. tt-metal
        │
        ├── tt_transformers (모델 구현)
        │       │
        │       ▼
        └── TTNN (연산 실행)
                │
                ▼
4. Tenstorrent Hardware
        │
        ▼
5. Response
```

### 미디어 모델 추론 흐름

```
1. Client Request
        │
        ▼
2. tt-media-server (FastAPI)
        │
        ├── tt_model_runners/ (모델별 러너)
        │       │
        │       ▼
        └── Diffusers + tt-metal
                │
                ▼
3. tt-metal
        │
        ├── demos/ (DiT, SDXL, Whisper 등)
        │       │
        │       ▼
        └── TTNN
                │
                ▼
4. Tenstorrent Hardware
        │
        ▼
5. Response (이미지/오디오/비디오)
```

## 지원 하드웨어

| 디바이스 | 코드명 | 설명 |
|----------|--------|------|
| N150 | `n150` | Wormhole B0 기반 |
| N300 | `n300` | Wormhole B0 기반 (듀얼 칩) |
| P100 | `p100` | Blackhole 기반 |
| P150 | `p150` | Blackhole 기반 |
| T3K | `t3k` | TT-QuietBox/TT-LoudBox |
| Galaxy | `galaxy` | 대규모 클러스터 |

## 개발 환경 설정

### 1. 시스템 소프트웨어 설치
```bash
# Tenstorrent 공식 문서 참고
# https://docs.tenstorrent.com/getting-started/
```

### 2. tt-metal 빌드
```bash
git clone https://github.com/tenstorrent-metal/tt-metal.git
cd tt-metal
git checkout <required_commit>
git submodule update --init --recursive
bash ./build_metal.sh
bash ./create_venv.sh
source python_env/bin/activate
```

### 3. tt-inference-server 설정
```bash
git clone https://github.com/tenstorrent/tt-inference-server.git
cd tt-inference-server

# 플러그인 방식
cd tt-vllm-plugin
pip install -e .

# 또는 Docker 방식
python run.py server --workflow vllm --docker
```

## 버전 호환성 매트릭스

각 릴리스에서 검증된 버전 조합:

| tt-inference-server | tt-metal | vLLM | Python |
|---------------------|----------|------|--------|
| v0.8.0 | 특정 커밋 | 특정 커밋 | 3.10-3.11 |

정확한 커밋은 `model_specs_output.json` 또는 Docker 이미지 태그에서 확인 가능합니다.

## 참고 링크

- **tt-inference-server**: https://github.com/tenstorrent/tt-inference-server
- **tt-metal**: https://github.com/tenstorrent-metal/tt-metal
- **vLLM Fork**: https://github.com/tenstorrent/vllm
- **Tenstorrent 문서**: https://docs.tenstorrent.com
- **Docker 이미지**: ghcr.io/tenstorrent/tt-inference-server/
