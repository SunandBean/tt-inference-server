# vLLM 수정 및 Tenstorrent 하드웨어 통합 가이드

이 문서는 tt-inference-server가 vLLM을 어떻게 수정하여 Tenstorrent 하드웨어에서 LLM 추론을 지원하는지 설명합니다.

## 개요

Tenstorrent는 vLLM을 두 가지 방식으로 통합하여 사용합니다:

1. **TT-vLLM Plugin (최신 방식)**: vLLM의 플러그인 시스템을 활용한 독립적인 플러그인
2. **vLLM Fork (레거시 방식)**: vLLM을 직접 포크하여 Tenstorrent 백엔드를 내장

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                         vLLM (v0.10.1.1)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Plugin System                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │ Platform Plugin │  │ Model Registry  │               │   │
│  │  │ (vllm.platform) │  │ (vllm.general)  │               │   │
│  │  └────────┬────────┘  └────────┬────────┘               │   │
│  └───────────┼────────────────────┼────────────────────────┘   │
└──────────────┼────────────────────┼────────────────────────────┘
               │                    │
               ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       tt-vllm-plugin                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │   TTPlatform   │  │    TTWorker    │  │   TTModelLoader    │ │
│  │ (platform.py)  │  │ (tt_worker.py) │  │  (tt_loader.py)    │ │
│  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘ │
└──────────┼───────────────────┼─────────────────────┼────────────┘
           │                   │                     │
           ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                          tt-metal                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    tt_transformers                          │ │
│  │  (Llama, Qwen, Mistral 등의 TT 최적화 모델 구현)           │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                         TTNN                                │ │
│  │  (Tenstorrent Neural Network API - 하드웨어 추상화 계층)   │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│              Tenstorrent Hardware (N150, N300, P100, etc.)       │
└──────────────────────────────────────────────────────────────────┘
```

## 1. TT-vLLM Plugin (권장 방식)

### 위치
```
tt-inference-server/tt-vllm-plugin/
```

### 주요 컴포넌트

#### 1.1 TTPlatform (`tt_vllm_plugin/platform.py`)

vLLM의 `Platform` 인터페이스를 구현하여 Tenstorrent 하드웨어를 위한 플랫폼을 정의합니다.

**주요 수정 사항:**

```python
class TTPlatform(Platform):
    _enum = PlatformEnum.OOT  # Out-of-tree 플랫폼
    device_name: str = "tt"
    device_type: str = "privateuseone"
```

**제약 사항 적용:**

| 기능 | 지원 여부 | 설명 |
|------|----------|------|
| Chunked Prefill | ❌ | TT 백엔드에서 아직 미지원 |
| Speculative Decoding | ❌ | TT 백엔드에서 아직 미지원 |
| LoRA | ❌ | TT 백엔드에서 아직 미지원 |
| Prefix Caching | ❌ | TT 백엔드에서 아직 미지원 |
| 분산 실행 (TP/PP) | ❌ | TT 백엔드에서 내부적으로 처리 |
| vLLM v1 아키텍처 | ✅ | 필수 (VLLM_USE_V1=1) |

**모델 아키텍처 변환:**

```python
# HuggingFace 모델 아키텍처에 "TT" 접두사 추가
# 예: "LlamaForCausalLM" → "TTLlamaForCausalLM"
arch_names = vllm_config.model_config.hf_config.architectures
for i in range(len(arch_names)):
    if not arch_names[i].startswith("TT"):
        arch_names[i] = "TT" + arch_names[i]
```

#### 1.2 TTWorker (`tt_vllm_plugin/v1/worker/tt_worker.py`)

vLLM v1의 `WorkerBase`를 상속받아 Tenstorrent 하드웨어에서 모델 실행을 담당합니다.

**핵심 기능:**

- **디바이스 초기화**: `init_device()` - TT mesh device 열기
- **모델 로드**: `load_model()` - TT 전용 모델 로더 사용
- **KV 캐시 관리**: `get_kv_cache_spec()`, `initialize_from_config()`
- **모델 실행**: `execute_model()` - 스케줄러 출력을 받아 실행

```python
def load_model(self):
    loader = TTModelLoader(self.load_config)
    model = loader.load_model(
        vllm_config=self.vllm_config,
        model_config=self.model_config
    )
    # Pooling 모델과 Generation 모델 자동 감지
    if is_pooling:
        self.model_runner = TTModelRunnerPooling(...)
    else:
        self.model_runner = TTModelRunner(...)
```

#### 1.3 모델 등록 (`tt_vllm_plugin/__init__.py`)

vLLM의 ModelRegistry에 TT 전용 모델을 등록합니다:

```python
def register_models():
    from vllm import ModelRegistry

    # Llama 모델 등록
    ModelRegistry.register_model(
        "TTLlamaForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
    )

    # BGE 임베딩 모델 등록
    ModelRegistry.register_model(
        "TTBertModel",
        "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
    )
```

### Entry Points (pyproject.toml)

```toml
[project.entry-points."vllm.platform_plugins"]
tt = "tt_vllm_plugin:register"

[project.entry-points."vllm.general_plugins"]
tt_model_registry = "tt_vllm_plugin:register_models"
```

## 2. vLLM Fork (레거시 방식)

### 위치
- **Fork 레포지토리**: https://github.com/tenstorrent/vllm (branch: `dev`)
- **사용 위치**: `tt-inference-server/vllm-tt-metal-llama3/`

### Dockerfile에서의 빌드

```dockerfile
# tt-metal 빌드
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && bash ./build_metal.sh \
    && bash ./create_venv.sh

# vLLM fork 빌드
RUN git clone https://github.com/tenstorrent/vllm.git ${vllm_dir} \
    && cd ${vllm_dir} \
    && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
    && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

### Fork의 주요 수정 사항

Tenstorrent vLLM fork에는 다음과 같은 수정이 포함되어 있습니다:

1. **TT 백엔드 통합**: `tt_metal/` 디렉토리 추가
2. **TT 플랫폼 감지**: 하드웨어 감지 및 초기화 로직
3. **override_tt_config**: TT 전용 설정 옵션 (Plugin에서는 미지원)
4. **AscendScheduler**: TT 플랫폼용 스케줄러

## 3. 주요 환경 변수

| 환경 변수 | 값 | 설명 |
|-----------|-----|------|
| `VLLM_USE_V1` | `1` | vLLM v1 아키텍처 활성화 (필수) |
| `VLLM_TARGET_DEVICE` | `tt` | Tenstorrent 플랫폼 대상 |
| `TT_METAL_HOME` | `/path/to/tt-metal` | tt-metal 설치 경로 |
| `PYTHON_ENV_DIR` | `${TT_METAL_HOME}/python_env` | Python 가상환경 경로 |

## 4. 지원되지 않는 샘플링 파라미터

TT 백엔드에서 "호환성 샘플링" (compatibility sampling)이 필요한 파라미터들:

```python
# 다음 파라미터들은 추가 처리가 필요함
- presence_penalty != 0.0
- frequency_penalty != 0.0
- repetition_penalty != 1.0
- min_p != 0.0
- bad_words
- logprobs
- prompt_logprobs
- logits_processors
- guided_decoding
- logit_bias
- allowed_token_ids
- seed (tt-metal #32209 해결 대기)
- min_tokens != 0
```

## 5. 지원 모델

### Generation 모델
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.3-70B-Instruct
- Qwen/Qwen2.5-72B-Instruct
- Qwen/Qwen3-32B
- mistralai/Mistral-7B-Instruct-v0.3

### Embedding 모델
- BAAI/bge-large-en-v1.5

## 6. 설치 및 실행

### Plugin 설치

```bash
# tt-metal 환경 활성화
source tt-metal/python_env/bin/activate

# Plugin 설치
cd tt-vllm-plugin
pip install -e .
```

### 서버 실행

```bash
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 4096 \
    --max-num-seqs 32
```

## 7. 주요 차이점: Plugin vs Fork

| 항목 | Plugin | Fork |
|------|--------|------|
| vLLM 버전 | 0.10.1.1 (고정) | dev 브랜치 |
| 설치 방식 | pip install | 소스 빌드 |
| override_tt_config | ❌ 미지원 | ✅ 지원 |
| 업데이트 용이성 | ✅ 쉬움 | ❌ 복잡함 |
| vLLM 버전 선택 | 제한적 | 자유로움 |

## 참고 자료

- [vLLM Plugin System Documentation](https://docs.vllm.ai/en/stable/design/plugin_system.html)
- [vLLM v1 Architecture Blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [Tenstorrent vLLM Fork](https://github.com/tenstorrent/vllm/tree/dev)
