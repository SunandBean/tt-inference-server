# vLLM과 tt-metal 연결 구조 상세 가이드

이 문서는 vLLM과 tt-metal이 코드 레벨에서 어떻게 연결되어 동작하는지 상세하게 설명합니다.

## 전체 연결 아키텍처

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              vLLM                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 1. ModelRegistry: "TTLlamaForCausalLM" 아키텍처 조회               │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          tt-vllm-plugin                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 2. TTModelLoader.load_model()                                       │  │
│  │    - get_model_architecture() → TT 모델 클래스 찾기                │  │
│  │    - model_class.initialize_vllm_model() 호출                      │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 3. open_mesh_device()                                               │  │
│  │    - ttnn.get_device_ids()     ← TTNN API 호출                     │  │
│  │    - ttnn.open_mesh_device()   ← TT 하드웨어 열기                  │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                             tt-metal                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 4. models.tt_transformers.tt.generator_vllm:LlamaForCausalLM       │  │
│  │    - initialize_vllm_model(hf_config, mesh_device, max_batch_size) │  │
│  │    - prefill_forward() / decode_forward()                          │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 5. TTNN (TT Neural Network API)                                     │  │
│  │    - ttnn.matmul(), ttnn.softmax(), ttnn.attention() 등            │  │
│  │    → Tenstorrent 하드웨어에서 실제 연산 수행                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 핵심 연결 포인트

### 1. 모델 등록 (Plugin → tt-metal 모델 매핑)

vLLM의 ModelRegistry를 통해 TT 모델 아키텍처와 tt-metal 구현을 연결합니다.

**파일 위치**: `tt-vllm-plugin/tt_vllm_plugin/__init__.py`

```python
def register_models():
    """vLLM ModelRegistry에 TT 모델 등록"""
    from vllm import ModelRegistry

    # vLLM이 "TTLlamaForCausalLM"을 요청하면
    # → tt-metal의 generator_vllm:LlamaForCausalLM으로 연결
    ModelRegistry.register_model(
        "TTLlamaForCausalLM",  # vLLM에서 찾는 아키텍처 이름
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",  # tt-metal 모듈 경로
    )

    # BGE 임베딩 모델 등록
    ModelRegistry.register_model(
        "TTBertModel",
        "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
    )
```

**동작 원리**:
1. HuggingFace 모델 설정에서 아키텍처 읽기: `"LlamaForCausalLM"`
2. TTPlatform이 "TT" 접두사 추가: `"TTLlamaForCausalLM"`
3. ModelRegistry에서 해당 아키텍처로 tt-metal 모델 클래스 찾기
4. tt-metal의 `models.tt_transformers.tt.generator_vllm:LlamaForCausalLM` 반환

### 2. 하드웨어 연결 (TTNN API)

tt-metal이 제공하는 TTNN(TT Neural Network) 라이브러리를 통해 하드웨어를 제어합니다.

**파일 위치**: `tt-vllm-plugin/tt_vllm_plugin/worker/tt_worker.py`

```python
import ttnn  # ← tt-metal의 TTNN 라이브러리 직접 import

def get_mesh_grid(dp_rank=0):
    """사용 가능한 TT 디바이스 확인 및 mesh 그리드 결정"""
    if dp_rank == 0:
        num_devices_available = len(ttnn.get_device_ids())

    # 디바이스 타입에 따른 그리드 매핑
    mesh_grid_dict = {
        "N150": (1, 1),      # 단일 Wormhole 칩
        "N300": (1, 2),      # 듀얼 Wormhole 칩
        "T3K": (1, 8),       # 8칩 구성
        "TG": (8, 4),        # Galaxy (32칩)
        "P100": (1, 1),      # Blackhole 단일
        "P150": (1, 1),      # Blackhole
    }
    return mesh_grid

def open_mesh_device(override_tt_config, trace_mode, dp_rank=0, model_config=None):
    """TT 하드웨어 mesh device 열기"""
    mesh_grid = get_mesh_grid(dp_rank)

    # 디바이스 파라미터 설정
    device_params = device_params_from_override_tt_config(
        override_tt_config, trace_mode, model_config
    )

    # Fabric 설정 (멀티 디바이스 통신)
    num_devices_requested = mesh_grid[0] * mesh_grid[1]
    set_fabric(override_tt_config, num_devices_requested)

    # TTNN API로 mesh device 열기
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_grid),
        dispatch_core_config=get_dispatch_core_config(override_tt_config),
        **device_params,
    )
    return mesh_device

def close_mesh_device(mesh_device, override_tt_config):
    """TT 하드웨어 mesh device 닫기"""
    ttnn.ReadDeviceProfiler(mesh_device)
    num_devices = mesh_device.get_num_devices()
    ttnn.close_mesh_device(mesh_device)
    reset_fabric(override_tt_config, num_devices)
```

**주요 TTNN API 호출**:

| API | 용도 |
|-----|------|
| `ttnn.get_device_ids()` | 사용 가능한 TT 디바이스 ID 목록 조회 |
| `ttnn.get_arch_name()` | 하드웨어 아키텍처 확인 (wormhole_b0 등) |
| `ttnn.open_mesh_device()` | mesh device 열기 |
| `ttnn.close_mesh_device()` | mesh device 닫기 |
| `ttnn.set_fabric_config()` | 멀티 디바이스 fabric 설정 |
| `ttnn.MeshShape()` | mesh 그리드 형태 정의 |

### 3. 모델 로드 (tt-metal 모델 초기화)

vLLM의 모델 로더를 확장하여 tt-metal 모델을 로드합니다.

**파일 위치**: `tt-vllm-plugin/tt_vllm_plugin/model_loader/tt_loader.py`

```python
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture

class TTModelLoader(BaseModelLoader):
    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
        """TT 플랫폼용 모델 로드"""

        device_config = vllm_config.device_config
        scheduler_config = vllm_config.scheduler_config

        # 1. vLLM 아키텍처 해석기로 모델 클래스 찾기
        model_class, _ = get_model_architecture(model_config)
        # → model_class = tt-metal의 LlamaForCausalLM

        # 2. TT 모델인지 확인 (initialize_vllm_model 메서드 존재 여부)
        if not hasattr(model_class, "initialize_vllm_model"):
            raise ValueError(
                f"Model class {model_class.__name__} does not have initialize_vllm_model method. "
                "TT plugin requires TT-specific model implementations."
            )

        # 3. 배치 크기 및 최적화 설정
        data_parallel = vllm_config.parallel_config.data_parallel_size
        max_batch_size = scheduler_config.max_num_seqs * data_parallel

        # 4. tt-metal 모델의 특수 초기화 메서드 호출
        model = model_class.initialize_vllm_model(
            model_config.hf_config,      # HuggingFace 설정
            device_config.device,         # mesh_device (TT 하드웨어)
            max_batch_size,               # 최대 배치 크기
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=data_parallel,
            optimizations=optimizations,
        )
        return model
```

**tt-metal 모델이 구현해야 하는 인터페이스**:

```python
# tt-metal/models/tt_transformers/tt/generator_vllm.py (예시)
class LlamaForCausalLM:
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,           # HuggingFace 모델 설정
        mesh_device,         # TT mesh device
        max_batch_size,      # 최대 배치 크기
        max_seq_len,         # 최대 시퀀스 길이
        tt_data_parallel,    # 데이터 병렬 크기
        optimizations=None,  # 최적화 옵션
    ):
        """vLLM용 모델 초기화"""
        # 모델 가중치 로드
        # TT 하드웨어에 맞게 텐서 변환
        # KV 캐시 할당
        return model_instance

    def prefill_forward(self, tokens, positions, ...):
        """프리필 (첫 토큰 생성)"""
        # TTNN 연산으로 forward pass
        pass

    def decode_forward(self, tokens, positions, ...):
        """디코드 (이후 토큰 생성)"""
        # TTNN 연산으로 forward pass
        pass
```

## 실행 흐름 상세

### 서버 시작 시 초기화 흐름

```
1. vLLM 서버 시작
   └── vllm serve meta-llama/Llama-3.1-8B-Instruct

2. 플러그인 로드
   └── tt-vllm-plugin 활성화
       ├── register() → TTPlatform 등록
       └── register_models() → TT 모델들 ModelRegistry에 등록

3. TTPlatform.check_and_update_config()
   ├── 아키텍처 이름에 "TT" 접두사 추가
   │   └── "LlamaForCausalLM" → "TTLlamaForCausalLM"
   ├── 제약 사항 적용 (chunked prefill 비활성화 등)
   └── TTWorker 클래스 설정

4. TTWorker.init_device()
   └── open_mesh_device()
       ├── ttnn.get_device_ids()
       ├── ttnn.set_fabric_config()
       └── ttnn.open_mesh_device()

5. TTWorker.load_model()
   └── TTModelLoader.load_model()
       ├── get_model_architecture() → tt-metal 모델 클래스
       └── model_class.initialize_vllm_model()
           ├── 가중치 로드 (HuggingFace → TT 포맷)
           └── KV 캐시 할당
```

### 추론 요청 처리 흐름

```
1. 클라이언트 요청
   └── POST /v1/chat/completions

2. vLLM API 서버
   └── 요청 파싱 및 검증

3. vLLM 엔진
   ├── 토큰화
   └── 스케줄러 → SchedulerOutput 생성

4. TTWorker.execute_model(scheduler_output)
   └── TTModelRunner.execute_model()

5. 프리필 단계 (첫 토큰)
   └── model.prefill_forward(tokens, positions, kv_cache)
       └── TTNN 연산 (tt-metal 하드웨어에서 실행)
           ├── ttnn.embedding()
           ├── ttnn.linear() / ttnn.matmul()
           ├── ttnn.layer_norm()
           ├── ttnn.scaled_dot_product_attention()
           └── ttnn.softmax()

6. 디코드 단계 (이후 토큰들)
   └── model.decode_forward(token, position, kv_cache)
       └── TTNN 연산 (반복)

7. 샘플링
   └── 다음 토큰 선택

8. 응답 반환
   └── 스트리밍 또는 전체 응답
```

## 계층별 역할 요약

| 계층 | 컴포넌트 | 역할 | 주요 파일 |
|------|----------|------|-----------|
| **API** | vLLM Server | HTTP API 제공, 요청 처리 | `run_vllm_api_server.py` |
| **엔진** | vLLM Engine | 스케줄링, 배치 관리 | vllm 패키지 |
| **플랫폼** | TTPlatform | TT 플랫폼 정의, 설정 검증 | `platform.py` |
| **워커** | TTWorker | 디바이스 관리, 실행 조율 | `v1/worker/tt_worker.py` |
| **로더** | TTModelLoader | TT 모델 로드 | `model_loader/tt_loader.py` |
| **런너** | TTModelRunner | 모델 실행, KV 캐시 관리 | `v1/worker/tt_model_runner.py` |
| **하드웨어 API** | TTNN | TT 하드웨어 추상화 | `import ttnn` (tt-metal) |
| **모델** | tt_transformers | TT 최적화 모델 구현 | tt-metal 레포 |

## 데이터 변환 흐름

```
┌─────────────────┐
│ HuggingFace     │
│ 모델 가중치     │
│ (PyTorch/SF)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TT 포맷 변환    │
│ (initialize_    │
│  vllm_model)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TTNN Tensor     │
│ (TT 하드웨어    │
│  메모리에 배치) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TT 하드웨어     │
│ 연산 실행       │
└─────────────────┘
```

## KV 캐시 관리

TT 백엔드에서의 KV 캐시 관리 방식:

```python
# tt_worker.py
def get_num_available_blocks_tt(vllm_config: VllmConfig) -> int:
    """TT KV 캐시 블록 수 계산"""

    # 하드웨어 타입 확인
    is_wormhole = "wormhole_b0" in ttnn.get_arch_name()

    # 모델과 디바이스에 따른 최대 토큰 수 결정
    if "Llama-3.1-8B" in model_config.model and is_wormhole:
        max_tokens_all_users = 65536  # N150에서 Llama8B
    elif "Llama-3.3-70B" in model_config.model:
        max_tokens_all_users = 131072  # T3K에서 70B 모델

    # 블록 수 계산
    num_tt_blocks = math.ceil(max_tokens_all_users / cache_config.block_size)
    return num_tt_blocks
```

## 환경 변수 및 설정

### 필수 환경 변수

```bash
# vLLM v1 아키텍처 활성화 (필수)
export VLLM_USE_V1=1

# TT 플랫폼 타겟
export VLLM_TARGET_DEVICE="tt"

# tt-metal 경로
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=${TT_METAL_HOME}:$PYTHONPATH

# Python 환경
source ${TT_METAL_HOME}/python_env/bin/activate
```

### 선택적 설정

```bash
# Mesh 디바이스 타입 지정
export MESH_DEVICE="N150"  # 또는 N300, T3K, TG 등

# 로그 레벨
export LOGURU_LEVEL=INFO
```

## 디버깅 팁

### 1. 플러그인 로드 확인

```bash
VLLM_USE_V1=1 python -c "import vllm; print('Plugin loaded')"
```

로그에서 다음 메시지 확인:
```
INFO - Available plugins for group vllm.platform_plugins:
INFO - - tt -> tt_vllm_plugin:register
INFO - Platform plugin tt is activated
```

### 2. TT 하드웨어 확인

```python
import ttnn
print(f"Devices: {ttnn.get_device_ids()}")
print(f"Architecture: {ttnn.get_arch_name()}")
```

### 3. 모델 등록 확인

```python
from vllm import ModelRegistry
print(ModelRegistry.get_supported_archs())
# ['TTLlamaForCausalLM', 'TTBertModel', ...]
```

## 참고 자료

- [vLLM Plugin System](https://docs.vllm.ai/en/stable/design/plugin_system.html)
- [TTNN API Documentation](https://github.com/tenstorrent-metal/tt-metal)
- [tt-vllm-plugin README](../tt-vllm-plugin/README.md)
- [vLLM 수정 가이드](./vllm_modification.md)
- [Tenstorrent 레포지토리 구조](./tenstorrent_repositories.md)
