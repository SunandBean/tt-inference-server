# tt-metal 포크 개발 및 연결 가이드

이 문서는 tt-metal을 포크하여 커스텀 모델을 개발하고 tt-inference-server에 연결하는 방법을 설명합니다.

## 목차

1. [tt-metal 포크 연결 방법](#1-tt-metal-포크-연결-방법)
2. [모델 인터페이스 요구사항](#2-모델-인터페이스-요구사항)
3. [새 모델 추가 가이드](#3-새-모델-추가-가이드)
4. [기존 모델 수정 가이드](#4-기존-모델-수정-가이드)
5. [디버깅 및 테스트](#5-디버깅-및-테스트)

---

## 1. tt-metal 포크 연결 방법

### 방법 1: 로컬 개발 (권장)

Docker 없이 직접 tt-metal을 빌드하고 연결하는 방법입니다.

```bash
# 1. 포크한 tt-metal 클론
git clone https://github.com/YOUR_USERNAME/tt-metal.git ~/my-tt-metal
cd ~/my-tt-metal
git checkout your-feature-branch
git submodule update --init --recursive

# 2. tt-metal 빌드
bash ./build_metal.sh
bash ./create_venv.sh

# 3. 환경 변수 설정
export TT_METAL_HOME=~/my-tt-metal
export PYTHONPATH=${TT_METAL_HOME}:$PYTHONPATH
export PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
export LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib:$LD_LIBRARY_PATH

# 4. Python 환경 활성화
source ${TT_METAL_HOME}/python_env/bin/activate

# 5. tt-vllm-plugin 설치
cd ~/tt-inference-server/tt-vllm-plugin
pip install -e .

# 6. 서버 실행
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### 방법 2: Docker 빌드 시 Dockerfile 수정

```dockerfile
# vllm-tt-metal-llama3/vllm.tt-metal.src.cloud.Dockerfile

# 80번째 줄을 수정:
# 기존:
RUN /bin/bash -c "git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \

# 변경 (본인 포크로):
RUN /bin/bash -c "git clone https://github.com/YOUR_USERNAME/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout your-feature-branch \
    ...
```

### 방법 3: 로컬 빌드 후 Docker 이미지 생성

```bash
# 1. 포크한 tt-metal 빌드
cd ~/my-tt-metal
export UBUNTU_VERSION="22.04"
export OS_VERSION="ubuntu-${UBUNTU_VERSION}-amd64"
export TT_METAL_COMMIT_SHA_OR_TAG=$(git rev-parse HEAD)

# 2. 로컬 베이스 이미지 빌드
docker build \
  -t local/tt-metal/tt-metalium/${OS_VERSION}:${TT_METAL_COMMIT_SHA_OR_TAG} \
  --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
  --target ci-build \
  -f dockerfile/Dockerfile .

# 3. tt-inference-server 빌드 시 로컬 이미지 사용
cd ~/tt-inference-server
export TT_METAL_DOCKERFILE_URL=local/tt-metal/tt-metalium/${OS_VERSION}:${TT_METAL_COMMIT_SHA_OR_TAG}

docker build \
  --build-arg TT_METAL_DOCKERFILE_URL=${TT_METAL_DOCKERFILE_URL} \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
  -f vllm-tt-metal-llama3/vllm.tt-metal.src.cloud.Dockerfile .
```

### 방법 4: Docker 컨테이너에 포크 마운트 (개발용)

```bash
docker run \
  --rm -it \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ~/my-tt-metal:/home/container_app_user/tt-metal:rw \
  --shm-size 32G \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:latest \
  bash

# 컨테이너 내부에서 재빌드
cd /home/container_app_user/tt-metal
bash ./build_metal.sh
```

---

## 2. 모델 인터페이스 요구사항

tt-vllm-plugin과 연동하려면 모델이 다음 인터페이스를 구현해야 합니다.

### 필수 메서드

```python
class YourModelForCausalLM:
    """vLLM과 연동하기 위한 TT 모델 클래스"""

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,              # HuggingFace 모델 설정
        mesh_device,            # ttnn.MeshDevice (TT 하드웨어)
        max_batch_size: int,    # 최대 배치 크기
        max_seq_len: int,       # 최대 시퀀스 길이
        tt_data_parallel: int = 1,  # 데이터 병렬 크기
        optimizations: str = None,  # "performance" 또는 "accuracy"
        **kwargs,
    ):
        """
        vLLM에서 호출하는 모델 초기화 메서드

        Returns:
            초기화된 모델 인스턴스
        """
        pass

    def prefill_forward(
        self,
        tokens: torch.Tensor,       # [batch_size, seq_len] 입력 토큰
        page_table: torch.Tensor,   # [batch_size, max_blocks] 페이지 테이블
        kv_cache: list,             # KV 캐시 텐서 리스트
        prompt_lens: list[int],     # 각 시퀀스의 프롬프트 길이
        **kwargs,
    ) -> torch.Tensor:
        """
        프리필 단계 (첫 토큰 생성)

        Returns:
            logits: [batch_size, seq_len, vocab_size] 또는 sampled tokens
        """
        pass

    def decode_forward(
        self,
        tokens: torch.Tensor,       # [batch_size, 1] 입력 토큰
        start_pos: torch.Tensor,    # [batch_size] 시작 위치
        page_table: torch.Tensor,   # [batch_size, max_blocks] 페이지 테이블
        kv_cache: list,             # KV 캐시 텐서 리스트
        enable_trace: bool = True,  # 트레이스 모드 활성화
        read_from_device: bool = True,  # 디바이스에서 결과 읽기
        **kwargs,
    ) -> torch.Tensor:
        """
        디코드 단계 (이후 토큰 생성)

        Returns:
            logits: [batch_size, vocab_size] 또는 sampled tokens
        """
        pass

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple,  # (num_blocks, num_kv_heads, block_size, head_size)
        dtype: torch.dtype,     # 데이터 타입
        num_layers: int,        # 레이어 수
    ) -> list:
        """
        KV 캐시 할당

        Returns:
            list of KV cache tensors for each layer
        """
        pass
```

### 메서드 호출 흐름

```
vLLM Engine
    │
    ▼
TTModelLoader.load_model()
    │
    ├── model_class.initialize_vllm_model()  ← 모델 초기화
    │       │
    │       ├── HuggingFace 가중치 로드
    │       ├── TT 하드웨어에 가중치 배치
    │       └── 모델 인스턴스 반환
    │
    ▼
TTModelRunner.initialize_kv_cache()
    │
    └── model.allocate_kv_cache()  ← KV 캐시 할당


추론 요청 시:
    │
    ▼
TTModelRunner.execute_model()
    │
    ├── 프리필 요청 → model.prefill_forward()
    │       │
    │       ├── tokens: 전체 프롬프트
    │       ├── page_table: KV 캐시 페이지 매핑
    │       └── prompt_lens: 각 시퀀스 길이
    │
    └── 디코드 요청 → model.decode_forward()
            │
            ├── tokens: 마지막 생성된 토큰
            ├── start_pos: 현재 위치
            └── page_table: KV 캐시 페이지 매핑
```

---

## 3. 새 모델 추가 가이드

### Step 1: tt-metal에 모델 구현

```
tt-metal/
└── models/
    └── tt_transformers/
        └── tt/
            ├── generator_vllm.py      # 기존 모델들
            └── your_model_vllm.py     # 새 모델 추가
```

### Step 2: 모델 클래스 구현

```python
# tt-metal/models/tt_transformers/tt/your_model_vllm.py

import ttnn
import torch
from torch import nn

class YourModelForCausalLM:
    """Your custom model for vLLM integration"""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hf_config,
        max_batch_size: int,
        max_seq_len: int,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # 모델 레이어 초기화
        self._init_layers()

    def _init_layers(self):
        """TTNN 연산을 사용한 레이어 초기화"""
        # 예: 임베딩, 어텐션, FFN 등
        pass

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int = 4096,
        tt_data_parallel: int = 1,
        optimizations: str = None,
        **kwargs,
    ):
        """vLLM 초기화 인터페이스"""

        # 1. HuggingFace 가중치 로드
        # (일반적으로 transformers 라이브러리 사용)

        # 2. 모델 인스턴스 생성
        model = cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # 3. 가중치를 TT 하드웨어로 전송
        model._load_weights_to_device()

        return model

    def _load_weights_to_device(self):
        """가중치를 TT 하드웨어에 배치"""
        # ttnn.from_torch() 등을 사용하여 변환
        pass

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        prompt_lens: list[int],
        **kwargs,
    ) -> torch.Tensor:
        """프리필 (전체 시퀀스 처리)"""

        # 1. 토큰 임베딩
        # 2. 각 레이어 순회
        #    - Self-attention (KV 캐시 저장)
        #    - FFN
        # 3. 출력 logits 계산

        # 예시 (실제 구현 필요):
        batch_size, seq_len = tokens.shape

        # TT 디바이스로 입력 전송
        tt_tokens = ttnn.from_torch(tokens, device=self.mesh_device)

        # Forward pass
        hidden_states = self.embedding(tt_tokens)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward(
                hidden_states,
                kv_cache=kv_cache[layer_idx],
                page_table=page_table,
                is_prefill=True,
            )

        logits = self.lm_head(hidden_states)

        # CPU로 결과 반환
        return ttnn.to_torch(logits)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        enable_trace: bool = True,
        read_from_device: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """디코드 (단일 토큰 처리)"""

        # 프리필과 유사하지만 단일 토큰만 처리
        # KV 캐시에서 읽고 새 K,V 추가

        tt_tokens = ttnn.from_torch(tokens, device=self.mesh_device)

        hidden_states = self.embedding(tt_tokens)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward(
                hidden_states,
                kv_cache=kv_cache[layer_idx],
                page_table=page_table,
                start_pos=start_pos,
                is_prefill=False,
            )

        logits = self.lm_head(hidden_states)

        if read_from_device:
            return ttnn.to_torch(logits)
        return logits

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple,
        dtype: torch.dtype,
        num_layers: int,
    ) -> list:
        """KV 캐시 메모리 할당"""

        num_blocks, num_kv_heads, block_size, head_size = kv_cache_shape

        kv_caches = []
        for _ in range(num_layers):
            # 각 레이어에 대해 K, V 캐시 할당
            k_cache = ttnn.zeros(
                kv_cache_shape,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
            )
            v_cache = ttnn.zeros(
                kv_cache_shape,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
            )
            kv_caches.append((k_cache, v_cache))

        return kv_caches
```

### Step 3: vLLM에 모델 등록

```python
# tt-vllm-plugin/tt_vllm_plugin/__init__.py 수정

def register_models():
    from vllm import ModelRegistry

    # 기존 모델들...

    # 새 모델 등록
    ModelRegistry.register_model(
        "TTYourModelForCausalLM",  # "TT" + HuggingFace 아키텍처 이름
        "models.tt_transformers.tt.your_model_vllm:YourModelForCausalLM",
    )
```

### Step 4: HuggingFace config 매핑

HuggingFace 모델의 `config.json`에 있는 아키텍처 이름이 자동으로 "TT" 접두사가 붙어 조회됩니다:

```json
// HuggingFace model config.json
{
  "architectures": ["YourModelForCausalLM"],
  ...
}
```

```
vLLM 로드 시:
"YourModelForCausalLM" → "TTYourModelForCausalLM" → your_model_vllm.py
```

---

## 4. 기존 모델 수정 가이드

### 수정 가능한 영역

```
tt-metal/models/tt_transformers/
├── tt/
│   ├── generator_vllm.py       # vLLM 인터페이스 (수정 가능)
│   ├── llama/                  # Llama 모델 구현
│   │   ├── tt_llama_model.py   # 메인 모델
│   │   ├── tt_llama_attention.py  # 어텐션 레이어
│   │   └── tt_llama_mlp.py     # FFN 레이어
│   └── common/                 # 공통 유틸리티
│       ├── tt_rope.py          # RoPE 구현
│       └── tt_layernorm.py     # LayerNorm 구현
```

### 예시: 어텐션 수정

```python
# tt-metal/models/tt_transformers/tt/llama/tt_llama_attention.py

class TTLlamaAttention:
    def __init__(self, ...):
        # 기존 초기화 코드
        pass

    def forward(self, hidden_states, kv_cache, page_table, ...):
        # 1. Q, K, V 프로젝션
        query = ttnn.linear(hidden_states, self.q_proj)
        key = ttnn.linear(hidden_states, self.k_proj)
        value = ttnn.linear(hidden_states, self.v_proj)

        # 2. RoPE 적용
        query = apply_rotary_pos_emb(query, ...)
        key = apply_rotary_pos_emb(key, ...)

        # 3. KV 캐시 업데이트
        # ★ 여기서 커스텀 캐시 로직 추가 가능

        # 4. 어텐션 계산
        # ★ 커스텀 어텐션 알고리즘 적용 가능
        attn_output = ttnn.scaled_dot_product_attention(
            query, key, value,
            is_causal=True,
        )

        # 5. 출력 프로젝션
        output = ttnn.linear(attn_output, self.o_proj)

        return output
```

### 예시: 새로운 최적화 추가

```python
# generator_vllm.py에서 optimizations 파라미터 활용

@classmethod
def initialize_vllm_model(cls, ..., optimizations: str = None, **kwargs):
    model = cls(...)

    if optimizations == "performance":
        # 성능 최적화: 낮은 정밀도, 더 많은 배치
        model.use_fp8_matmul = True
        model.fuse_qkv = True
    elif optimizations == "accuracy":
        # 정확도 최적화: 높은 정밀도
        model.use_fp8_matmul = False
        model.fuse_qkv = False

    return model
```

---

## 5. 디버깅 및 테스트

### 모델 로드 확인

```python
# 모델이 제대로 등록되었는지 확인
from vllm import ModelRegistry

# 등록된 모델 목록
print(ModelRegistry.get_supported_archs())
# ['TTLlamaForCausalLM', 'TTYourModelForCausalLM', ...]
```

### 단위 테스트

```python
# test_your_model.py
import torch
import ttnn

def test_model_initialization():
    # mesh device 열기
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    try:
        # 모델 초기화
        from models.tt_transformers.tt.your_model_vllm import YourModelForCausalLM

        model = YourModelForCausalLM.initialize_vllm_model(
            hf_config=mock_config,
            mesh_device=mesh_device,
            max_batch_size=32,
            max_seq_len=2048,
        )

        # prefill 테스트
        tokens = torch.randint(0, 32000, (1, 128))
        output = model.prefill_forward(
            tokens=tokens,
            page_table=torch.zeros(1, 64, dtype=torch.int32),
            kv_cache=model.allocate_kv_cache(...),
            prompt_lens=[128],
        )

        assert output.shape == (1, 128, 32000)  # vocab_size

    finally:
        ttnn.close_mesh_device(mesh_device)
```

### vLLM 통합 테스트

```bash
# 서버 시작
VLLM_USE_V1=1 LOGURU_LEVEL=DEBUG vllm serve your-model \
    --max-model-len 2048 \
    --max-num-seqs 8

# 다른 터미널에서 테스트 요청
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "your-model",
        "prompt": "Hello, world!",
        "max_tokens": 50
    }'
```

### 일반적인 디버깅 포인트

| 문제 | 확인 사항 |
|------|-----------|
| 모델 찾을 수 없음 | `ModelRegistry`에 등록 확인, 아키텍처 이름 확인 |
| `initialize_vllm_model` 없음 | 메서드 이름 철자 확인, classmethod 데코레이터 확인 |
| KV 캐시 오류 | `allocate_kv_cache` 반환 형식 확인 |
| Shape 불일치 | `prefill_forward`, `decode_forward` 출력 shape 확인 |
| 디바이스 오류 | `TT_METAL_HOME` 환경 변수, 하드웨어 연결 확인 |

---

## 참고 자료

- [vLLM 수정 가이드](./vllm_modification.md)
- [vLLM-tt-metal 연결 구조](./vllm_tt_metal_connection.md)
- [Tenstorrent 레포지토리 구조](./tenstorrent_repositories.md)
- [tt-metal 공식 문서](https://github.com/tenstorrent-metal/tt-metal)
- [TTNN API 가이드](https://docs.tenstorrent.com)
