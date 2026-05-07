# Compare Existing Only Workspace

이 워크스페이스는 `scripts/compare_existing.py` 실행에 필요한 파일만 남긴 슬림 구성입니다.

## 실행

```bash
python scripts/compare_existing.py
```

자주 쓰는 옵션 예시:

```bash
python scripts/compare_existing.py \
  --instance-dir data/existing/egl \
  --num-instances 5 \
  --full-instance 0 \
  --bnp-variant global_rmp
```

## 남겨둔 핵심 경로
- `scripts/compare_existing.py`: 메인 실행 스크립트
- `scripts/existing_instance.py`: EGL `.dat` 로더/인스턴스 구성
- `src/master/`: arc 기반 비교 + branch-and-price
- `src/pricing/`: pricing 및 C++ pricer 연동
- `src/util/initial_heuristic.py`: 초기 컬럼/인공변수 유틸
- `data/existing/egl/`: 기존 공개 인스턴스 데이터

## 참고
- C++ pricer는 필요 시 `src/pricing/native/libcpp_pricing_core.so`를 자동 빌드/로드합니다.
