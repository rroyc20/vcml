# Refactor Algorithm

이 디렉토리는 기존 코드를 직접 건드리지 않고, 인터페이스 계층과 solver 엔진 계층을 분리한 독립 실행 패키지입니다.

## 구조

- `cli/`
  - 사용자가 직접 실행하는 진입점
- `app/`
  - 실행 흐름 조합, 배치 orchestration, 리포트 출력
- `engine/`
  - 인스턴스 로더, solver entry, 공용 support 코드
- `core/`
  - 기존 `src/master`, `src/pricing`, `src/util`을 내부 패키지로 이관한 실행 엔진

## 현재 상태

- `compare_existing`:
  - CLI 파싱, 배치 orchestration, 리포트 출력이 새 계층으로 분리됨
  - 실행은 `refactor_algorithm.engine` + `refactor_algorithm.core`만 사용
- `inspect_bnp`:
  - 새 CLI 진입점 제공
  - inspection 실행도 `refactor_algorithm` 내부 모듈만 사용

## 실행

리포지토리 루트에서 실행:

```bash
python -m refactor_algorithm.cli.compare_existing --help
python -m refactor_algorithm.cli.inspect_bnp --help
```

## 다음 리팩토링 후보

1. `app/inspection.py`를 더 작게 나눠 recorder / printer / runner 책임 분리
2. `core/master`, `core/pricing` 내부의 순환 참조를 더 줄여서 모듈 경계 명확화
3. `engine/instances.py`, `app/inspection.py`를 더 작은 loader / runner / parser 단위로 분리

## 최근 정리

- `compare_existing`, `inspect_bnp` 모두 `refactor_algorithm` 내부 모듈만 사용
- inspection 지원 코드를 `inspection_records.py`, `inspection_output.py`로 분리
- 코어 마스터 모듈의 불필요한 `sys.path` 보정 제거
