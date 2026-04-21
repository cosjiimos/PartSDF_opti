# GA Chair Design — Colab Quickstart

30초 세팅 가이드. 이 노트북은 비공개 repo에 있어서 GitHub 토큰이 필요합니다.

## 1. Fine-grained PAT 발급 (1회)
1. https://github.com/settings/tokens?type=beta 접속
2. **Generate new token** 클릭
3. **Repository access** → *Only select repositories* → 이 repo만 체크
4. **Repository permissions** → **Contents: Read-only**
5. **Generate token** → 긴 문자열 복사 (한 번만 보여줌!)

## 2. Colab 시크릿 등록 (1회)
1. Colab에서 노트북 열기 → 왼쪽 사이드바의 **🔑 (key) 아이콘** 클릭
2. **+ Add new secret**
3. Name: `GITHUB_TOKEN`
4. Value: 1단계에서 복사한 토큰
5. **Notebook access** 토글 ON

## 3. 노트북 열기
GitHub repo 페이지 → `notebooks/ga_chair_generation/ga_chair_design.ipynb` → 상단의 **Open in Colab** 배지 클릭.

또는 직접:
```
https://colab.research.google.com/github/USER/REPO/blob/main/notebooks/ga_chair_generation/ga_chair_design.ipynb
```
(사설 repo라도 Colab은 토큰 있는 사용자에게 로드해줍니다)

## 4. 실행
- **런타임 > 런타임 유형 변경 > GPU (T4)** 설정
- **런타임 > 모두 실행**
- 끝나면 왼쪽 파일 탭의 `/content/partsdf-ga/notebooks/ga_chair_generation/output/` 에서 OBJ/PNG 다운로드

## 생성물
디자인 랭크마다:
- `ga_rank{K}.png` — 4뷰 스냅샷
- `ga_rank{K}_sources.png` — 파트 출처 콜라주 (어느 체어에서 어느 파트가 왔는지)
- `ga_rank{K}_full.obj` / `_parts.obj` / `_part{P}.obj` — 3D 메쉬
- `ga_rank{K}_info.json` — 유전자 + fitness 상세

## GA 설정 변경
`ga_chair_design.ipynb` 셀 17 (`GA Configuration`):
- `population_size: 30` / `n_generations: 20` — 제대로 된 런
- `10 / 5` — 스모크 테스트 (7초)

## 문제 해결
- `Colab secret GITHUB_TOKEN not found` → 2단계 Notebook access 토글 확인
- `Permission denied` → PAT가 해당 repo에 Contents: Read 권한 있는지 확인
- `CUDA out of memory` → 런타임 > 런타임 재시작
