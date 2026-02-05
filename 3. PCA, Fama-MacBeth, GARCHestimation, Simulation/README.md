# PCA + Fama-MacBeth + GARCH 시뮬레이션

ETF/펀드/REITs 포트폴리오의 기대수익률 추정 및 Monte Carlo 시뮬레이션을 위한 분석 파이프라인입니다.

## 파일 구조

```
3. PCA, Fama-MacBeth, GARCHestimation, Simulation/
├── config.py                              # 설정 파일 (경로, 파라미터)
├── 1_PCA_Fama-MacBeth.py                  # PCA + Fama-MacBeth 분석
├── 2_t-GJR-GARCH_MonteCarloSimulation.py  # GARCH + Monte Carlo 시뮬레이션
├── requirements.txt                        # 의존성 패키지
└── README.md                              # 이 파일
```

## 설치

```bash
pip install -r requirements.txt

# GPU 가속 사용 시 (선택)
pip install cupy-cuda11x  # CUDA 11.x
# 또는
pip install cupy-cuda12x  # CUDA 12.x
```

## 사전 요구사항

이 폴더의 스크립트를 실행하기 전에 `2. 데이터전처리` 폴더의 파이프라인을 먼저 실행해야 합니다.

**필요 파일 (2번 폴더에서 생성):**
- `output/상품별일별초과수익률_분배금반영.csv`
- `상품명.xlsx`

## 사용법

### 1단계: PCA + Fama-MacBeth 분석

```bash
python 1_PCA_Fama-MacBeth.py
```

**처리 내용:**
1. 초과수익률 데이터 로드 및 피벗 테이블 생성
2. 상품 필터링 (PCA: 7년 이상, FM: 1년 이상)
3. PCA 수행 (T x T Covariance 방식)
4. PC 설명력 분석 및 시각화
5. Fama-MacBeth 회귀로 기대수익률 추정

**출력 파일:**
- `output/pca_results.npz`: PCA 결과 (eigenvalues, eigenvectors)
- `output/pca_loadings.csv`: PCA loadings
- `output/factor_returns.csv`: Factor 수익률 시계열
- `output/pc_portfolio_returns.csv`: PC 포트폴리오 수익률
- `output/fama_macbeth_results.csv`: FM 결과 (beta, t_stat, r_hat)
- `output/gamma_estimates.csv`: gamma 추정치
- `output/pc_variance_explained.png`: PC 설명력 그래프

### 2단계: GARCH + Monte Carlo 시뮬레이션

```bash
python 2_t-GJR-GARCH_MonteCarloSimulation.py
```

**처리 내용:**
1. Part 1 결과 로드
2. t-GJR-GARCH 모델로 변동성 추정
3. Monte Carlo 시뮬레이션 (100,000 paths)
4. VaR/CVaR 산출
5. 스트레스 테스트

**출력 파일:**
- `output/simulation_results.csv`: 시뮬레이션 결과 요약
- `output/var_cvar_results.csv`: VaR/CVaR 결과
- `output/garch_params/`: GARCH 파라미터 (상품별)
- `output/simulations/`: 시뮬레이션 경로 (상품별)

## 설정 파일 (config.py)

### AnalysisConfig (PCA + Fama-MacBeth)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `pca_min_years` | 7.0 | PCA 대상 최소 데이터 기간 (년) |
| `max_factors` | 30 | 최대 factor 수 |
| `n_factors` | 10 | 사용할 factor 수 |
| `fm_min_obs` | 250 | FM 대상 최소 관측치 |
| `min_obs_for_beta` | 60 | Beta 추정 최소 관측치 |

### SimulationConfig (GARCH + Monte Carlo)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `n_simulations` | 100,000 | 시뮬레이션 횟수 |
| `forecast_days` | 250 | 예측 기간 (일) |
| `var_confidence` | 0.95 | VaR 신뢰수준 |
| `garch_min_obs` | 250 | GARCH 최소 관측치 |
| `filter_insignificant_beta` | False | 유의하지 않은 beta 필터링 |

## 주요 클래스

### 1_PCA_Fama-MacBeth.py

- `DataLoader`: 데이터 로드 및 전처리
- `CommonPeriodPCA`: PCA 수행 (T x T Covariance)
- `FamaMacBethRegression`: Fama-MacBeth 회귀분석
- `ProgressTracker`: 진행률 표시

### 2_t-GJR-GARCH_MonteCarloSimulation.py

- `Part1ResultLoader`: Part 1 결과 로드
- `tGJRGARCH`: t-GJR-GARCH 모델
- `MonteCarloSimulator`: Monte Carlo 시뮬레이션
- `VaRCalculator`: VaR/CVaR 계산
- `StressTestRunner`: 스트레스 테스트

## 참고사항

- **수익률**: 단순수익률 (로그수익률 X)
- **연율화**: r_daily × 250 (단리)
- **누적수익률**: prod(1+r) - 1
- **PCA 방식**: T × T Covariance Matrix (demean, 분산 정규화 X)
- **gamma_0**: 0으로 고정 (초과수익률 사용)
- **GPU 가속**: CuPy 설치 시 자동 감지

## 폴더 연결 구조

```
1. 데이터크롤링/
    └── output/
         └── (가격, 배당 데이터)
              ↓
2. 데이터전처리/
    └── output/
         └── 상품별일별초과수익률_분배금반영.csv
              ↓
3. PCA, Fama-MacBeth, GARCHestimation, Simulation/  ← 현재 폴더
    └── output/
         └── (PCA, FM, 시뮬레이션 결과)
```
