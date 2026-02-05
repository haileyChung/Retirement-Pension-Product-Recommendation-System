# -*- coding: utf-8 -*-
"""
PCA, Fama-MacBeth, GARCHestimation, Simulation 설정 파일
- 모든 분석 모듈에서 사용하는 경로 및 파라미터 관리
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class AnalysisConfig:
    """PCA + Fama-MacBeth 분석 설정"""

    # ===== 기준 경로 (상대경로 계산용) =====
    _base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _current_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def preprocess_dir(self) -> str:
        """전처리 폴더 경로"""
        return os.path.join(self._base_dir, "2. 데이터전처리")

    @property
    def preprocess_output_dir(self) -> str:
        """전처리 output 폴더"""
        return os.path.join(self.preprocess_dir, "output")

    # ===== 입력 파일 =====
    # 초과수익률 파일 (2번 폴더 output에서 가져옴)
    file_excess_return: str = "상품별일별초과수익률_분배금반영.csv"

    # 상품 정보 파일 (2번 폴더의 상품명.xlsx)
    file_product_info: str = "상품명.xlsx"

    @property
    def excess_return_path(self) -> str:
        """초과수익률 파일 전체 경로"""
        return os.path.join(self.preprocess_output_dir, self.file_excess_return)

    @property
    def product_info_path(self) -> str:
        """상품 정보 파일 전체 경로"""
        return os.path.join(self.preprocess_output_dir, self.file_product_info)

    @property
    def selected_products_path(self) -> str:
        """선별 상품 파일 경로 (상품 정보와 동일)"""
        return self.product_info_path

    # ===== 출력 경로 =====
    @property
    def output_dir(self) -> str:
        """결과 저장 폴더"""
        return os.path.join(self._current_dir, "output")

    # ===== PCA 설정 =====
    pca_min_years: float = 7.0       # PCA 대상 최소 데이터 기간 (년)
    max_factors: int = 30            # 최대 factor 수
    n_factors: int = 10              # 사용할 factor 수
    simulation_factors: int = 10     # 시뮬레이션에 사용할 factor 수

    # ===== Fama-MacBeth 설정 =====
    fm_min_obs: int = 250            # FM 대상 최소 관측치
    min_obs_for_beta: int = 60       # Beta 추정 최소 관측치

    # ===== 캐시 설정 =====
    use_cache: bool = True
    force_recalculate: bool = False

    # ===== 정규화 설정 =====
    normalize_factors: bool = False


@dataclass
class SimulationConfig:
    """t-GJR-GARCH + Monte Carlo 시뮬레이션 설정"""

    # ===== 기준 경로 (상대경로 계산용) =====
    _base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _current_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def preprocess_dir(self) -> str:
        """전처리 폴더 경로"""
        return os.path.join(self._base_dir, "2. 데이터전처리")

    @property
    def preprocess_output_dir(self) -> str:
        """전처리 output 폴더"""
        return os.path.join(self.preprocess_dir, "output")

    # ===== 입력 파일 =====
    # 초과수익률 파일 (2번 폴더 output에서 가져옴)
    file_excess_return: str = "상품별일별초과수익률_분배금반영.csv"

    # 상품 정보 파일 (2번 폴더의 상품명.xlsx)
    file_product_info: str = "상품명.xlsx"

    @property
    def excess_return_path(self) -> str:
        """초과수익률 파일 전체 경로"""
        return os.path.join(self.preprocess_output_dir, self.file_excess_return)

    @property
    def selected_products_path(self) -> str:
        """선별 상품 파일 경로"""
        return os.path.join(self.preprocess_output_dir, self.file_product_info)

    # ===== Part 1 결과 경로 (동일 폴더 output) =====
    @property
    def output_dir(self) -> str:
        """결과 저장 폴더"""
        return os.path.join(self._current_dir, "output")

    # ===== 시뮬레이션 설정 =====
    n_simulations: int = 100000      # 시뮬레이션 횟수
    forecast_days: int = 250         # 예측 기간 (일)
    var_confidence: float = 0.95     # VaR 신뢰수준
    garch_min_obs: int = 250         # GARCH 최소 관측치

    # ===== 캐시 설정 =====
    use_cache: bool = True
    force_recalculate: bool = False

    # ===== GPU 설정 =====
    use_gpu: bool = False            # GPU 사용 여부 (런타임에 자동 감지)
    seed: int = 42                   # 랜덤 시드

    # ===== Beta 유의성 필터링 설정 =====
    filter_insignificant_beta: bool = False
    beta_significance_level: float = 0.05  # 유의수준 (0.05 = 95% 신뢰수준)

    # ===== 상품 타입별 일일 수익률 상하한 =====
    return_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'ETF': (-0.30, 0.30),        # KRX Circuit Breaker
        'REITs': (-0.30, 0.30),
        'FUND': (-0.50, 0.50)        # 장외파생상품 기준
    })

    # ===== 스트레스 테스트 설정 =====
    stress_test_days: int = 23       # 위기 기간 일수 (코로나: 2020-02-20 ~ 2020-03-23)
    save_epsilon_stats: bool = True  # epsilon 통계 저장 여부


# =============================================================================
# 폴더 자동 생성
# =============================================================================

def create_directories():
    """필요한 폴더가 없으면 생성"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"폴더 생성: {output_dir}")


# 모듈 로드 시 폴더 자동 생성
create_directories()
