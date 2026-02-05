# %% [0] t-GJR-GARCH + Monte Carlo 시뮬레이션 (Part 2)
#
# 파이프라인:
#   1. Part 1 결과 로드 (PCA, FM)
#   2. t-GJR-GARCH 모델로 변동성 추정
#   3. Monte Carlo 시뮬레이션으로 수익률 분포 생성
#   4. VaR/CVaR 산출
#
# 주요 기능:
#   - GPU 가속 (CuPy, NumPy API 호환)
#   - 벡터화된 시뮬레이션 (100,000 paths 동시 처리)
#   - Beta 유의성 필터링 옵션
#   - 상품 타입별 Winsorization (KRX 가격제한폭 기준)
#
# 수익률 처리:
#   - 입력: 단순수익률 (일반수익률)
#   - 누적수익률: prod(1+r) - 1
#   - 연율화: r_daily x 250 (단리)
#
# 필요 파일 (Part 1에서 생성):
#   - pca_results.npz
#   - factor_returns.csv
#   - fama_macbeth_results.csv
#   - gamma_estimates.csv
#

# %% [1] 라이브러리 임포트 및 GPU 설정

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
import time
import warnings
import os

warnings.filterwarnings('ignore')

# 설정 파일 임포트
from config import SimulationConfig

# 1-1. GPU 가속 설정 (CuPy)
USE_GPU = False
cp = None  # CuPy 모듈 (GPU 사용 시)
xp = np    # 배열 라이브러리 (GPU: cupy, CPU: numpy)

try:
    import cupy as cp_module
    cp = cp_module
    if cp.cuda.is_available():
        USE_GPU = True
        xp = cp
        # GPU 정보 출력
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        total_mem = props['totalGlobalMem'] / 1e9
        print(f"[GPU 감지] {props['name'].decode()}")
        print(f"  - CUDA Compute: {props['major']}.{props['minor']}")
        print(f"  - 메모리: {total_mem:.1f} GB")
    else:
        print("[GPU 미감지] CPU 모드로 실행")
except ImportError:
    print("[CuPy 미설치] CPU 모드로 실행")
    print("  GPU 사용하려면: pip install cupy-cuda11x 또는 cupy-cuda12x")

# 1-2. 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# %% [1-1] 공통 유틸리티 함수

def normalize_code(code) -> str:
    """상품코드 정규화"""
    code_str = str(code).strip()
    if code_str.isdigit():
        return code_str.zfill(6)
    return code_str


# %% [2] 진행률 표시 유틸리티

class ProgressTracker:
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.last_print_time = 0
    
    def update(self, n: int = 1) -> None:
        self.current += n
        current_time = time.time()
        
        if current_time - self.last_print_time >= 0.5 or self.current == self.total:
            self._print_progress()
            self.last_print_time = current_time
    
    def _print_progress(self) -> None:
        elapsed = time.time() - self.start_time
        progress = self.current / self.total
        
        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            remaining = 0
        
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "=" * filled + "-" * (bar_length - filled)
        
        print(f"\r  {self.desc}: [{bar}] {progress*100:5.1f}% "
              f"({self.current}/{self.total}) "
              f"경과: {self._format_time(elapsed)} "
              f"남은시간: {self._format_time(remaining)}    ", end="", flush=True)
        
        if self.current == self.total:
            print()
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}초"
        elif seconds < 3600:
            return f"{seconds/60:.1f}분"
        else:
            return f"{seconds/3600:.1f}시간"


# %% [4] Part 1 결과 로더

class Part1ResultLoader:
    """Part 1 (PCA + FM) 결과 로드"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.output_dir = config.output_dir
        
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.pca_products: Optional[List[str]] = None
        self.factor_returns: Optional[pd.DataFrame] = None
        self.fm_results: Optional[pd.DataFrame] = None
        self.gamma_estimates: Optional[pd.DataFrame] = None
        self.avg_gamma: Optional[pd.Series] = None
    
    def load_all(self) -> bool:
        """모든 Part 1 결과 로드"""
        print("\n" + "="*70)
        print("[1] Part 1 결과 로드")
        print("="*70)
        
        try:
            # 4-1. PCA 결과
            pca_path = f"{self.output_dir}\\pca_results.npz"
            if not os.path.exists(pca_path):
                print(f"  오류: {pca_path} 파일이 없습니다.")
                print("  Part 1 (pca_fama_macbeth.py)을 먼저 실행하세요.")
                return False
            
            pca_data = np.load(pca_path, allow_pickle=True)
            self.eigenvalues = pca_data['eigenvalues']
            self.eigenvectors = pca_data['eigenvectors']
            self.pca_products = pca_data['pca_products'].tolist()
            print(f"  PCA 결과 로드: {len(self.pca_products)}개 상품, {len(self.eigenvalues)}개 PC")
            
            # 4-2. Factor Returns
            factor_path = f"{self.output_dir}\\factor_returns.csv"
            if not os.path.exists(factor_path):
                print(f"  오류: {factor_path} 파일이 없습니다.")
                return False
            
            self.factor_returns = pd.read_csv(factor_path, index_col=0, parse_dates=True)
            print(f"  Factor Returns 로드: {self.factor_returns.shape}")
            
            # 4-3. FM 결과
            fm_path = f"{self.output_dir}\\fama_macbeth_results.csv"
            if not os.path.exists(fm_path):
                print(f"  오류: {fm_path} 파일이 없습니다.")
                return False
            
            self.fm_results = pd.read_csv(fm_path, encoding='utf-8-sig')
            self.fm_results['상품코드'] = self.fm_results['상품코드'].apply(normalize_code)
            print(f"  FM 결과 로드: {len(self.fm_results)}개 상품")
            
            # 4-4. gamma 추정치
            gamma_path = f"{self.output_dir}\\gamma_estimates.csv"
            if not os.path.exists(gamma_path):
                print(f"  오류: {gamma_path} 파일이 없습니다.")
                return False
            
            self.gamma_estimates = pd.read_csv(gamma_path, encoding='utf-8-sig')
            
            gamma_dict = {}
            for _, row in self.gamma_estimates.iterrows():
                pc = row['PC']
                if pc == 'gamma_0':
                    gamma_dict['gamma_0'] = row['gamma_daily']
                else:
                    gamma_dict[f'gamma_{pc}'] = row['gamma_daily']
            self.avg_gamma = pd.Series(gamma_dict)
            print(f"  gamma 추정치 로드: {len(self.avg_gamma)}개")
            # 단리 연율화: r_daily x 250
            gamma_0_annual = self.avg_gamma['gamma_0'] * 250 * 100
            print(f"  gamma_0: {gamma_0_annual:.4f}% (연율화)")
            
            print("\n  Part 1 결과 로드 완료")
            return True
            
        except Exception as e:
            print(f"  오류 발생: {e}")
            return False
    
    def get_betas_for_simulation(self) -> pd.DataFrame:
        """시뮬레이션용 beta DataFrame"""
        if self.fm_results is None:
            raise ValueError("FM 결과가 로드되지 않았습니다.")
        
        betas_df = self.fm_results.set_index('상품코드')
        beta_cols = [col for col in betas_df.columns if col.startswith('beta_')]
        betas_for_sim = betas_df[beta_cols].copy()
        rename_map = {col: col.replace('beta_', '') for col in beta_cols}
        betas_for_sim = betas_for_sim.rename(columns=rename_map)
        
        return betas_for_sim
    
    def get_beta_t_stats(self) -> pd.DataFrame:
        """시뮬레이션용 beta t-stat DataFrame"""
        if self.fm_results is None:
            raise ValueError("FM 결과가 로드되지 않았습니다.")
        
        betas_df = self.fm_results.set_index('상품코드')
        
        # t_stat 컬럼 찾기 (t_stat_PC1, t_stat_PC2, ... 형식)
        t_stat_cols = [col for col in betas_df.columns if col.startswith('t_stat_')]
        
        if len(t_stat_cols) == 0:
            print("  경고: FM 결과에 t_stat 컬럼이 없습니다.")
            print("         beta 유의성 필터링을 사용하려면 Part 1을 다시 실행하세요.")
            return None
        
        t_stats_for_sim = betas_df[t_stat_cols].copy()
        rename_map = {col: col.replace('t_stat_', '') for col in t_stat_cols}
        t_stats_for_sim = t_stats_for_sim.rename(columns=rename_map)
        
        return t_stats_for_sim
    
    def get_ts_alphas(self) -> pd.Series:
        """ts_alpha Series"""
        if self.fm_results is None:
            raise ValueError("FM 결과가 로드되지 않았습니다.")
        
        return self.fm_results.set_index('상품코드')['ts_alpha']
    
    def get_product_stds(self) -> pd.Series:
        """상품별 표준편차 Series (beta 정규화용)"""
        if self.fm_results is None:
            raise ValueError("FM 결과가 로드되지 않았습니다.")
        
        if 'std' not in self.fm_results.columns:
            print("  경고: FM 결과에 std 컬럼이 없습니다. Part 1을 다시 실행하세요.")
            return None
        
        return self.fm_results.set_index('상품코드')['std']


# %% [5] t-GJR-GARCH 모델 클래스

class tGJRGARCH:
    """
    t-GJR-GARCH(1,1) 모델
    
    개선사항:
      - 초기값 처리: 마지막 관측 return + 분산 모두 사용
      - t=0부터 분산 재귀 수행
    """
    
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.T = len(returns)
        
        self.mu: float = 0.0
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.gamma: float = 0.0
        self.beta: float = 0.0
        self.nu: float = 10.0
        
        self.sigma2: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.fitted: bool = False
    
    def _compute_variance_series(self, data: np.ndarray, omega: float, 
                                   alpha: float, beta: float, gamma: float) -> np.ndarray:
        """분산 시계열 계산"""
        T = len(data)
        H = np.zeros(T)
        H[0] = np.var(data)
        
        for t in range(1, T):
            indicator = 1.0 if data[t-1] < 0 else 0.0
            H[t] = omega + alpha * data[t-1]**2 + gamma * data[t-1]**2 * indicator + beta * H[t-1]
            H[t] = max(H[t], 1e-10)
        
        return H
    
    def _neg_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """음의 로그 우도"""
        omega, alpha, beta, gamma, nu = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
            return 1e10
        if alpha + 0.5*gamma + beta >= 1:
            return 1e10
        if nu <= 2.01:
            return 1e10
        
        H = self._compute_variance_series(data, omega, alpha, beta, gamma)
        
        const = gammaln((nu + 1.) / 2.) - gammaln(nu / 2.) - 0.5 * np.log((nu - 2.) * np.pi)
        std_resid_sq = (data ** 2) / H
        loglik = const - 0.5 * np.log(H) - ((nu + 1) / 2) * np.log(1 + std_resid_sq / (nu - 2))
        
        return -np.sum(loglik)
    
    def fit(self, verbose: bool = False) -> bool:
        """모델 추정"""
        data = self.returns - np.mean(self.returns)
        self.mu = np.mean(self.returns)
        
        sample_var = np.var(data)
        init_candidates = [
            [sample_var * 0.05, 0.05, 0.90, 0.05, 8.0],
            [sample_var * 0.10, 0.08, 0.85, 0.08, 6.0],
            [sample_var * 0.02, 0.03, 0.94, 0.03, 10.0],
        ]
        
        bounds = [
            (1e-10, sample_var * 10),
            (1e-6, 0.5),
            (0.5, 0.999),
            (0.0, 0.5),
            (2.1, 50.0)
        ]
        
        best_result = None
        best_lik = np.inf
        
        for init_params in init_candidates:
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    init_params,
                    args=(data,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'disp': False}
                )
                
                if result.fun < best_lik:
                    best_lik = result.fun
                    best_result = result
                    
            except Exception:
                continue
        
        if best_result is not None and best_result.fun < 1e9:
            self.omega = best_result.x[0]
            self.alpha = best_result.x[1]
            self.beta = best_result.x[2]
            self.gamma = best_result.x[3]
            self.nu = best_result.x[4]
            
            self.sigma2 = self._compute_variance_series(data, self.omega, self.alpha, 
                                                         self.beta, self.gamma)
            self.residuals = data
            self.fitted = True
            
            return True
        else:
            return False
    
    def simulate(self, n_paths: int, horizon: int, seed: int = None) -> np.ndarray:
        """
        벡터화된 시뮬레이션 (CPU)
        
        개선사항:
          - 초기값: 마지막 관측 return + 분산 모두 사용
          - t=0부터 분산 재귀 수행
        """
        if not self.fitted:
            raise ValueError("모델이 추정되지 않았습니다.")
        
        if seed is not None:
            np.random.seed(seed)
        
        # 5-1. 난수 미리 생성 (정규화된 t-분포)
        t_scale = np.sqrt(self.nu / (self.nu - 2)) if self.nu > 2 else 1.0
        z_all = stats.t.rvs(df=self.nu, size=(n_paths, horizon)) / t_scale
        
        # 5-2. 결과 배열
        simulated_returns = np.zeros((n_paths, horizon), dtype=np.float32)
        
        # 5-3. 초기값 (마지막 관측값 사용) - 개선된 부분
        last_ret = self.residuals[-1]
        last_sigma2 = self.sigma2[-1]
        
        ret_t = np.full(n_paths, last_ret, dtype=np.float32)
        sigma2_t = np.full(n_paths, last_sigma2, dtype=np.float32)
        
        # 5-4. t=0부터 시뮬레이션 - 개선된 부분
        for t in range(horizon):
            # 분산 업데이트 (이전 수익률 기반)
            indicator = (ret_t < 0).astype(np.float32)
            sigma2_t = (self.omega + 
                       self.alpha * ret_t**2 + 
                       self.gamma * ret_t**2 * indicator + 
                       self.beta * sigma2_t)
            sigma2_t = np.maximum(sigma2_t, 1e-10)
            
            # 새 수익률 생성
            ret_t = np.sqrt(sigma2_t) * z_all[:, t]
            simulated_returns[:, t] = self.mu + ret_t
        
        return simulated_returns


# %% [6] CuPy 기반 GPU 시뮬레이터

class CuPySimulator:
    """
    CuPy 기반 GPU 가속 시뮬레이터
    
    특징:
      - NumPy API 호환으로 코드 단순화
      - 초기값 처리 개선 (마지막 return + 분산)
      - CPU 폴백 지원
    """
    
    def __init__(self, use_gpu: bool = USE_GPU):
        self.use_gpu = use_gpu and (cp is not None)
        self.xp = cp if self.use_gpu else np
    
    def simulate_garch(self, omega: float, alpha: float, beta: float, 
                       gamma_garch: float, nu: float, mu: float,
                       last_ret: float, last_sigma2: float,
                       n_paths: int, horizon: int, seed: int = None) -> np.ndarray:
        """
        t-GJR-GARCH 시뮬레이션 (GPU/CPU)
        
        Parameters
        ----------
        omega, alpha, beta, gamma_garch, nu : float
            GARCH 파라미터
        mu : float
            평균 수익률
        last_ret : float
            마지막 관측 수익률 (demeaned)
        last_sigma2 : float
            마지막 관측 분산
        n_paths : int
            시뮬레이션 경로 수
        horizon : int
            예측 기간 (일)
        seed : int, optional
            난수 시드
        
        Returns
        -------
        np.ndarray : shape (n_paths, horizon)
            시뮬레이션된 수익률
        """
        xp = self.xp
        
        # 6-1. CPU에서 난수 생성 (재현성 보장)
        if seed is not None:
            np.random.seed(seed)
        
        t_scale = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0
        z_cpu = np.random.standard_t(df=nu, size=(n_paths, horizon)).astype(np.float32)
        z_cpu = z_cpu / t_scale
        
        # 6-2. GPU로 전송 (GPU 모드인 경우)
        if self.use_gpu:
            z_all = cp.asarray(z_cpu)
            del z_cpu
        else:
            z_all = z_cpu
        
        # 6-3. 결과 배열 초기화
        simulated_returns = xp.zeros((n_paths, horizon), dtype=xp.float32)
        
        # 6-4. 초기값 설정 (마지막 관측값)
        ret_t = xp.full(n_paths, last_ret, dtype=xp.float32)
        sigma2_t = xp.full(n_paths, last_sigma2, dtype=xp.float32)
        
        # 6-5. t=0부터 시뮬레이션
        for t in range(horizon):
            # 분산 업데이트 (GJR-GARCH)
            indicator = (ret_t < 0).astype(xp.float32)
            sigma2_t = (omega + 
                       alpha * ret_t**2 + 
                       gamma_garch * ret_t**2 * indicator + 
                       beta * sigma2_t)
            sigma2_t = xp.maximum(sigma2_t, 1e-10)
            
            # 수익률 생성
            ret_t = xp.sqrt(sigma2_t) * z_all[:, t]
            simulated_returns[:, t] = mu + ret_t
        
        # 6-6. CPU로 반환
        if self.use_gpu:
            result = cp.asnumpy(simulated_returns)
            # GPU 메모리 정리
            del simulated_returns, z_all, ret_t, sigma2_t
            cp.get_default_memory_pool().free_all_blocks()
            return result
        else:
            return simulated_returns


# %% [7] Monte Carlo 시뮬레이션 클래스

class MonteCarloSimulator:
    """Factor 기반 t-GJR-GARCH Monte Carlo 시뮬레이션"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.n_simulations = config.n_simulations
        self.forecast_days = config.forecast_days
        
        # 7-1. GPU 시뮬레이터
        self.gpu_sim = CuPySimulator(use_gpu=config.use_gpu)
        
        # 7-2. Factor 관련
        self.factor_returns: Optional[pd.DataFrame] = None
        self.factor_garch_models: Dict[str, tGJRGARCH] = {}
        self.simulated_factors: Optional[np.ndarray] = None
        
        # 7-3. Fama-MacBeth gamma
        self.gamma_0: float = 0.0  # 상수항 (추정값)
        self.gamma_factors: Optional[np.ndarray] = None
        
        # 7-4. 상품별 관련
        self.betas: Optional[pd.DataFrame] = None
        self.beta_t_stats: Optional[pd.DataFrame] = None  # beta t-stat (유의성 검정용)
        self.ts_alphas: Optional[pd.Series] = None
        self.product_stds: Optional[pd.Series] = None
        self.residuals: Dict[str, np.ndarray] = {}
        self.residual_garch_models: Dict[str, tGJRGARCH] = {}
        self.historical_returns: Optional[pd.DataFrame] = None
        
        # 7-5. 최종 결과
        self.simulated_paths: Dict[str, np.ndarray] = {}
        self.product_var: Dict[str, float] = {}
        self.product_cvar: Dict[str, float] = {}
        self.product_r_hat: Dict[str, float] = {}
        self.product_names: Dict[str, str] = {}
        self.product_types: Dict[str, str] = {}  # 상품 타입 (ETF/REITs/FUND)

        # 7-6. Beta 필터링 통계
        self.beta_filter_stats: Dict[str, Dict] = {}

        # 7-7. Winsorization 상세 로그
        # 각 항목: (상품코드, 상품명, 원래값, 대체값, 방향)
        self.winsorization_logs: List[Dict] = []
        self.winsorization_stats: Dict[str, Dict] = {}  # 상품별 통계

        # 7-8. 스트레스 테스트용 epsilon 통계
        # 각 상품별로 epsilon 23일 합의 5th percentile 저장
        self.epsilon_stress_stats: Dict[str, Dict] = {}

    def set_product_names(self, product_names: Dict[str, str]) -> None:
        """상품명 설정"""
        self.product_names = product_names

    def set_product_types(self, product_types: Dict[str, str]) -> None:
        """상품 타입 설정 (ETF/REITs/FUND)"""
        self.product_types = product_types
    
    def set_factor_data(self, factor_returns: pd.DataFrame, 
                        betas: pd.DataFrame, ts_alphas: pd.Series,
                        avg_gamma: pd.Series, product_stds: pd.Series = None,
                        beta_t_stats: pd.DataFrame = None) -> None:
        """Factor 데이터 설정
        
        Args:
            factor_returns: Factor 수익률 데이터
            betas: 상품별 ts_beta
            ts_alphas: 상품별 ts_alpha
            avg_gamma: gamma 추정치 (gamma_0 = 0 고정)
            product_stds: 상품별 표준편차 (참고용)
            beta_t_stats: 상품별 beta t-stat (유의성 필터링용)
        
        Covariance PCA (demean 방식):
            - ts_beta ~ eigenvector
            - r_hat = ts_beta x gamma (gamma_0 = 0 고정)
        """
        self.factor_returns = factor_returns
        self.betas = betas
        self.ts_alphas = ts_alphas
        self.product_stds = product_stds
        self.beta_t_stats = beta_t_stats
        
        # gamma_0 = 0 (초과수익률이므로 고정)
        self.gamma_0 = avg_gamma['gamma_0']  # = 0
        factor_names = factor_returns.columns.tolist()
        self.gamma_factors = np.array([avg_gamma[f'gamma_{fn}'] for fn in factor_names])
        
        # r_hat = ts_beta x gamma (gamma_0 = 0)
        r_hat_list = []
        for product in betas.index:
            beta = betas.loc[product].values
            r_hat = self.gamma_0 + np.dot(beta, self.gamma_factors)
            r_hat_list.append(r_hat)
        # 단리 연율화: r_daily x 250
        r_hat_mean_annual = np.mean(r_hat_list) * 250 * 100
        
        print(f"\n  Factor 데이터 설정 완료")
        print(f"    - Factor 수: {factor_returns.shape[1]}개")
        print(f"    - 관측치 수: {len(factor_returns)}일")
        print(f"    - Beta 추정 상품: {len(betas)}개")
        print(f"    - gamma_0: {self.gamma_0:.6f} (초과수익률 -> 0 고정)")
        print(f"    - ts_alpha 평균: {ts_alphas.mean()*250*100:.2f}% (연율화)")
        print(f"    - r_hat (beta x gamma) 평균: {r_hat_mean_annual:.2f}% (연율화)")
        
        # beta t-stat 정보 출력
        if beta_t_stats is not None:
            print(f"    - Beta t-stat 로드됨: 유의성 필터링 가능")
        else:
            print(f"    - Beta t-stat 없음: 유의성 필터링 불가")
    
    def _filter_beta_by_significance(self, product: str, beta: np.ndarray, 
                                      factor_names: List[str]) -> np.ndarray:
        """
        Beta 유의성 필터링
        
        유의하지 않은 beta (|t-stat| < critical value)를 0으로 처리
        
        Args:
            product: 상품코드
            beta: 원본 beta 배열
            factor_names: Factor 이름 리스트
        
        Returns:
            필터링된 beta 배열
        """
        if not self.config.filter_insignificant_beta:
            return beta
        
        if self.beta_t_stats is None:
            return beta
        
        if product not in self.beta_t_stats.index:
            return beta
        
        # 임계값 계산 (양측 검정)
        # 95% 신뢰수준 -> alpha = 0.05 -> critical value ~= 1.96
        alpha = self.config.beta_significance_level
        critical_value = stats.norm.ppf(1 - alpha/2)  # 양측 검정
        
        # 필터링
        filtered_beta = beta.copy()
        t_stats = self.beta_t_stats.loc[product, factor_names].values
        
        significant_count = 0
        filtered_count = 0
        
        for i, (b, t) in enumerate(zip(beta, t_stats)):
            if np.isnan(t) or np.abs(t) < critical_value:
                filtered_beta[i] = 0.0
                filtered_count += 1
            else:
                significant_count += 1
        
        # 통계 저장
        self.beta_filter_stats[product] = {
            'total': len(beta),
            'significant': significant_count,
            'filtered': filtered_count,
            'filter_rate': filtered_count / len(beta) * 100
        }
        
        return filtered_beta
    
    def fit_factor_garch(self, min_obs: int = 250, output_dir: str = None, 
                         use_cache: bool = True) -> None:
        """Factor별 t-GJR-GARCH 추정 (캐시 지원)"""
        print("\n" + "="*70)
        print("[2] Factor별 t-GJR-GARCH 추정")
        print("="*70)
        
        if self.factor_returns is None:
            print("  Factor 데이터가 설정되지 않았습니다.")
            return
        
        # 2-1. 캐시 확인
        cache_path = f"{output_dir}\\factor_garch_parameters.csv" if output_dir else None
        if use_cache and cache_path and os.path.exists(cache_path):
            print(f"  캐시 발견: {cache_path}")
            cached_df = pd.read_csv(cache_path)
            
            # 캐시에서 모델 복원
            for _, row in cached_df.iterrows():
                factor_name = row['factor']
                if factor_name not in self.factor_returns.columns:
                    continue
                
                factor_data = self.factor_returns[factor_name].dropna().values
                model = tGJRGARCH(factor_data)
                model.mu = row['mu']
                model.omega = row['omega']
                model.alpha = row['alpha']
                model.gamma = row['gamma']
                model.beta = row['beta']
                model.nu = row['nu']
                model.sigma2 = model._compute_variance_series(
                    factor_data - model.mu, model.omega, model.alpha, model.beta, model.gamma
                )
                model.residuals = factor_data - model.mu
                model.fitted = True
                self.factor_garch_models[factor_name] = model
            
            print(f"  캐시에서 로드 완료: {len(self.factor_garch_models)}개 Factor")
            self._print_factor_garch_params()
            return
        
        # 2-2. 새로 추정
        n_factors = self.factor_returns.shape[1]
        progress = ProgressTracker(n_factors, "Factor GARCH")
        success_count = 0
        
        for col in self.factor_returns.columns:
            factor_data = self.factor_returns[col].dropna().values
            
            if len(factor_data) < min_obs:
                progress.update(1)
                continue
            
            model = tGJRGARCH(factor_data)
            success = model.fit(verbose=False)
            
            if success:
                self.factor_garch_models[col] = model
                success_count += 1
            
            progress.update(1)
        
        print(f"\n  Factor GARCH 추정 완료: {success_count}/{n_factors}개")
        self._print_factor_garch_params()
    
    def _print_factor_garch_params(self) -> None:
        """Factor GARCH 파라미터 출력"""
        print(f"\n  [Factor GARCH 파라미터]")
        print(f"  {'Factor':<8} {'alpha':<8} {'gamma':<8} {'beta':<8} {'nu':<8} {'persist':<8}")
        print("-"*55)
        for name, model in self.factor_garch_models.items():
            persistence = model.alpha + 0.5*model.gamma + model.beta
            print(f"  {name:<8} {model.alpha:<8.4f} {model.gamma:<8.4f} "
                  f"{model.beta:<8.4f} {model.nu:<8.2f} {persistence:<8.4f}")
    
    def compute_residuals(self, returns_pivot: pd.DataFrame, products: List[str]) -> None:
        """상품별 잔차 계산"""
        print("\n" + "="*70)
        print("[3] 상품별 잔차 계산")
        print("="*70)
        
        if self.betas is None or self.factor_returns is None:
            print("  Beta 또는 Factor 데이터가 없습니다.")
            return
        
        common_dates = returns_pivot.index.intersection(self.factor_returns.index)
        factor_cols = list(self.factor_garch_models.keys())
        
        # 과거 수익률 저장 (그래프용)
        self.historical_returns = returns_pivot.loc[common_dates].copy()
        
        F = self.factor_returns.loc[common_dates, factor_cols].values
        
        progress = ProgressTracker(len(products), "잔차 계산")
        computed = 0
        
        for product in products:
            if product not in returns_pivot.columns:
                progress.update(1)
                continue
            if product not in self.betas.index:
                progress.update(1)
                continue
            
            ts_alpha = 0.0
            if self.ts_alphas is not None and product in self.ts_alphas.index:
                ts_alpha = self.ts_alphas.loc[product]
            
            r = returns_pivot.loc[common_dates, product].values
            beta = self.betas.loc[product, factor_cols].values
            
            # 잔차 계산에서는 beta 필터링 적용 안함 (원본 beta 사용)
            # 시뮬레이션에서만 필터링 적용
            systematic = F @ beta
            residual = r - ts_alpha - systematic
            
            valid_mask = ~np.isnan(residual)
            self.residuals[product] = residual[valid_mask]
            computed += 1
            progress.update(1)
        
        print(f"\n  잔차 계산 완료: {computed}개 상품")
    
    def fit_residual_garch(self, products: List[str], min_obs: int = 250,
                            output_dir: str = None, use_cache: bool = True) -> None:
        """상품별 잔차 t-GJR-GARCH 추정 (캐시 지원)"""
        print("\n" + "="*70)
        print("[4] 상품별 잔차 t-GJR-GARCH 추정")
        print("="*70)
        
        if len(self.residuals) == 0:
            print("  잔차 데이터가 없습니다.")
            return
        
        valid_products = [p for p in products if p in self.residuals]
        n_products = len(valid_products)
        
        if n_products == 0:
            print("  유효한 상품이 없습니다.")
            return
        
        # 4-1. 캐시 확인
        cache_path = f"{output_dir}\\residual_garch_parameters.csv" if output_dir else None
        if use_cache and cache_path and os.path.exists(cache_path):
            print(f"  캐시 발견: {cache_path}")
            cached_df = pd.read_csv(cache_path)
            cached_df['상품코드'] = cached_df['상품코드'].apply(normalize_code)
            
            loaded_count = 0
            for _, row in cached_df.iterrows():
                product = row['상품코드']
                if product not in self.residuals:
                    continue
                
                residual = self.residuals[product]
                model = tGJRGARCH(residual)
                model.mu = row['mu']
                model.omega = row['omega']
                model.alpha = row['alpha']
                model.gamma = row['gamma']
                model.beta = row['beta']
                model.nu = row['nu']
                model.sigma2 = model._compute_variance_series(
                    residual - model.mu, model.omega, model.alpha, model.beta, model.gamma
                )
                model.residuals = residual - model.mu
                model.fitted = True
                self.residual_garch_models[product] = model
                loaded_count += 1
            
            print(f"  캐시에서 로드 완료: {loaded_count}개 상품")
            return
        
        # 4-2. 새로 추정
        progress = ProgressTracker(n_products, "잔차 GARCH")
        success_count = 0
        
        for product in valid_products:
            residual = self.residuals[product]
            
            if len(residual) < min_obs:
                progress.update(1)
                continue
            
            model = tGJRGARCH(residual)
            success = model.fit(verbose=False)
            
            if success:
                self.residual_garch_models[product] = model
                success_count += 1
            
            progress.update(1)
        
        print(f"\n  잔차 GARCH 추정 완료: {success_count}/{n_products}개 상품")
    
    def simulate_factors(self, seed: int = None) -> np.ndarray:
        """Factor 시뮬레이션"""
        print("\n" + "="*70)
        print("[5] Factor 시뮬레이션")
        print("="*70)
        
        if len(self.factor_garch_models) == 0:
            print("  Factor GARCH 모델이 없습니다.")
            return None
        
        n_factors = len(self.factor_garch_models)
        factor_names = list(self.factor_garch_models.keys())
        
        self.simulated_factors = np.zeros((self.n_simulations, self.forecast_days, n_factors), 
                                          dtype=np.float32)
        
        print(f"  시뮬레이션: {self.n_simulations:,}회 x {self.forecast_days}일 x {n_factors} factors")
        print(f"  모드: {'GPU (CuPy)' if self.gpu_sim.use_gpu else 'CPU (NumPy)'}")
        
        start_time = time.time()
        
        for i, name in enumerate(factor_names):
            model = self.factor_garch_models[name]
            
            # 시드 생성 (Factor별 고유)
            factor_seed = (seed + i * 10000) if seed else None
            
            paths = self.gpu_sim.simulate_garch(
                omega=model.omega,
                alpha=model.alpha,
                beta=model.beta,
                gamma_garch=model.gamma,
                nu=model.nu,
                mu=0.0,  # Factor는 demeaned 상태 유지 (E[F_sim]=0)
                last_ret=model.residuals[-1],
                last_sigma2=model.sigma2[-1],
                n_paths=self.n_simulations,
                horizon=self.forecast_days,
                seed=factor_seed
            )
            
            self.simulated_factors[:, :, i] = paths
        
        elapsed = time.time() - start_time
        print(f"\n  Factor 시뮬레이션 완료 ({elapsed:.1f}초)")
        
        return self.simulated_factors
    
    def simulate_products(self, seed: int = None, output_dir: str = None) -> None:
        """
        상품별 수익률 시뮬레이션 (메모리 효율화: 즉시 저장)

        처리 내용:
          - 단순수익률 누적: prod(1+r) - 1
          - r_hat 연율화: r_daily x 250 (단리)
          - Beta 유의성 필터링 옵션 적용
          - 잔차 시뮬레이션에서 mu=0 고정 (변동성만 사용)
        """
        print("\n" + "="*70)
        print("[6] 상품별 수익률 시뮬레이션")
        print("="*70)
        
        if self.simulated_factors is None:
            print("  Factor 시뮬레이션을 먼저 실행하세요.")
            return
        
        if len(self.residual_garch_models) == 0:
            print("  잔차 GARCH 모델이 없습니다.")
            return
        
        if self.gamma_factors is None:
            print("  gamma가 설정되지 않았습니다.")
            return
        
        factor_names = list(self.factor_garch_models.keys())
        n_products = len(self.residual_garch_models)
        
        print(f"  시뮬레이션 대상: {n_products}개 상품")
        print(f"  구조: r = r_hat + beta x F + epsilon  (r_hat = gamma_0 + beta x gamma)")
        print(f"  누적수익률: 단순수익률 -> prod(1+r) - 1")
        print(f"  잔차: mu=0 고정 (변동성 구조만 사용)")
        print(f"  극단치 처리: Winsorization (KRX 가격제한폭 +-50%)")
        print(f"  모드: {'GPU (CuPy)' if self.gpu_sim.use_gpu else 'CPU (NumPy)'}")
        
        # Beta 필터링 설정 출력
        if self.config.filter_insignificant_beta:
            alpha = self.config.beta_significance_level
            critical_value = stats.norm.ppf(1 - alpha/2)
            print(f"\n  [Beta 유의성 필터링: 활성화]")
            print(f"    유의수준: {alpha*100:.0f}% (|t| >= {critical_value:.2f})")
            if self.beta_t_stats is None:
                print(f"    경고: t-stat 데이터 없음 -> 필터링 불가")
        else:
            print(f"\n  [Beta 유의성 필터링: 비활성화]")
        
        # 저장 폴더 생성
        if output_dir:
            sim_dir = f"{output_dir}\\simulations"
            os.makedirs(sim_dir, exist_ok=True)
            print(f"  저장 경로: {sim_dir}")
        
        start_time = time.time()
        progress = ProgressTracker(n_products, "상품 시뮬레이션")
        
        product_list = list(self.residual_garch_models.keys())
        
        # 통계 저장용 리스트
        self._product_stats = []
        
        # Winsorization 통계 저장용
        total_clipped_count = 0
        total_simulated_count = 0
        
        # Winsorization 상세 로그 초기화
        self.winsorization_logs = []
        self.winsorization_stats = {}
        
        # 샘플 로그 저장 설정 (상품당 최대 5개 극단치만 기록)
        MAX_SAMPLES_PER_PRODUCT = 5
        
        for idx, product in enumerate(product_list):
            if product not in self.betas.index:
                progress.update(1)
                continue
            
            # 원본 beta 가져오기
            beta_original = self.betas.loc[product, factor_names].values.copy()
            
            # Beta 유의성 필터링 적용
            beta = self._filter_beta_by_significance(product, beta_original, factor_names)
            
            # r_hat = ts_beta x gamma (gamma_0 = 0)
            # 필터링된 beta로 r_hat 계산
            r_hat = self.gamma_0 + np.dot(beta, self.gamma_factors)
            
            # systematic: beta x F_sim (필터링된 beta 사용)
            systematic = np.einsum('ijk,k->ij', self.simulated_factors, beta)
            
            # 잔차 시뮬레이션
            residual_model = self.residual_garch_models[product]
            product_seed = (seed + idx * 1000) if seed else None
            
            # 수정: mu=0으로 고정 (잔차의 평균은 이론적으로 0이어야 함)
            idiosyncratic = self.gpu_sim.simulate_garch(
                omega=residual_model.omega,
                alpha=residual_model.alpha,
                beta=residual_model.beta,
                gamma_garch=residual_model.gamma,
                nu=residual_model.nu,
                mu=0.0,  # mu=0 고정 (변동성 구조만 사용)
                last_ret=residual_model.residuals[-1],
                last_sigma2=residual_model.sigma2[-1],
                n_paths=self.n_simulations,
                horizon=self.forecast_days,
                seed=product_seed
            )

            # 6-0. 스트레스 테스트용 epsilon 통계 계산
            # 각 path의 처음 stress_test_days일 동안의 epsilon 합계의 5th percentile
            if self.config.save_epsilon_stats:
                stress_days = min(self.config.stress_test_days, self.forecast_days)
                # epsilon 합계 (단리): 각 path별로 처음 stress_days일의 합
                epsilon_sum = idiosyncratic[:, :stress_days].sum(axis=1)  # shape: (n_paths,)

                # 통계 계산
                epsilon_5th = np.percentile(epsilon_sum, 5)  # 5th percentile (하위 극단)
                epsilon_1st = np.percentile(epsilon_sum, 1)  # 1st percentile
                epsilon_mean = np.mean(epsilon_sum)
                epsilon_std = np.std(epsilon_sum)
                epsilon_min = np.min(epsilon_sum)
                epsilon_max = np.max(epsilon_sum)

                self.epsilon_stress_stats[product] = {
                    'product_name': self.product_names.get(product, product),
                    'stress_days': stress_days,
                    'epsilon_sum_5th': float(epsilon_5th),
                    'epsilon_sum_1st': float(epsilon_1st),
                    'epsilon_sum_mean': float(epsilon_mean),
                    'epsilon_sum_std': float(epsilon_std),
                    'epsilon_sum_min': float(epsilon_min),
                    'epsilon_sum_max': float(epsilon_max)
                }

            # r = r_hat + beta x F + epsilon
            # 단순수익률
            paths = (r_hat + systematic + idiosyncratic).astype(np.float32)

            # 6-1. Winsorization (상품 타입별 가격제한폭 기준)
            # ETF/REITs: ±30% (KRX Circuit Breaker)
            # FUND: ±50% (장외파생상품 기준)
            if product in self.historical_returns.columns:
                product_type = self.product_types.get(product, 'FUND')  # 기본값 FUND (보수적)
                bounds = self.config.return_bounds.get(product_type, (-0.50, 0.50))
                lower_bound, upper_bound = bounds
                
                # clipping 전 극단치 인덱스 찾기
                below_mask = paths < lower_bound
                above_mask = paths > upper_bound
                n_below = np.sum(below_mask)
                n_above = np.sum(above_mask)
                n_clipped = n_below + n_above
                
                # 상품명 조회
                product_name = self.product_names.get(product, product)
                
                # 상세 로그 수집 (상품당 최대 MAX_SAMPLES_PER_PRODUCT개)
                if n_clipped > 0:
                    sample_count = 0
                    
                    # 하한 초과 극단치 샘플
                    if n_below > 0:
                        below_indices = np.where(below_mask)
                        for i in range(min(n_below, MAX_SAMPLES_PER_PRODUCT - sample_count)):
                            sim_idx = below_indices[0][i]
                            day_idx = below_indices[1][i]
                            original_val = paths[sim_idx, day_idx]
                            self.winsorization_logs.append({
                                'product_code': product,
                                'product_name': product_name,
                                'original_value': float(original_val),
                                'replaced_value': float(lower_bound),
                                'direction': 'lower',
                                'simulation_idx': int(sim_idx),
                                'day_idx': int(day_idx)
                            })
                            sample_count += 1
                    
                    # 상한 초과 극단치 샘플
                    if n_above > 0 and sample_count < MAX_SAMPLES_PER_PRODUCT:
                        above_indices = np.where(above_mask)
                        for i in range(min(n_above, MAX_SAMPLES_PER_PRODUCT - sample_count)):
                            sim_idx = above_indices[0][i]
                            day_idx = above_indices[1][i]
                            original_val = paths[sim_idx, day_idx]
                            self.winsorization_logs.append({
                                'product_code': product,
                                'product_name': product_name,
                                'original_value': float(original_val),
                                'replaced_value': float(upper_bound),
                                'direction': 'upper',
                                'simulation_idx': int(sim_idx),
                                'day_idx': int(day_idx)
                            })
                            sample_count += 1
                    
                    # 상품별 통계 저장
                    self.winsorization_stats[product] = {
                        'product_name': product_name,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'n_below': int(n_below),
                        'n_above': int(n_above),
                        'n_total': int(n_clipped),
                        'total_values': int(paths.size),
                        'clip_rate': float(n_clipped / paths.size * 100)
                    }
                
                # clipping 적용
                paths = np.clip(paths, lower_bound, upper_bound)
                
                # 통계 업데이트
                total_clipped_count += n_clipped
                total_simulated_count += paths.size
            
            # 누적수익률 계산 (단순수익률)
            # cumulative = (1+r_1)(1+r_2)...(1+r_n) - 1 = prod(1+r) - 1
            cumulative_returns = np.prod(1 + paths, axis=1) - 1
            
            # r_hat 연율화 (단리)
            r_hat_daily = r_hat
            r_hat_annual = r_hat_daily * self.forecast_days
            
            # sim_mean_annual: 단순수익률 누적의 평균
            sim_mean_annual = cumulative_returns.mean()
            
            # VaR: 양수로 표시 (손실 크기)
            var_95_quantile = np.percentile(cumulative_returns, 5)
            var_99_quantile = np.percentile(cumulative_returns, 1)
            var_95 = -var_95_quantile  # 양수로 변환
            var_99 = -var_99_quantile  # 양수로 변환
            
            # CVaR: 하위 tail 평균 (양수로 표시)
            tail_95 = cumulative_returns[cumulative_returns <= var_95_quantile]
            tail_99 = cumulative_returns[cumulative_returns <= var_99_quantile]
            cvar_95 = -tail_95.mean() if len(tail_95) > 0 else var_95
            cvar_99 = -tail_99.mean() if len(tail_99) > 0 else var_99
            
            self.product_var[product] = var_95
            self.product_cvar[product] = cvar_95
            self.product_r_hat[product] = r_hat_daily  # r_hat 저장
            
            # 통계 저장
            product_type = self.product_types.get(product, 'FUND')
            stat_dict = {
                '상품코드': product,
                '상품타입': product_type,
                'r_hat_daily': r_hat_daily,
                'r_hat_annual': r_hat_annual,
                'sim_mean_daily': paths.mean(),
                'sim_mean_annual': sim_mean_annual,
                'sim_std_annual': cumulative_returns.std(),
                'VaR_95': var_95,
                'VaR_99': var_99,
                'CVaR_95': cvar_95,
                'CVaR_99': cvar_99,
                'median_return': np.median(cumulative_returns),
                'min_return': cumulative_returns.min(),
                'max_return': cumulative_returns.max(),
                'skewness': stats.skew(cumulative_returns),
                'kurtosis': stats.kurtosis(cumulative_returns)
            }
            
            # Beta 필터링 통계 추가
            if product in self.beta_filter_stats:
                stat_dict['beta_significant'] = self.beta_filter_stats[product]['significant']
                stat_dict['beta_filtered'] = self.beta_filter_stats[product]['filtered']
                stat_dict['beta_filter_rate'] = self.beta_filter_stats[product]['filter_rate']
            
            self._product_stats.append(stat_dict)
            
            # 파일 저장 (output_dir 있을 경우)
            if output_dir:
                # 일별 수익률 저장 (.npz)
                sim_path = f"{sim_dir}\\{product}.npz"
                np.savez_compressed(sim_path, returns=paths)
                
                # 누적수익률 별도 저장 (.npy)
                cumulative_path = f"{sim_dir}\\{product}_cumulative.npy"
                np.save(cumulative_path, cumulative_returns.astype(np.float32))
            
            # 메모리 해제 (simulated_paths에 저장 안함)
            del paths, systematic, idiosyncratic, cumulative_returns
            
            progress.update(1)
        
        elapsed = time.time() - start_time
        print(f"\n  상품 시뮬레이션 완료: {len(self._product_stats)}개 ({elapsed:.1f}초)")
        
        # Beta 필터링 요약 출력
        if self.config.filter_insignificant_beta and len(self.beta_filter_stats) > 0:
            avg_significant = np.mean([s['significant'] for s in self.beta_filter_stats.values()])
            avg_filtered = np.mean([s['filtered'] for s in self.beta_filter_stats.values()])
            avg_rate = np.mean([s['filter_rate'] for s in self.beta_filter_stats.values()])
            print(f"\n  [Beta 필터링 요약]")
            print(f"    평균 유의한 beta: {avg_significant:.1f}개")
            print(f"    평균 필터링된 beta: {avg_filtered:.1f}개")
            print(f"    평균 필터링 비율: {avg_rate:.1f}%")
        
        # Winsorization 통계 출력
        if total_simulated_count > 0:
            clip_rate = total_clipped_count / total_simulated_count * 100
            print(f"\n  [Winsorization 통계]")
            print(f"    총 시뮬레이션 값: {total_simulated_count:,}개")
            print(f"    Clipping된 값: {total_clipped_count:,}개")
            print(f"    Clipping 비율: {clip_rate:.4f}%")

            # 타입별 Winsorization 요약
            if len(self.winsorization_stats) > 0:
                type_stats = {}  # {타입: {'count': 상품수, 'clipped': clipping수, 'total': 전체값수}}
                for product, w_stat in self.winsorization_stats.items():
                    ptype = self.product_types.get(product, 'FUND')
                    if ptype not in type_stats:
                        type_stats[ptype] = {'count': 0, 'clipped': 0, 'total': 0}
                    type_stats[ptype]['count'] += 1
                    type_stats[ptype]['clipped'] += w_stat['n_total']
                    type_stats[ptype]['total'] += w_stat['total_values']

                print(f"\n  [타입별 Winsorization 요약]")
                print(f"    {'타입':<8} {'상품수':>8} {'상하한':>12} {'Clipping비율':>12}")
                print("    " + "-"*44)
                for ptype in ['ETF', 'REITs', 'FUND']:
                    if ptype in type_stats:
                        ts = type_stats[ptype]
                        bounds = self.config.return_bounds.get(ptype, (-0.50, 0.50))
                        bound_str = f"±{abs(bounds[0])*100:.0f}%"
                        rate = ts['clipped'] / ts['total'] * 100 if ts['total'] > 0 else 0
                        print(f"    {ptype:<8} {ts['count']:>8} {bound_str:>12} {rate:>11.4f}%")

            # Winsorization 적용된 상품 목록 출력
            if len(self.winsorization_stats) > 0:
                print(f"\n  [Winsorization 적용 상품: {len(self.winsorization_stats)}개]")
                print("-"*110)
                print(f"  {'코드':<12} {'종목명':<30} {'타입':<8} {'하한':<10} {'상한':<10} {'하한초과':<10} {'상한초과':<10} {'비율':<8}")
                print("-"*110)

                # clipping 비율 순으로 정렬
                sorted_stats = sorted(self.winsorization_stats.items(),
                                      key=lambda x: x[1]['clip_rate'], reverse=True)

                for product, w_stat in sorted_stats[:20]:  # 상위 20개만 출력
                    name = w_stat['product_name'][:25] + '...' if len(w_stat['product_name']) > 25 else w_stat['product_name']
                    product_type = self.product_types.get(product, 'FUND')
                    print(f"  {product:<12} {name:<30} {product_type:<8} {w_stat['lower_bound']*100:>8.0f}% {w_stat['upper_bound']*100:>8.0f}% "
                          f"{w_stat['n_below']:>10,} {w_stat['n_above']:>10,} {w_stat['clip_rate']:>6.3f}%")
                
                if len(sorted_stats) > 20:
                    print(f"  ... 외 {len(sorted_stats) - 20}개 상품")
                
                # 극단치 샘플 출력
                if len(self.winsorization_logs) > 0:
                    print(f"\n  [극단치 샘플 (상품당 최대 5개)]")
                    print("-"*110)
                    print(f"  {'코드':<12} {'종목명':<30} {'방향':<8} {'원래값':<15} {'대체값':<15} {'시뮬#':<10} {'일자#':<8}")
                    print("-"*110)
                    
                    for log in self.winsorization_logs[:50]:  # 최대 50개 출력
                        name = log['product_name'][:25] + '...' if len(log['product_name']) > 25 else log['product_name']
                        direction = '하한' if log['direction'] == 'lower' else '상한'
                        print(f"  {log['product_code']:<12} {name:<30} {direction:<8} "
                              f"{log['original_value']*100:>13.4f}% {log['replaced_value']*100:>13.4f}% "
                              f"{log['simulation_idx']:>10,} {log['day_idx']:>8}")
                    
                    if len(self.winsorization_logs) > 50:
                        print(f"  ... 외 {len(self.winsorization_logs) - 50}개 샘플")
    
    def run_simulation(self, seed: int = None, output_dir: str = None, 
                        use_cache: bool = True) -> None:
        """전체 시뮬레이션 실행 (캐시 지원)"""
        
        # 캐시 확인: risk_metrics.csv가 있고 simulations 폴더가 있으면 스킵
        risk_cache = f"{output_dir}\\risk_metrics.csv" if output_dir else None
        sim_dir = f"{output_dir}\\simulations" if output_dir else None
        
        if use_cache and risk_cache and os.path.exists(risk_cache):
            if sim_dir and os.path.exists(sim_dir):
                sim_files = [f for f in os.listdir(sim_dir) if f.endswith('.npz')]
                if len(sim_files) > 0:
                    print("\n" + "="*70)
                    print("[5-6] 시뮬레이션 캐시 발견")
                    print("="*70)
                    print(f"  캐시 파일: {risk_cache}")
                    print(f"  시뮬레이션 파일: {len(sim_files)}개")
                    
                    # risk_metrics에서 _product_stats 복원
                    cached_df = pd.read_csv(risk_cache)
                    cached_df['상품코드'] = cached_df['상품코드'].apply(normalize_code)
                    
                    self._product_stats = cached_df.to_dict('records')
                    
                    # product_var, product_cvar, product_r_hat 복원
                    for _, row in cached_df.iterrows():
                        self.product_var[row['상품코드']] = row['VaR_95']
                        self.product_cvar[row['상품코드']] = row['CVaR_95']
                        self.product_r_hat[row['상품코드']] = row['r_hat_daily']
                    
                    print(f"  캐시에서 로드 완료: {len(self._product_stats)}개 상품")
                    return
        
        # 새로 시뮬레이션 실행
        total_start = time.time()
        self.simulate_factors(seed=seed)
        self.simulate_products(seed=seed, output_dir=output_dir)
        total_elapsed = time.time() - total_start
        print(f"\n  [총 시뮬레이션 시간: {total_elapsed:.1f}초]")
    
    def compute_expected_returns_comparison(self) -> pd.DataFrame:
        """r_hat vs 시뮬레이션 기대수익률 비교"""
        print("\n" + "="*70)
        print("[7] 기대수익률 비교: r_hat (FM) vs 시뮬레이션")
        print("="*70)
        
        if not hasattr(self, '_product_stats') or len(self._product_stats) == 0:
            print("  시뮬레이션 결과가 없습니다.")
            return pd.DataFrame()
        
        # _product_stats에서 비교용 데이터 추출
        results = []
        for stat in self._product_stats:
            results.append({
                '상품코드': stat['상품코드'],
                'r_hat_daily': stat['r_hat_daily'],
                'r_hat_annual': stat['r_hat_annual'],
                'sim_mean_daily': stat['sim_mean_daily'],
                'sim_mean_annual': stat['sim_mean_annual'],
                'sim_std_annual': stat['sim_std_annual'],
                'diff_daily': stat['sim_mean_daily'] - stat['r_hat_daily'],
                'diff_annual': stat['sim_mean_annual'] - stat['r_hat_annual'],
                'diff_pct': (stat['sim_mean_annual'] - stat['r_hat_annual']) / abs(stat['r_hat_annual']) * 100 if stat['r_hat_annual'] != 0 else 0
            })
        
        df_results = pd.DataFrame(results)
        
        if len(df_results) > 0:
            print(f"\n  [기대수익률 비교] {len(df_results)}개 상품")
            print("-"*70)
            
            r_hat_daily_mean = df_results['r_hat_daily'].mean()
            sim_daily_mean = df_results['sim_mean_daily'].mean()
            diff_daily = sim_daily_mean - r_hat_daily_mean
            
            print(f"\n  [일별]")
            print(f"  r_hat 평균: {r_hat_daily_mean*100:.6f}%")
            print(f"  E[r_sim] 평균: {sim_daily_mean*100:.6f}%")
            print(f"  차이: {diff_daily*100:.6f}%")
            
            r_hat_annual_mean = df_results['r_hat_annual'].mean()
            sim_annual_mean = df_results['sim_mean_annual'].mean()
            diff_annual = sim_annual_mean - r_hat_annual_mean
            
            print(f"\n  [연율화]")
            print(f"  r_hat 평균: {r_hat_annual_mean*100:.2f}%")
            print(f"  E[r_sim] 평균: {sim_annual_mean*100:.2f}%")
            print(f"  차이: {diff_annual*100:.4f}%")
            
            corr = df_results['r_hat_annual'].corr(df_results['sim_mean_annual'])
            print(f"\n  [검증]")
            print(f"  상관계수: {corr:.6f}")
            
            if abs(diff_annual) < 0.005:
                print(f"  결론: r_hat ~ E[r_sim] - 검증 통과")
            else:
                print(f"  주의: 차이 >= 0.5% - 점검 필요")
        
        return df_results
    
    def compute_risk_metrics(self) -> pd.DataFrame:
        """VaR, CVaR 계산"""
        print("\n" + "="*70)
        print("[8] 위험 지표 계산 (VaR/CVaR)")
        print("="*70)
        
        if not hasattr(self, '_product_stats') or len(self._product_stats) == 0:
            print("  시뮬레이션 결과가 없습니다.")
            return pd.DataFrame()
        
        # _product_stats를 DataFrame으로 변환
        df_results = pd.DataFrame(self._product_stats)
        
        if len(df_results) > 0:
            print(f"\n  위험 지표 계산 완료: {len(df_results)}개 상품")
            print(f"\n  [전체 요약]")
            print(f"  {'지표':<15} {'평균':>12} {'중위수':>12}")
            print("-"*45)
            print(f"  {'r_hat (FM)':<15} {df_results['r_hat_annual'].mean()*100:>11.2f}% "
                  f"{df_results['r_hat_annual'].median()*100:>11.2f}%")
            print(f"  {'VaR 95%':<15} {df_results['VaR_95'].mean()*100:>11.2f}% "
                  f"{df_results['VaR_95'].median()*100:>11.2f}%")
            print(f"  {'CVaR 95%':<15} {df_results['CVaR_95'].mean()*100:>11.2f}% "
                  f"{df_results['CVaR_95'].median()*100:>11.2f}%")

            # 타입별 위험 지표 요약
            if '상품타입' in df_results.columns:
                print(f"\n  [타입별 위험 지표 요약]")
                print(f"  {'타입':<8} {'상품수':>6} {'r_hat평균':>12} {'VaR95평균':>12} {'CVaR95평균':>12}")
                print("  " + "-"*54)
                for ptype in ['ETF', 'REITs', 'FUND']:
                    type_df = df_results[df_results['상품타입'] == ptype]
                    if len(type_df) > 0:
                        print(f"  {ptype:<8} {len(type_df):>6} "
                              f"{type_df['r_hat_annual'].mean()*100:>11.2f}% "
                              f"{type_df['VaR_95'].mean()*100:>11.2f}% "
                              f"{type_df['CVaR_95'].mean()*100:>11.2f}%")

        return df_results
    
    def plot_simulation_results(self, product: str, output_dir: str = None, 
                                 show: bool = True) -> None:
        """시뮬레이션 결과 시각화 (과거 수익률 + 시뮬레이션)"""
        
        # 0. 상품명 가져오기
        product_name = self.product_names.get(product, product)
        title_prefix = f'{product} ({product_name})'
        
        # 1. 시뮬레이션 데이터 로드
        if output_dir:
            sim_path = f"{output_dir}\\simulations\\{product}.npz"
            if os.path.exists(sim_path):
                data = np.load(sim_path)
                paths = data['returns']
            else:
                print(f"  상품 {product}의 시뮬레이션 파일이 없습니다.")
                return
        elif product in self.simulated_paths:
            paths = self.simulated_paths[product]
        else:
            print(f"  상품 {product}의 시뮬레이션 결과가 없습니다.")
            return
        
        # 2. 과거 수익률 로드 (전체 기간)
        hist_cumulative = None
        if self.historical_returns is not None and product in self.historical_returns.columns:
            hist_returns = self.historical_returns[product].dropna().values
            # 누적 수익률 계산 (단순수익률: cumprod(1+r))
            hist_cumulative = np.cumprod(1 + hist_returns)
        
        # 3. 시뮬레이션 누적 수익률 (단순수익률: cumprod(1+r))
        sim_cumulative = np.cumprod(1 + paths, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # 4. 왼쪽: 과거 + 시뮬레이션 경로 (y축 0% 기준)
        ax1 = axes[0]
        
        # 과거 수익률이 있으면 연결
        if hist_cumulative is not None:
            n_hist = len(hist_cumulative)
            # x축: 과거는 음수, 시뮬레이션은 양수
            hist_x = np.arange(-n_hist, 0)
            sim_x = np.arange(0, self.forecast_days)
            
            # 과거 수익률 (% 단위로 변환, 0 기준)
            hist_pct = (hist_cumulative - 1) * 100
            ax1.plot(hist_x, hist_pct, 'b-', linewidth=1.5, label='과거 실제 수익률')
            
            # 시뮬레이션 시작점을 과거 끝점에 연결
            last_hist_value = hist_cumulative[-1]
            last_hist_pct = (last_hist_value - 1) * 100
            sim_cumulative_adjusted = sim_cumulative * last_hist_value
            sim_pct_adjusted = (sim_cumulative_adjusted - 1) * 100
            
            # 시뮬레이션 경로 (100개 샘플)
            sample_idx = np.random.choice(len(paths), min(100, len(paths)), replace=False)
            for idx in sample_idx:
                ax1.plot(sim_x, sim_pct_adjusted[idx], alpha=0.05, color='gray')
            
            # 백분위수 라인
            p5 = np.percentile(sim_pct_adjusted, 5, axis=0)
            p50 = np.percentile(sim_pct_adjusted, 50, axis=0)
            p95 = np.percentile(sim_pct_adjusted, 95, axis=0)
            
            # r_hat 기대수익률 라인 (단리 연율화, % 단위)
            r_hat_daily = self.product_r_hat.get(product, 0)
            r_hat_annual = r_hat_daily * 250  # 단리 연율화
            r_hat_cumulative = last_hist_value * (1 + r_hat_daily) ** (sim_x + 1)
            r_hat_pct = (r_hat_cumulative - 1) * 100
            r_hat_end_pct = r_hat_pct[-1]  # 250일 후 누적 수익률
            
            ax1.plot(sim_x, r_hat_pct, 'b-', linewidth=2, 
                    label=f'기대수익률 (연 {r_hat_annual*100:.1f}%, 250일후 {r_hat_end_pct:.1f}%)')
            ax1.plot(sim_x, p50, color='orange', linewidth=2, label='P50 (중앙값)')
            ax1.plot(sim_x, p5, 'r--', linewidth=1.5, label='P5 (하위 5%)')
            ax1.plot(sim_x, p95, 'g--', linewidth=1.5, label='P95 (상위 5%)')
            
            # 구분선 (현재 시점)
            ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
            
            ax1.set_xlim(-n_hist, self.forecast_days)
            ax1.set_xlabel(f'거래일 (과거 {n_hist}일 / 미래 {self.forecast_days}일)')
        else:
            # 과거 데이터 없으면 시뮬레이션만
            sim_x = np.arange(0, self.forecast_days)
            sim_pct = (sim_cumulative - 1) * 100
            
            sample_idx = np.random.choice(len(paths), min(100, len(paths)), replace=False)
            for idx in sample_idx:
                ax1.plot(sim_x, sim_pct[idx], alpha=0.05, color='gray')
            
            p5 = np.percentile(sim_pct, 5, axis=0)
            p50 = np.percentile(sim_pct, 50, axis=0)
            p95 = np.percentile(sim_pct, 95, axis=0)
            
            # r_hat 기대수익률 라인 (단리 연율화, % 단위)
            r_hat_daily = self.product_r_hat.get(product, 0)
            r_hat_annual = r_hat_daily * 250  # 단리 연율화
            r_hat_cumulative = (1 + r_hat_daily) ** (sim_x + 1)
            r_hat_pct = (r_hat_cumulative - 1) * 100
            r_hat_end_pct = r_hat_pct[-1]
            
            ax1.plot(sim_x, r_hat_pct, 'b-', linewidth=2, 
                    label=f'기대수익률 (연 {r_hat_annual*100:.1f}%, 250일후 {r_hat_end_pct:.1f}%)')
            ax1.plot(sim_x, p50, color='orange', linewidth=2, label='P50 (중앙값)')
            ax1.plot(sim_x, p5, 'r--', linewidth=1.5, label='P5 (하위 5%)')
            ax1.plot(sim_x, p95, 'g--', linewidth=1.5, label='P95 (상위 5%)')
            ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax1.set_xlabel('거래일')
        
        ax1.set_ylabel('누적 수익률 (%)')
        ax1.set_title(f'{title_prefix}\n누적 수익률 + 1년 시뮬레이션 전망')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 5. 오른쪽: 최종 수익률 분포
        ax2 = axes[1]
        final_returns = sim_cumulative[:, -1] - 1
        ax2.hist(final_returns * 100, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
        
        var = self.product_var.get(product, 0)
        cvar = self.product_cvar.get(product, 0)
        r_hat_daily = self.product_r_hat.get(product, 0)
        r_hat_annual = r_hat_daily * self.forecast_days  # 단리 연율화
        
        # VaR/CVaR은 양수로 저장되어 있으므로, 그래프에서는 음수 위치에 표시
        ax2.axvline(-var * 100, color='red', linestyle='--', linewidth=2, 
                   label=f'VaR 95%: {var*100:.2f}%')
        ax2.axvline(-cvar * 100, color='darkred', linestyle='-', linewidth=2, 
                   label=f'CVaR 95%: {cvar*100:.2f}%')
        ax2.axvline(r_hat_annual * 100, color='blue', linestyle='--', linewidth=2, 
                   label=f'기대수익률: {r_hat_annual*100:.2f}%')
        
        ax2.set_xlabel('연간 수익률 (%)')
        ax2.set_ylabel('빈도')
        ax2.set_title(f'{title_prefix}\n수익률 분포 ({self.forecast_days}일 후)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장 (graphs 폴더에)
        if output_dir:
            graphs_dir = f"{output_dir}\\graphs"
            os.makedirs(graphs_dir, exist_ok=True)
            save_path = f"{graphs_dir}\\{product}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # 출력 (show=True일 때만)
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # 메모리 해제
        del paths, sim_cumulative
    
    def save_results(self, output_dir: str) -> None:
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_df = self.compute_expected_returns_comparison()
        if len(comparison_df) > 0:
            comparison_path = f"{output_dir}\\expected_returns_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
            print(f"\n  기대수익률 비교 저장: {comparison_path}")
        
        risk_df = self.compute_risk_metrics()
        if len(risk_df) > 0:
            risk_path = f"{output_dir}\\risk_metrics.csv"
            risk_df.to_csv(risk_path, index=False, encoding='utf-8-sig')
            print(f"  위험 지표 저장: {risk_path}")
        
        factor_params = []
        for name, model in self.factor_garch_models.items():
            factor_params.append({
                'factor': name,
                'mu': model.mu,
                'omega': model.omega,
                'alpha': model.alpha,
                'gamma': model.gamma,
                'beta': model.beta,
                'nu': model.nu,
                'persistence': model.alpha + 0.5*model.gamma + model.beta
            })
        
        if factor_params:
            factor_df = pd.DataFrame(factor_params)
            factor_path = f"{output_dir}\\factor_garch_parameters.csv"
            factor_df.to_csv(factor_path, index=False, encoding='utf-8-sig')
            print(f"  Factor GARCH 파라미터 저장: {factor_path}")
        
        residual_params = []
        for product, model in self.residual_garch_models.items():
            product_type = self.product_types.get(product, 'FUND')
            residual_params.append({
                '상품코드': product,
                '상품타입': product_type,
                'mu': model.mu,
                'omega': model.omega,
                'alpha': model.alpha,
                'gamma': model.gamma,
                'beta': model.beta,
                'nu': model.nu,
                'persistence': model.alpha + 0.5*model.gamma + model.beta
            })
        
        if residual_params:
            residual_df = pd.DataFrame(residual_params)
            residual_path = f"{output_dir}\\residual_garch_parameters.csv"
            residual_df.to_csv(residual_path, index=False, encoding='utf-8-sig')
            print(f"  Residual GARCH 파라미터 저장: {residual_path}")
        
        # Beta 필터링 통계 저장 (필터링 활성화 시)
        if self.config.filter_insignificant_beta and len(self.beta_filter_stats) > 0:
            filter_stats = []
            for product, b_stat in self.beta_filter_stats.items():
                filter_stats.append({
                    '상품코드': product,
                    'total_betas': b_stat['total'],
                    'significant_betas': b_stat['significant'],
                    'filtered_betas': b_stat['filtered'],
                    'filter_rate': b_stat['filter_rate']
                })
            filter_df = pd.DataFrame(filter_stats)
            filter_path = f"{output_dir}\\beta_filter_stats.csv"
            filter_df.to_csv(filter_path, index=False, encoding='utf-8-sig')
            print(f"  Beta 필터링 통계 저장: {filter_path}")
        
        # 시뮬레이션 경로는 simulate_products에서 이미 저장됨
        sim_dir = f"{output_dir}\\simulations"
        if os.path.exists(sim_dir):
            npz_files = len([f for f in os.listdir(sim_dir) if f.endswith('.npz')])
            npy_files = len([f for f in os.listdir(sim_dir) if f.endswith('.npy')])
            print(f"  시뮬레이션 경로: {sim_dir}")
            print(f"    - 일별 수익률: {npz_files}개 (.npz)")
            print(f"    - 누적 수익률: {npy_files}개 (.npy)")
        
        # Winsorization 통계 저장
        if len(self.winsorization_stats) > 0:
            winsor_stats_list = []
            for product, w_stat in self.winsorization_stats.items():
                product_type = self.product_types.get(product, 'FUND')
                winsor_stats_list.append({
                    '상품코드': product,
                    '상품명': w_stat['product_name'],
                    '상품타입': product_type,
                    '하한': w_stat['lower_bound'],
                    '상한': w_stat['upper_bound'],
                    '하한초과_개수': w_stat['n_below'],
                    '상한초과_개수': w_stat['n_above'],
                    '총_clipping_개수': w_stat['n_total'],
                    '총_시뮬레이션_값': w_stat['total_values'],
                    'clipping_비율_pct': w_stat['clip_rate']
                })
            winsor_stats_df = pd.DataFrame(winsor_stats_list)
            winsor_stats_df = winsor_stats_df.sort_values('clipping_비율_pct', ascending=False)
            winsor_stats_path = f"{output_dir}\\winsorization_stats.csv"
            winsor_stats_df.to_csv(winsor_stats_path, index=False, encoding='utf-8-sig')
            print(f"  Winsorization 통계 저장: {winsor_stats_path}")
        
        # Winsorization 상세 로그 저장
        if len(self.winsorization_logs) > 0:
            winsor_logs_df = pd.DataFrame(self.winsorization_logs)
            winsor_logs_df.columns = ['상품코드', '상품명', '원래값', '대체값', '방향', '시뮬레이션번호', '일자번호']
            winsor_logs_path = f"{output_dir}\\winsorization_samples.csv"
            winsor_logs_df.to_csv(winsor_logs_path, index=False, encoding='utf-8-sig')
            print(f"  Winsorization 샘플 저장: {winsor_logs_path}")

        # 스트레스 테스트용 epsilon 통계 저장
        if self.config.save_epsilon_stats and len(self.epsilon_stress_stats) > 0:
            epsilon_stats_list = []
            for product, e_stat in self.epsilon_stress_stats.items():
                product_type = self.product_types.get(product, 'FUND')
                epsilon_stats_list.append({
                    '상품코드': product,
                    '상품명': e_stat['product_name'],
                    '상품타입': product_type,
                    '스트레스_일수': e_stat['stress_days'],
                    'epsilon_합_5th_percentile': e_stat['epsilon_sum_5th'],
                    'epsilon_합_1st_percentile': e_stat['epsilon_sum_1st'],
                    'epsilon_합_평균': e_stat['epsilon_sum_mean'],
                    'epsilon_합_표준편차': e_stat['epsilon_sum_std'],
                    'epsilon_합_최소': e_stat['epsilon_sum_min'],
                    'epsilon_합_최대': e_stat['epsilon_sum_max']
                })
            epsilon_stats_df = pd.DataFrame(epsilon_stats_list)
            epsilon_stats_path = f"{output_dir}\\epsilon_stress_stats.csv"
            epsilon_stats_df.to_csv(epsilon_stats_path, index=False, encoding='utf-8-sig')
            print(f"  스트레스 테스트용 epsilon 통계 저장: {epsilon_stats_path}")
            print(f"    - 스트레스 기간: {self.config.stress_test_days}일")
            print(f"    - 상품 수: {len(epsilon_stats_list)}개")

            # epsilon 통계 요약 출력
            epsilon_5th_mean = np.mean([e['epsilon_sum_5th'] for e in self.epsilon_stress_stats.values()])
            epsilon_5th_std = np.std([e['epsilon_sum_5th'] for e in self.epsilon_stress_stats.values()])
            print(f"    - epsilon {self.config.stress_test_days}일 합 5th percentile 평균: {epsilon_5th_mean*100:.2f}%")
            print(f"    - epsilon {self.config.stress_test_days}일 합 5th percentile 표준편차: {epsilon_5th_std*100:.2f}%")


# %% [8] 메인 실행 함수

def run_simulation():
    """t-GJR-GARCH + Monte Carlo 시뮬레이션 실행"""
    config = SimulationConfig()
    
    print("\n" + "#"*70)
    print("#  t-GJR-GARCH + Monte Carlo 시뮬레이션 (Part 2)")
    print("#"*70)
    print(f"#  모드: {'GPU (CuPy)' if USE_GPU else 'CPU (NumPy)'}")
    print(f"#  시드: {config.seed}")
    print(f"#  수익률: 단순수익률 (일반수익률)")
    print(f"#  Beta 필터링: {'활성화' if config.filter_insignificant_beta else '비활성화'}")
    print(f"#  잔차 mu: 0 고정 (변동성만 사용)")
    print("#"*70)
    
    loader = Part1ResultLoader(config)
    if not loader.load_all():
        print("\n  Part 1 결과 로드 실패. 종료합니다.")
        return None
    
    print("\n" + "="*70)
    print("[1-1] 초과수익률 데이터 로드")
    print("="*70)
    
    returns_raw = pd.read_csv(config.excess_return_path, encoding='utf-8-sig')
    
    date_col = None
    for col in returns_raw.columns:
        if '일자' in str(col) or 'date' in str(col).lower():
            date_col = col
            break
    
    code_col = None
    for col in returns_raw.columns:
        if '코드' in str(col):
            code_col = col
            break
    
    return_col = None
    for col in returns_raw.columns:
        if '초과' in str(col) and '수익' in str(col):
            return_col = col
            break
    
    returns_raw[date_col] = pd.to_datetime(returns_raw[date_col])
    returns_raw[code_col] = returns_raw[code_col].apply(normalize_code)
    returns_raw = returns_raw.groupby([code_col, date_col]).agg({return_col: 'mean'}).reset_index()
    
    returns_pivot = returns_raw.pivot(index=date_col, columns=code_col, values=return_col)
    print(f"  데이터 shape: {returns_pivot.shape}")
    
    print("\n" + "="*70)
    print("[1-2] 시뮬레이션 대상 상품 로드")
    print("="*70)
    
    selected_df = pd.read_excel(config.selected_products_path)
    code_col_sel = None
    name_col_sel = None
    for col in selected_df.columns:
        if '상품' in str(col) and '코드' in str(col):
            code_col_sel = col
        if '상품' in str(col) and '명' in str(col):
            name_col_sel = col
    if code_col_sel is None:
        code_col_sel = selected_df.columns[0]
    if name_col_sel is None:
        # 상품명 컬럼이 없으면 두 번째 컬럼 시도
        if len(selected_df.columns) > 1:
            name_col_sel = selected_df.columns[1]
    
    selected_products = [normalize_code(c) for c in selected_df[code_col_sel].tolist()]
    available_products = set(returns_pivot.columns)
    simulation_products = [p for p in selected_products if p in available_products]
    
    # 상품코드-상품명 매핑 생성
    product_names = {}
    if name_col_sel is not None:
        for _, row in selected_df.iterrows():
            code = normalize_code(row[code_col_sel])
            name = str(row[name_col_sel]) if pd.notna(row[name_col_sel]) else code
            product_names[code] = name

    # 상품코드-상품타입 매핑 생성 (비고 열 기준)
    product_types = {}
    type_col = None
    for col in selected_df.columns:
        if '비고' in str(col):
            type_col = col
            break
    if type_col is not None:
        for _, row in selected_df.iterrows():
            code = normalize_code(row[code_col_sel])
            ptype = str(row[type_col]).strip() if pd.notna(row[type_col]) else 'FUND'
            product_types[code] = ptype
        # 타입별 개수 출력
        type_counts = {}
        for ptype in product_types.values():
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        print(f"  상품 타입별 개수: {type_counts}")
        print(f"  적용 상하한: ETF/REITs=±30%, FUND=±50%")
    else:
        print("  [경고] '비고' 열을 찾을 수 없어 모든 상품에 FUND 기준(±50%) 적용")

    print(f"  선별된 상품: {len(selected_products)}개")
    print(f"  데이터 있는 상품: {len(simulation_products)}개")

    mc_simulator = MonteCarloSimulator(config)
    mc_simulator.set_product_names(product_names)  # 상품명 설정
    mc_simulator.set_product_types(product_types)  # 상품 타입 설정
    
    betas_for_sim = loader.get_betas_for_simulation()
    ts_alphas = loader.get_ts_alphas()
    product_stds = loader.get_product_stds()  # 상품별 표준편차
    beta_t_stats = loader.get_beta_t_stats()  # beta t-stat (유의성 필터링용)
    
    mc_simulator.set_factor_data(
        loader.factor_returns, 
        betas_for_sim, 
        ts_alphas,
        loader.avg_gamma,
        product_stds,  # 표준편차 전달
        beta_t_stats   # t-stat 전달
    )
    
    mc_simulator.fit_factor_garch(
        min_obs=config.garch_min_obs, 
        output_dir=config.output_dir, 
        use_cache=config.use_cache
    )
    mc_simulator.compute_residuals(returns_pivot, simulation_products)
    mc_simulator.fit_residual_garch(
        simulation_products, 
        min_obs=config.garch_min_obs,
        output_dir=config.output_dir,
        use_cache=config.use_cache
    )
    mc_simulator.run_simulation(
        seed=config.seed, 
        output_dir=config.output_dir,
        use_cache=config.use_cache
    )
    
    mc_simulator.save_results(config.output_dir)
    
    # 모든 상품 그래프 저장
    print("\n" + "="*70)
    print("[9] 시뮬레이션 그래프 저장")
    print("="*70)
    
    if len(mc_simulator._product_stats) > 0:
        all_products = [s['상품코드'] for s in mc_simulator._product_stats]
        graphs_dir = f"{config.output_dir}\\graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        print(f"  저장 경로: {graphs_dir}")
        print(f"  대상 상품: {len(all_products)}개")
        
        progress = ProgressTracker(len(all_products), "그래프 저장")
        for product in all_products:
            mc_simulator.plot_simulation_results(
                product, 
                output_dir=config.output_dir, 
                show=False
            )
            progress.update(1)
        
        print(f"\n  그래프 저장 완료: {len(all_products)}개")
        
        # 샘플 3개만 화면 출력
        print("\n  [샘플 그래프 출력 (3개)]")
        sample_products = all_products[:3]
        for product in sample_products:
            mc_simulator.plot_simulation_results(
                product, 
                output_dir=config.output_dir, 
                show=True
            )
    
    print("\n" + "="*70)
    print("  [Part 2 완료] t-GJR-GARCH + Monte Carlo 시뮬레이션")
    print("="*70)
    print(f"\n  시뮬레이션 완료: {len(mc_simulator._product_stats)}개 상품")
    print(f"  시뮬레이션 횟수: {config.n_simulations:,}회")
    print(f"  예측 기간: {config.forecast_days}일")
    print(f"  실행 모드: {'GPU (CuPy)' if USE_GPU else 'CPU (NumPy)'}")
    print(f"  수익률: 단순수익률 (일반수익률)")
    print(f"  Beta 필터링: {'활성화' if config.filter_insignificant_beta else '비활성화'}")
    print(f"  잔차 mu: 0 고정")
    print(f"  극단치 처리: Winsorization (KRX 가격제한폭 +-50%)")
    print(f"\n  저장 파일:")
    print(f"    - expected_returns_comparison.csv")
    print(f"    - risk_metrics.csv")
    print(f"    - factor_garch_parameters.csv")
    print(f"    - residual_garch_parameters.csv")
    if config.filter_insignificant_beta:
        print(f"    - beta_filter_stats.csv")
    print(f"    - winsorization_stats.csv (상품별 통계)")
    print(f"    - winsorization_samples.csv (극단치 샘플)")
    print(f"    - simulations/ (일별: .npz, 누적: .npy)")
    print(f"    - graphs/ ({len(mc_simulator._product_stats)}개 .png)")
    print(f"\n  출력 폴더: {config.output_dir}")
    print("="*70)
    
    return {
        'config': config,
        'loader': loader,
        'mc_simulator': mc_simulator,
        'simulation_products': simulation_products
    }


# %% [9] 실행

if __name__ == "__main__":
    results = run_simulation()
# %%