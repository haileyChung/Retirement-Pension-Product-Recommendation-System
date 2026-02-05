# %% [0] PCA + Fama-MacBeth 분석 (Part 1)
#
# 파이프라인:
#   1. 데이터 로드 및 전처리
#   2. 청산 상품 제외
#   3. PCA 수행 (가능한 모든 PC 추출)
#   4. PC 설명력 분석 및 시각화
#   5. Fama-MacBeth 회귀로 기대수익률 추정
#
# 입력 데이터:
#   - 초과수익률이 단순수익률 형태
#   - 연율화: r_daily x 250 (단리)
#
# 결과 저장:
#   - pca_results.npz: PCA 결과
#   - fama_macbeth_results.csv: FM 결과 (beta, r_hat 등)
#   - gamma_estimates.csv: gamma 추정치
#   - factor_returns.csv: Factor 수익률 시계열
#

# %% [1] 라이브러리 임포트

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import time
import warnings
import os

warnings.filterwarnings('ignore')

# 설정 파일 임포트
from config import AnalysisConfig

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# %% [1-1] 공통 유틸리티 함수

def normalize_code(code) -> str:
    """
    상품코드 정규화 함수
    
    숫자 코드는 6자리로 맞춤 (앞에 0 채움)
    """
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


# %% [4] 데이터 로더 클래스

class DataLoader:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.excess_returns_raw: Optional[pd.DataFrame] = None
        self.excess_returns_pivot: Optional[pd.DataFrame] = None
        self.pca_products: Optional[List[str]] = None
        self.fm_products: Optional[List[str]] = None
        self.pca_common_data: Optional[pd.DataFrame] = None
        self.recent_data: Optional[pd.DataFrame] = None
        self.selected_products: Optional[List[str]] = None
        self.selected_products_info: Optional[pd.DataFrame] = None
    
    def load_selected_products(self) -> List[str]:
        """시뮬레이션 대상 상품 목록 로드"""
        print("\n[0] 시뮬레이션 대상 상품 로드")
        print("-"*60)
        
        df = pd.read_excel(self.config.selected_products_path)
        
        code_col = None
        for col in df.columns:
            col_str = str(col)
            if '상품' in col_str and '코드' in col_str:
                code_col = col
                break
        
        if code_col is None:
            code_col = df.columns[0]
        
        self.selected_products = [normalize_code(c) for c in df[code_col].tolist()]
        self.selected_products_info = df
        
        print(f"  파일: {self.config.selected_products_path}")
        print(f"  선별된 상품 수: {len(self.selected_products)}개")
        
        return self.selected_products
    
    def load_excess_returns(self) -> pd.DataFrame:
        print("="*70)
        print("[1] 초과수익률 데이터 로드")
        print("="*70)
        
        self.excess_returns_raw = pd.read_csv(
            self.config.excess_return_path,
            encoding='utf-8-sig'
        )
        
        print(f"  원본 데이터 shape: {self.excess_returns_raw.shape}")
        
        return self.excess_returns_raw
    
    def create_pivot_table(self) -> pd.DataFrame:
        print("\n[2] 피벗 테이블 생성")
        
        df = self.excess_returns_raw.copy()
        
        # 2-1. 날짜 컬럼 자동 탐지
        date_col = None
        date_candidates = ['기준일자', 'date', '날짜', '일자', 'Date', 'DATE']
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        
        if date_col is None:
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower or '일자' in str(col) or '날짜' in str(col):
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError(f"날짜 컬럼을 찾을 수 없습니다. 컬럼 목록: {df.columns.tolist()}")
        
        print(f"  날짜 컬럼: {date_col}")
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 2-2. 상품코드 컬럼 자동 탐지
        code_col = None
        code_candidates = ['코드', '상품코드', 'code', 'Code', 'CODE', '종목코드']
        for candidate in code_candidates:
            if candidate in df.columns:
                code_col = candidate
                break
        
        if code_col is None:
            for col in df.columns:
                if '코드' in str(col):
                    code_col = col
                    break
        
        if code_col is None:
            raise ValueError(f"상품코드 컬럼을 찾을 수 없습니다.")
        
        print(f"  상품코드 컬럼: {code_col}")
        
        # 2-3. 초과수익률 컬럼 자동 탐지
        return_col = None
        return_candidates = ['초과수익률', 'excess_return', 'ExcessReturn', 'excess']
        for candidate in return_candidates:
            if candidate in df.columns:
                return_col = candidate
                break
        
        if return_col is None:
            for col in df.columns:
                if '초과' in str(col) and '수익' in str(col):
                    return_col = col
                    break
        
        if return_col is None:
            raise ValueError(f"초과수익률 컬럼을 찾을 수 없습니다.")
        
        print(f"  수익률 컬럼: {return_col}")
        
        # 2-4. 상품코드 정규화
        df[code_col] = df[code_col].apply(normalize_code)
        
        # 2-5. 중복 처리
        dup_check = df.groupby([code_col, date_col]).size()
        dup_count = (dup_check > 1).sum()
        
        if dup_count > 0:
            print(f"\n  [경고] 중복 발견: {dup_count}건 -> 평균 처리")
            df = df.groupby([code_col, date_col]).agg({
                return_col: 'mean'
            }).reset_index()
        
        # 2-6. 피벗 테이블 생성
        self.excess_returns_pivot = df.pivot(
            index=date_col,
            columns=code_col,
            values=return_col
        )
        
        print(f"\n  피벗 shape: {self.excess_returns_pivot.shape}")
        print(f"  기간: {self.excess_returns_pivot.index.min()} ~ {self.excess_returns_pivot.index.max()}")
        
        return self.excess_returns_pivot
    
    def filter_products(self) -> Tuple[List[str], List[str]]:
        print("\n[3] 상품 필터링")
        
        obs_counts = self.excess_returns_pivot.notna().sum()
        
        min_obs_pca = int(self.config.pca_min_years * 250)
        self.pca_products = obs_counts[obs_counts >= min_obs_pca].index.tolist()
        self.fm_products = obs_counts[obs_counts >= self.config.fm_min_obs].index.tolist()
        
        print(f"\n  전체 상품: {len(self.excess_returns_pivot.columns)}개")
        print(f"  PCA 대상 ({min_obs_pca}일 이상): {len(self.pca_products)}개")
        print(f"  FM 대상 ({self.config.fm_min_obs}일 이상): {len(self.fm_products)}개")
        
        self.recent_data = self.excess_returns_pivot[self.pca_products].dropna()
        
        return self.pca_products, self.fm_products
    
    def get_pca_common_data(self) -> pd.DataFrame:
        print(f"\n[4] PCA 데이터 추출")
        
        self.pca_common_data = self.recent_data
        
        print(f"  PCA 상품 수: {len(self.pca_products)}개")
        print(f"  공통 기간 날짜 수: {len(self.pca_common_data)}일")
        
        return self.pca_common_data


# %% [5] PCA 클래스

class CommonPeriodPCA:
    def __init__(self, pca_common_data: pd.DataFrame, config: AnalysisConfig):
        self.returns = pca_common_data
        self.config = config
        self.pca_products = pca_common_data.columns.tolist()
        self.n_products = len(self.pca_products)
        self.n_time = len(pca_common_data)
        
        # demean 파라미터
        self.demeaned_returns: Optional[pd.DataFrame] = None
        self.returns_mean: Optional[pd.Series] = None
        self.returns_std: Optional[pd.Series] = None
        
        self.cov_matrix: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.loadings: Optional[pd.DataFrame] = None
        self.factor_returns: Optional[pd.DataFrame] = None
        self.pc_portfolio_returns: Optional[pd.DataFrame] = None  # PC 포트폴리오 실제 수익률

        self.max_possible_factors = min(self.n_time, self.n_products) - 1
    
    def demean_returns(self) -> pd.DataFrame:
        """
        수익률 demean (평균=0, 분산은 원본 유지)

        X_demean = R - mean

        T x T 공분산 행렬 기반 PCA 수행
        분산=1 정규화는 하지 않음 (Covariance PCA)
        """
        print("\n[5-1] 수익률 demean (평균=0, 분산 유지)")
        
        self.returns_mean = self.returns.mean()
        self.returns_std = self.returns.std()  # std는 저장 (나중에 필요할 수 있음)
        
        # demean만: R - mean (분산 정규화 안 함)
        self.demeaned_returns = self.returns - self.returns_mean
        
        # 검증
        demean_mean = self.demeaned_returns.mean().abs().max()
        demean_std = self.demeaned_returns.std()
        
        print(f"  원본 데이터: {self.n_time} x {self.n_products} (T x N)")
        print(f"  demean 검증:")
        print(f"    - 평균 (max abs): {demean_mean:.6f} (목표: 0)")
        print(f"    - 표준편차 범위: {demean_std.min()*100:.4f}% ~ {demean_std.max()*100:.4f}% (원본 유지)")
        
        return self.demeaned_returns
    
    def compute_covariance_tbt(self) -> np.ndarray:
        """
        T x T 공분산 행렬 계산 (Covariance Matrix)
        
        Cov = X_demean @ X_demean' / (N-1)  where X_demean is T x N
        결과: T x T 행렬
        
        주의: 표준화(분산=1)를 하면 Correlation Matrix가 됨
              여기서는 demean만 하여 Covariance Matrix 사용
        """
        print("\n[5-2] 공분산 행렬 계산 (T x T Covariance)")
        
        if self.demeaned_returns is None:
            self.demean_returns()
        
        X = self.demeaned_returns.values  # T x N (demean된 데이터)
        
        # T x T 공분산: X @ X' / (N-1)
        self.cov_matrix = X @ X.T / (self.n_products - 1)
        
        print(f"  demean 데이터 shape: {X.shape} (T x N)")
        print(f"  공분산 행렬 shape: {self.cov_matrix.shape} (T x T)")
        
        return self.cov_matrix
    
    def run_pca(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        PCA 수행 (T x T Covariance 방식)

        절차:
        1. T x T 공분산 행렬의 eigen decomposition
        2. Loadings 계산: L = X_demean' @ U / sqrt(eigenvalues * (N-1))
        3. eigenvector 정규화 (norm=1)

        주의: gamma 부호 조정은 Fama-MacBeth 단계에서 수행
        """
        if self.demeaned_returns is None:
            self.demean_returns()
        if self.cov_matrix is None:
            self.compute_covariance_tbt()
        
        print("\n[6] PCA 수행 (T x T Covariance 방식)")
        
        # Eigen decomposition of T x T covariance
        eigenvalues_raw, U = np.linalg.eigh(self.cov_matrix)
        
        # 내림차순 정렬
        idx = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues_raw = eigenvalues_raw[idx]
        U = U[:, idx]
        
        # 양수 eigenvalue만 사용
        positive_mask = eigenvalues_raw > 1e-10
        self.eigenvalues = eigenvalues_raw[positive_mask]
        U = U[:, positive_mask]
        
        n_components = len(self.eigenvalues)
        print(f"  유효 PC 개수: {n_components}개 (양수 eigenvalue)")
        
        # Loadings 계산: L = X_demean' @ U / sqrt(eigenvalues * (N-1))
        # L은 N x K (상품 x PC)
        X = self.demeaned_returns.values  # demean된 데이터
        scaling = np.sqrt(self.eigenvalues * (self.n_products - 1))
        self.eigenvectors = X.T @ U / scaling  # N x K
        
        # eigenvector 정규화: norm = 1
        # 부호는 gamma 추정 후 조정
        for k in range(self.eigenvectors.shape[1]):
            norm = np.linalg.norm(self.eigenvectors[:, k])
            if norm > 1e-10:
                self.eigenvectors[:, k] /= norm
        
        # 설명력 계산
        total_var = np.sum(self.eigenvalues)
        var_explained = self.eigenvalues / total_var
        cum_var = np.cumsum(var_explained)
        
        print(f"\n  총 PC 개수: {n_components}개")
        print(f"  데이터 차원: T={self.n_time}, N={self.n_products}")
        
        print(f"\n  상위 10개 PC 설명력:")
        for i in range(min(10, len(var_explained))):
            print(f"    PC{i+1}: {var_explained[i]*100:.2f}% (누적: {cum_var[i]*100:.2f}%)")
        
        # Loadings DataFrame
        self.loadings = pd.DataFrame(
            self.eigenvectors[:, :n_components],
            index=self.pca_products,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Loading 검증
        self._verify_loadings()
        
        return self.eigenvalues, self.eigenvectors
    
    def _verify_loadings(self) -> None:
        """Loading 검증 - PC1의 가중평균 확인"""
        print(f"\n  [Loading 검증]")
        
        n_check = min(5, self.eigenvectors.shape[1])
        
        for k in range(n_check):
            loading_k = self.eigenvectors[:, k]
            
            # 제곱합 (norm)
            norm_sq = np.sum(loading_k ** 2)
            
            # 단순 합
            loading_sum = np.sum(loading_k)
            
            # 단순 평균
            loading_mean = np.mean(loading_k)
            
            # 절대값 가중 평균
            abs_weights = np.abs(loading_k) / np.sum(np.abs(loading_k))
            weighted_mean = np.sum(loading_k * abs_weights)
            
            print(f"    PC{k+1}: norm^2={norm_sq:.4f}, sum={loading_sum:.4f}, "
                  f"mean={loading_mean:.4f}, weighted_mean={weighted_mean:.4f}")
    
    def compute_factor_returns(self, full_returns: pd.DataFrame, n_factors: int,
                                normalize: bool = False) -> pd.DataFrame:
        """
        Factor Returns 계산
        
        F = X_demean @ V (demean된 수익률 x eigenvector)
        
        주의: Factor를 추가로 demean하지 않음!
              demean하면 gamma = E[F] = 0이 됨
        """
        if self.eigenvectors is None:
            self.run_pca()
        
        print(f"\n[7] Factor Returns 계산 (PC {n_factors}개)")
        
        # 전체 기간 데이터에 대해 demean 적용 (PCA 기간의 mean 사용)
        pca_data = full_returns[self.pca_products].dropna()
        
        # PCA 기간의 평균으로 demean (표준화 안 함)
        X_demeaned = pca_data - self.returns_mean
        
        V = self.eigenvectors[:, :n_factors]
        F = X_demeaned.values @ V
        
        self.factor_returns = pd.DataFrame(
            F,
            index=pca_data.index,
            columns=[f'PC{k+1}' for k in range(n_factors)]
        )
        
        # Factor demean 안 함! (demean하면 gamma = 0이 됨)
        print(f"    계산 완료: {len(self.factor_returns)}일")
        print(f"    Factor 평균: {self.factor_returns.mean().values[:3]}...")
        print(f"    Factor 표준편차: {self.factor_returns.std().mean():.6f}")

        return self.factor_returns

    def compute_pc_portfolio_returns(self, full_returns: pd.DataFrame, n_factors: int) -> pd.DataFrame:
        """
        PC 포트폴리오 실제 수익률 계산

        PC_return = R @ V (demean 없이 원본 수익률 사용!)

        이것이 "PC 포트폴리오를 실제로 보유했을 때의 수익률"
        - Factor Returns (F = (R-mean)@V): 변동성/분산 캡처, 방향 정보 없음
        - PC Portfolio Returns (R@V): 실제 수익률 경로, 방향 정보 있음

        스트레스 테스트에서는 이 값을 사용해야 코로나 때 음수(손실)가 나옴
        """
        if self.eigenvectors is None:
            self.run_pca()

        print(f"\n[7-1] PC 포트폴리오 수익률 계산 (PC {n_factors}개)")
        print(f"  → PC_return = R @ V (demean 없이 원본 수익률 사용)")
        print(f"  → 실제 PC 포트폴리오 보유 시 수익률 경로")

        # 전체 기간 데이터 (demean 없이!)
        pca_data = full_returns[self.pca_products].dropna()

        V = self.eigenvectors[:, :n_factors]

        # PC 포트폴리오 수익률 = R @ V (원본 수익률 사용)
        PC_returns = pca_data.values @ V

        self.pc_portfolio_returns = pd.DataFrame(
            PC_returns,
            index=pca_data.index,
            columns=[f'PC{k+1}' for k in range(n_factors)]
        )

        print(f"    계산 완료: {len(self.pc_portfolio_returns)}일")
        print(f"    PC 포트폴리오 수익률 평균: {self.pc_portfolio_returns.mean().values[:3]}...")
        print(f"    PC 포트폴리오 수익률 표준편차: {self.pc_portfolio_returns.std().mean():.6f}")

        # Factor Returns와 비교
        if self.factor_returns is not None:
            print(f"\n  [비교] Factor Returns vs PC Portfolio Returns")
            print(f"  {'PC':<6} {'Factor평균':>12} {'PC수익률평균':>14} {'차이':>12}")
            print("  " + "-"*50)
            for i in range(min(5, n_factors)):
                f_mean = self.factor_returns[f'PC{i+1}'].mean()
                pc_mean = self.pc_portfolio_returns[f'PC{i+1}'].mean()
                diff = pc_mean - f_mean
                print(f"  PC{i+1:<4} {f_mean:>12.6f} {pc_mean:>14.6f} {diff:>12.6f}")

        return self.pc_portfolio_returns

    def compute_crisis_stats(self, n_factors: int,
                             crisis_start: str = '2020-02-20',
                             crisis_end: str = '2020-03-23',
                             output_dir: str = None) -> pd.DataFrame:
        """
        PC별 위기 기간 통계 계산

        코로나, 금리인상 등 특정 위기 기간의 PC 수익률 통계
        - 평균, 합계, 표준편차, 최소, 최대, MDD 등
        """
        if self.pc_portfolio_returns is None:
            raise ValueError("먼저 compute_pc_portfolio_returns()를 실행하세요")

        print(f"\n[7-2] PC별 위기 기간 통계 ({crisis_start} ~ {crisis_end})")

        # 위기 기간 데이터 추출
        pc_crisis = self.pc_portfolio_returns.loc[crisis_start:crisis_end]
        factor_crisis = self.factor_returns.loc[crisis_start:crisis_end]
        n_days = len(pc_crisis)

        print(f"  위기 기간: {n_days}일")

        stats_list = []
        for i in range(n_factors):
            pc_name = f'PC{i+1}'

            # PC Portfolio Returns (R@V) 통계
            pc_data = pc_crisis[pc_name]
            pc_cumsum = pc_data.cumsum()
            pc_mdd = (pc_cumsum - pc_cumsum.cummax()).min()

            # Factor Returns ((R-mean)@V) 통계
            f_data = factor_crisis[pc_name]

            stats = {
                'PC': pc_name,
                # PC Portfolio Returns (실제 수익률)
                'pc_mean': pc_data.mean(),
                'pc_sum': pc_data.sum(),
                'pc_std': pc_data.std(),
                'pc_min': pc_data.min(),
                'pc_max': pc_data.max(),
                'pc_mdd': pc_mdd,
                # Factor Returns (변동성)
                'factor_mean': f_data.mean(),
                'factor_sum': f_data.sum(),
                'factor_std': f_data.std(),
            }
            stats_list.append(stats)

        crisis_stats_df = pd.DataFrame(stats_list)

        # 출력
        print(f"\n  [PC Portfolio Returns 통계] (R@V, 실제 수익률)")
        print(f"  {'PC':<6} {'평균':>12} {'합계':>12} {'MDD':>12}")
        print("  " + "-"*45)
        for _, row in crisis_stats_df.iterrows():
            print(f"  {row['PC']:<6} {row['pc_mean']*100:>11.4f}% {row['pc_sum']*100:>11.2f}% {row['pc_mdd']*100:>11.2f}%")

        print(f"\n  [Factor Returns 통계] ((R-mean)@V, 변동성)")
        print(f"  {'PC':<6} {'평균':>12} {'합계':>12} {'표준편차':>12}")
        print("  " + "-"*45)
        for _, row in crisis_stats_df.iterrows():
            print(f"  {row['PC']:<6} {row['factor_mean']:>12.6f} {row['factor_sum']:>12.4f} {row['factor_std']:>12.6f}")

        # 저장
        if output_dir:
            crisis_path = f"{output_dir}\\pc_crisis_stats.csv"
            crisis_stats_df.to_csv(crisis_path, index=False, encoding='utf-8-sig')
            print(f"\n  PC 위기 통계 저장: {crisis_path}")

        return crisis_stats_df

    def plot_variance_explained(self, n_factors: int, output_dir: str = None) -> None:
        """PC 설명력 그래프"""
        if self.eigenvalues is None:
            raise ValueError("PCA를 먼저 실행하세요")
        
        total_var = np.sum(self.eigenvalues)
        var_explained = self.eigenvalues / total_var
        cum_var = np.cumsum(var_explained)
        
        n_display = min(n_factors, len(self.eigenvalues))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        x = range(1, n_display + 1)
        ax1.plot(x, var_explained[:n_display] * 100, 'b-o', markersize=4, linewidth=1.5)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('개별 설명력 (%)')
        ax1.set_title(f'PC 개별 설명력 (PC 1~{n_display})')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(x, cum_var[:n_display] * 100, 'g-o', markersize=4, linewidth=1.5)
        ax2.axhline(y=80, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='80%')
        ax2.axhline(y=90, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='90%')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('누적 설명력 (%)')
        ax2.set_title(f'PC 누적 설명력 (PC 1~{n_display})')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = f"{output_dir}\\pc_variance_explained.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  PC 설명력 그래프 저장: {save_path}")
        
        plt.show()


# %% [6] Fama-MacBeth 회귀분석 클래스
#
# 2단계 접근법:
#   1단계: PCA 대상 상품으로 gamma 추정 (Beta = Eigenvector)
#   2단계: 모든 FM 상품에 대해 시계열 회귀로 Beta 추정
#          r_hat_i = ts_beta_i x gamma (gamma_0 = 0)
#
# 주의: PCA는 표준화 기반이지만, FM 회귀는 원본 스케일 사용

class FamaMacBethRegression:
    def __init__(self, excess_returns: pd.DataFrame, factor_returns: pd.DataFrame, 
                 fm_products: List[str], config: AnalysisConfig,
                 pca_products: List[str] = None, eigenvectors: np.ndarray = None,
                 eigenvalues: np.ndarray = None,
                 returns_mean: pd.Series = None, returns_std: pd.Series = None):
        # 6-1. 데이터 저장
        self.returns = excess_returns
        self.factors = factor_returns
        self.fm_products = fm_products
        self.config = config
        self.pca_products = pca_products
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        
        # PCA 상품의 평균/표준편차 (표준화용)
        self.pca_returns_mean = returns_mean
        self.pca_returns_std = returns_std
        
        # 6-2. 결과 저장용
        self.betas: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        
        self.avg_gamma: Optional[pd.Series] = None
        self.se_gamma: Optional[pd.Series] = None
        self.t_stat_gamma: Optional[pd.Series] = None
    
    def estimate_gamma(self, full_returns: pd.DataFrame) -> pd.Series:
        """
        6-3. [1단계] PCA 대상 상품으로 gamma 추정
        
        횡단면 회귀 (상수항 없음, gamma_0 = 0 고정):
            E[r_i] = V_i x gamma
        
        여기서:
        - E[r_i] = 상품 i의 평균 초과수익률
        - V_i = eigenvector_i (PCA loading)
        - gamma = factor risk premium
        
        초과수익률이므로 gamma_0 = 0으로 고정 (상수항 없는 OLS)
        """
        print("\n[8] Fama-MacBeth 1단계: gamma 추정 (PCA 상품 기반)")
        
        if self.pca_products is None or self.eigenvectors is None:
            raise ValueError("gamma 추정을 위해 pca_products와 eigenvectors가 필요합니다.")
        
        n_factors = self.factors.shape[1]
        n_pca_products = len(self.pca_products)
        
        # 원본 수익률의 상품별 평균
        pca_returns = self.returns[self.pca_products]
        common_dates = pca_returns.index.intersection(self.factors.index)
        pca_returns_common = pca_returns.loc[common_dates].dropna()
        
        # 상품별 평균 수익률 (원본 스케일)
        r_bar = pca_returns_common.mean().values  # N x 1
        
        print(f"  PCA 상품: {n_pca_products}개")
        print(f"  공통 기간: {len(pca_returns_common)}일")
        print(f"  방법: 상수항 없는 횡단면 회귀 (r_bar = V x gamma, gamma_0 = 0)")
        
        # Beta = Eigenvector (norm=1)
        V = self.eigenvectors[:, :n_factors].copy()
        
        # 상수항 없는 회귀: r_bar = V x gamma (gamma_0 = 0 고정)
        XtX_inv = np.linalg.pinv(V.T @ V)
        gamma = XtX_inv @ V.T @ r_bar  # K x 1
        
        # gamma가 음수인 경우 eigenvector와 gamma 모두 부호 반전
        sign_flipped = []
        for k in range(n_factors):
            if gamma[k] < 0:
                self.eigenvectors[:, k] *= -1
                gamma[k] *= -1
                sign_flipped.append(k + 1)
        
        if len(sign_flipped) > 0:
            print(f"\n  [gamma 부호 조정] PC {sign_flipped}의 gamma/eigenvector 부호 반전")
            V = self.eigenvectors[:, :n_factors].copy()
            
            # Factor returns 재계산 (부호 반전된 eigenvector 사용)
            print("  [Factor Returns 재계산] 부호 반전된 eigenvector 적용")
            pca_data = full_returns[self.pca_products].dropna()
            X_demeaned = pca_data - self.pca_returns_mean
            F = X_demeaned.values @ self.eigenvectors[:, :n_factors]
            
            self.factors = pd.DataFrame(
                F,
                index=pca_data.index,
                columns=[f'PC{k+1}' for k in range(n_factors)]
            )
        
        # r_hat 계산 (gamma_0 = 0)
        r_hat_pca = V @ gamma
        
        # t-stat 계산 (잔차 기반)
        residuals = r_bar - r_hat_pca
        mse = np.sum(residuals**2) / (n_pca_products - n_factors)
        var_gamma = mse * np.diag(XtX_inv)
        se_gamma = np.sqrt(np.maximum(var_gamma, 1e-20))
        t_stat = gamma / se_gamma
        
        # 결과 저장
        factor_names = self.factors.columns.tolist()
        gamma_names = [f'gamma_{fn}' for fn in factor_names]
        
        self.avg_gamma = pd.Series([0.0] + list(gamma), 
                                   index=['gamma_0'] + gamma_names)
        self.se_gamma = pd.Series([0.0] + list(se_gamma), 
                                  index=['gamma_0'] + gamma_names)
        self.t_stat_gamma = pd.Series([np.nan] + list(np.abs(t_stat)), 
                                      index=['gamma_0'] + gamma_names)
        
        # 검증
        corr_pca = np.corrcoef(r_bar, r_hat_pca)[0, 1]
        
        # 단리 연율화: r_daily x 250
        annual_factor = 250 * 100
        
        print(f"\n  [gamma 추정 결과] (부호 조정 후)")
        print(f"  {'Factor':<12} {'gamma(일별)':<15} {'gamma(연율화)':<15} {'t-stat':<10}")
        print("  " + "-"*55)
        print(f"  {'gamma_0':<12} {'0 (고정)':<15} {'0.00%':>15} {'N/A':>10}")
        for i, fn in enumerate(factor_names):
            print(f"  {'gamma_'+fn:<12} {gamma[i]:.8f} {gamma[i]*annual_factor:>10.2f}% {np.abs(t_stat[i]):>8.2f}")
        
        print(f"\n  [1단계 검증] PCA 상품 ({n_pca_products}개)")
        print(f"    r_bar 평균: {r_bar.mean()*annual_factor:.2f}%")
        print(f"    r_hat 평균: {r_hat_pca.mean()*annual_factor:.2f}%")
        print(f"    차이: {(r_bar.mean()-r_hat_pca.mean())*annual_factor:.4f}%")
        print(f"    상관계수: {corr_pca:.4f}")
        
        return self.avg_gamma
    
    def run_time_series_regression(self) -> pd.DataFrame:
        """
        6-4. [2단계] 모든 FM 상품에 대해 시계열 회귀로 Beta 추정
        
        원본 수익률 사용:
          r_i,t = alpha_i + sum(beta_i,k x F_k,t) + epsilon_i,t
        
        여기서 F는 표준화 수익률 @ eigenvector (평균 0)
        ts_alpha는 상품의 평균 수준 포함
        
        t-stat도 함께 계산하여 저장 (beta 유의성 필터링용)
        """
        print("\n[9] Fama-MacBeth 2단계: Beta 추정 (시계열 회귀)")
        print("  원본 수익률 사용, 상수항 포함")
        
        factor_names = self.factors.columns.tolist()
        n_products = len(self.fm_products)
        n_factors = len(factor_names)
        
        # PCA 상품 인덱스 맵
        pca_idx_map = {}
        if self.pca_products is not None:
            for i, p in enumerate(self.pca_products):
                pca_idx_map[p] = i
        
        progress = ProgressTracker(n_products, "베타 추정")
        
        betas_list = []
        
        # PCA 상품 검증용 (demean 방식: ts_beta vs eigenvector)
        pca_ts_beta = []
        pca_eigenvector = []
        
        for product in self.fm_products:
            y = self.returns[product].dropna()
            product_dates = y.index.intersection(self.factors.index)
            
            if len(product_dates) < self.config.min_obs_for_beta:
                progress.update(1)
                continue
            
            y = y.loc[product_dates].values
            X = self.factors.loc[product_dates].values
            
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            y = y[valid_mask]
            X = X[valid_mask]
            
            if len(y) < self.config.min_obs_for_beta:
                progress.update(1)
                continue
            
            # 상수항 포함 회귀
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            try:
                coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                ts_alpha = coeffs[0]
                ts_betas = coeffs[1:]
                
                y_pred = X_with_const @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # t-stat 계산 (beta 유의성 필터링용)
                n_obs = len(y)
                dof = n_obs - n_factors - 1  # 자유도 (n - k - 1)
                mse = ss_res / dof if dof > 0 else 0
                
                # (X'X)^-1 계산
                XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
                
                # 각 계수의 표준오차
                se_coeffs = np.sqrt(np.maximum(mse * np.diag(XtX_inv), 1e-20))
                
                # t-stat (alpha 제외, beta만)
                t_stats = ts_betas / se_coeffs[1:]
                
                n_obs_original = self.returns[product].dropna().shape[0]
                
                result = {
                    '상품코드': product,
                    'n_obs': n_obs_original,
                    'n_obs_regression': len(y),
                    'ts_alpha': ts_alpha,
                    'ts_r_squared': r_squared
                }
                
                # beta 저장
                for i, fname in enumerate(factor_names):
                    result[f'beta_{fname}'] = ts_betas[i]
                
                # t-stat 저장 (beta 유의성 필터링용)
                for i, fname in enumerate(factor_names):
                    result[f't_stat_{fname}'] = t_stats[i]
                
                betas_list.append(result)
                
                # PCA 상품 검증: ts_beta vs eigenvector (demean 방식)
                # Covariance PCA에서는 ts_beta ≈ eigenvector (직접)
                if product in pca_idx_map:
                    pca_i = pca_idx_map[product]
                    eigenvec = self.eigenvectors[pca_i, :n_factors]
                    pca_ts_beta.append(ts_betas)
                    pca_eigenvector.append(eigenvec)
                
            except Exception:
                pass
            
            progress.update(1)
        
        self.betas = pd.DataFrame(betas_list)
        print(f"\n  베타 추정 완료: {len(self.betas)}개 상품")
        
        # t-stat 저장 확인
        t_stat_cols = [col for col in self.betas.columns if col.startswith('t_stat_')]
        print(f"  t-stat 컬럼: {len(t_stat_cols)}개 (beta 유의성 필터링용)")
        
        # PCA 상품 검증 출력 (demean 방식: ts_beta ≈ eigenvector)
        if len(pca_ts_beta) > 0:
            pca_ts_beta = np.array(pca_ts_beta)
            pca_eigenvector = np.array(pca_eigenvector)
            
            print(f"\n  [검증] ts_beta vs eigenvector (PCA 상품 {len(pca_ts_beta)}개)")
            print(f"  (Covariance PCA에서는 ts_beta ≈ eigenvector)")
            for k in range(min(3, n_factors)):
                corr = np.corrcoef(pca_ts_beta[:, k], pca_eigenvector[:, k])[0, 1]
                ratio = np.mean(pca_ts_beta[:, k] / (pca_eigenvector[:, k] + 1e-10))
                print(f"    PC{k+1}: 상관={corr:.4f}, 평균비율={ratio:.4f}")
        
        return self.betas
    
    def compute_expected_returns(self) -> pd.DataFrame:
        """
        6-5. 기대수익률 계산
        
        Covariance PCA (demean 방식):
        - F = (R - mu) x V
        - ts_beta ~ V (직접)
        - r_hat = ts_beta x gamma (gamma_0 = 0 고정)
        
        초과수익률이므로 gamma_0 = 0으로 고정
        
        단리 연율화: r_daily x 250
        """
        print("\n[10] 기대수익률 계산")
        
        if self.avg_gamma is None:
            raise ValueError("gamma가 추정되지 않았습니다.")
        
        factor_names = self.factors.columns.tolist()
        n_factors = len(factor_names)
        
        # gamma_0 = 0 (고정)
        gamma_0 = self.avg_gamma['gamma_0']  # = 0
        
        # gamma 벡터 (factor premium)
        gamma_vec = np.array([self.avg_gamma[f'gamma_{fn}'] for fn in factor_names])
        
        results = self.betas.copy()
        
        # PCA 상품 set
        pca_set = set(self.pca_products) if self.pca_products else set()
        
        # 각 상품의 표준편차 계산 (저장용)
        common_dates = self.returns.index.intersection(self.factors.index)
        
        predicted_returns = []
        product_stds = []  # 표준편차 저장 (참고용)
        
        for idx, row in results.iterrows():
            product = row['상품코드']
            beta_vec = np.array([row[f'beta_{fn}'] for fn in factor_names])
            
            # 표준편차 계산 (저장용)
            if product in pca_set and self.pca_returns_std is not None:
                std_i = self.pca_returns_std[product]
            else:
                r = self.returns[product].dropna()
                r_common = r.loc[r.index.intersection(common_dates)]
                std_i = r_common.std() if len(r_common) > 0 else 1.0
            
            if std_i < 1e-10:
                std_i = 1.0
            
            product_stds.append(std_i)
            
            # r_hat = gamma_0 + beta x gamma (일별)
            r_hat = gamma_0 + np.dot(beta_vec, gamma_vec)
            predicted_returns.append(r_hat)
        
        results['std'] = product_stds  # 표준편차 컬럼 추가 (참고용)
        results['predicted_return'] = predicted_returns
        
        # 단리 연율화: r_daily x 250
        results['r_hat_annual'] = np.array(predicted_returns) * 250
        
        # 실제 평균 수익률
        avg_returns = []
        for product in results['상품코드']:
            r = self.returns[product].dropna()
            common = r.index.intersection(self.factors.index)
            avg_returns.append(r.loc[common].mean() if len(common) > 0 else np.nan)
        
        results['avg_return'] = avg_returns
        results['fm_alpha'] = results['avg_return'] - results['predicted_return']
        
        # PCA 상품 여부 표시
        results['is_pca'] = results['상품코드'].apply(lambda x: x in pca_set)
        
        self.results = results
        
        pca_count = results['is_pca'].sum()
        non_pca_count = len(results) - pca_count
        
        print(f"  기대수익률 계산 완료: {len(results)}개 상품")
        print(f"    - PCA 상품 (7년 이상): {pca_count}개")
        print(f"    - 일반 상품 (7년 미만): {non_pca_count}개")
        print(f"    - 연율화 방식: 단리 (r_daily x 250)")
        
        return results
    
    def print_return_comparison(self, results: pd.DataFrame) -> pd.DataFrame:
        """6-6. 실제수익률 vs 예상수익률 비교 분석"""
        print("\n" + "="*70)
        print("[FM 분석 결과] 실제수익률 vs 예상수익률 비교")
        print("="*70)
        
        min_obs_7yr = 250 * 7
        
        # 단리 연율화: r_daily x 250
        annual_factor = 250
        
        def calc_stats(r_bar_s, r_hat_s):
            residual = r_bar_s - r_hat_s
            return {
                'r_bar_mean': r_bar_s.mean() * annual_factor * 100,
                'r_bar_std': r_bar_s.std() * np.sqrt(250) * 100,
                'r_bar_min': r_bar_s.min() * annual_factor * 100,
                'r_bar_max': r_bar_s.max() * annual_factor * 100,
                'r_hat_mean': r_hat_s.mean() * annual_factor * 100,
                'r_hat_std': r_hat_s.std() * np.sqrt(250) * 100,
                'r_hat_min': r_hat_s.min() * annual_factor * 100,
                'r_hat_max': r_hat_s.max() * annual_factor * 100,
                'residual_mean': residual.mean() * annual_factor * 100,
                'residual_std': residual.std() * np.sqrt(250) * 100,
                'correlation': r_bar_s.corr(r_hat_s),
                'n_products': len(r_bar_s)
            }
        
        def print_table(title, stats):
            print(f"\n{title}")
            print("-"*65)
            print(f"{'Stat':<12} {'r_bar':<18} {'r_hat':<18} {'Residual':<18}")
            print("-"*65)
            print(f"{'Mean':<12} {stats['r_bar_mean']:>14.2f}% {stats['r_hat_mean']:>14.2f}% {stats['residual_mean']:>14.2f}%")
            print(f"{'Std':<12} {stats['r_bar_std']:>14.2f}% {stats['r_hat_std']:>14.2f}% {stats['residual_std']:>14.2f}%")
            print(f"{'Min':<12} {stats['r_bar_min']:>14.2f}% {stats['r_hat_min']:>14.2f}%")
            print(f"{'Max':<12} {stats['r_bar_max']:>14.2f}% {stats['r_hat_max']:>14.2f}%")
            print("-"*65)
            print(f"Correlation: {stats['correlation']:.4f}")
        
        # 전체 상품
        valid_mask = results['avg_return'].notna() & results['predicted_return'].notna()
        results_valid = results[valid_mask]
        r_bar_all = results_valid['avg_return']
        r_hat_all = results_valid['predicted_return']
        stats_all = calc_stats(r_bar_all, r_hat_all)
        print_table(f"[All Products] ({stats_all['n_products']})", stats_all)
        
        # 7년 이상
        pca_mask = results_valid['n_obs'] >= min_obs_7yr
        results_pca = results_valid[pca_mask]
        if len(results_pca) > 0:
            stats_pca = calc_stats(results_pca['avg_return'], results_pca['predicted_return'])
            print_table(f"[7yr+ Products] ({stats_pca['n_products']})", stats_pca)
        
        # 7년 미만
        results_short = results_valid[~pca_mask]
        if len(results_short) > 0:
            stats_short = calc_stats(results_short['avg_return'], results_short['predicted_return'])
            print_table(f"[7yr- Products] ({stats_short['n_products']})", stats_short)
        
        # CSV 저장
        summary_path = f"{self.config.output_dir}\\fm_return_comparison.csv"
        results_valid.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n  비교 통계 저장: {summary_path}")
        
        return results_valid


# %% [7] 메인 실행 함수

def run_pca_fm_analysis():
    """
    PCA + Fama-MacBeth 분석 실행
    
    결과물:
      - pca_results.npz: eigenvalues, eigenvectors
      - pca_loadings.csv: PCA loadings
      - factor_returns.csv: Factor 수익률 시계열
      - fama_macbeth_results.csv: FM 결과 (beta, t_stat, r_hat 등)
      - gamma_estimates.csv: gamma 추정치
    """
    config = AnalysisConfig()
    
    print("\n" + "#"*70)
    print("#  PCA + Fama-MacBeth 분석 (Part 1)")
    print("#"*70)
    print(f"#  수익률: 단순수익률 (일반수익률)")
    print("#"*70)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ==========================================================================
    # [1] 데이터 로드
    # ==========================================================================
    data_loader = DataLoader(config)
    data_loader.load_selected_products()
    data_loader.load_excess_returns()
    
    # ==========================================================================
    # [2] 피벗 테이블 생성 및 필터링
    # ==========================================================================
    data_loader.create_pivot_table()
    data_loader.filter_products()
    data_loader.get_pca_common_data()
    
    # ==========================================================================
    # [3] PCA 수행 (표준화 + T x T 공분산 + eigen decomposition)
    # ==========================================================================
    pca = CommonPeriodPCA(data_loader.pca_common_data, config)
    pca.run_pca()  # 내부에서 standardize_returns, compute_covariance_tbt 자동 호출
    
    # PCA 결과 저장
    pca_path = f"{config.output_dir}\\pca_results.npz"
    np.savez_compressed(
        pca_path,
        eigenvalues=pca.eigenvalues,
        eigenvectors=pca.eigenvectors,
        pca_products=np.array(pca.pca_products),
        max_possible_factors=pca.max_possible_factors
    )
    print(f"\n  PCA 결과 저장: {pca_path}")
    
    # loadings 저장
    loadings_path = f"{config.output_dir}\\pca_loadings.csv"
    pca.loadings.to_csv(loadings_path, encoding='utf-8-sig')
    print(f"  PCA loadings 저장: {loadings_path}")
    
    # ==========================================================================
    # [4] PC 설명력 분석
    # ==========================================================================
    total_var = np.sum(pca.eigenvalues)
    var_explained = pca.eigenvalues / total_var
    cum_var = np.cumsum(var_explained)
    
    print(f"\n[PC 설명력 분석] PC 1~{config.simulation_factors}")
    print(f"  {'PC':<6} {'개별 설명력':<15} {'누적 설명력':<15}")
    print("  " + "-"*40)
    for i in range(config.simulation_factors):
        print(f"  PC{i+1:<4} {var_explained[i]*100:>10.2f}% {cum_var[i]*100:>10.2f}%")
    
    pca.plot_variance_explained(config.simulation_factors, output_dir=config.output_dir)
    
    # ==========================================================================
    # [5] Factor Returns 계산
    # ==========================================================================
    factor_returns = pca.compute_factor_returns(
        data_loader.excess_returns_pivot,
        n_factors=config.simulation_factors,
        normalize=config.normalize_factors
    )
    
    # Factor Returns 저장
    factor_path = f"{config.output_dir}\\factor_returns.csv"
    factor_returns.to_csv(factor_path, encoding='utf-8-sig')
    print(f"\n  Factor Returns 저장: {factor_path}")

    # ==========================================================================
    # [6] Fama-MacBeth 회귀
    # ==========================================================================
    fm = FamaMacBethRegression(
        data_loader.excess_returns_pivot,
        factor_returns,
        data_loader.fm_products,
        config,
        pca_products=data_loader.pca_products,
        eigenvectors=pca.eigenvectors,
        eigenvalues=pca.eigenvalues,
        returns_mean=pca.returns_mean,
        returns_std=pca.returns_std
    )
    
    # 1단계: gamma 추정 (부호 반전 시 Factor returns 재계산)
    fm.estimate_gamma(data_loader.excess_returns_pivot)
    
    # 2단계: Beta 추정 (재계산된 Factor 사용, t-stat 포함)
    fm.run_time_series_regression()
    
    # 기대수익률 계산
    fm_results = fm.compute_expected_returns()
    
    # 비교 분석 출력
    fm.print_return_comparison(fm_results)
    
    # FM 결과 저장
    fm_path = f"{config.output_dir}\\fama_macbeth_results.csv"
    fm_results.to_csv(fm_path, index=False, encoding='utf-8-sig')
    print(f"\n  FM 결과 저장: {fm_path}")
    
    # gamma 저장
    gamma_data = []
    for i in range(config.simulation_factors):
        pc_name = f'PC{i+1}'
        gamma_key = f'gamma_{pc_name}'
        gamma_daily = fm.avg_gamma[gamma_key]
        # 단리 연율화: r_daily x 250
        gamma_annual = gamma_daily * 250 * 100
        gamma_data.append({
            'PC': pc_name,
            'gamma_daily': gamma_daily,
            'gamma_annual': gamma_annual,
            'se': fm.se_gamma[gamma_key],
            't_stat': fm.t_stat_gamma[gamma_key]
        })
    
    gamma_df = pd.DataFrame(gamma_data)
    gamma_0_daily = fm.avg_gamma['gamma_0']
    # 단리 연율화: r_daily x 250
    gamma_0_annual = gamma_0_daily * 250 * 100
    gamma_df.loc[len(gamma_df)] = {
        'PC': 'gamma_0',
        'gamma_daily': gamma_0_daily,
        'gamma_annual': gamma_0_annual,
        'se': fm.se_gamma['gamma_0'],
        't_stat': fm.t_stat_gamma['gamma_0']
    }
    
    gamma_path = f"{config.output_dir}\\gamma_estimates.csv"
    gamma_df.to_csv(gamma_path, index=False, encoding='utf-8-sig')
    print(f"  gamma 추정치 저장: {gamma_path}")
    
    # 부호 반전된 PCA 결과 다시 저장
    # (estimate_gamma에서 eigenvector 부호가 반전되었으므로)
    pca_path = f"{config.output_dir}\\pca_results.npz"
    np.savez_compressed(
        pca_path,
        eigenvalues=pca.eigenvalues,
        eigenvectors=pca.eigenvectors,  # 부호 반전 적용됨
        pca_products=np.array(pca.pca_products),
        max_possible_factors=pca.max_possible_factors
    )
    print(f"  PCA 결과 재저장 (부호 반전 적용): {pca_path}")
    
    # loadings도 다시 저장
    pca.loadings = pd.DataFrame(
        pca.eigenvectors[:, :len(pca.loadings.columns)],
        index=pca.pca_products,
        columns=pca.loadings.columns
    )
    loadings_path = f"{config.output_dir}\\pca_loadings.csv"
    pca.loadings.to_csv(loadings_path, encoding='utf-8-sig')
    print(f"  PCA loadings 재저장 (부호 반전 적용): {loadings_path}")

    # ==========================================================================
    # [6-1] PC 포트폴리오 수익률 계산 (부호 반전된 eigenvector 사용)
    # ==========================================================================
    # 중요: FM의 estimate_gamma()에서 eigenvector 부호가 반전되었으므로,
    # 이 시점에서 계산해야 beta와 일관된 부호를 가짐
    pc_portfolio_returns = pca.compute_pc_portfolio_returns(
        data_loader.excess_returns_pivot,
        n_factors=config.simulation_factors
    )

    # PC 포트폴리오 수익률 저장
    pc_return_path = f"{config.output_dir}\\pc_portfolio_returns.csv"
    pc_portfolio_returns.to_csv(pc_return_path, encoding='utf-8-sig')
    print(f"\n  PC 포트폴리오 수익률 저장 (부호 반전 적용): {pc_return_path}")

    # ==========================================================================
    # [6-2] PC별 위기 기간 통계 (코로나)
    # ==========================================================================
    crisis_stats = pca.compute_crisis_stats(
        n_factors=config.simulation_factors,
        crisis_start='2020-02-20',
        crisis_end='2020-03-23',
        output_dir=config.output_dir
    )

    # ==========================================================================
    # [7] 결과 요약
    # ==========================================================================
    print("\n" + "="*70)
    print("  [Part 1 완료] PCA + Fama-MacBeth 분석")
    print("="*70)
    print(f"\n  수익률: 단순수익률 (일반수익률)")
    print(f"  연율화: r_daily x 250 (단리)")
    print(f"\n  저장된 파일:")
    print(f"    - pca_results.npz")
    print(f"    - pca_loadings.csv")
    print(f"    - factor_returns.csv (F = (R-mean)@V, 변동성 캡처)")
    print(f"    - pc_portfolio_returns.csv (R@V, 실제 수익률 경로) ★ 스트레스테스트용")
    print(f"    - pc_crisis_stats.csv (PC별 코로나 기간 통계) ★ 스트레스테스트용")
    print(f"    - fama_macbeth_results.csv (t_stat 컬럼 포함)")
    print(f"    - gamma_estimates.csv")
    print(f"    - pc_variance_explained.png")
    print(f"    - fm_return_comparison.csv")
    print(f"\n  출력 폴더: {config.output_dir}")
    print("\n  다음 단계: garch_simulation.py 실행")
    print("="*70)

    return {
        'config': config,
        'data_loader': data_loader,
        'pca': pca,
        'fm': fm,
        'fm_results': fm_results,
        'factor_returns': factor_returns,
        'pc_portfolio_returns': pc_portfolio_returns
    }


# %% [8] 실행

if __name__ == "__main__":
    results = run_pca_fm_analysis()
# %%