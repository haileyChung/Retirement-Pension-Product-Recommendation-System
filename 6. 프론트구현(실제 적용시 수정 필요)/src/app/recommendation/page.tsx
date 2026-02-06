"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { useCart } from "@/context/CartContext";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  TooltipItem,
} from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { Doughnut, Line } from "react-chartjs-2";

// Chart.js 등록
ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  ChartDataLabels,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler
);

// API 응답 타입
interface PortfolioData {
  portfolioId: number;
  conditions: {
    region: string;
    theme: string;
    targetReturn: number;
    retireYear: number;
  };
  metrics: {
    expectedReturn: number;
    var95: number;
  };
  allocation: {
    riskAssetWeight: number;
    safeAssetWeight: number;
    tdfWeight: number;
  };
  products: {
    total: number;
    top10: Array<{
      rank: number;
      code: string;
      name: string;
      weight_pct: number;
      productRegion?: string;
      productType?: string;
      productTheme?: string;
      isTDF?: boolean;
    }>;
  };
  breakdown: {
    region: Record<string, number>;
    theme: Record<string, number>;
  };
}

// 색상 팔레트
const CHART_COLORS = {
  bothMatch: "#0A2972",      // 국가&테마 둘다 - 네이비
  regionOnly: "#3b82f6",     // 국가만 - 블루
  themeOnly: "#D5B45C",      // 테마만 - 골드
  tdf: "#10b981",            // TDF - 그린
  other: "#94a3b8",          // 나머지 - 그레이
};

const PRODUCT_COLORS = [
  "#0A2972", "#1e40af", "#3b82f6", "#60a5fa", "#93c5fd",
  "#D5B45C", "#f59e0b", "#10b981", "#6366f1", "#8b5cf6",
];

function RecommendationContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { addToCart, isInCart } = useCart();

  // URL 파라미터에서 사용자 선택 정보 가져오기
  const retireYear = searchParams.get("retireYear") || "2040";
  const targetReturn = searchParams.get("targetReturn") || "0.07";
  const country = searchParams.get("country") || "";
  const theme = searchParams.get("theme") || "";

  // 상태 관리
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [showCartModal, setShowCartModal] = useState(false);

  // 국가 표시명 변환
  const getCountryDisplay = (value: string) => {
    if (value === "지역기타") return "기타 지역";
    return value || "글로벌";
  };

  // API 호출
  useEffect(() => {
    const fetchPortfolio = async () => {
      setLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams({
          region: country,
          theme: theme,
          targetReturn: targetReturn,
          retireYear: retireYear,
        });

        const response = await fetch(`/api/portfolio?${params.toString()}`);
        const result = await response.json();

        if (!result.success) {
          setError(result.error || "포트폴리오를 불러올 수 없습니다.");
          setPortfolioData(null);
        } else {
          setPortfolioData(result.data);

          // 시연용: 포트폴리오 조회 시 자동으로 마이페이지에 저장
          const portfolioForMypage = {
            portfolioId: result.data.portfolioId,
            conditions: result.data.conditions,
            metrics: result.data.metrics,
            allocation: result.data.allocation,
            products: result.data.products.top10.map((p: { code: string; name: string; weight_pct: number; productRegion?: string; productTheme?: string; productType?: string; isTDF?: boolean }) => ({
              code: p.code,
              name: p.name,
              weight_pct: p.weight_pct,
              productRegion: p.productRegion,
              productTheme: p.productTheme,
              productType: p.productType,
              isTDF: p.isTDF,
            })),
            totalProducts: result.data.products.total,
            purchasedAt: new Date().toISOString(),
          };
          localStorage.setItem("purchasedPortfolios", JSON.stringify([portfolioForMypage]));
        }
      } catch (err) {
        setError("서버 연결에 실패했습니다.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();
  }, [country, theme, targetReturn, retireYear]);

  // 도넛 차트 데이터 계산 (국가&테마 기준)
  const calculateChartData = () => {
    if (!portfolioData) return null;

    const { breakdown, allocation } = portfolioData;
    const selectedRegion = portfolioData.conditions.region;
    const selectedTheme = portfolioData.conditions.theme;

    // 선택된 국가의 비중
    const regionWeight = breakdown.region[selectedRegion] || 0;
    // 선택된 테마의 비중
    const themeWeight = breakdown.theme[selectedTheme] || 0;
    // TDF 비중
    const tdfWeight = allocation.tdfWeight || 0;

    // 국가&테마 둘다 해당하는 비중 (교집합 추정)
    // 실제로는 상품별로 계산해야 하지만, 간단히 min 값으로 추정
    const bothMatchWeight = Math.min(regionWeight, themeWeight) * 0.5;

    // 국가만 해당 (국가 비중 - 교집합)
    const regionOnlyWeight = Math.max(0, regionWeight - bothMatchWeight);

    // 테마만 해당 (테마 비중 - 교집합)
    const themeOnlyWeight = Math.max(0, themeWeight - bothMatchWeight);

    // 나머지 (100 - 국가/테마 관련 - TDF)
    const otherWeight = Math.max(
      0,
      100 - regionOnlyWeight - themeOnlyWeight - bothMatchWeight - tdfWeight
    );

    return {
      labels: [
        `${selectedRegion} & ${selectedTheme}`,
        `${selectedRegion}`,
        `${selectedTheme}`,
        "TDF",
        "기타",
      ],
      datasets: [
        {
          data: [
            bothMatchWeight,
            regionOnlyWeight,
            themeOnlyWeight,
            tdfWeight,
            otherWeight,
          ],
          backgroundColor: [
            CHART_COLORS.bothMatch,
            CHART_COLORS.regionOnly,
            CHART_COLORS.themeOnly,
            CHART_COLORS.tdf,
            CHART_COLORS.other,
          ],
          borderColor: "#ffffff",
          borderWidth: 2,
        },
      ],
    };
  };

  // 12개월 수익률 시뮬레이션 데이터 계산
  const calculateSimulationData = () => {
    if (!portfolioData) return null;

    const { metrics } = portfolioData;
    const expectedReturn = metrics.expectedReturn; // 연간 기대 수익률 (%)
    const var95 = metrics.var95; // VaR 95% (최대 손실)

    // 월별 수익률 계산 (연간 수익률을 월별로 환산)
    const monthlyReturn = expectedReturn / 12;

    // 변동성 추정 (VaR를 기반으로 표준편차 추정)
    // VaR 95% ≈ μ - 1.645σ 이므로, σ ≈ (μ - VaR) / 1.645
    const estimatedVolatility = Math.abs(expectedReturn - var95) / 1.645;
    const monthlyVol = estimatedVolatility / Math.sqrt(12);

    // 기간 (개월)
    const months = [0, 3, 6, 9, 12];
    const labels = ["현재", "3개월", "6개월", "9개월", "12개월"];

    // 시나리오별 수익률 계산 (복리 기준)
    const expectedLine = months.map((m) => {
      if (m === 0) return 100;
      return 100 * Math.pow(1 + monthlyReturn / 100, m);
    });

    // 낙관적 시나리오 (기대수익률 + 1.5 표준편차)
    const optimisticLine = months.map((m) => {
      if (m === 0) return 100;
      const optimisticMonthly = (monthlyReturn + monthlyVol * 1.5);
      return 100 * Math.pow(1 + optimisticMonthly / 100, m);
    });

    // 비관적 시나리오 (기대수익률 - 1.5 표준편차)
    const pessimisticLine = months.map((m) => {
      if (m === 0) return 100;
      const pessimisticMonthly = (monthlyReturn - monthlyVol * 1.5);
      return 100 * Math.pow(1 + pessimisticMonthly / 100, m);
    });

    // VaR 라인 (95% 확률로 이 이상의 손실은 없음)
    // DB에서 var95는 양수로 저장됨 (예: 3.15는 -3.15% 손실을 의미)
    const varLine = months.map((m) => {
      if (m === 0) return 100;
      // VaR를 월별로 환산 (시간에 비례하여 감소) - 손실이므로 빼줌
      const monthlyVarRate = Math.abs(var95) / 12;
      return 100 - (monthlyVarRate * m);
    });

    return {
      labels,
      datasets: [
        {
          label: "낙관적 시나리오",
          data: optimisticLine,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.3,
          pointRadius: 4,
          pointBackgroundColor: "#10b981",
        },
        {
          label: "기대 수익률",
          data: expectedLine,
          borderColor: "#0A2972",
          backgroundColor: "rgba(10, 41, 114, 0.1)",
          borderWidth: 3,
          fill: false,
          tension: 0.3,
          pointRadius: 5,
          pointBackgroundColor: "#0A2972",
        },
        {
          label: "비관적 시나리오",
          data: pessimisticLine,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.3,
          pointRadius: 4,
          pointBackgroundColor: "#f59e0b",
        },
        {
          label: "손실한계선(VaR 95%)",
          data: varLine,
          borderColor: "#dc2626",
          backgroundColor: "rgba(220, 38, 38, 0.05)",
          borderWidth: 2,
          borderDash: [10, 5],
          fill: "origin",
          tension: 0,
          pointRadius: 3,
          pointBackgroundColor: "#dc2626",
        },
      ],
    };
  };

  // 시뮬레이션 차트 옵션
  const simulationChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index" as const,
      intersect: false,
    },
    plugins: {
      legend: {
        display: true,
        position: "bottom" as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            size: 11,
          },
        },
      },
      tooltip: {
        callbacks: {
          label: function (context: TooltipItem<"line">) {
            const value = context.parsed.y;
            if (value === null) return "";
            const change = value - 100;
            const sign = change >= 0 ? "+" : "";
            return `${context.dataset.label}: 약 ${Math.round(value)}만원 (${sign}${change.toFixed(1)}%)`;
          },
        },
      },
      datalabels: {
        display: false,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
        ticks: {
          font: {
            size: 11,
          },
        },
      },
      y: {
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          font: {
            size: 11,
          },
          callback: function (value: number | string) {
            return `${value}`;
          },
        },
        suggestedMin: 90,
        suggestedMax: 115,
      },
    },
  };

  // 도넛 차트 옵션
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: "55%",
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (context: { label?: string; parsed: number }) {
            return `${context.label}: ${context.parsed.toFixed(1)}%`;
          },
        },
      },
      datalabels: {
        color: "#fff",
        font: {
          weight: "bold" as const,
          size: 11,
        },
        formatter: (value: number) => {
          // 5% 이상인 경우만 표시
          if (value >= 5) {
            return `${value.toFixed(0)}%`;
          }
          return "";
        },
        textAlign: "center" as const,
      },
    },
  };

  // 로딩 상태
  if (loading) {
    return (
      <div className="container">
        <section className="section-wrap fade-in">
          <div className="section-title">맞춤 포트폴리오 추천 결과</div>
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--muted)" }}>
            포트폴리오를 불러오는 중...
          </div>
        </section>
      </div>
    );
  }

  // 에러 상태 (해당 조건의 포트폴리오가 없는 경우)
  if (error) {
    return (
      <div className="container">
        <section className="section-wrap fade-in">
          <div className="section-title">맞춤 포트폴리오 추천 결과</div>

          {/* 사용자 선택 요약 */}
          <div
            style={{
              background: "#f8f9fc",
              padding: "12px 16px",
              borderRadius: "10px",
              marginBottom: "24px",
              fontSize: "13px",
            }}
          >
            <strong>선택하신 조건:</strong> 은퇴시점 {retireYear}년 | 목표수익률{" "}
            {(() => { const v = Math.round(Number(targetReturn) * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}% | 지역: {getCountryDisplay(country)} | 테마:{" "}
            {theme || "미선택"}
          </div>

          {/* 에러 메시지 */}
          <div
            style={{
              background: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: "12px",
              padding: "40px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "48px", marginBottom: "16px" }}>:(</div>
            <div
              style={{
                fontSize: "18px",
                fontWeight: 700,
                color: "#dc2626",
                marginBottom: "8px",
              }}
            >
              {error}
            </div>
            <div style={{ fontSize: "14px", color: "var(--muted)" }}>
              다른 조건으로 다시 시도해 주세요.
            </div>
          </div>

          {/* 버튼 */}
          <div
            className="confirm"
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: "24px",
            }}
          >
            <button className="btn gold" onClick={() => router.push("/")}>
              ← 다시 추천받기
            </button>
          </div>
        </section>
      </div>
    );
  }

  // 데이터가 있는 경우
  const chartData = calculateChartData();
  const { allocation, products, metrics, portfolioId } = portfolioData!;

  return (
    <div className="container">
      <section className="section-wrap fade-in">
        <div className="section-title">맞춤 포트폴리오 추천 결과</div>

        {/* 사용자 선택 요약 */}
        <div
          style={{
            background: "#f8f9fc",
            padding: "12px 16px",
            borderRadius: "10px",
            marginBottom: "24px",
            fontSize: "13px",
          }}
        >
          <strong>선택하신 조건:</strong> 은퇴시점 {retireYear}년 | 목표수익률{" "}
          {(() => { const v = Math.round(Number(targetReturn) * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}% | 지역: {getCountryDisplay(country)} | 테마:{" "}
          {theme || "분산형"}
        </div>

        {/* 도넛 차트 + 범례 */}
        <div className="chart-box">
          <div className="chart-title">포트폴리오 구성</div>
          {/* 기대 수익률 & 손실한계선 */}
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: "40px",
              marginBottom: "24px",
              padding: "20px 24px",
            }}
          >
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "12px", color: "var(--muted)", marginBottom: "6px", fontWeight: 500 }}>기대 수익률</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "#0A2972" }}>{metrics.expectedReturn.toFixed(2)}%</div>
            </div>
            <div style={{ width: "1px", background: "#d1d5db" }} />
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "12px", color: "var(--muted)", marginBottom: "6px", fontWeight: 500 }}>손실한계선(VaR 95%)</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "#dc2626" }}>-{Math.abs(metrics.var95).toFixed(2)}%</div>
            </div>
          </div>
          <div
            style={{
              display: "flex",
              gap: "30px",
              alignItems: "flex-start",
              flexWrap: "wrap",
            }}
          >
            {/* 도넛 차트 */}
            <div style={{ flex: "1", minWidth: "250px", height: "280px" }}>
              {chartData && <Doughnut data={chartData} options={chartOptions} />}

              {/* 차트 범례 - 0% 항목 제외 */}
              <div style={{ marginTop: "16px" }}>
                <div style={{ display: "flex", gap: "12px", justifyContent: "center", flexWrap: "wrap" }}>
                  {chartData?.labels.map((label, index) => {
                    const value = chartData.datasets[0].data[index];
                    if (value < 1) return null;
                    return (
                      <div key={label} style={{ display: "flex", alignItems: "center", fontSize: "11px" }}>
                        <div
                          style={{
                            width: "10px",
                            height: "10px",
                            borderRadius: "2px",
                            backgroundColor: chartData.datasets[0].backgroundColor[index],
                            marginRight: "4px",
                          }}
                        />
                        <span style={{ color: "var(--muted)" }}>{label}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* 상위 10개 상품 리스트 */}
            <div
              style={{
                flex: "2",
                minWidth: "500px",
              }}
            >
              <div style={{ fontSize: "13px", fontWeight: 700, color: "var(--navy)", marginBottom: "12px" }}>
                상위 {products.top10.length}개 상품 (총 {products.total}개)
              </div>
              <div style={{ maxHeight: "300px", overflowY: "auto" }}>
                {products.top10.map((product) => {
                  // API에서 제공하는 분류 정보 사용
                  const productType = product.productType || "";
                  const productRegion = product.productRegion || "";
                  const productTheme = product.productTheme || "";
                  const isTDF = product.isTDF || false;

                  // 안전자산 여부 (채권형이거나 TDF이면 안전자산)
                  const isSafe = productType === "채권" || isTDF;

                  // 선택된 국가/테마 매칭 확인
                  const selectedRegion = portfolioData?.conditions.region || "";
                  const selectedTheme = portfolioData?.conditions.theme || "";
                  const matchesRegion = productRegion === selectedRegion;
                  const matchesTheme = productTheme === selectedTheme;

                  // 태그 생성 (국가, 테마, TDF 각각 별도로)
                  const tags: { label: string; color: string; bg: string }[] = [];
                  if (matchesRegion) {
                    tags.push({ label: selectedRegion, color: "#3b82f6", bg: "#dbeafe" });
                  }
                  if (matchesTheme) {
                    tags.push({ label: selectedTheme, color: "#92400e", bg: "#fef3c7" });
                  }
                  if (isTDF) {
                    tags.push({ label: "TDF", color: "#065f46", bg: "#d1fae5" });
                  }

                  return (
                    <div
                      key={product.code}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        padding: "8px 10px",
                        borderBottom: "1px solid var(--line)",
                        fontSize: "12px",
                        backgroundColor: isSafe ? "#f0f9ff" : "#fef2f2",
                        borderRadius: "4px",
                        marginBottom: "4px",
                      }}
                    >
                      <div style={{ flex: 1, color: "#334155", lineHeight: 1.4, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span>{product.name}</span>
                        <div style={{ display: "flex", gap: "4px", flexShrink: 0, marginLeft: "8px" }}>
                          {tags.map((tag, i) => (
                            <span
                              key={i}
                              style={{
                                padding: "2px 6px",
                                borderRadius: "4px",
                                fontSize: "10px",
                                fontWeight: 600,
                                color: tag.color,
                                backgroundColor: tag.bg,
                                whiteSpace: "nowrap",
                              }}
                            >
                              {tag.label}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div
                        style={{
                          fontWeight: 700,
                          color: "var(--navy)",
                          marginLeft: "10px",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {product.weight_pct.toFixed(1)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 성장추구/안전자산 비중 바 */}
          <div style={{ marginTop: "24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontSize: "12px" }}>
              <span style={{ color: "#dc2626", fontWeight: 600 }}>성장추구 {allocation.riskAssetWeight.toFixed(1)}%</span>
              <span style={{ color: "#2563eb", fontWeight: 600 }}>안전자산 {allocation.safeAssetWeight.toFixed(1)}%</span>
            </div>
            <div
              style={{
                display: "flex",
                height: "32px",
                borderRadius: "8px",
                overflow: "hidden",
                boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
              }}
            >
              <div
                style={{
                  width: `${allocation.riskAssetWeight}%`,
                  background: "linear-gradient(90deg, #ef4444, #f87171)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: "12px",
                }}
              >
                {allocation.riskAssetWeight >= 15 && `${allocation.riskAssetWeight.toFixed(0)}%`}
              </div>
              <div
                style={{
                  width: `${allocation.safeAssetWeight}%`,
                  background: "linear-gradient(90deg, #60a5fa, #3b82f6)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: "12px",
                }}
              >
                {allocation.safeAssetWeight >= 15 && `${allocation.safeAssetWeight.toFixed(0)}%`}
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: "24px", marginTop: "8px", fontSize: "11px", color: "var(--muted)" }}>
              <span>🔴 성장추구: 주식, 해외펀드 등</span>
              <span>🔵 안전자산: 채권, TDF, MMF 등</span>
            </div>
          </div>
        </div>

        {/* 12개월 수익률 시뮬레이션 차트 */}
        <div className="chart-box" style={{ marginTop: "24px" }}>
          <div className="chart-title">12개월 수익률 시뮬레이션</div>
          <div style={{ fontSize: "13px", color: "var(--muted)", marginBottom: "16px", textAlign: "center" }}>
            100만원 투자 시 예상 자산 변화 (시나리오별)
          </div>
          <div style={{ fontSize: "11px", color: "var(--muted)", marginBottom: "4px" }}>
            단위: 만원
          </div>
          <div style={{ height: "400px", position: "relative" }}>
            {calculateSimulationData() && (
              <Line data={calculateSimulationData()!} options={simulationChartOptions} />
            )}
          </div>
          <div
            style={{
              marginTop: "16px",
              padding: "12px",
              background: "#f8f9fc",
              borderRadius: "8px",
              fontSize: "12px",
              color: "var(--muted)",
            }}
          >
            <strong style={{ color: "var(--navy)" }}>시뮬레이션 안내:</strong>
            <br />
            • <span style={{ color: "#0A2972", fontWeight: 600 }}>기대 수익률</span>: 과거 데이터 기반 예상 수익률
            <br />
            • <span style={{ color: "#10b981", fontWeight: 600 }}>낙관적</span> / <span style={{ color: "#f59e0b", fontWeight: 600 }}>비관적</span>: 변동성을 고려한 상/하단 시나리오
            <br />
            • <span style={{ color: "#dc2626", fontWeight: 600 }}>손실한계선(VaR 95%)</span>: 5% 확률의 예상 최대 손실선
          </div>
        </div>

        {/* 추천 배경 */}
        <div className="reason-box">
          <div className="title">추천 배경</div>
          <div>
            본 포트폴리오는 목표 수익률 <strong>{(() => { const v = Math.round(Number(targetReturn) * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}%</strong>를 초과하는{" "}
            <strong>{metrics.expectedReturn.toFixed(2)}%</strong>의 기대 수익률을 제공하며 손실 가능성을 최소화하는 전략으로 구성되었습니다.{" "}
            <strong>{getCountryDisplay(country)}</strong> 지역과 <strong>{theme || "분산형"}</strong> 테마에 우선 투자하여 고객님의 투자 성향을 반영하였으며,{" "}
            TDF를 통해 <strong>{retireYear}년</strong> 은퇴시점까지 안정적인 자산 전환이 이루어지도록 구성하였습니다.
          </div>
        </div>

        {/* 버튼 영역 */}
        <div
          className="confirm"
          style={{
            display: "flex",
            justifyContent: "flex-end",
            marginTop: "24px",
            gap: "10px",
          }}
        >
          <button className="btn ghost" onClick={() => router.push("/")}>
            ← 다시 추천받기
          </button>
          <button
            className={`btn ${isInCart(portfolioId) ? "disabled" : "primary"}`}
            onClick={() => {
              if (portfolioData && !isInCart(portfolioId)) {
                addToCart({
                  portfolioId: portfolioId,
                  conditions: {
                    region: country,
                    theme: theme,
                    targetReturn: Number(targetReturn),
                    retireYear: Number(retireYear),
                  },
                  metrics: {
                    expectedReturn: metrics.expectedReturn,
                    var95: metrics.var95,
                  },
                  allocation: {
                    riskAssetWeight: allocation.riskAssetWeight,
                    safeAssetWeight: allocation.safeAssetWeight,
                    tdfWeight: allocation.tdfWeight,
                  },
                  products: products.top10.map((p) => ({
                    code: p.code,
                    name: p.name,
                    weight_pct: p.weight_pct,
                    productRegion: p.productRegion,
                    productTheme: p.productTheme,
                    productType: p.productType,
                    isTDF: p.isTDF,
                  })),
                  totalProducts: products.total,
                });
                setShowCartModal(true);
              }
            }}
            disabled={isInCart(portfolioId)}
          >
            {isInCart(portfolioId) ? "장바구니에 담김" : "장바구니에 담기"}
          </button>
          <button
            className="btn gold"
            onClick={() => {
              const params = new URLSearchParams({
                retireYear: retireYear,
                targetReturn: targetReturn,
                country: country,
                theme: theme,
              });
              router.push(`/report?${params.toString()}`);
            }}
          >
            자세한 분석 보기 →
          </button>
        </div>
      </section>

      {/* 장바구니 담기 완료 모달 */}
      {showCartModal && (
        <div
          className="modal-overlay"
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
          onClick={() => setShowCartModal(false)}
        >
          <div
            className="modal"
            style={{
              background: "white",
              borderRadius: "20px",
              padding: "40px",
              maxWidth: "400px",
              width: "90%",
              textAlign: "center",
              boxShadow: "0 20px 60px rgba(0, 0, 0, 0.2)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: "64px", marginBottom: "20px" }}>✅</div>
            <div style={{ fontSize: "22px", fontWeight: 700, color: "#1f2937", marginBottom: "12px" }}>
              장바구니에 담겼습니다
            </div>
            <div style={{ fontSize: "15px", color: "#6b7280", marginBottom: "28px", lineHeight: 1.6 }}>
              선택하신 포트폴리오가<br />장바구니에 추가되었습니다.
            </div>
            <div style={{ display: "flex", gap: "12px", justifyContent: "center" }}>
              <button
                style={{
                  flex: 1,
                  padding: "14px 20px",
                  borderRadius: "10px",
                  fontSize: "15px",
                  fontWeight: 600,
                  cursor: "pointer",
                  border: "none",
                  background: "#f3f4f6",
                  color: "#374151",
                }}
                onClick={() => setShowCartModal(false)}
              >
                계속 쇼핑하기
              </button>
              <button
                style={{
                  flex: 1,
                  padding: "14px 20px",
                  borderRadius: "10px",
                  fontSize: "15px",
                  fontWeight: 600,
                  cursor: "pointer",
                  border: "none",
                  background: "linear-gradient(135deg, #0A2972 0%, #1e40af 100%)",
                  color: "white",
                }}
                onClick={() => router.push("/cart")}
              >
                장바구니 보기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function RecommendationPage() {
  return (
    <Suspense
      fallback={
        <div className="container">
          <div className="section-wrap">로딩 중...</div>
        </div>
      }
    >
      <RecommendationContent />
    </Suspense>
  );
}
