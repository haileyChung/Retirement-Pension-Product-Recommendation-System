"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect, Suspense } from "react";
import { useCart } from "@/context/CartContext";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TooltipItem,
  RadialLinearScale,
} from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { Line, Radar } from "react-chartjs-2";
import "./report.css";

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartDataLabels,
  RadialLinearScale
);

// 포트폴리오 데이터 타입
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

// 리포트 섹션 데이터 타입
interface ReportSections {
  section1: string; // 포트폴리오 구성 상품 설명
  section2: string; // 기대 손실감수수준 및 예상 수익률
  section3: string; // 시장 전망
  section4: string; // 종합 평가
  timeline: Record<number, string>; // 타임라인 설명
  references?: {
    insights: Array<{
      title: string;
      date: string;
      docId: string;
    }>;
    news: Array<{
      title: string;
      date: string;
      link: string;
    }>;
  };
}

// VaR 비교 데이터 타입
interface VarComparisonData {
  targetReturn: number;
  totalCount: number;
  portfolios: Array<{
    portfolioId: number;
    region: string;
    theme: string;
    var95: number;
  }>;
  stats: {
    minVar: number;
    maxVar: number;
    avgVar: number;
  };
}

function ReportContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { addToCart, isInCart } = useCart();

  // URL 파라미터
  const retireYear = searchParams.get("retireYear") || "2040";
  const targetReturn = searchParams.get("targetReturn") || "0.07";
  const country = searchParams.get("country") || "";
  const theme = searchParams.get("theme") || "";

  // 참고 자료 타입
  interface ReferencesData {
    insights: Array<{ title: string; date: string; docId: string }>;
    news: Array<{ title: string; date: string; link: string }>;
  }

  // 상태 관리
  const [loading, setLoading] = useState(true);
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [reportSections, setReportSections] = useState<ReportSections | null>(null);
  const [varComparisonData, setVarComparisonData] = useState<VarComparisonData | null>(null);
  const [referencesData, setReferencesData] = useState<ReferencesData | null>(null);
  const [sectionsLoading, setSectionsLoading] = useState({
    section1: true,
    section2: true,
    section3: true,
    section4: true,
  });
  const [openAccordions, setOpenAccordions] = useState<Record<number, boolean>>({
    1: true,
    2: false,
    3: false,
    4: false,
  });
  const [showCartModal, setShowCartModal] = useState(false);

  // 국가 표시명 변환
  const getCountryDisplay = (value: string) => {
    if (value === "지역기타") return "기타 지역";
    return value || "글로벌";
  };

  // 12개월 수익률 시뮬레이션 데이터 계산
  const calculateSimulationData = () => {
    if (!portfolioData) return null;

    const { metrics } = portfolioData;
    const expectedReturn = metrics.expectedReturn; // 연간 기대 수익률 (%)
    const var95 = metrics.var95; // VaR 95% (최대 손실, 음수)

    // 월별 수익률 계산 (연간 수익률을 월별로 환산)
    const monthlyReturn = expectedReturn / 12;

    // 변동성 추정 (VaR를 기반으로 표준편차 추정)
    const estimatedVolatility = Math.abs(expectedReturn - var95) / 1.645;
    const monthlyVol = estimatedVolatility / Math.sqrt(12);

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
      const optimisticMonthly = monthlyReturn + monthlyVol * 1.5;
      return 100 * Math.pow(1 + optimisticMonthly / 100, m);
    });

    // 비관적 시나리오 (기대수익률 - 1.5 표준편차)
    const pessimisticLine = months.map((m) => {
      if (m === 0) return 100;
      const pessimisticMonthly = monthlyReturn - monthlyVol * 1.5;
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

  // ================================================================================
  // 포트폴리오 종합 평가 점수 계산 (5가지 항목)
  // ================================================================================
  const calculatePortfolioScores = () => {
    if (!portfolioData) return null;

    const { conditions, metrics, allocation, products } = portfolioData;
    const currentYear = new Date().getFullYear();

    // 1. 기대수익률 점수: 목표 수익률 대비 달성률
    // 목표 초과 달성 시 높은 점수, 미달성 시 낮은 점수
    const targetReturnPct = conditions.targetReturn * 100;
    const expectedReturnPct = metrics.expectedReturn;
    let scoreReturn: number;
    if (expectedReturnPct >= targetReturnPct) {
      // 목표 달성: 80점 기본 + 초과분 보너스 (최대 20점)
      const bonus = Math.min(20, (expectedReturnPct - targetReturnPct) * 5);
      scoreReturn = Math.min(100, 80 + bonus);
    } else {
      // 목표 미달성: 80점에서 차이만큼 감점
      const penalty = (targetReturnPct - expectedReturnPct) * 10;
      scoreReturn = Math.max(0, 80 - penalty);
    }

    // 2. 안정성 점수: 동일 기대수익률 포트폴리오 중 VaR 순위 기반
    // VaR가 낮을수록 좋으므로, 순위가 높을수록(1위에 가까울수록) 높은 점수
    const actualVar = Math.abs(metrics.var95);
    let scoreStability: number;
    let stabilityRank = 0;
    let stabilityTotal = 0;

    if (varComparisonData && varComparisonData.portfolios.length > 0) {
      // 동일 기대수익률 포트폴리오 중 VaR 순위 계산
      const sortedPortfolios = [...varComparisonData.portfolios].sort((a, b) => a.var95 - b.var95);
      stabilityTotal = sortedPortfolios.length;

      // 현재 포트폴리오의 순위 찾기 (VaR가 낮을수록 좋은 순위)
      stabilityRank = sortedPortfolios.findIndex(p =>
        p.region === conditions.region && p.theme === conditions.theme
      ) + 1;

      if (stabilityRank === 0) {
        // 현재 포트폴리오를 찾지 못한 경우 VaR 값으로 순위 추정
        stabilityRank = sortedPortfolios.filter(p => p.var95 < actualVar).length + 1;
      }

      // 순위를 점수로 변환 (1위 = 100점, 꼴찌 = 30점)
      // 점수 = 100 - (순위-1) / (전체-1) * 70
      if (stabilityTotal === 1) {
        scoreStability = 85; // 비교 대상이 없으면 기본 85점
      } else {
        const rankRatio = (stabilityRank - 1) / (stabilityTotal - 1);
        scoreStability = Math.round(100 - rankRatio * 70);
      }
    } else {
      // VaR 비교 데이터가 없는 경우 기본값
      scoreStability = 70;
    }

    // 3. 분산투자 점수: HHI(허핀달 지수) 기반 집중도 역수
    // HHI가 낮을수록(분산될수록) 높은 점수
    const weights = products.top10.map(p => p.weight_pct);
    const hhi = weights.reduce((sum, w) => sum + w * w, 0);
    // HHI: 완전집중=10000, 동일비중10개=1000
    const scoreDiversification = Math.min(100, Math.max(0, (10000 - hhi) / 90));

    // 4. 장기적합성 점수: 글라이드패스 적정성
    const yearsToRetirement = conditions.retireYear - currentYear;
    const riskWeight = allocation.riskAssetWeight;

    // 은퇴까지 남은 기간별 적정 위험자산 비중 범위
    let optimalMin: number, optimalMax: number;
    if (yearsToRetirement >= 20) {
      optimalMin = 60; optimalMax = 80;
    } else if (yearsToRetirement >= 15) {
      optimalMin = 50; optimalMax = 70;
    } else if (yearsToRetirement >= 10) {
      optimalMin = 40; optimalMax = 60;
    } else if (yearsToRetirement >= 5) {
      optimalMin = 25; optimalMax = 45;
    } else {
      optimalMin = 15; optimalMax = 30;
    }

    let scoreLongterm: number;
    if (optimalMin <= riskWeight && riskWeight <= optimalMax) {
      scoreLongterm = 100; // 적정 범위 내
    } else {
      // 범위 벗어난 정도에 따라 감점
      const deviation = Math.min(Math.abs(riskWeight - optimalMin), Math.abs(riskWeight - optimalMax));
      scoreLongterm = Math.max(0, 100 - deviation * 2);
    }

    // 5. 선택조건 부합 점수: 국가&테마 + 국가 + 테마 + TDF 비중 합
    const selectedRegion = conditions.region || "";
    const selectedTheme = conditions.theme || "";
    const regionWeight = portfolioData.breakdown.region[selectedRegion] || 0;
    const themeWeight = portfolioData.breakdown.theme[selectedTheme] || 0;
    const tdfWeight = allocation.tdfWeight || 0;

    // 국가&테마 동시 부합 비중 (중복 계산 방지를 위해 최소값의 절반)
    const bothMatchWeight = Math.min(regionWeight, themeWeight) * 0.5;
    const regionOnlyWeight = Math.max(0, regionWeight - bothMatchWeight);
    const themeOnlyWeight = Math.max(0, themeWeight - bothMatchWeight);

    // 선택 조건 부합 비중 합계
    const conditionMatchWeight = bothMatchWeight + regionOnlyWeight + themeOnlyWeight + tdfWeight;
    const scoreConditionMatch = Math.min(100, conditionMatchWeight); // 비중 합계가 곧 점수 (최대 100)

    // 종합 평점 계산 (5개 점수 평균 → 5점 만점 환산)
    const avgScore = (scoreReturn + scoreStability + scoreDiversification + scoreLongterm + scoreConditionMatch) / 5;
    const overallRating = (avgScore / 100) * 5;

    return {
      scores: {
        return: Math.round(scoreReturn * 10) / 10,
        stability: Math.round(scoreStability * 10) / 10,
        diversification: Math.round(scoreDiversification * 10) / 10,
        longterm: Math.round(scoreLongterm * 10) / 10,
        conditionMatch: Math.round(scoreConditionMatch * 10) / 10,
      },
      explanations: {
        return: `목표 ${(() => { const v = Math.round(targetReturnPct * 10) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}% vs 기대 ${expectedReturnPct.toFixed(1)}%`,
        stability: stabilityTotal > 0
          ? `VaR 순위 ${stabilityRank}/${stabilityTotal}위 (${actualVar.toFixed(1)}%)`
          : `VaR ${actualVar.toFixed(1)}%`,
        diversification: `HHI ${Math.round(hhi)}`,
        longterm: `적정 ${optimalMin}-${optimalMax}% vs 실제 ${riskWeight.toFixed(0)}%`,
        conditionMatch: `선택 조건 부합 비중 ${conditionMatchWeight.toFixed(0)}%`,
      },
      overallRating: Math.round(overallRating * 10) / 10,
      hhi: Math.round(hhi),
      optimalRange: { min: optimalMin, max: optimalMax },
      yearsToRetirement,
    };
  };

  // 레이더 차트 데이터
  const getRadarChartData = () => {
    const scores = calculatePortfolioScores();
    if (!scores) return null;

    return {
      labels: ["기대수익률", "안정성", "분산투자", "장기적합성", "선택조건부합"],
      datasets: [
        {
          label: "포트폴리오 점수",
          data: [
            scores.scores.return,
            scores.scores.stability,
            scores.scores.diversification,
            scores.scores.longterm,
            scores.scores.conditionMatch,
          ],
          backgroundColor: "rgba(10, 41, 114, 0.15)",
          borderColor: "#0A2972",
          borderWidth: 2,
          pointBackgroundColor: "#D4A853",
          pointBorderColor: "#fff",
          pointBorderWidth: 2,
          pointRadius: 5,
          pointHoverBackgroundColor: "#D4A853",
          pointHoverBorderColor: "#fff",
        },
      ],
    };
  };

  // 레이더 차트 옵션
  const radarChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      datalabels: {
        display: false,
      },
    },
    scales: {
      r: {
        angleLines: {
          color: "rgba(0, 0, 0, 0.1)",
        },
        grid: {
          color: "rgba(0, 0, 0, 0.1)",
        },
        pointLabels: {
          font: {
            size: 12,
            weight: 600 as const,
          },
          color: "#333",
        },
        ticks: {
          stepSize: 20,
          font: {
            size: 10,
          },
          color: "#666",
          backdropColor: "transparent",
        },
        suggestedMin: 0,
        suggestedMax: 100,
      },
    },
  };

  // 포트폴리오 데이터 로드
  useEffect(() => {
    const fetchPortfolio = async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          region: country,
          theme: theme,
          targetReturn: targetReturn,
          retireYear: retireYear,
        });

        const response = await fetch(`/api/portfolio?${params.toString()}`);
        const result = await response.json();

        if (result.success) {
          setPortfolioData(result.data);
          // 리포트 섹션 생성 요청
          fetchReportSections(result.data);
          // VaR 비교 데이터 로드
          fetchVarComparison(result.data.conditions.targetReturn);
          // 참고 자료 로드 (별도 API로 빠르게)
          fetchReferences(result.data.conditions.region, result.data.conditions.theme);
        }
      } catch (err) {
        console.error("포트폴리오 로드 실패:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();
  }, [country, theme, targetReturn, retireYear]);

  // 백엔드 API URL
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // VaR 비교 데이터 로드
  const fetchVarComparison = async (targetReturnValue: number) => {
    try {
      const response = await fetch(
        `${API_URL}/api/var-comparison?targetReturn=${targetReturnValue}`
      );
      const result = await response.json();
      if (result.success) {
        setVarComparisonData(result.data);
      }
    } catch (err) {
      console.error("VaR 비교 데이터 로드 실패:", err);
    }
  };

  // 참고 자료 로드 (별도 API - GPT와 독립적으로 빠르게 로드)
  const fetchReferences = async (region: string, themeValue: string) => {
    try {
      const response = await fetch(
        `${API_URL}/api/references?region=${encodeURIComponent(region)}&theme=${encodeURIComponent(themeValue)}`
      );
      const result = await response.json();
      if (result.success && result.data) {
        setReferencesData(result.data);
      } else {
        // 실패 시 빈 데이터로 설정 (로딩 완료 표시)
        setReferencesData({ insights: [], news: [] });
      }
    } catch (err) {
      console.error("참고 자료 로드 실패:", err);
      // 에러 시에도 빈 데이터로 설정
      setReferencesData({ insights: [], news: [] });
    }
  };

  // 리포트 섹션 생성 (GPT API 호출)
  const fetchReportSections = async (data: PortfolioData) => {
    try {
      const response = await fetch(`${API_URL}/api/generate-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          portfolioId: data.portfolioId,
          region: data.conditions.region,
          theme: data.conditions.theme,
          targetReturn: data.conditions.targetReturn,
          retireYear: data.conditions.retireYear,
          expectedReturn: data.metrics.expectedReturn,
          var95: data.metrics.var95,
          riskAssetWeight: data.allocation.riskAssetWeight,
          safeAssetWeight: data.allocation.safeAssetWeight,
          tdfWeight: data.allocation.tdfWeight,
          totalProducts: data.products.total,
          top10Products: data.products.top10,
        }),
      });
      const result = await response.json();
      if (result.success) {
        setReportSections(result.data);
      }
    } catch (err) {
      console.error("리포트 생성 실패:", err);
      // 폴백: 기본 템플릿 사용
      setReportSections(generateFallbackSections(data));
    } finally {
      setSectionsLoading({
        section1: false,
        section2: false,
        section3: false,
        section4: false,
      });
    }
  };

  // 폴백 섹션 생성
  const generateFallbackSections = (data: PortfolioData): ReportSections => {
    const { conditions, metrics, allocation, products } = data;
    const regionDisplay = getCountryDisplay(conditions.region);
    const themeDisplay = conditions.theme || "분산형";

    return {
      section1: `현재 포트폴리오는 <strong>${regionDisplay} ${themeDisplay}</strong> 테마를 중심으로 구성되어 있습니다. 총 <strong>${products.total}개</strong>의 상품으로 다양하게 분산 투자되어 있으며, TDF(타겟데이트펀드)를 <strong>${allocation.tdfWeight.toFixed(1)}%</strong> 편입하여 은퇴 시점에 맞춰 자동으로 안전자산 비중이 조절됩니다.`,
      section2: `고객님이 설정하신 목표 수익률 <strong>${(() => { const v = Math.round(conditions.targetReturn * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}%</strong>를 초과하는 <strong>${metrics.expectedReturn.toFixed(2)}%</strong>의 기대 수익률을 확보하였습니다. 손실한계선은 <strong>-${Math.abs(metrics.var95).toFixed(2)}%</strong>로, 100만원 투자 시 최악의 경우 약 <strong>${Math.abs(metrics.var95 * 10000).toFixed(0)}원</strong>의 손실이 예상됩니다.`,
      section3: `현재 ${regionDisplay} ${themeDisplay} 시장은 성장과 조정이 공존하는 상황입니다. 현대차증권 리서치팀에 따르면, 중장기적 관점에서 매력적인 투자 기회가 있으나 단기 변동성에 대한 대비가 필요합니다.`,
      section4: `고객님의 포트폴리오는 <strong>${regionDisplay} ${themeDisplay}</strong> 선호와 <strong>${conditions.retireYear}년 은퇴</strong> 목표에 전반적으로 잘 부합합니다. 성장추구 자산 ${allocation.riskAssetWeight.toFixed(1)}%, 안전자산 ${allocation.safeAssetWeight.toFixed(1)}%의 균형 잡힌 구성으로 장기 수익과 리스크 관리를 동시에 추구합니다.`,
      timeline: generateTimelineDescriptions(conditions.retireYear),
    };
  };

  // 타임라인 설명 생성
  const generateTimelineDescriptions = (retireYr: number): Record<number, string> => {
    const currentYear = 2026;
    const years = [];
    for (let y = currentYear; y <= retireYr; y += 5) {
      years.push(y);
    }
    if (years[years.length - 1] !== retireYr) {
      years.push(retireYr);
    }

    const descriptions: Record<number, string> = {};
    years.forEach((year, idx) => {
      if (idx === 0) {
        descriptions[year] = `${theme || "테마"} 중심 집중투자 · 장기 성장동력 선점`;
      } else if (idx === years.length - 1) {
        descriptions[year] = "인출전략 중심 재구성 · 안정적 현금흐름 확보";
      } else if (idx === 1) {
        descriptions[year] = "성장자산 비중 유지 · 변동 속 우상향 수익 기대";
      } else if (idx === 2) {
        descriptions[year] = "TDF와 균형 조정 · 수익성과 안전성 동시 확보";
      } else {
        descriptions[year] = "안전자산 단계적 확대 · 은퇴자금 변동성 완화";
      }
    });

    return descriptions;
  };

  // 아코디언 토글
  const toggleAccordion = (num: number) => {
    setOpenAccordions((prev) => ({ ...prev, [num]: !prev[num] }));
  };

  // 오늘 날짜
  const today = new Date();
  const formattedDate = `${today.getFullYear()}년 ${String(today.getMonth() + 1).padStart(2, "0")}월 ${String(today.getDate()).padStart(2, "0")}일`;

  if (loading || !portfolioData) {
    return (
      <div className="container">
        <div className="section-wrap">로딩 중...</div>
      </div>
    );
  }

  const { conditions, metrics, allocation, products, breakdown, portfolioId } = portfolioData;

  // 타임라인 연도 계산
  const getTimelineYears = () => {
    const currentYear = 2026;
    const years = [];
    for (let y = currentYear; y <= conditions.retireYear; y += 5) {
      years.push(y);
    }
    if (years[years.length - 1] !== conditions.retireYear) {
      years.push(conditions.retireYear);
    }
    return years;
  };

  const timelineYears = getTimelineYears();

  return (
    <div className="report-container">
      {/* 리포트 헤더 */}
      <div className="report-header">
        <div className="report-title">포트폴리오 분석 리포트</div>
        <div className="report-sub">개인 맞춤형 퇴직연금 포트폴리오 상세 분석</div>
        <div className="report-meta">
          <div className="meta-item">
            <span className="meta-label">작성일:</span>
            <span className="meta-value">{formattedDate}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">투자 지역/테마:</span>
            <span className="meta-value">
              {getCountryDisplay(country)} / {theme || "분산형"}
            </span>
          </div>
          <div className="meta-item">
            <span className="meta-label">목표 은퇴:</span>
            <span className="meta-value">{retireYear}년</span>
          </div>
        </div>
      </div>

      {/* 핵심 지표 */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value positive">+{metrics.expectedReturn.toFixed(2)}%</div>
          <div className="metric-label">기대 수익률 (연)</div>
          <div className="metric-badge">목표 초과 달성</div>
        </div>
        <div className="metric-card">
          <div className="metric-value negative">-{Math.abs(metrics.var95).toFixed(2)}%</div>
          <div className="metric-label">손실한계선(VaR 95%)</div>
        </div>
        <div className="metric-card">
          <div className="metric-value score">
            {calculatePortfolioScores()?.overallRating.toFixed(1) || "0.0"}/5.0
          </div>
          <div className="metric-label">종합 평점</div>
        </div>
      </div>

      <div className="section-wrap">
        {/* 섹션 1: 포트폴리오 구성 상품 설명 */}
        <div className={`accordion ${openAccordions[1] ? "open" : ""}`}>
          <div className="accordion-header" onClick={() => toggleAccordion(1)}>
            <span>1. 포트폴리오 구성 상품 설명</span>
            <span className="arrow">▼</span>
          </div>
          <div className="accordion-content">
            <div className="accordion-inner">
              {/* 포트폴리오 구성 바 차트 */}
              <div style={{ marginBottom: "24px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--navy)", marginBottom: "12px" }}>
                  포트폴리오 구성 비중
                </div>
                {(() => {
                  const selectedRegion = conditions.region || "";
                  const selectedTheme = conditions.theme || "";
                  const regionWeight = breakdown.region[selectedRegion] || 0;
                  const themeWeight = breakdown.theme[selectedTheme] || 0;
                  const tdfWeight = allocation.tdfWeight || 0;
                  const bothMatchWeight = Math.min(regionWeight, themeWeight) * 0.5;
                  const regionOnlyWeight = Math.max(0, regionWeight - bothMatchWeight);
                  const themeOnlyWeight = Math.max(0, themeWeight - bothMatchWeight);
                  const otherWeight = Math.max(0, 100 - regionOnlyWeight - themeOnlyWeight - bothMatchWeight - tdfWeight);

                  const barData = [
                    { label: `${selectedRegion} & ${selectedTheme}`, value: bothMatchWeight, color: "#0A2972" },
                    { label: `${selectedRegion}`, value: regionOnlyWeight, color: "#3b82f6" },
                    { label: `${selectedTheme}`, value: themeOnlyWeight, color: "#D5B45C" },
                    { label: "TDF", value: tdfWeight, color: "#10b981" },
                    { label: "기타", value: otherWeight, color: "#94a3b8" },
                  ].filter(item => item.value >= 1);

                  return (
                    <>
                      <div style={{ display: "flex", height: "32px", borderRadius: "8px", overflow: "hidden", marginBottom: "12px" }}>
                        {barData.map((item, idx) => (
                          <div
                            key={idx}
                            title={`${item.label}: ${item.value.toFixed(1)}%`}
                            style={{
                              width: `${item.value}%`,
                              backgroundColor: item.color,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              color: "#fff",
                              fontSize: "11px",
                              fontWeight: 600,
                              cursor: "pointer",
                            }}
                          >
                            {item.value >= 10 && `${item.value.toFixed(0)}%`}
                          </div>
                        ))}
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", justifyContent: "center" }}>
                        {barData.map((item, idx) => (
                          <div key={idx} style={{ display: "flex", alignItems: "center", fontSize: "11px" }}>
                            <div style={{ width: "10px", height: "10px", borderRadius: "2px", backgroundColor: item.color, marginRight: "4px" }} />
                            <span style={{ color: "var(--muted)" }}>{item.label}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  );
                })()}
              </div>

              {/* 자산 배분 요약 */}
              <div className="summary-circles">
                <div className="summary-item">
                  <div className="summary-circle risk">
                    <span>{allocation.riskAssetWeight.toFixed(1)}%</span>
                  </div>
                  <div className="summary-label">성장추구</div>
                </div>
                <div className="summary-item">
                  <div className="summary-circle safe">
                    <span>{allocation.safeAssetWeight.toFixed(1)}%</span>
                  </div>
                  <div className="summary-label">안전자산</div>
                </div>
              </div>

              {/* 상품 테이블 */}
              <table className="rec-table" style={{ marginTop: "24px" }}>
                <thead>
                  <tr>
                    <th style={{ width: "70px" }}>유형</th>
                    <th>상품명</th>
                    <th style={{ width: "70px" }}>비중</th>
                  </tr>
                </thead>
                <tbody>
                  {products.top10.map((product) => {
                    const isSafe = product.productType === "채권" || product.isTDF;
                    const productRegion = product.productRegion || "";
                    const productTheme = product.productTheme || "";
                    const selectedRegion = conditions.region || "";
                    const selectedTheme = conditions.theme || "";
                    const matchesRegion = productRegion === selectedRegion;
                    const matchesTheme = productTheme === selectedTheme;

                    // 태그 생성
                    const tags: { label: string; color: string; bg: string }[] = [];
                    if (matchesRegion) {
                      tags.push({ label: selectedRegion, color: "#3b82f6", bg: "#dbeafe" });
                    }
                    if (matchesTheme) {
                      tags.push({ label: selectedTheme, color: "#92400e", bg: "#fef3c7" });
                    }
                    if (product.isTDF) {
                      tags.push({ label: "TDF", color: "#065f46", bg: "#d1fae5" });
                    }

                    return (
                      <tr key={product.code}>
                        <td>
                          <span className={`type-badge ${isSafe ? "safe" : "risk"}`}>
                            {isSafe ? "안전자산" : "성장추구"}
                          </span>
                        </td>
                        <td>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span>{product.name}</span>
                            <div style={{ display: "flex", gap: "4px", flexShrink: 0 }}>
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
                                  }}
                                >
                                  {tag.label}
                                </span>
                              ))}
                            </div>
                          </div>
                        </td>
                        <td className="weight">{product.weight_pct.toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* AI 생성 설명 */}
              <div style={{ marginTop: "20px", lineHeight: 1.8 }}>
                {sectionsLoading.section1 ? (
                  <div style={{ color: "var(--muted)", fontStyle: "italic" }}>
                    AI가 상품 설명을 생성하고 있습니다...
                  </div>
                ) : (
                  <div dangerouslySetInnerHTML={{ __html: reportSections?.section1 || "" }} />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* 섹션 2: 기대 손실감수수준 및 예상 수익률 */}
        <div className={`accordion ${openAccordions[2] ? "open" : ""}`}>
          <div className="accordion-header" onClick={() => toggleAccordion(2)}>
            <span>2. 기대 손실감수수준 및 예상 수익률</span>
            <span className="arrow">▼</span>
          </div>
          <div className="accordion-content">
            <div className="accordion-inner">
              {/* 투자 결과 바 - 중앙(0%) 기준 좌우로 손실/수익 표시 */}
              <div className="result-box">
                <div className="chart-title">100만원 투자 시 예상 결과 (1년 후)</div>

                {/* 단일 바 차트 - 중앙이 0%(100만원) */}
                <div style={{ marginTop: "20px", position: "relative" }}>
                  {/* 상단 라벨 (손실한계선 / 기대 수익) */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "13px", fontWeight: 600, color: "#dc2626" }}>손실한계선(VaR 95%)</span>
                    <span style={{ fontSize: "13px", fontWeight: 600, color: "#0A2972" }}>기대 수익</span>
                  </div>

                  {/* 바 컨테이너 - 중앙 기준 좌우 배치 */}
                  <div style={{
                    display: "flex",
                    height: "40px",
                    borderRadius: "8px",
                    overflow: "visible",
                    position: "relative",
                    backgroundColor: "#e8e8e8",
                    alignItems: "center"
                  }}>
                    {/* 왼쪽 영역 (손실 방향) - 50% */}
                    <div style={{
                      width: "50%",
                      height: "100%",
                      position: "relative",
                      display: "flex",
                      justifyContent: "flex-end",
                      alignItems: "center",
                      borderRadius: "8px 0 0 8px",
                      overflow: "visible"
                    }}>
                      {/* 왼쪽 퍼센트 라벨 (바 끝에 붙임) */}
                      <span style={{
                        position: "absolute",
                        right: `${Math.min(Math.abs(metrics.var95) * 5, 100)}%`,
                        transform: "translateX(-4px)",
                        fontSize: "14px",
                        fontWeight: 600,
                        color: "#dc2626",
                        whiteSpace: "nowrap"
                      }}>
                        -{Math.abs(metrics.var95).toFixed(2)}%
                      </span>
                      {/* 손실 바 (빨간색 - 시뮬레이션과 통일) */}
                      <div style={{
                        width: `${Math.min(Math.abs(metrics.var95) * 5, 100)}%`,
                        height: "100%",
                        backgroundColor: "#dc2626",
                        borderRadius: "4px 0 0 4px"
                      }} />
                    </div>

                    {/* 중앙선 (0% = 100만원) */}
                    <div style={{
                      width: "2px",
                      height: "100%",
                      backgroundColor: "#666",
                      zIndex: 2,
                      position: "relative"
                    }}>
                      {/* 0% 라벨 */}
                      <div style={{
                        position: "absolute",
                        top: "-20px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        fontSize: "10px",
                        color: "#666",
                        whiteSpace: "nowrap"
                      }}>
                        0%
                      </div>
                    </div>

                    {/* 오른쪽 영역 (수익 방향) - 50% */}
                    <div style={{
                      width: "50%",
                      height: "100%",
                      position: "relative",
                      display: "flex",
                      justifyContent: "flex-start",
                      alignItems: "center",
                      borderRadius: "0 8px 8px 0",
                      overflow: "visible"
                    }}>
                      {/* 수익 바 (파란색 - 시뮬레이션과 통일) */}
                      <div style={{
                        width: `${Math.min(metrics.expectedReturn * 5, 100)}%`,
                        height: "100%",
                        backgroundColor: "#0A2972",
                        borderRadius: "0 4px 4px 0"
                      }} />
                      {/* 오른쪽 퍼센트 라벨 (바 끝에 붙임) */}
                      <span style={{
                        position: "absolute",
                        left: `${Math.min(metrics.expectedReturn * 5, 100)}%`,
                        transform: "translateX(4px)",
                        fontSize: "14px",
                        fontWeight: 600,
                        color: "#0A2972",
                        whiteSpace: "nowrap"
                      }}>
                        +{metrics.expectedReturn.toFixed(2)}%
                      </span>
                    </div>

                    {/* 목표 수익률 점선 마커 */}
                    <div style={{
                      position: "absolute",
                      left: `${50 + (conditions.targetReturn * 100 * 2.5)}%`,
                      top: 0,
                      bottom: 0,
                      width: "0px",
                      borderLeft: "2px dashed #10b981",
                      zIndex: 3
                    }}>
                      {/* 목표 수익률 라벨 (상단) */}
                      <div style={{
                        position: "absolute",
                        top: "-20px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        fontSize: "11px",
                        color: "#10b981",
                        whiteSpace: "nowrap",
                        fontWeight: 600
                      }}>
                        목표 {(() => { const v = Math.round(conditions.targetReturn * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}%
                      </div>
                    </div>
                  </div>

                  {/* 하단 금액 라벨 */}
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: "12px" }}>
                    <span style={{ fontSize: "14px", fontWeight: 600, color: "#dc2626" }}>
                      -{Math.round(Math.abs(metrics.var95) * 10000).toLocaleString()}원
                    </span>
                    <span style={{ fontSize: "14px", fontWeight: 600, color: "#0A2972" }}>
                      +{Math.round(metrics.expectedReturn * 10000).toLocaleString()}원
                    </span>
                  </div>
                </div>

              </div>

              {/* 12개월 수익률 시뮬레이션 차트 */}
              <div className="chart-box" style={{ marginTop: "24px" }}>
                <div className="chart-title" style={{ textAlign: "center" }}>12개월 수익률 시뮬레이션</div>
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
              </div>

              {/* AI 생성 설명 */}
              <div style={{ marginTop: "20px", lineHeight: 1.8 }}>
                {sectionsLoading.section2 ? (
                  <div style={{ color: "var(--muted)", fontStyle: "italic" }}>
                    AI가 위험/수익 분석을 생성하고 있습니다...
                  </div>
                ) : (
                  <div dangerouslySetInnerHTML={{ __html: reportSections?.section2 || "" }} />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* 섹션 3: 시장 전망 */}
        <div className={`accordion ${openAccordions[3] ? "open" : ""}`}>
          <div className="accordion-header" onClick={() => toggleAccordion(3)}>
            <span>
              3. {getCountryDisplay(country)} {theme || "분산형"} 시장 전망
            </span>
            <span className="arrow">▼</span>
          </div>
          <div className="accordion-content">
            <div className="accordion-inner">
              {sectionsLoading.section3 ? (
                <div style={{ color: "var(--muted)", fontStyle: "italic" }}>
                  AI가 시장 전망을 분석하고 있습니다...
                </div>
              ) : (
                <div dangerouslySetInnerHTML={{ __html: reportSections?.section3 || "" }} />
              )}
            </div>
          </div>
        </div>

        {/* 섹션 4: 종합 평가 */}
        <div className={`accordion ${openAccordions[4] ? "open" : ""}`}>
          <div className="accordion-header" onClick={() => toggleAccordion(4)}>
            <span>4. 포트폴리오 종합 평가</span>
            <span className="arrow">▼</span>
          </div>
          <div className="accordion-content">
            <div className="accordion-inner">
              {/* 포트폴리오 종합 평가 레이더 차트 */}
              <div className="chart-box">
                <div className="chart-title" style={{ textAlign: "center", marginBottom: "24px" }}>포트폴리오 종합 평가</div>

                {(() => {
                  const scores = calculatePortfolioScores();
                  const radarData = getRadarChartData();

                  if (!scores || !radarData) return null;

                  return (
                    <div style={{
                      display: "flex",
                      gap: "32px",
                      alignItems: "center",
                      justifyContent: "center",
                      flexWrap: "wrap"
                    }}>
                      {/* 레이더 차트 */}
                      <div style={{ width: "340px", height: "340px", flex: "0 0 340px" }}>
                        <Radar data={radarData} options={radarChartOptions} />
                      </div>

                      {/* 점수 산정 기준 테이블 */}
                      <div style={{
                        minWidth: "280px",
                        flex: "0 0 280px",
                        backgroundColor: "#fff",
                        borderRadius: "12px",
                        padding: "20px 24px",
                        boxShadow: "0 2px 8px rgba(0, 0, 0, 0.06)"
                      }}>
                        {/* 테이블 헤더 */}
                        <div style={{
                          marginBottom: "16px"
                        }}>
                          <span style={{
                            fontSize: "15px",
                            fontWeight: 700,
                            color: "#0A2972"
                          }}>
                            점수 산정 기준
                          </span>
                        </div>

                        {/* 기대수익률 */}
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "12px 0",
                          borderBottom: "1px solid #eee"
                        }}>
                          <span style={{ fontSize: "13px", fontWeight: 700, color: "#333" }}>기대수익률</span>
                          <span style={{ fontSize: "13px", color: "#333", fontWeight: 400 }}>{scores.explanations.return}</span>
                        </div>

                        {/* 안정성 */}
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "12px 0",
                          borderBottom: "1px solid #eee"
                        }}>
                          <span style={{ fontSize: "13px", fontWeight: 700, color: "#333" }}>안정성</span>
                          <span style={{ fontSize: "13px", color: "#333", fontWeight: 400 }}>{scores.explanations.stability}</span>
                        </div>

                        {/* 분산투자 */}
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "12px 0",
                          borderBottom: "1px solid #eee"
                        }}>
                          <span style={{ fontSize: "13px", fontWeight: 700, color: "#333" }}>분산투자</span>
                          <span style={{ fontSize: "13px", color: "#333", fontWeight: 400 }}>{scores.explanations.diversification}</span>
                        </div>

                        {/* 장기적합성 */}
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "12px 0",
                          borderBottom: "1px solid #eee"
                        }}>
                          <span style={{ fontSize: "13px", fontWeight: 700, color: "#333" }}>장기적합성</span>
                          <span style={{ fontSize: "13px", color: "#333", fontWeight: 400 }}>{scores.explanations.longterm}</span>
                        </div>

                        {/* 선택조건부합 */}
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "12px 0"
                        }}>
                          <span style={{ fontSize: "13px", fontWeight: 700, color: "#333" }}>선택조건부합</span>
                          <span style={{ fontSize: "13px", color: "#333", fontWeight: 400 }}>{scores.explanations.conditionMatch}</span>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* AI 생성 설명 */}
              <div style={{ marginTop: "24px", lineHeight: 1.8 }}>
                {sectionsLoading.section4 ? (
                  <div style={{ color: "var(--muted)", fontStyle: "italic" }}>
                    AI가 종합 평가를 작성하고 있습니다...
                  </div>
                ) : (
                  <div dangerouslySetInnerHTML={{ __html: reportSections?.section4 || "" }} />
                )}
              </div>

              {/* 타임라인 */}
              <div className="chart-box" style={{ marginTop: "24px" }}>
                <div className="chart-title">은퇴까지의 투자 여정</div>
                <div className="timeline">
                  <div className="timeline-line"></div>

                  {timelineYears.map((year, idx) => {
                    const isFirst = idx === 0;
                    const isLast = idx === timelineYears.length - 1;
                    const dotColor = isFirst ? "navy" : isLast ? "green" : "gold";
                    const emoji = isFirst ? "📈" : isLast ? "🏖️" : idx === 1 ? "⚖️" : "🛡️";

                    return (
                      <div className="timeline-item" key={year}>
                        <div className={`timeline-dot ${dotColor}`}>{emoji}</div>
                        <div className="timeline-content">
                          <div className="timeline-year">
                            {year}년 {isFirst ? "(현재)" : isLast ? "(은퇴)" : ""}
                          </div>
                          <div className="timeline-desc">
                            {reportSections?.timeline?.[year] ||
                              generateTimelineDescriptions(conditions.retireYear)[year] ||
                              "투자 전략 진행 중"}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 버튼 영역 */}
        <div className="confirm" style={{ marginTop: "24px" }}>
          <button className="btn ghost" onClick={() => router.back()}>
            ← 추천 결과로
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
        </div>
      </div>

      {/* 참고자료 */}
      <div className="references">
        <h3>참고 자료 출처</h3>
        <div className="ref-grid">
          <div>
            <div className="ref-title">현대차증권 리서치 인사이트</div>
            <ul className="ref-list">
              {referencesData === null ? (
                <li style={{ color: "var(--muted)", fontStyle: "italic" }}>리서치 인사이트 로딩 중...</li>
              ) : referencesData.insights.length > 0 ? (
                referencesData.insights.map((insight, idx) => (
                  <li key={idx}>[{insight.date}] {insight.title}</li>
                ))
              ) : (
                <li style={{ color: "var(--muted)" }}>관련 리서치 인사이트가 없습니다.</li>
              )}
            </ul>
          </div>
          <div>
            <div className="ref-title">시장 뉴스</div>
            <ul className="ref-list">
              {referencesData === null ? (
                <li style={{ color: "var(--muted)", fontStyle: "italic" }}>시장 뉴스 로딩 중...</li>
              ) : referencesData.news.length > 0 ? (
                referencesData.news.map((news, idx) => (
                  <li key={idx}>
                    <a href={news.link} target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "none" }}>
                      [{news.date}] {news.title}
                    </a>
                  </li>
                ))
              ) : (
                <li style={{ color: "var(--muted)" }}>관련 시장 뉴스가 없습니다.</li>
              )}
            </ul>
          </div>
        </div>
      </div>

      {/* 푸터 */}
      <div className="report-footer">
        <div className="footer-disclaimer">
          ※ 본 리포트는 고려대학교 MSBA 캡스톤 프로젝트를 통해 만들어진 현대차증권 Future Shield Advisor AI가 생성한 리포트입니다.
          <br />화면 내 수치와 분석 결과는 실제 시장 상황과 다를 수 있으며, 현대차증권의 공식적인 의견이 아님을 밝힙니다.
          <br />※ 투자 결정 전 반드시 전문가와 상담하시기 바랍니다. 과거 수익률이 미래 수익률을 보장하지 않습니다.
        </div>
        <div className="footer-logo">현대차증권 리서치센터</div>
      </div>

      {/* 장바구니 담기 완료 모달 */}
      {showCartModal && (
        <div
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
                계속 보기
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

export default function ReportPage() {
  return (
    <Suspense
      fallback={
        <div className="container">
          <div className="section-wrap">리포트 로딩 중...</div>
        </div>
      }
    >
      <ReportContent />
    </Suspense>
  );
}
