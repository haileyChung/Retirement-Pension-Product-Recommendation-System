"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface PortfolioData {
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
}

// 사분면 데이터 (목표수익률 10% 기준)
const QUADRANT_DATA = [
  { region: "미국", x: 3.52, y: 37.1, color: "#22c55e" },
  { region: "한국", x: 3.51, y: 37.25, color: "#3b82f6" },
  { region: "아시아", x: 3.55, y: 37.35, color: "#a855f7" },
  { region: "중국", x: 4.08, y: 36.2, color: "#ef4444" },
];

export default function RebalancingPage() {
  const router = useRouter();
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [currentTheme, setCurrentTheme] = useState<string>("반도체");
  const [currentCountry, setCurrentCountry] = useState<string>("미국");
  const [var95, setVar95] = useState<number>(3.15);
  const [yearsToRetire, setYearsToRetire] = useState<number>(10);

  useEffect(() => {
    // localStorage에서 포트폴리오 데이터 로드
    const savedData = localStorage.getItem("purchasedPortfolios");
    if (savedData) {
      const portfolios = JSON.parse(savedData);
      if (portfolios.length > 0) {
        const latest = portfolios[0];
        setPortfolioData(latest);
        setCurrentTheme(latest.conditions?.theme || "반도체");
        setCurrentCountry(latest.conditions?.region || "미국");
        setVar95(Math.abs(latest.metrics?.var95 || 3.15));
        // 은퇴년도에서 현재 연도(2026) 빼기
        const retireYear = latest.conditions?.retireYear || 2036;
        setYearsToRetire(retireYear - 2026);
      }
    }
  }, []);

  // 사분면 차트에서 좌표 계산
  const getChartPosition = (x: number, y: number) => {
    // x: 3.2 ~ 4.4 -> 0% ~ 100%
    // y: 35.5 ~ 38.0 -> 100% ~ 0%
    const xPercent = ((x - 3.2) / (4.4 - 3.2)) * 100;
    const yPercent = ((38.0 - y) / (38.0 - 35.5)) * 100;
    return { left: `${xPercent}%`, top: `${yPercent}%` };
  };

  return (
    <>
      <style jsx>{`
        .rebalancing-container {
          max-width: 900px;
          margin: 0 auto;
          padding: 40px 24px;
          min-height: calc(100vh - 80px);
        }
        .page-title {
          font-size: 28px;
          font-weight: 700;
          color: #111827;
          margin-bottom: 24px;
        }
        .intro-card {
          background: linear-gradient(135deg, #fef2f2 0%, #fff7ed 100%);
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 32px;
          border-left: 4px solid #dc2626;
        }
        .intro-card p {
          font-size: 15px;
          color: #374151;
          line-height: 1.8;
          margin: 0 0 12px 0;
        }
        .intro-card p:last-child {
          margin-bottom: 0;
        }
        .intro-card strong {
          color: #dc2626;
          font-weight: 600;
        }
        .section-title {
          font-size: 20px;
          font-weight: 700;
          color: #111827;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .chart-container {
          background: #ffffff;
          border-radius: 16px;
          padding: 40px;
          padding-left: 100px;
          padding-bottom: 210px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
          margin-bottom: 32px;
        }
        .chart-wrapper {
          position: relative;
        }
        .chart-title {
          font-size: 16px;
          font-weight: 600;
          color: #374151;
          text-align: center;
          margin-bottom: 20px;
        }
        .quadrant-chart {
          position: relative;
          width: 100%;
          height: 400px;
        }
        .quadrant-inner {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          overflow: hidden;
        }
        .quadrant {
          position: absolute;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          padding: 16px;
        }
        .quadrant-tl {
          top: 0;
          left: 0;
          width: 38.33%;
          height: 41.2%;
          background: #fef9c3;
        }
        .quadrant-tr {
          top: 0;
          left: 38.33%;
          width: 61.67%;
          height: 41.2%;
          background: #fecaca;
        }
        .quadrant-bl {
          top: 41.2%;
          left: 0;
          width: 38.33%;
          height: 58.8%;
          background: #dcfce7;
        }
        .quadrant-br {
          top: 41.2%;
          left: 38.33%;
          width: 61.67%;
          height: 58.8%;
          background: #dbeafe;
        }
        .quadrant-label {
          font-size: 14px;
          font-weight: 600;
          color: #374151;
          margin-bottom: 4px;
        }
        .quadrant-desc {
          font-size: 11px;
          color: #6b7280;
          text-align: center;
        }
        .chart-point {
          position: absolute;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          transform: translate(-50%, -50%);
          display: flex;
          justify-content: center;
          align-items: center;
          font-size: 8px;
          color: white;
          font-weight: 600;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
          z-index: 10;
        }
        .chart-point-label {
          position: absolute;
          transform: translate(-50%, -150%);
          font-size: 12px;
          font-weight: 600;
          white-space: nowrap;
          z-index: 11;
        }
        .chart-point-label-right {
          position: absolute;
          transform: translate(70%, -50%);
          font-size: 12px;
          font-weight: 600;
          white-space: nowrap;
          z-index: 11;
        }
        .chart-axis-x {
          position: absolute;
          bottom: -55px;
          left: 0;
          right: 0;
          text-align: center;
          font-size: 12px;
          color: #6b7280;
        }
        .chart-axis-y {
          position: absolute;
          top: 50%;
          left: -140px;
          transform: translateY(-50%) rotate(-90deg);
          font-size: 12px;
          color: #6b7280;
          white-space: nowrap;
        }
        .y-axis-ticks {
          position: absolute;
          left: -45px;
          top: 0;
          bottom: 0;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          font-size: 11px;
          color: #374151;
          text-align: right;
          width: 35px;
        }
        .x-axis-ticks {
          position: absolute;
          bottom: -25px;
          left: 0;
          right: 0;
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #374151;
        }
        .x-axis-ticks span:first-child {
          margin-left: -8px;
        }
        .x-axis-ticks span:last-child {
          margin-right: -8px;
        }
        .chart-center-line-h {
          position: absolute;
          top: 41.2%;
          left: 0;
          right: 0;
          height: 1px;
          background: rgba(156, 163, 175, 0.5);
          border-top: 1px dashed #9ca3af;
        }
        .chart-center-line-v {
          position: absolute;
          left: 38.33%;
          top: 0;
          bottom: 0;
          width: 1px;
          background: rgba(156, 163, 175, 0.5);
          border-left: 1px dashed #9ca3af;
        }
        .mean-label-x {
          position: absolute;
          top: 10px;
          left: 8px;
          font-size: 11px;
          font-weight: 600;
          color: #6b7280;
          white-space: nowrap;
          background: rgba(254, 202, 202, 0.8);
          padding: 2px 6px;
          border-radius: 4px;
        }
        .mean-label-y {
          position: absolute;
          right: 10px;
          top: 8px;
          font-size: 11px;
          font-weight: 600;
          color: #6b7280;
          white-space: nowrap;
          background: rgba(219, 234, 254, 0.8);
          padding: 2px 6px;
          border-radius: 4px;
        }
        .recommend-arrow {
          position: absolute;
          font-size: 20px;
          z-index: 12;
        }
        .analysis-card {
          background: #f0f9ff;
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 32px;
          border: 1px solid #bae6fd;
        }
        .analysis-title {
          font-size: 16px;
          font-weight: 600;
          color: #0369a1;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .analysis-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 0;
          border-bottom: 1px solid #e0f2fe;
        }
        .analysis-row:last-child {
          border-bottom: none;
        }
        .analysis-label {
          font-size: 14px;
          color: #374151;
        }
        .analysis-value {
          font-size: 14px;
          font-weight: 600;
          color: #111827;
        }
        .analysis-value.positive {
          color: #16a34a;
        }
        .analysis-value.negative {
          color: #dc2626;
        }
        .comparison-box {
          display: flex;
          gap: 16px;
          margin-top: 16px;
        }
        .comparison-item {
          flex: 1;
          background: white;
          border-radius: 12px;
          padding: 16px;
          text-align: center;
        }
        .comparison-item.current {
          border: 2px solid #fca5a5;
        }
        .comparison-item.recommend {
          border: 2px solid #86efac;
        }
        .comparison-badge {
          font-size: 11px;
          font-weight: 600;
          padding: 4px 8px;
          border-radius: 12px;
          margin-bottom: 8px;
          display: inline-block;
        }
        .comparison-badge.current {
          background: #fef2f2;
          color: #dc2626;
        }
        .comparison-badge.recommend {
          background: #f0fdf4;
          color: #16a34a;
        }
        .comparison-region {
          font-size: 18px;
          font-weight: 700;
          color: #111827;
          margin-bottom: 4px;
        }
        .comparison-quadrant {
          font-size: 12px;
          color: #6b7280;
        }
        .btn-primary {
          width: 100%;
          padding: 16px 24px;
          background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
          color: white;
          border: none;
          border-radius: 12px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          margin-bottom: 12px;
          transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(10, 41, 114, 0.3);
        }
        .btn-secondary {
          width: 100%;
          padding: 14px 24px;
          background: white;
          color: #374151;
          border: 1px solid #d1d5db;
          border-radius: 12px;
          font-size: 15px;
          font-weight: 500;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: background 0.2s;
        }
        .btn-secondary:hover {
          background: #f9fafb;
        }
        .recommend-arrow {
          position: absolute;
          font-size: 20px;
          z-index: 12;
        }
      `}</style>

      <div className="rebalancing-container">
        <h1 className="page-title">리밸런싱 전략 제안</h1>

        {/* 상단 멘트 */}
        <div className="intro-card">
          <p>
            고객님의 포트폴리오가 <strong>손실한계선(VaR 95%)인 {var95.toFixed(2)}%</strong>을
            초과하여 리밸런싱이 필요한 상황입니다.
          </p>
          <p>
            현재 보유하신 <strong>{currentTheme}</strong> 테마는 최근 시장
            변동성이 높아 위험 수준이 상승하였습니다.
          </p>
          <p>
            은퇴까지 <strong>{yearsToRetire}년</strong> 남은 시점에서, 주요 투자국가인 <strong>{currentCountry}</strong> 시장의 리스크를 고려하여 포트폴리오 상품 변경을 권장 드립니다.
          </p>
        </div>

        {/* 사분면 차트 */}
        <div className="section-title">
          <span>📊</span> 리스크 프로파일 사분면 분석
        </div>
        <div className="chart-container">
          <div className="chart-title">리스크 프로파일 사분면 분석 (목표수익률 10%)</div>
          <div className="chart-wrapper">
          <div className="quadrant-chart">
            {/* 축 레이블 - 차트 바깥 */}
            <div className="chart-axis-x">
              평상시 손실 한계선 ← 안정 | 불안정 →
            </div>
            <div className="chart-axis-y">
              위기 시 손실 가능성 ← 강함 | 취약 →
            </div>

            {/* Y축 눈금값 */}
            <div className="y-axis-ticks">
              <span>38.0</span>
              <span>37.5</span>
              <span>37.0</span>
              <span>36.5</span>
              <span>36.0</span>
              <span>35.5</span>
            </div>

            {/* X축 눈금값 */}
            <div className="x-axis-ticks">
              <span>3.2</span>
              <span>3.4</span>
              <span>3.6</span>
              <span>3.8</span>
              <span>4.0</span>
              <span>4.2</span>
              <span>4.4</span>
            </div>

            {/* 사분면 영역 (클리핑됨) */}
            <div className="quadrant-inner">
              {/* 사분면 배경 */}
              <div className="quadrant quadrant-tl"></div>
              <div className="quadrant quadrant-tr"></div>
              <div className="quadrant quadrant-bl"></div>
              <div className="quadrant quadrant-br"></div>

              {/* 중심선 */}
              <div className="chart-center-line-h">
                <span className="mean-label-y">평균 36.97%</span>
              </div>
              <div className="chart-center-line-v">
                <span className="mean-label-x">평균 3.66%</span>
              </div>

              {/* 데이터 포인트 */}
              {QUADRANT_DATA.map((point) => {
                const pos = getChartPosition(point.x, point.y);
                const isUSA = point.region === "미국";
                return (
                  <div key={point.region}>
                    <div
                      className="chart-point"
                      style={{
                        ...pos,
                        backgroundColor: point.color,
                      }}
                    />
                    <div
                      className={isUSA ? "chart-point-label-right" : "chart-point-label"}
                      style={{
                        left: pos.left,
                        top: pos.top,
                        color: point.color,
                      }}
                    >
                      {point.region}
                    </div>
                  </div>
                );
              })}

            </div>
          </div>
          </div>
        </div>

        {/* 분석 결과 */}
        <div className="analysis-card">
          <div className="analysis-title">
            <span>💡</span> 분석 결과
          </div>

          <div className="comparison-box">
            <div className="comparison-item current">
              <span className="comparison-badge current">현재 포트폴리오</span>
              <div className="comparison-region">{currentTheme}</div>
              <div className="comparison-quadrant">숨은 위험 영역</div>
            </div>
            <div className="comparison-item recommend">
              <span className="comparison-badge recommend">추천 포트폴리오</span>
              <div className="comparison-region">중국 테마</div>
              <div className="comparison-quadrant">위기에 강함 영역</div>
            </div>
          </div>

          <div style={{ marginTop: "20px" }}>
            <div className="analysis-row">
              <span className="analysis-label">손실 한계선</span>
              <span className="analysis-value">-3.5% → -4.1% (소폭 감소)</span>
            </div>
            <div className="analysis-row">
              <span className="analysis-label">위기시 손실</span>
              <span className="analysis-value positive">37.3% → 36.2% (1.1%p 개선)</span>
            </div>
          </div>
        </div>

        {/* 버튼 */}
        <button className="btn-primary" onClick={() => router.push("/survey")}>
          <span>📋</span> 포트폴리오 리밸런싱 하러가기
        </button>
        <button className="btn-secondary" onClick={() => router.push("/mypage")}>
          <span>🏠</span> 마이페이지로 돌아가기
        </button>
      </div>
    </>
  );
}
