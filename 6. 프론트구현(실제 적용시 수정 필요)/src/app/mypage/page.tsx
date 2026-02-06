"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

// 목표 수익률 포맷팅
const formatTargetReturn = (value: number) => {
  const v = Math.round(value * 1000) / 10;
  return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1);
};

// 국가 표시명 변환
const getCountryDisplay = (value: string) => {
  const map: Record<string, string> = {
    "미국": "미국",
    "한국": "한국",
    "중국": "중국",
    "아시아": "아시아",
    "지역기타": "기타 지역",
  };
  return map[value] || value;
};

// 구매한 포트폴리오 타입 (로컬 스토리지에서 가져옴)
interface PurchasedProduct {
  code: string;
  name: string;
  weight_pct: number;
  productRegion?: string;
  productTheme?: string;
  productType?: string;
  isTDF?: boolean;
}

interface PurchasedPortfolio {
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
  products: PurchasedProduct[];
  totalProducts: number;
  purchasedAt: string;
}

export default function MyPage() {
  const router = useRouter();
  const [purchasedData, setPurchasedData] = useState<PurchasedPortfolio[] | null>(null);
  const [isProductListExpanded, setIsProductListExpanded] = useState(false);
  const [showAlertModal, setShowAlertModal] = useState(false);

  useEffect(() => {
    // 로컬 스토리지에서 구매 데이터 가져오기
    const stored = localStorage.getItem("purchasedPortfolios");
    if (stored) {
      setPurchasedData(JSON.parse(stored));
    }
  }, []);

  // 데이터가 없는 경우
  if (purchasedData === null || purchasedData.length === 0) {
    return (
      <>
        <style jsx>{`
          .mypage-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 24px;
            min-height: calc(100vh - 80px);
          }
          .empty-state {
            text-align: center;
            padding: 100px 20px;
          }
          .empty-icon {
            font-size: 80px;
            margin-bottom: 24px;
          }
          .empty-title {
            font-size: 24px;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 12px;
          }
          .empty-desc {
            font-size: 16px;
            color: #6b7280;
            margin-bottom: 32px;
          }
          .btn {
            padding: 14px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
            background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
            color: white;
          }
          .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(10, 41, 114, 0.3);
          }
        `}</style>
        <div className="mypage-container">
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <div className="empty-title">보유 중인 포트폴리오가 없습니다</div>
            <div className="empty-desc">포트폴리오를 추천받고 구매해보세요</div>
            <button className="btn" onClick={() => router.push("/")}>
              포트폴리오 추천받기
            </button>
          </div>
        </div>
      </>
    );
  }

  // 첫 번째 포트폴리오 데이터 사용 (복수 포트폴리오 구매 시 합산 로직 필요 시 추가)
  const portfolio = purchasedData[0];
  const { conditions, metrics, allocation, products, totalProducts } = portfolio;

  // 전체 상품 목록 (복수 포트폴리오 합산, 비중순 정렬)
  const allProducts = purchasedData.flatMap(p => p.products)
    .sort((a, b) => b.weight_pct - a.weight_pct);

  // 총 상품 수
  const totalProductsCount = purchasedData.reduce((sum, p) => sum + p.totalProducts, 0);

  // 평균 기대 수익률, VaR 계산 (복수 포트폴리오 시)
  const avgExpectedReturn = purchasedData.reduce((sum, p) => sum + p.metrics.expectedReturn, 0) / purchasedData.length;
  const avgVar95 = purchasedData.reduce((sum, p) => sum + Math.abs(p.metrics.var95), 0) / purchasedData.length;

  // 현재 수익률 (손실한계선 + 1.16% 하락)
  const currentReturn = (-(avgVar95 + 1.16)).toFixed(2);

  return (
    <>
      <style jsx>{`
        .mypage-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 40px 24px;
          min-height: calc(100vh - 80px);
        }
        .page-header {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 8px;
        }
        .page-title {
          font-size: 28px;
          font-weight: 700;
          color: #0A2972;
        }
        .alert-btn {
          background: #dc2626;
          color: white;
          border: none;
          border-radius: 8px;
          padding: 8px 16px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 6px;
          transition: all 0.2s;
          box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
        }
        .alert-btn:hover {
          background: #b91c1c;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
        }
        .page-subtitle {
          font-size: 16px;
          color: #6b7280;
          margin-bottom: 32px;
        }

        /* 상단 요약 카드 */
        .summary-cards {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 20px;
          margin-bottom: 32px;
        }
        .summary-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 24px;
          text-align: center;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        .card-icon {
          font-size: 32px;
          margin-bottom: 12px;
        }
        .card-label {
          font-size: 14px;
          color: #6b7280;
          margin-bottom: 8px;
        }
        .card-value {
          font-size: 28px;
          font-weight: 700;
        }
        .card-value.positive {
          color: #0A2972;
        }
        .card-value.negative {
          color: #dc2626;
        }
        .card-value.neutral {
          color: #1f2937;
        }

        /* 상품 리스트 섹션 */
        .products-section {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 32px;
        }
        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        .section-title {
          font-size: 20px;
          font-weight: 700;
          color: #1f2937;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .toggle-btn {
          background: #f3f4f6;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 8px 16px;
          font-size: 14px;
          font-weight: 500;
          color: #374151;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .toggle-btn:hover {
          background: #e5e7eb;
        }
        .products-table {
          width: 100%;
          border-collapse: collapse;
        }
        .products-table th {
          text-align: left;
          padding: 12px 16px;
          background: #f9fafb;
          font-size: 14px;
          font-weight: 600;
          color: #374151;
          border-bottom: 1px solid #e5e7eb;
        }
        .products-table td {
          padding: 14px 16px;
          font-size: 14px;
          color: #4b5563;
          border-bottom: 1px solid #f3f4f6;
        }
        .products-table tr:hover {
          background: #f9fafb;
        }
        .product-name {
          font-weight: 500;
          color: #1f2937;
        }
        .weight-bar {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .weight-fill {
          height: 8px;
          background: linear-gradient(90deg, #0A2972, #3b82f6);
          border-radius: 4px;
        }
        .more-indicator {
          text-align: center;
          padding: 16px;
          color: #6b7280;
          font-size: 14px;
        }
        .products-table-container {
          max-height: 500px;
          overflow-y: auto;
        }
        .products-table-container::-webkit-scrollbar {
          width: 8px;
        }
        .products-table-container::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 4px;
        }
        .products-table-container::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 4px;
        }
        .products-table-container::-webkit-scrollbar-thumb:hover {
          background: #a1a1a1;
        }

        /* 목표 적합도 섹션 */
        .suitability-section {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 32px;
        }
        .suitability-content {
          display: flex;
          align-items: center;
          gap: 40px;
          margin-top: 20px;
        }
        .suitability-item {
          flex: 1;
          text-align: center;
          padding: 20px;
          background: #f9fafb;
          border-radius: 12px;
        }
        .suitability-label {
          font-size: 14px;
          color: #6b7280;
          margin-bottom: 8px;
        }
        .suitability-value {
          font-size: 24px;
          font-weight: 700;
          color: #0A2972;
        }
        .suitability-badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 12px 24px;
          border-radius: 50px;
          font-size: 18px;
          font-weight: 700;
        }
        .suitability-badge.suitable {
          background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
          color: #166534;
        }
        .suitability-badge.moderate {
          background: linear-gradient(135deg, #fef9c3 0%, #fde047 100%);
          color: #854d0e;
        }
        .suitability-badge.unsuitable {
          background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
          color: #991b1b;
        }
        .suitability-arrow {
          font-size: 32px;
          color: #d1d5db;
        }

        /* 포트폴리오 정보 */
        .portfolio-info {
          background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
          border-radius: 12px;
          padding: 16px 20px;
          margin-bottom: 24px;
          font-size: 14px;
          color: #0369a1;
        }

        @media (max-width: 768px) {
          .summary-cards {
            grid-template-columns: repeat(2, 1fr);
          }
          .suitability-content {
            flex-direction: column;
            gap: 20px;
          }
          .suitability-arrow {
            transform: rotate(90deg);
          }
        }

        /* 알림 모달 */
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 20px;
        }
        .alert-modal {
          background: white;
          border-radius: 20px;
          padding: 32px;
          max-width: 600px;
          width: 100%;
          text-align: center;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        .alert-modal-icon {
          font-size: 60px;
          margin-bottom: 16px;
        }
        .alert-modal-title {
          font-size: 24px;
          font-weight: 700;
          color: #dc2626;
          margin-bottom: 20px;
        }
        .alert-modal-desc {
          font-size: 15px;
          color: #4b5563;
          line-height: 1.7;
          margin-bottom: 24px;
          text-align: left;
          background: #fef2f2;
          padding: 16px;
          border-radius: 12px;
          border-left: 4px solid #dc2626;
        }
        .alert-graph {
          background: #f9fafb;
          border-radius: 12px;
          padding: 24px;
          margin-bottom: 24px;
        }
        .graph-container {
          position: relative;
          height: 220px;
          margin-top: 20px;
          margin-left: 60px;
          margin-bottom: 10px;
        }
        .graph-y-axis {
          position: absolute;
          left: 0px;
          top: 20px;
          bottom: 40px;
          width: 2px;
          background: #000000;
        }
        .graph-y-axis::before {
          content: "";
          position: absolute;
          top: -6px;
          left: -4px;
          border-left: 5px solid transparent;
          border-right: 5px solid transparent;
          border-bottom: 8px solid #000000;
        }
        .graph-x-axis {
          position: absolute;
          left: 0px;
          right: 0px;
          top: 100px;
          height: 2px;
          background: #000000;
        }
        .graph-x-axis::after {
          content: "";
          position: absolute;
          right: -6px;
          top: -4px;
          border-top: 5px solid transparent;
          border-bottom: 5px solid transparent;
          border-left: 8px solid #000000;
        }
        .graph-label-y {
          position: absolute;
          left: -50px;
          top: 12px;
          font-size: 12px;
          font-weight: 500;
          color: #374151;
          text-align: right;
          width: 45px;
        }
        .graph-label-zero {
          position: absolute;
          left: -30px;
          top: 92px;
          font-size: 11px;
          color: #6b7280;
        }
        .graph-var-line {
          position: absolute;
          left: 0px;
          right: 0px;
          top: 170px;
          height: 0px;
          background: transparent;
          border-top: 2px dashed #dc2626;
        }
        .graph-label-var {
          position: absolute;
          left: -95px;
          top: 162px;
          font-size: 12px;
          font-weight: 500;
          color: #dc2626;
          line-height: 1.2;
          text-align: right;
          width: 90px;
          white-space: nowrap;
        }
        .graph-label-x {
          position: absolute;
          right: 0px;
          top: 108px;
          font-size: 12px;
          font-weight: 500;
          color: #374151;
        }
        .graph-label-current {
          position: absolute;
          right: 55px;
          top: 15px;
          font-size: 12px;
          font-weight: 500;
          color: #374151;
        }
        .graph-current-line {
          position: absolute;
          right: 75px;
          top: 30px;
          bottom: 40px;
          width: 0px;
          border-left: 1px dashed #3b82f6;
        }
        .graph-line-path {
          position: absolute;
          left: 0px;
          top: 20px;
          right: 10px;
          bottom: 30px;
        }
        .graph-line-path svg {
          width: 100%;
          height: 100%;
        }
        .new-proposal-btn {
          width: 100%;
          padding: 16px 24px;
          background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
          color: white;
          border: none;
          border-radius: 12px;
          font-size: 18px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }
        .new-proposal-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(10, 41, 114, 0.3);
        }
        .modal-close-btn {
          position: absolute;
          top: 16px;
          right: 16px;
          background: #f3f4f6;
          border: none;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          font-size: 18px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .modal-close-btn:hover {
          background: #e5e7eb;
        }
      `}</style>

      <div className="mypage-container">
        <div className="page-header">
          <h1 className="page-title">마이페이지</h1>
          <button className="alert-btn" onClick={() => setShowAlertModal(true)}>
            <span>🔔</span> 포트폴리오 알림
          </button>
        </div>
        <p className="page-subtitle">보유 중인 포트폴리오 현황을 확인하세요</p>

        {/* 포트폴리오 정보 */}
        <div className="portfolio-info">
          <strong>보유 포트폴리오:</strong> {getCountryDisplay(conditions.region)} · {conditions.theme || "분산형"} |
          은퇴 목표 {conditions.retireYear}년 |
          구매일 {new Date(portfolio.purchasedAt).toLocaleDateString("ko-KR")}
        </div>

        {/* 상단 요약 카드 */}
        <div className="summary-cards">
          <div className="summary-card">
            <div className="card-icon">📦</div>
            <div className="card-label">보유 상품 수</div>
            <div className="card-value neutral">{totalProductsCount}개</div>
          </div>
          <div className="summary-card">
            <div className="card-icon">📈</div>
            <div className="card-label">현재 수익률</div>
            <div className="card-value negative">{currentReturn}%</div>
          </div>
          <div className="summary-card">
            <div className="card-icon">🎯</div>
            <div className="card-label">연간 수익률 (예상)</div>
            <div className="card-value positive">+{avgExpectedReturn.toFixed(2)}%</div>
          </div>
          <div className="summary-card">
            <div className="card-icon">🛡️</div>
            <div className="card-label">손실한계선(VaR 95%, 예상)</div>
            <div className="card-value negative">-{avgVar95.toFixed(2)}%</div>
          </div>
        </div>

        {/* 상품 리스트 */}
        <div className="products-section">
          <div className="section-header">
            <div className="section-title">
              📋 보유 상품 목록
            </div>
            <button
              className="toggle-btn"
              onClick={() => setIsProductListExpanded(!isProductListExpanded)}
            >
              {isProductListExpanded ? (
                <>접어두기 <span>▲</span></>
              ) : (
                <>모두 확인하기 <span>▼</span></>
              )}
            </button>
          </div>
          <div className={isProductListExpanded ? "products-table-container" : ""}>
            <table className="products-table">
              <thead>
                <tr>
                  <th style={{ width: "50%" }}>상품명</th>
                  <th style={{ width: "30%" }}>비중</th>
                  <th style={{ width: "20%" }}>코드</th>
                </tr>
              </thead>
              <tbody>
                {(isProductListExpanded ? allProducts : allProducts.slice(0, 10)).map((product, index) => (
                  <tr key={`${product.code}-${index}`}>
                    <td className="product-name">{product.name}</td>
                    <td>
                      <div className="weight-bar">
                        <div
                          className="weight-fill"
                          style={{ width: `${Math.min(product.weight_pct * 3, 100)}px` }}
                        />
                        <span>{product.weight_pct.toFixed(2)}%</span>
                      </div>
                    </td>
                    <td>{product.code}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!isProductListExpanded && allProducts.length > 10 && (
            <div className="more-indicator">
              외 {allProducts.length - 10}개 상품
            </div>
          )}
        </div>

        {/* 목표 적합도 */}
        <div className="suitability-section">
          <div className="section-title">
            🎯 포트폴리오 목표 적합도
          </div>
          <div className="suitability-content">
            <div className="suitability-item">
              <div className="suitability-label">내 목표 수익률</div>
              <div className="suitability-value">{formatTargetReturn(conditions.targetReturn)}%</div>
            </div>
            <div className="suitability-arrow">→</div>
            <div className="suitability-item">
              <div className="suitability-label">포트폴리오 기대 수익률</div>
              <div className="suitability-value">{avgExpectedReturn.toFixed(2)}%</div>
            </div>
            <div className="suitability-arrow">→</div>
            <div className="suitability-item">
              <div className="suitability-label">목표 달성 적합도</div>
              <div className="suitability-badge suitable">
                ✅ 적합
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 손실 위험 알림 모달 */}
      {showAlertModal && (
        <div className="modal-overlay" onClick={() => setShowAlertModal(false)}>
          <div className="alert-modal" onClick={(e) => e.stopPropagation()} style={{ position: "relative" }}>
            <button className="modal-close-btn" onClick={() => setShowAlertModal(false)}>
              ✕
            </button>
            <div className="alert-modal-icon">⚠️</div>
            <div className="alert-modal-title">손실 위험이 발생했습니다</div>
            <div className="alert-modal-desc">
              <p style={{ margin: "0 0 12px 0" }}>
                고객님이 보유하신 포트폴리오는 최근 위기 상황으로 인해 <strong>극단적 손실</strong>이 발생하였습니다.
              </p>
              <p style={{ margin: "0 0 12px 0" }}>
                최근 <strong>{conditions.theme || "분산형"}</strong>의 산업 이슈로 인해 추가 수익률 저하가 우려됩니다.
              </p>
              <p style={{ margin: 0 }}>
                고객님을 위한 맞춤 리밸런싱 전략을 확인해보세요.
              </p>
            </div>
            <div className="alert-graph">
              <div style={{ fontSize: "14px", fontWeight: 600, color: "#374151", marginBottom: "8px" }}>
                포트폴리오 변동 추이
              </div>
              <div className="graph-container">
                <div className="graph-label-y">수익률</div>
                <div className="graph-label-zero">0%</div>
                <div className="graph-y-axis" />
                <div className="graph-x-axis" />
                <div className="graph-var-line" />
                <div className="graph-label-var">손실한계선<br/>(-{avgVar95.toFixed(2)}%)</div>
                <div className="graph-label-x">보유기간</div>
                <div className="graph-current-line" />
                <div className="graph-label-current">현재 시점</div>
                <div className="graph-line-path">
                  <svg viewBox="0 0 300 170" preserveAspectRatio="none">
                    <path
                      d="M 0 80
                         Q 15 70, 25 55
                         Q 40 40, 50 60
                         Q 60 80, 75 90
                         Q 90 100, 100 85
                         Q 115 65, 130 50
                         Q 145 35, 160 45
                         Q 175 55, 185 70
                         Q 195 85, 210 75
                         Q 220 65, 230 80
                         Q 238 100, 244 125
                         Q 248 150, 250 165"
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              </div>
            </div>
            <button className="new-proposal-btn" onClick={() => router.push("/survey")}>
              <span>📋</span> 포트폴리오 리밸런싱 하기
            </button>
          </div>
        </div>
      )}
    </>
  );
}
