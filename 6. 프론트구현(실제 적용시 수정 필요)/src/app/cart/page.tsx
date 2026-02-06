"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useCart, CartPortfolio } from "@/context/CartContext";

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

// 목표 수익률 포맷팅
const formatTargetReturn = (value: number) => {
  const v = Math.round(value * 1000) / 10;
  return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1);
};

export default function CartPage() {
  const router = useRouter();
  const { cartItems, removeFromCart, clearCart } = useCart();
  const [showPurchaseModal, setShowPurchaseModal] = useState(false);
  const [isPurchased, setIsPurchased] = useState(false);

  // 전체 상품 목록 (모든 포트폴리오의 상품 합산)
  const getAllProducts = () => {
    const productMap = new Map<string, { name: string; weight_pct: number; portfolioCount: number }>();

    cartItems.forEach((portfolio) => {
      portfolio.products.forEach((product) => {
        const existing = productMap.get(product.code);
        if (existing) {
          existing.weight_pct += product.weight_pct;
          existing.portfolioCount += 1;
        } else {
          productMap.set(product.code, {
            name: product.name,
            weight_pct: product.weight_pct,
            portfolioCount: 1,
          });
        }
      });
    });

    return Array.from(productMap.entries())
      .map(([code, data]) => ({ code, ...data }))
      .sort((a, b) => b.weight_pct - a.weight_pct);
  };

  const handlePurchase = () => {
    setShowPurchaseModal(true);
  };

  const confirmPurchase = () => {
    // 구매 데이터를 로컬 스토리지에 저장
    const purchasedData = cartItems.map((item) => ({
      ...item,
      purchasedAt: new Date().toISOString(),
    }));
    localStorage.setItem("purchasedPortfolios", JSON.stringify(purchasedData));
    setIsPurchased(true);
  };

  const goToMyPage = () => {
    clearCart();
    router.push("/mypage");
  };

  const closeModalAndStay = () => {
    setShowPurchaseModal(false);
    setIsPurchased(false);
    clearCart();
  };

  if (cartItems.length === 0) {
    return (
      <>
        <style jsx>{`
          .cart-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 24px;
            min-height: calc(100vh - 80px);
          }
          .empty-cart {
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
          }
          .btn.primary {
            background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
            color: white;
          }
          .btn.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(10, 41, 114, 0.3);
          }
        `}</style>
        <div className="cart-container">
          <div className="empty-cart">
            <div className="empty-icon">🛒</div>
            <div className="empty-title">장바구니가 비어있습니다</div>
            <div className="empty-desc">마음에 드는 포트폴리오를 담아보세요</div>
            <button className="btn primary" onClick={() => router.push("/")}>
              포트폴리오 추천받기
            </button>
          </div>
        </div>
      </>
    );
  }

  const allProducts = getAllProducts();
  const totalProductsCount = cartItems.reduce((sum, item) => sum + item.totalProducts, 0);

  return (
    <>
      <style jsx>{`
        .cart-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 40px 24px;
          min-height: calc(100vh - 80px);
        }
        .page-title {
          font-size: 28px;
          font-weight: 700;
          color: #0A2972;
          margin-bottom: 8px;
        }
        .page-subtitle {
          font-size: 16px;
          color: #6b7280;
          margin-bottom: 32px;
        }
        .cart-summary {
          background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 32px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .summary-left {
          display: flex;
          gap: 40px;
        }
        .summary-item {
          text-align: center;
        }
        .summary-label {
          font-size: 14px;
          color: #6b7280;
          margin-bottom: 4px;
        }
        .summary-value {
          font-size: 24px;
          font-weight: 700;
          color: #0A2972;
        }
        .portfolio-list {
          margin-bottom: 32px;
        }
        .portfolio-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 16px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        .portfolio-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 16px;
        }
        .portfolio-title {
          font-size: 18px;
          font-weight: 700;
          color: #1f2937;
          margin-bottom: 8px;
        }
        .portfolio-conditions {
          font-size: 14px;
          color: #6b7280;
        }
        .remove-btn {
          background: none;
          border: none;
          color: #ef4444;
          font-size: 14px;
          cursor: pointer;
          padding: 8px 12px;
          border-radius: 6px;
          transition: background 0.2s;
        }
        .remove-btn:hover {
          background: #fef2f2;
        }
        .portfolio-metrics {
          display: flex;
          gap: 24px;
          padding: 16px;
          background: #f9fafb;
          border-radius: 12px;
          margin-bottom: 16px;
        }
        .metric {
          flex: 1;
        }
        .metric-label {
          font-size: 13px;
          color: #6b7280;
          margin-bottom: 4px;
        }
        .metric-value {
          font-size: 18px;
          font-weight: 600;
        }
        .metric-value.positive {
          color: #0A2972;
        }
        .metric-value.negative {
          color: #dc2626;
        }
        .product-preview {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .product-tag {
          background: #e0e7ff;
          color: #3730a3;
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 13px;
        }
        .product-more {
          background: #f3f4f6;
          color: #6b7280;
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 13px;
        }
        .products-section {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 16px;
          padding: 24px;
          margin-bottom: 32px;
        }
        .section-title {
          font-size: 20px;
          font-weight: 700;
          color: #1f2937;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 8px;
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
        .bottom-actions {
          position: sticky;
          bottom: 0;
          background: white;
          border-top: 1px solid #e5e7eb;
          padding: 20px 24px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin: 0 -24px;
          box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.05);
        }
        .total-info {
          font-size: 16px;
          color: #374151;
        }
        .total-amount {
          font-size: 24px;
          font-weight: 700;
          color: #0A2972;
          margin-left: 8px;
        }
        .btn {
          padding: 16px 48px;
          border-radius: 12px;
          font-size: 18px;
          font-weight: 700;
          cursor: pointer;
          border: none;
          transition: all 0.2s;
        }
        .btn.primary {
          background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
          color: white;
        }
        .btn.primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(10, 41, 114, 0.35);
        }
        .btn.secondary {
          background: #f3f4f6;
          color: #374151;
          margin-right: 12px;
        }
        .btn.secondary:hover {
          background: #e5e7eb;
        }

        /* 모달 스타일 */
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }
        .modal {
          background: white;
          border-radius: 20px;
          padding: 40px;
          max-width: 450px;
          width: 90%;
          text-align: center;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        }
        .modal-icon {
          font-size: 64px;
          margin-bottom: 20px;
        }
        .modal-title {
          font-size: 24px;
          font-weight: 700;
          color: #1f2937;
          margin-bottom: 12px;
        }
        .modal-desc {
          font-size: 16px;
          color: #6b7280;
          margin-bottom: 32px;
          line-height: 1.6;
        }
        .modal-buttons {
          display: flex;
          gap: 12px;
          justify-content: center;
        }
        .modal-btn {
          flex: 1;
          padding: 14px 24px;
          border-radius: 10px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          border: none;
          transition: all 0.2s;
        }
        .modal-btn.primary {
          background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
          color: white;
        }
        .modal-btn.primary:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(10, 41, 114, 0.3);
        }
        .modal-btn.secondary {
          background: #f3f4f6;
          color: #374151;
        }
        .modal-btn.secondary:hover {
          background: #e5e7eb;
        }
      `}</style>

      <div className="cart-container">
        <h1 className="page-title">장바구니</h1>
        <p className="page-subtitle">선택하신 포트폴리오를 확인하고 구매를 진행하세요</p>

        {/* 요약 정보 */}
        <div className="cart-summary">
          <div className="summary-left">
            <div className="summary-item">
              <div className="summary-label">포트폴리오</div>
              <div className="summary-value">{cartItems.length}개</div>
            </div>
            <div className="summary-item">
              <div className="summary-label">총 상품 수</div>
              <div className="summary-value">{totalProductsCount}개</div>
            </div>
          </div>
        </div>

        {/* 포트폴리오 목록 */}
        <div className="portfolio-list">
          {cartItems.map((portfolio) => (
            <div key={portfolio.portfolioId} className="portfolio-card">
              <div className="portfolio-header">
                <div>
                  <div className="portfolio-title">
                    {getCountryDisplay(portfolio.conditions.region)} · {portfolio.conditions.theme || "분산형"} 포트폴리오
                  </div>
                  <div className="portfolio-conditions">
                    목표 수익률 {formatTargetReturn(portfolio.conditions.targetReturn)}% |
                    은퇴 시점 {portfolio.conditions.retireYear}년
                  </div>
                </div>
                <button className="remove-btn" onClick={() => removeFromCart(portfolio.portfolioId)}>
                  삭제
                </button>
              </div>

              <div className="portfolio-metrics">
                <div className="metric">
                  <div className="metric-label">기대 수익률</div>
                  <div className="metric-value positive">+{portfolio.metrics.expectedReturn.toFixed(2)}%</div>
                </div>
                <div className="metric">
                  <div className="metric-label">손실한계선(VaR 95%)</div>
                  <div className="metric-value negative">-{Math.abs(portfolio.metrics.var95).toFixed(2)}%</div>
                </div>
                <div className="metric">
                  <div className="metric-label">TDF 비중</div>
                  <div className="metric-value">{portfolio.allocation.tdfWeight.toFixed(1)}%</div>
                </div>
              </div>

              <div className="product-preview">
                {portfolio.products.slice(0, 4).map((product) => (
                  <span key={product.code} className="product-tag">
                    {product.name.length > 15 ? product.name.slice(0, 15) + "..." : product.name}
                  </span>
                ))}
                {portfolio.products.length > 4 && (
                  <span className="product-more">+{portfolio.products.length - 4}개</span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* 전체 상품 목록 */}
        <div className="products-section">
          <div className="section-title">
            📋 구매 예정 상품 목록 (상위 10개 요약)
          </div>
          <table className="products-table">
            <thead>
              <tr>
                <th style={{ width: "50%" }}>상품명</th>
                <th style={{ width: "30%" }}>비중</th>
                <th style={{ width: "20%" }}>코드</th>
              </tr>
            </thead>
            <tbody>
              {allProducts.slice(0, 20).map((product, index) => (
                <tr key={product.code}>
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
          {allProducts.length > 20 && (
            <div style={{ textAlign: "center", padding: "16px", color: "#6b7280" }}>
              외 {allProducts.length - 20}개 상품
            </div>
          )}
        </div>

        {/* 하단 구매 버튼 */}
        <div className="bottom-actions">
          <div className="total-info">
            총 <span className="total-amount">{cartItems.length}개</span> 포트폴리오
          </div>
          <div>
            <button className="btn secondary" onClick={() => router.push("/")}>
              계속 쇼핑하기
            </button>
            <button className="btn primary" onClick={handlePurchase}>
              구매하기
            </button>
          </div>
        </div>
      </div>

      {/* 구매 확인 모달 */}
      {showPurchaseModal && !isPurchased && (
        <div className="modal-overlay" onClick={() => setShowPurchaseModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-icon">🛒</div>
            <div className="modal-title">구매를 진행하시겠습니까?</div>
            <div className="modal-desc">
              {cartItems.length}개의 포트폴리오에 포함된<br />
              총 {totalProductsCount}개의 상품을 구매합니다.
            </div>
            <div className="modal-buttons">
              <button className="modal-btn secondary" onClick={() => setShowPurchaseModal(false)}>
                취소
              </button>
              <button className="modal-btn primary" onClick={confirmPurchase}>
                구매하기
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 구매 완료 모달 */}
      {showPurchaseModal && isPurchased && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-icon">✅</div>
            <div className="modal-title">결제가 완료되었습니다</div>
            <div className="modal-desc">
              마이페이지에서 구매하신 포트폴리오를<br />
              확인하실 수 있습니다.
            </div>
            <div className="modal-buttons">
              <button className="modal-btn secondary" onClick={closeModalAndStay}>
                계속 쇼핑하기
              </button>
              <button className="modal-btn primary" onClick={goToMyPage}>
                마이페이지로 이동
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
