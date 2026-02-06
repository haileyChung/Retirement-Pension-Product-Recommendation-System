"use client";

import { useRouter, usePathname } from "next/navigation";
import { useCart } from "@/context/CartContext";

export default function Header() {
  const router = useRouter();
  const pathname = usePathname();
  const { cartItems } = useCart();

  // 랜딩 페이지에서는 프로젝트 배너만 표시
  const isLandingPage = pathname === "/" || pathname === "/landing";

  return (
    <>
      {/* 프로젝트 안내 배너 - 모든 페이지에서 표시 */}
      <div className="project-banner">
        <div className="project-banner-content">
          <span>본 화면은 고려대학교 MSBA 캡스톤 프로젝트의 결과물로, 학술적 시연을 목적으로 제작되었습니다.</span>
          <span className="project-authors">작성자 : 이다정, 이동훈, 정혜윤</span>
        </div>
      </div>

      {/* 메인 헤더 - 랜딩 페이지 제외 */}
      {!isLandingPage && (
        <header className="main-header">
          <div className="header-content">
            <div className="header-left" onClick={() => router.push("/")} style={{ cursor: "pointer" }}>
              <h1>Future Shield Advisor</h1>
              <div className="header-sub">
                현대차증권 AI 퇴직연금 포트폴리오 어드바이저
              </div>
            </div>
            <div className="header-right">
              <button className="cart-btn" onClick={() => router.push("/cart")}>
                <span className="cart-icon">🛒</span>
                {cartItems.length > 0 && (
                  <span className="cart-badge">{cartItems.length}</span>
                )}
              </button>
              <button className="mypage-btn" onClick={() => router.push("/mypage")}>
                <span className="mypage-icon">👨‍💼</span>
              </button>
            </div>
          </div>
        </header>
      )}

      <style jsx>{`
        .project-banner {
          background: #6b7280;
          color: white;
          padding: 8px 24px;
          font-size: 12px;
          text-align: center;
        }
        .project-banner-content {
          max-width: 1200px;
          margin: 0 auto;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 16px;
          flex-wrap: wrap;
        }
        .project-authors {
          font-weight: 600;
        }
        .main-header {
          background: linear-gradient(135deg, #0A2972 0%, #061a4a 100%);
          color: white;
          padding: 16px 24px;
          position: sticky;
          top: 0;
          z-index: 100;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .header-content {
          max-width: 1200px;
          margin: 0 auto;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .header-left {
          text-align: left;
        }
        .main-header h1 {
          font-size: 20px;
          font-weight: 800;
          margin: 0;
        }
        .header-sub {
          font-size: 12px;
          opacity: 0.8;
          margin-top: 4px;
        }
        .header-right {
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .cart-btn {
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 8px;
          padding: 8px 12px;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 4px;
          position: relative;
          transition: all 0.2s;
        }
        .cart-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        .cart-icon {
          font-size: 20px;
        }
        .cart-badge {
          position: absolute;
          top: -6px;
          right: -6px;
          background: #ef4444;
          color: white;
          font-size: 11px;
          font-weight: 700;
          min-width: 18px;
          height: 18px;
          border-radius: 9px;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0 4px;
        }
        .mypage-btn {
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 8px;
          padding: 8px 12px;
          cursor: pointer;
          display: flex;
          align-items: center;
          transition: all 0.2s;
        }
        .mypage-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        .mypage-icon {
          font-size: 20px;
        }
      `}</style>
    </>
  );
}
