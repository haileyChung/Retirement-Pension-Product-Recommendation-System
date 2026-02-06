"use client";

import { useRouter } from "next/navigation";

export default function LandingPage() {
  const router = useRouter();

  return (
    <>
      <style jsx>{`
        .landing-container {
          position: relative;
          width: 100%;
          min-height: 100vh;
        }
        .background-image {
          width: 100%;
          height: auto;
          display: block;
        }
        .fsa-button {
          position: absolute;
          top: 42%;
          right: 8%;
          width: 280px;
          background: linear-gradient(135deg, #0A2972 0%, #1e40af 100%);
          border-radius: 16px;
          padding: 24px;
          cursor: pointer;
          transition: all 0.3s;
          box-shadow: 0 8px 32px rgba(10, 41, 114, 0.4);
          border: 3px solid rgba(255, 255, 255, 0.3);
        }
        .fsa-button:hover {
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 12px 40px rgba(10, 41, 114, 0.5);
        }
        .fsa-icon {
          font-size: 40px;
          margin-bottom: 12px;
        }
        .fsa-title {
          font-size: 20px;
          font-weight: 800;
          color: white;
          margin-bottom: 8px;
        }
        .fsa-subtitle {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.85);
          line-height: 1.5;
          margin-bottom: 16px;
        }
        .fsa-cta {
          background: white;
          color: #0A2972;
          border: none;
          border-radius: 8px;
          padding: 10px 20px;
          font-size: 14px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.2s;
          width: 100%;
        }
        .fsa-cta:hover {
          background: #f0f9ff;
        }
        .fsa-badge {
          position: absolute;
          top: -10px;
          right: -10px;
          background: #dc2626;
          color: white;
          font-size: 11px;
          font-weight: 700;
          padding: 4px 10px;
          border-radius: 20px;
        }
      `}</style>

      <div className="landing-container">
        {/* 현대차증권 홈페이지 배경 이미지 */}
        <img
          src="/hyundai-securities-bg.png"
          alt="현대차증권 홈페이지"
          className="background-image"
        />

        {/* Future Shield Advisor 버튼 */}
        <div className="fsa-button" onClick={() => router.push("/survey")}>
          <div className="fsa-badge">NEW</div>
          <div className="fsa-icon">🛡️</div>
          <div className="fsa-title">Future Shield Advisor</div>
          <div className="fsa-subtitle">
            현대차증권 AI 퇴직연금<br />
            포트폴리오 어드바이저
          </div>
          <button className="fsa-cta">
            시작하기 →
          </button>
        </div>
      </div>
    </>
  );
}
