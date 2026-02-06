"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

// 국가 옵션 (화면 표시용 -> DB 저장용)
const COUNTRIES = [
  { display: "미국", value: "미국" },
  { display: "한국", value: "한국" },
  { display: "중국", value: "중국" },
  { display: "아시아", value: "아시아" },
  { display: "기타 지역", value: "지역기타" },
];

// 테마 옵션 (지수추종 제외 - 별도 처리)
const THEMES = [
  "AI테크",
  "ESG",
  "금",
  "바이오헬스케어",
  "반도체",
  "배터리전기차",
  "소비재",
];

// 지수추종 테마 매핑 (국가별)
const INDEX_THEME_MAP: { [key: string]: string } = {
  "미국": "지수추종_미국",
  "한국": "지수추종_한국",
  "중국": "지수추종_지역특화",
  "아시아": "지수추종_지역특화",
  "지역기타": "지수추종_지역특화",
};

// 목표 수익률 옵션
const TARGET_RETURN_OPTIONS = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12];

// 목표 은퇴 연도 옵션 (5년 단위)
const TARGET_RETIREMENT_YEAR_OPTIONS = [2030, 2035, 2040, 2045, 2050, 2055, 2060];

export default function SurveyPage() {
  const router = useRouter();

  // 상태 관리
  const [retireYear, setRetireYear] = useState(2040);
  const [targetReturn, setTargetReturn] = useState(0.07);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [selectedTheme, setSelectedTheme] = useState<string | null>(null);
  const [includeIndexTheme, setIncludeIndexTheme] = useState(false);

  // 버블 위치 계산 (슬라이더 thumb 위치에 맞게 보정)
  const getBubblePosition = (value: number, min: number, max: number) => {
    const percent = (value - min) / (max - min);
    // thumb 크기(20px)를 고려한 보정: calc(percent% - thumb반지름 + thumb반지름*2*percent)
    // 간단히: percent * (100% - thumbSize) + thumbSize/2 형태로 보정
    return percent * 100;
  };

  // 버블 스타일 계산 (thumb 위치 보정 포함)
  const getBubbleStyle = (value: number, min: number, max: number) => {
    const percent = (value - min) / (max - min);
    // thumb 너비 20px 기준, 양쪽 10px씩 보정
    return {
      left: `calc(${percent * 100}% - ${percent * 20 - 10}px)`
    };
  };

  // 가장 가까운 옵션 찾기 (은퇴연도)
  const findClosestRetireYear = (value: number) => {
    return TARGET_RETIREMENT_YEAR_OPTIONS.reduce((prev, curr) =>
      Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
    );
  };

  // 가장 가까운 옵션 찾기 (목표수익률)
  const findClosestReturn = (value: number) => {
    return TARGET_RETURN_OPTIONS.reduce((prev, curr) =>
      Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
    );
  };

  // 국가 선택 핸들러
  const handleCountryClick = (countryValue: string) => {
    setSelectedCountry(selectedCountry === countryValue ? null : countryValue);
  };

  // 테마 선택 핸들러 (단일 선택)
  const handleThemeClick = (theme: string) => {
    setSelectedTheme(selectedTheme === theme ? null : theme);
    // 일반 테마 선택 시 지수추종 해제
    if (selectedTheme !== theme) {
      setIncludeIndexTheme(false);
    }
  };

  // 지수추종 선택 핸들러
  const handleIndexThemeClick = () => {
    if (!includeIndexTheme) {
      // 지수추종 선택 시 일반 테마 해제
      setSelectedTheme(null);
    }
    setIncludeIndexTheme(!includeIndexTheme);
  };

  // 실제 DB에서 조회할 지수추종 테마 값 계산
  const getIndexThemeValue = () => {
    if (!includeIndexTheme || !selectedCountry) return null;
    return INDEX_THEME_MAP[selectedCountry];
  };

  // 확인 버튼 클릭
  const handleSubmit = () => {
    // 최종 테마 결정 (일반 테마 or 지수추종)
    let finalTheme = selectedTheme || "";
    const indexThemeValue = getIndexThemeValue();
    if (indexThemeValue) {
      finalTheme = indexThemeValue;
    }

    // 선택 데이터를 쿼리 파라미터로 전달
    const params = new URLSearchParams({
      retireYear: retireYear.toString(),
      targetReturn: targetReturn.toString(),
      country: selectedCountry || "",
      theme: finalTheme,
    });

    router.push(`/recommendation?${params.toString()}`);
  };

  return (
    <div className="container">
      <section className="section-wrap fade-in">
        <div className="section-title">나의 투자성향 알아보기</div>

        {/* 목표 은퇴 연도 */}
        <div className="q">예상 퇴직연금 수령 시점은?</div>
        <div className="slider-wrap">
          <input
            type="range"
            min={2030}
            max={2060}
            step={1}
            value={retireYear}
            onChange={(e) => {
              const closest = findClosestRetireYear(Number(e.target.value));
              setRetireYear(closest);
            }}
          />
          <div
            className="bubble"
            style={getBubbleStyle(retireYear, 2030, 2060)}
          >
            {retireYear}년
          </div>
          {/* 눈금 표시 */}
          <div className="slider-ticks">
            {TARGET_RETIREMENT_YEAR_OPTIONS.map((year) => {
              const percent = ((year - 2030) / (2060 - 2030)) * 100;
              return (
                <div
                  key={year}
                  className={`tick ${retireYear === year ? "active" : ""}`}
                  style={{ left: `calc(${percent}% - ${percent * 20 / 100 - 10}px)` }}
                  onClick={() => setRetireYear(year)}
                >
                  <div className="tick-mark" />
                  <div className="tick-label">{year}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 목표 수익률 */}
        <div className="q">목표 수익률은?</div>
        <div className="slider-wrap">
          <input
            type="range"
            min={0.03}
            max={0.12}
            step={0.005}
            value={targetReturn}
            onChange={(e) => {
              const closest = findClosestReturn(Number(e.target.value));
              setTargetReturn(closest);
            }}
          />
          <div
            className="bubble"
            style={getBubbleStyle(targetReturn, 0.03, 0.12)}
          >
            {(() => { const v = Math.round(targetReturn * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}%
          </div>
          {/* 눈금 표시 */}
          <div className="slider-ticks">
            {TARGET_RETURN_OPTIONS.map((ret) => {
              const percent = ((ret - 0.03) / (0.12 - 0.03)) * 100;
              return (
                <div
                  key={ret}
                  className={`tick ${targetReturn === ret ? "active" : ""}`}
                  style={{ left: `calc(${percent}% - ${percent * 20 / 100 - 10}px)` }}
                  onClick={() => setTargetReturn(ret)}
                >
                  <div className="tick-mark" />
                  <div className="tick-label">{(() => { const v = Math.round(ret * 1000) / 10; return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1); })()}%</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 투자 국가 */}
        <div className="q">투자하고 싶은 국가</div>
        <div className="chips">
          {COUNTRIES.map((country) => (
            <div
              key={country.value}
              className={`chip ${selectedCountry === country.value ? "active" : ""}`}
              onClick={() => handleCountryClick(country.value)}
            >
              {country.display}
            </div>
          ))}
        </div>

        {/* 투자 테마 */}
        <div className="q">관심 있는 투자 테마</div>
        <div className="chips">
          {THEMES.map((theme) => (
            <div
              key={theme}
              className={`chip ${selectedTheme === theme ? "active" : ""}`}
              onClick={() => handleThemeClick(theme)}
            >
              {theme}
            </div>
          ))}
          {/* 지수추종 (국가에 따라 다른 값으로 매핑) */}
          <div
            className={`chip ${includeIndexTheme ? "active" : ""}`}
            onClick={handleIndexThemeClick}
          >
            지수추종
          </div>
        </div>

        {/* 지수추종 선택 시 안내 메시지 */}
        {includeIndexTheme && selectedCountry && (
          <div className="note" style={{ color: "var(--navy)", fontWeight: 500 }}>
            → 선택된 국가({COUNTRIES.find(c => c.value === selectedCountry)?.display})에 맞는 지수추종 테마가 적용됩니다: {INDEX_THEME_MAP[selectedCountry]}
          </div>
        )}
        {includeIndexTheme && !selectedCountry && (
          <div className="note" style={{ color: "var(--orange)" }}>
            * 지수추종을 선택하려면 먼저 국가를 선택해주세요.
          </div>
        )}

        <div className="note">
          * 선호하는 항목이 없다면 선택 없이 바로 &apos;확인&apos;을 눌러주세요.
        </div>

        {/* 확인 버튼 */}
        <div className="confirm" style={{ display: "flex", justifyContent: "flex-end", marginTop: "24px", gap: "10px" }}>
          <button className="btn gold" onClick={handleSubmit}>
            확인
          </button>
        </div>
      </section>
    </div>
  );
}
