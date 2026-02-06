import { NextRequest, NextResponse } from "next/server";

// 백엔드 API URL (환경변수에서 가져오기)
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;

  // 쿼리 파라미터 가져오기
  const region = searchParams.get("region") || "";
  const theme = searchParams.get("theme") || "";
  const targetReturn = searchParams.get("targetReturn") || "0.07";
  const retireYear = searchParams.get("retireYear") || "2040";

  try {
    // Railway 백엔드로 프록시
    const params = new URLSearchParams({
      region,
      theme,
      targetReturn,
      retireYear,
    });

    const response = await fetch(`${API_URL}/api/portfolio?${params.toString()}`);
    const data = await response.json();

    return NextResponse.json(data);
  } catch (error) {
    console.error("백엔드 API 호출 에러:", error);
    return NextResponse.json(
      {
        success: false,
        error: "백엔드 서버에 연결할 수 없습니다.",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
