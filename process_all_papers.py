import os
import time
import pandas as pd
from summarize_paper import summarize_paper

def process_all_papers(
    input_csv: str = "papers.csv",
    output_csv: str = "papers_with_summary.csv",
    delay_seconds: float = 2.0,
    max_retries: int = 3
):
    """
    papers.csv를 읽어 Groq API로 일괄 요약하고 papers_with_summary.csv로 저장합니다.
    중단 시 이어하기(Resume) 및 재시도(Retry) 로직이 포함되어 있습니다.
    """
    if not os.path.exists(input_csv):
        print(f"[오류] 입력 파일 '{input_csv}'이 존재하지 않습니다.")
        return

    df_input = pd.read_csv(input_csv)
    
    # 이어하기 처리
    if os.path.exists(output_csv):
        df_output = pd.read_csv(output_csv)
        print(f"기존 작업 파일('{output_csv}')을 감지했습니다. 이어서 처리를 시작합니다.")
    else:
        df_output = df_input.copy()
        for col in ["summary_one_line", "summary_easy", "real_world", "limitations"]:
            if col not in df_output.columns:
                df_output[col] = None

    total_count = len(df_output)
    
    for idx, row in df_output.iterrows():
        # 이미 요약된 항목 건너뛰기
        if pd.notna(row.get("summary_one_line")) and str(row.get("summary_one_line")).strip() != "":
            continue

        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        
        if not title or not abstract:
            continue

        print(f"[{idx + 1}/{total_count}] 요약 생성 중: {title[:40]}...")

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                summary = summarize_paper(title, abstract)
                df_output.at[idx, "summary_one_line"] = summary.get("summary_one_line")
                df_output.at[idx, "summary_easy"] = summary.get("summary_easy")
                df_output.at[idx, "real_world"] = summary.get("real_world")
                df_output.at[idx, "limitations"] = summary.get("limitations")
                success = True
                break
            except Exception as e:
                print(f"  └ 재시도 {attempt}/{max_retries} 실패: {e}")
                time.sleep(3)

        if not success:
            df_output.at[idx, "summary_one_line"] = "처리 실패"
            df_output.at[idx, "summary_easy"] = "API 응답 오류로 요약 실패"
            df_output.at[idx, "real_world"] = "정보 없음"
            df_output.at[idx, "limitations"] = "정보 없음"

        # 중간 저장 (데이터 유실 방지)
        df_output.to_csv(output_csv, index=False, encoding="utf-8-sig")
        time.sleep(delay_seconds)

    print(f"\n모든 작업이 완료되었습니다. 결과 저장: {output_csv}")

if __name__ == "__main__":
    process_all_papers()