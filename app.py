import os
import time
import pandas as pd
import streamlit as st
import arxiv
from summarize_paper import summarize_paper

# 1. 페이지 설정
st.set_page_config(
    page_title="AI 논문 브리핑 대시보드",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 커스텀 CSS (사이드바 밝은 민트/하늘색 그라데이션 및 카드 스타일)
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E6FFFA 0%, #E0F2FE 100%);
        border-right: 1px solid #BAE6FD;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .paper-box {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# 3. 데이터 로드 헬퍼 함수
INPUT_FILE = "papers.csv"
OUTPUT_FILE = "papers_with_summary.csv"

def load_data():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE)
    elif os.path.exists(INPUT_FILE):
        return pd.read_csv(INPUT_FILE)
    return pd.DataFrame(columns=["title", "abstract", "summary_one_line", "summary_easy", "real_world", "limitations"])

# 4. 사이드바 내비게이션
st.sidebar.title("📚 논문 관리 시스템")
menu = st.sidebar.radio(
    "메뉴 이동",
    ["📊 대시보드", "🔍 논문 크롤링", "⚡ AI 일괄 요약", "📖 논문 탐색"]
)

# 5. 메인 뷰 구현

# 1) 대시보드
if menu == "📊 대시보드":
    st.header("📊 논문 요약 관리 대시보드")
    df = load_data()
    
    total_papers = len(df)
    summarized_papers = len(df[df["summary_one_line"].notna() & (df["summary_one_line"] != "")]) if "summary_one_line" in df.columns else 0
    pending_papers = total_papers - summarized_papers

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#475569; margin:0;">📄 크롤링된 논문</h3>
            <h1 style="color:#0F172A; margin:10px 0 0 0;">{total_papers}건</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#059669; margin:0;">✅ 요약 완료</h3>
            <h1 style="color:#059669; margin:10px 0 0 0;">{summarized_papers}건</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#D97706; margin:0;">⏳ 요약 대기</h3>
            <h1 style="color:#D97706; margin:10px 0 0 0;">{pending_papers}건</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📚 최근 논문")
    if not df.empty:
        st.dataframe(
            df[["title", "summary_one_line"] if "summary_one_line" in df.columns else ["title"]].tail(5),
            use_container_width=True
        )
    else:
        st.info("현재 등록된 논문 데이터가 없습니다. 크롤링 메뉴에서 논문을 수집하세요.")

# 2) 논문 크롤링
elif menu == "🔍 논문 크롤링":
    st.header("🔍 arXiv 논문 크롤링")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("검색 키워드 (영문)", value="LLM Agent")
    with col2:
        max_results = st.number_input("수집 개수", min_value=1, max_value=50, value=5)

    if st.button("수집 시작", type="primary"):
        with st.spinner("arXiv에서 논문을 검색 중입니다..."):
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers_list = []
            for r in client.results(search):
                papers_list.append({
                    "title": r.title.replace("\n", " "),
                    "abstract": r.summary.replace("\n", " "),
                    "url": r.entry_id,
                    "published": str(r.published.date())
                })
            
            if papers_list:
                df_new = pd.DataFrame(papers_list)
                if os.path.exists(INPUT_FILE):
                    df_existing = pd.read_csv(INPUT_FILE)
                    df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=["title"]).reset_index(drop=True)
                else:
                    df_combined = df_new
                
                df_combined.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")
                st.success(f"{len(papers_list)}건의 논문을 수집하여 '{INPUT_FILE}'에 저장했습니다.")
                st.dataframe(df_new[["title", "published"]])
            else:
                st.warning("검색 결과가 없습니다.")

# 3) AI 일괄 요약
elif menu == "⚡ AI 일괄 요약":
    st.header("⚡ Groq 기반 AI 일괄 요약")
    
    if not os.path.exists(INPUT_FILE):
        st.warning(f"먼저 논문 크롤링을 통해 '{INPUT_FILE}'을 생성하세요.")
    else:
        df = load_data()
        for col in ["summary_one_line", "summary_easy", "real_world", "limitations"]:
            if col not in df.columns:
                # 새 컬럼 생성 시 명시적으로 object(문자열) 타입 지정
                df[col] = pd.Series(dtype='object')
            else:
                # 기존 컬럼이 float64로 잡혀있을 경우를 대비해 타입 강제 변환
                df[col] = df[col].astype(object)

        pending_mask = df["summary_one_line"].isna() | (df["summary_one_line"] == "")
        pending_count = pending_mask.sum()
        
        st.write(f"총 **{len(df)}**건 중 요약 대기 논문: **{pending_count}**건")
        
        if pending_count > 0:
            if st.button("요약 실행 시작", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                indices_to_process = df[pending_mask].index.tolist()
                
                for i, idx in enumerate(indices_to_process):
                    title = df.at[idx, "title"]
                    abstract = df.at[idx, "abstract"]
                    
                    status_text.text(f"요약 처리 중 ({i+1}/{len(indices_to_process)}): {title[:35]}...")
                    
                    try:
                        res = summarize_paper(title, abstract)
                        df.at[idx, "summary_one_line"] = res.get("summary_one_line")
                        df.at[idx, "summary_easy"] = res.get("summary_easy")
                        df.at[idx, "real_world"] = res.get("real_world")
                        df.at[idx, "limitations"] = res.get("limitations")
                    except Exception as e:
                        st.error(f"오류 발생 ({title[:20]}): {e}")
                    
                    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                    progress_bar.progress((i + 1) / len(indices_to_process))
                    time.sleep(1.0)
                
                status_text.text("모든 요약 작업이 완료되었습니다.")
                st.success("요약 완료 및 저장 완료!")
                st.rerun()
        else:
            st.info("모든 논문의 요약이 완료되었습니다.")

# 4) 논문 탐색
elif menu == "📖 논문 탐색":
    st.header("📖 논문 요약 탐색")
    df = load_data()
    
    if df.empty:
        st.info("표시할 논문 데이터가 없습니다.")
    else:
        search_kw = st.text_input("논문 제목 검색")
        if search_kw:
            df = df[df["title"].str.contains(search_kw, case=False, na=False)]
            
        for _, row in df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="paper-box">
                    <h3 style="margin-top:0; color:#1E293B;">{row.get('title', '제목 없음')}</h3>
                    <p><strong>💡 한 줄 요약:</strong> {row.get('summary_one_line', '요약 대기 중')}</p>
                    <p><strong>📝 쉬운 설명:</strong> {row.get('summary_easy', '요약 대기 중')}</p>
                    <p><strong>🌐 현실 적용:</strong> {row.get('real_world', '요약 대기 중')}</p>
                    <p><strong>⚠️ 한계점:</strong> {row.get('limitations', '요약 대기 중')}</p>
                </div>
                """, unsafe_allow_html=True)