import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Groq에서 사용 가능한 최저가/고속 텍스트 모델로 수정
DEFAULT_MODEL = "llama3-8b-8192" 
ENV_FILE = ".env.local"

# 다른 스크립트와의 호환성을 위해 summary_one_line으로 통일
REQUIRED_KEYS = (
    "summary_one_line",
    "summary_easy",
    "real_world",
    "limitations",
)

SYSTEM_PROMPT = """You are a science communicator who explains research papers in Korean.
You must always respond with valid JSON only.
Do not include markdown, code fences, or extra text outside the JSON object."""

USER_PROMPT_TEMPLATE = """아래 논문 제목과 초록을 읽고, 반드시 아래 JSON 형식으로만 답하세요.

[논문 제목]
{title}

[논문 초록]
{abstract}

[출력 규칙]
1. JSON 객체 1개만 출력하세요.
2. 키 이름은 정확히 다음 4개만 사용하세요:
   - "summary_one_line"
   - "summary_easy"
   - "real_world"
   - "limitations"
3. 모든 값은 한국어 문자열이어야 합니다.
4. "summary_one_line": 핵심만 담은 한 줄 요약, 공백 포함 20자 이내
5. "summary_easy": 중학생도 이해할 수 있는 쉬운 설명, 3~4문장
6. "real_world": 현실 적용 예시, 2~3줄
7. "limitations": 연구 한계점, 1~2줄
8. JSON 외 다른 텍스트, 주석, 마크다운을 절대 넣지 마세요.

[출력 형식 예시]
{{
  "summary_one_line": "세션 간 AI 기억 이어주기",
  "summary_easy": "AI가 대화를 이어갈 때 이전 내용을 어떻게 넘길지 설명합니다. ...",
  "real_world": "긴 문서 작성 AI, 고객 상담 챗봇, 협업 AI 에이전트에 활용될 수 있습니다.",
  "limitations": "이론 중심 연구라 실제 서비스 검증은 아직 부족합니다."
}}"""


def load_groq_api_key(env_file: str = ENV_FILE) -> str:
    """`.env.local`에서 Groq API 키를 불러온다."""
    load_dotenv(Path(env_file))
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(f"{env_file}에서 GROQ_API_KEY를 찾을 수 없습니다.")

    return api_key


def _parse_summary_response(content: str) -> dict[str, str]:
    """모델 응답을 JSON으로 파싱하고 필수 키를 검증한다."""
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 파싱 실패: {error}") from error

    if not isinstance(result, dict):
        raise ValueError("응답이 JSON 객체가 아닙니다.")

    missing_keys = [key for key in REQUIRED_KEYS if key not in result]
    if missing_keys:
        raise ValueError(f"필수 키 누락: {', '.join(missing_keys)}")

    return {key: str(result[key]).strip() for key in REQUIRED_KEYS}


def summarize_paper(
    title: str,
    abstract: str,
    model: str = DEFAULT_MODEL,
    env_file: str = ENV_FILE,
) -> dict[str, str]:
    """Groq API로 논문 제목과 초록을 쉬운 요약 딕셔너리로 변환한다."""
    client = Groq(api_key=load_groq_api_key(env_file))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    title=title.strip(),
                    abstract=abstract.strip(),
                ),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq API가 빈 응답을 반환했습니다.")

    return _parse_summary_response(content)


def print_summary(summary: dict[str, str]) -> None:
    """요약 결과를 보기 좋게 출력한다."""
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize_first_paper_from_csv(
    csv_path: str = "cs_ai_papers.csv",
    env_file: str = ENV_FILE,
) -> dict[str, str]:
    """CSV 첫 번째 논문을 읽어 요약한다."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    first_paper = df.iloc[0]

    # 컬럼명이 arxiv_id인지 확인 후 접근 (일반적으로 url 또는 id 등일 수 있음)
    paper_id = first_paper.get('arxiv_id', first_paper.get('url', 'ID 없음'))
    print(f"논문 ID: {paper_id}")
    print(f"제목: {first_paper['title']}\n")

    return summarize_paper(
        title=str(first_paper["title"]),
        abstract=str(first_paper["abstract"]),
        env_file=env_file,
    )


if __name__ == "__main__":
    # 프로젝트 내 존재하는 CSV 파일 자동 탐색
    csv_candidates = ["arxiv_paper.csv", "cs_ai_papers.csv", "papers.csv", "paper.csv"]
    csv_path = next(
        (path for path in csv_candidates if Path(path).exists()),
        "cs_ai_papers.csv",
    )

    try:
        summary = summarize_first_paper_from_csv(csv_path)
        print_summary(summary)
    except Exception as e:
        print(f"오류 발생: {e}")