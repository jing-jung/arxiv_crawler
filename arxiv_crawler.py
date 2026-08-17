import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ARXIV_ID_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
DEFAULT_HEADERS = {"User-Agent": "educational crawler (contact: test@email.com)"}
REQUEST_DELAY_SECONDS = 15
DEFAULT_LOG_FILE = "crawler.log"
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "have", "has", "had", "not", "but", "can", "our", "their", "they", "them",
    "its", "into", "using", "use", "used", "via", "also", "such", "than", "then",
    "when", "where", "which", "while", "who", "will", "would", "about", "across",
    "based", "between", "both", "each", "more", "most", "other", "over", "show",
    "through", "under", "within", "without", "paper", "approach", "method", "model",
    "models", "results", "study", "work", "data", "learning", "language", "large",
}


class ParsingError(Exception):
    """HTML 파싱 중 필수 정보를 찾지 못했을 때 발생한다."""


def setup_logger(log_file: str = DEFAULT_LOG_FILE) -> logging.Logger:
    """크롤링 로그를 파일과 콘솔에 기록한다."""
    logger = logging.getLogger("arxiv_crawler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _text_without_descriptor(element) -> str:
    if element is None:
        return ""

    descriptor = element.find("span", class_="descriptor")
    if descriptor:
        descriptor.decompose()

    return element.get_text(" ", strip=True)


def _fetch_html(url: str) -> str:
    """URL에서 HTML을 가져온다. HTTP/네트워크 오류는 그대로 전달한다."""
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def _parse_paper_html(arxiv_id: str, html: str) -> dict[str, str]:
    """HTML에서 논문 정보를 파싱한다."""
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("h1", class_="title")

    if title_el is None:
        raise ParsingError(f"{arxiv_id}: h1.title 태그를 찾을 수 없습니다.")

    authors_el = soup.find("div", class_="authors")
    abstract_el = soup.find("blockquote", class_="abstract")
    subjects_el = soup.find("td", class_="tablecell subjects")

    if abstract_el is None:
        raise ParsingError(f"{arxiv_id}: abstract 태그를 찾을 수 없습니다.")

    return {
        "arxiv_id": arxiv_id,
        "title": _text_without_descriptor(title_el),
        "authors": ", ".join(
            author.get_text(strip=True) for author in authors_el.find_all("a")
        )
        if authors_el
        else "",
        "abstract": _text_without_descriptor(abstract_el),
        "submitted_date": _text_without_descriptor(soup.find("div", class_="dateline")),
        "categories": subjects_el.get_text(" ", strip=True) if subjects_el else "",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def extract_arxiv_ids_from_list_page(url: str) -> list[str]:
    """카테고리 목록 페이지에서 최신 논문의 arXiv ID 리스트를 반환한다."""
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    ids: list[str] = []

    for dt in soup.find_all("dt"):
        match = ARXIV_ID_PATTERN.search(dt.get_text(" ", strip=True))
        if match:
            ids.append(match.group(1))

    return ids


def fetch_paper_title(url: str) -> str:
    """논문 페이지 URL에서 h1.title 태그의 제목을 추출한다."""
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    return _text_without_descriptor(soup.find("h1", class_="title"))


def extract_paper_info(arxiv_id: str) -> dict[str, str]:
    """arXiv ID로 논문의 모든 정보를 추출해 딕셔너리로 반환한다."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    html = _fetch_html(url)
    return _parse_paper_html(arxiv_id, html)


def print_paper_info(paper: dict[str, str]) -> None:
    """논문 정보 딕셔너리를 보기 좋게 출력한다."""
    print(json.dumps(paper, indent=2, ensure_ascii=False))


def crawl_paper_safely(
    arxiv_id: str,
    logger: logging.Logger,
) -> dict[str, str] | None:
    """논문 1건을 안전하게 크롤링하고, 오류 시 None을 반환한다."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        html = _fetch_html(url)
        paper = _parse_paper_html(arxiv_id, html)
        logger.info(f"크롤링 성공 | arxiv_id={arxiv_id} | time={crawled_at}")
        return paper

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        logger.error(
            f"HTTP 오류 | arxiv_id={arxiv_id} | status={status_code} | "
            f"time={crawled_at} | message={error}"
        )
        print(f"  [HTTP 오류] {arxiv_id}: {status_code} {error}")

    except requests.RequestException as error:
        logger.error(
            f"네트워크 오류 | arxiv_id={arxiv_id} | time={crawled_at} | message={error}"
        )
        print(f"  [네트워크 오류] {arxiv_id}: {error}")

    except ParsingError as error:
        logger.error(
            f"파싱 오류 | arxiv_id={arxiv_id} | time={crawled_at} | message={error}"
        )
        print(f"  [파싱 오류] {arxiv_id}: {error}")

    return None


def crawl_papers(
    arxiv_ids: list[str],
    log_file: str = DEFAULT_LOG_FILE,
) -> dict[str, list[dict[str, str]] | int]:
    """여러 논문을 안전하게 크롤링한다.

    각 요청 전 15초 대기, 진행 상황 표시, 오류 처리, 로그 기록을 수행한다.
    """
    logger = setup_logger(log_file)
    total_count = len(arxiv_ids)
    papers: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    logger.info(f"크롤링 시작 | total={total_count}")

    for index, arxiv_id in enumerate(arxiv_ids, start=1):
        print(f"[{index}/{total_count}] {arxiv_id} - {REQUEST_DELAY_SECONDS}초 대기 중...")
        time.sleep(REQUEST_DELAY_SECONDS)

        print(f"[{index}/{total_count}] {arxiv_id} - 크롤링 중...")
        paper = crawl_paper_safely(arxiv_id, logger)

        if paper:
            papers.append(paper)
            print(f"[{index}/{total_count}] {arxiv_id} - 완료")
        else:
            failures.append({"arxiv_id": arxiv_id})
            print(f"[{index}/{total_count}] {arxiv_id} - 실패")

    success_count = len(papers)
    failure_count = len(failures)

    print(
        f"\n크롤링 완료: 총 {total_count}개 중 "
        f"성공 {success_count}개, 실패 {failure_count}개"
    )
    logger.info(
        f"크롤링 완료 | total={total_count} | success={success_count} | "
        f"failure={failure_count}"
    )

    return {
        "papers": papers,
        "failures": failures,
        "success_count": success_count,
        "failure_count": failure_count,
    }


def load_existing_papers(output_csv: str) -> pd.DataFrame:
    """CSV 파일이 있으면 기존 논문 데이터를 읽어 반환한다."""
    csv_path = Path(output_csv)
    if not csv_path.exists():
        return pd.DataFrame()

    return pd.read_csv(csv_path, encoding="utf-8-sig")


def get_existing_arxiv_ids(output_csv: str) -> set[str]:
    """CSV에 이미 저장된 arxiv_id 집합을 반환한다."""
    existing_df = load_existing_papers(output_csv)
    if existing_df.empty or "arxiv_id" not in existing_df.columns:
        return set()

    return set(existing_df["arxiv_id"].astype(str))


def crawl_and_save_papers(
    arxiv_ids: list[str],
    output_csv: str = "paper.csv",
    log_file: str = DEFAULT_LOG_FILE,
) -> pd.DataFrame:
    """중복을 제외한 새 논문만 안전하게 크롤링해 기존 CSV에 추가 저장한다."""
    total_count = len(arxiv_ids)
    existing_df = load_existing_papers(output_csv)
    existing_ids = get_existing_arxiv_ids(output_csv)

    new_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in existing_ids]
    skipped_count = total_count - len(new_ids)

    if skipped_count > 0:
        print(f"{total_count}개 중 {skipped_count}개는 이미 있어서 건너뜀")

    if not new_ids:
        print(f"새로운 논문이 없습니다. {output_csv} ({len(existing_df)}개)")
        return existing_df

    crawl_result = crawl_papers(new_ids, log_file=log_file)
    new_papers = crawl_result["papers"]
    new_df = pd.DataFrame(new_papers)

    if existing_df.empty:
        df = new_df
    elif new_df.empty:
        df = existing_df
    else:
        df = pd.concat([existing_df, new_df], ignore_index=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if new_papers:
        print(
            f"{len(new_papers)}개 논문을 추가해 {output_csv}에 저장했습니다. "
            f"(총 {len(df)}개, 실패 {crawl_result['failure_count']}개)"
        )
    else:
        print(f"새로운 논문을 저장하지 못했습니다. {output_csv} ({len(df)}개)")

    return df


def count_authors(authors: str) -> int:
    """저자 문자열에서 저자 수를 계산한다."""
    if not authors.strip():
        return 0
    return len([author for author in authors.split(",") if author.strip()])


def extract_top_keywords(texts: list[str], top_n: int = 5) -> list[tuple[str, int]]:
    """제목과 초록에서 가장 많이 등장한 키워드를 추출한다."""
    words: list[str] = []

    for text in texts:
        for word in re.findall(r"[a-zA-Z]{3,}", text.lower()):
            if word not in STOP_WORDS:
                words.append(word)

    return Counter(words).most_common(top_n)


def print_statistics(df: pd.DataFrame) -> None:
    """크롤링 결과에 대한 간단한 통계를 출력한다."""
    total_papers = len(df)
    author_counts = df["authors"].apply(count_authors)
    avg_authors = author_counts.mean() if total_papers else 0.0
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    top_keywords = extract_top_keywords(texts)

    print("\n=== 크롤링 통계 ===")
    print(f"총 논문 수: {total_papers}개")
    print(f"저자 수 평균: {avg_authors:.1f}명")

    if top_keywords:
        print("가장 많이 나온 키워드:")
        for keyword, count in top_keywords:
            print(f"  - {keyword}: {count}회")
    else:
        print("가장 많이 나온 키워드: 없음")


def crawl_category_recent(
    category_url: str = "https://arxiv.org/list/cs.AI/recent",
    limit: int = 10,
    output_csv: str = "paper.csv",
    log_file: str = DEFAULT_LOG_FILE,
) -> pd.DataFrame:
    """카테고리 최신 논문을 지정 개수만큼 크롤링하고 CSV 저장 및 통계를 출력한다."""
    arxiv_ids = extract_arxiv_ids_from_list_page(category_url)[:limit]
    print(f"cs.AI 최신 논문 {len(arxiv_ids)}개 크롤링을 시작합니다.\n")

    crawl_result = crawl_papers(arxiv_ids, log_file=log_file)
    df = pd.DataFrame(crawl_result["papers"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n{output_csv}에 저장했습니다.")

    print_statistics(df)
    return df


def crawl_paper(arxiv_id: str) -> dict[str, str]:
    """arXiv ID 하나에 대한 논문 정보를 수집한다."""
    return extract_paper_info(arxiv_id)


def main(
    list_url: str = "https://arxiv.org/list/cs.AI/recent",
    output_csv: str = "paper.csv",
    log_file: str = DEFAULT_LOG_FILE,
) -> pd.DataFrame:
    """목록 페이지에서 ID를 수집하고, 중복을 제외한 논문만 크롤링해 CSV로 저장한다."""
    arxiv_ids = extract_arxiv_ids_from_list_page(list_url)
    return crawl_and_save_papers(arxiv_ids, output_csv, log_file)


if __name__ == "__main__":
    main()
