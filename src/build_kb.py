import os
import json
import re
import requests

from bs4 import BeautifulSoup
from pypdf import PdfReader


SOURCES_FILE = "data/knowledge_base/sources.json"

DOCUMENTS_DIR = "data/knowledge_base/documents"

PASSAGES_FILE = "data/knowledge_base/passages.json"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/154.0 Safari/537.36"
    )
}

os.makedirs(DOCUMENTS_DIR, exist_ok=True)


with open(SOURCES_FILE, "r", encoding="utf-8") as f:
    sources = json.load(f)


def clean_text(text):

    text = text.replace("\xa0", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def extract_html_text(content):

    soup = BeautifulSoup(
        content,
        "html.parser"
    )

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header"
    ]):
        tag.decompose()

    main = soup.find("main")

    if main is None:
        main = soup.find("article")

    if main is None:
        main = soup.body

    if main is None:
        return ""

    text = main.get_text(
        separator=" ",
        strip=True
    )

    return clean_text(text)

def extract_pdf_text(content):

    import io

    pdf_stream = io.BytesIO(content)

    reader = PdfReader(pdf_stream)

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    text = "\n".join(pages)

    return clean_text(text)

def download_source(source):

    source_id = source["id"]
    url = source["url"]

    print(f"\nDownloading: {source['title']}")
    print(f"URL: {url}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if (
        "application/pdf" in content_type
        or url.lower().endswith(".pdf")
    ):
        text = extract_pdf_text(
            response.content
        )

    else:
        text = extract_html_text(
            response.content
        )

    if len(text) < 500:

        raise ValueError(
            f"Extracted text is too short "
            f"({len(text)} characters)."
        )

    filename = f"{source_id}.txt"

    output_path = os.path.join(
        DOCUMENTS_DIR,
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

    print(
        f"Extracted {len(text):,} characters"
    )

    print(
        f"Saved: {output_path}"
    )

    return text

def create_passages(
    text,
    source
):

    words = text.split()

    passages = []

    start = 0
    passage_number = 1

    while start < len(words):

        end = min(
            start + CHUNK_SIZE,
            len(words)
        )

        passage_words = words[start:end]

        passage_text = " ".join(
            passage_words
        )

        passages.append(
            {
                "passage_id": (
                    f"{source['id']}_"
                    f"{passage_number}"
                ),
                "source_id": source["id"],
                "title": source["title"],
                "publisher": source["publisher"],
                "url": source["url"],
                "text": passage_text
            }
        )

        passage_number += 1

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return passages

all_passages = []

successful_sources = 0
failed_sources = []


for source in sources:

    try:

        text = download_source(source)

        passages = create_passages(
            text,
            source
        )

        all_passages.extend(passages)

        successful_sources += 1

        print(
            f"Created {len(passages)} passages"
        )

    except Exception as e:

        print(
            f"ERROR: {source['id']}"
        )

        print(
            f"       {e}"
        )

        failed_sources.append(
            {
                "id": source["id"],
                "title": source["title"],
                "error": str(e)
            }
        )


with open(
    PASSAGES_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_passages,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 60)

print("KNOWLEDGE BASE BUILD COMPLETE")

print("=" * 60)

print(
    f"Sources listed    : {len(sources)}"
)

print(
    f"Sources successful: {successful_sources}"
)

print(
    f"Total passages    : {len(all_passages)}"
)

print(
    f"Passages file     : {PASSAGES_FILE}"
)

print(
    f"Documents folder  : {DOCUMENTS_DIR}"
)


if failed_sources:

    print("\nFailed sources:")

    for item in failed_sources:

        print(
            f"- {item['id']}: "
            f"{item['error']}"
        )


if successful_sources < 6:

    raise RuntimeError(
        "\nFewer than 6 sources were successfully "
        "added to the knowledge base. "
        "Fix the failed sources before continuing."
    )

else:

    print(
        "\n✓ Requirement satisfied: "
        "at least 6 sources successfully indexed."
    )
