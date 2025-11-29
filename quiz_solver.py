import os
import re
import time
import json
from pathlib import Path
from textwrap import dedent
from urllib.parse import urljoin  # for handling /submit relative URL

import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

WORKDIR = Path("workdir")
WORKDIR.mkdir(exist_ok=True)


# =========================
# Helper functions
# =========================

def download(url: str) -> Path:
    """
    Kisi bhi URL se file download karke local path return karo.
    """
    print(f"[DOWNLOAD] Fetching: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    name = url.split("/")[-1] or "downloaded_file"
    path = WORKDIR / name
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"[DOWNLOAD] Saved to: {path}")
    return path


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def extract_visible_text(html: str) -> str:
    """
    HTML se readable text nikaalne ke liye helper.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_urls_from_text(text: str):
    """
    Page ke text me jitne bhi http/https URLs hai unko list karo.
    """
    url_pattern = r"https?://[^\s\"'<>]+"
    return list(set(re.findall(url_pattern, text)))


def extract_submit_url_from_text(text: str, quiz_url: str):
    """
    Quiz page ke text se submit URL nikaalne ki koshish karo.

    Priority:
    1. Absolute URLs jisme 'submit' ho.  (https://.../submit...)
    2. Relative '/submit' ya '/submit?...' ko quiz_url ke base se join karo.
       (demo case: "POST this JSON to /submit")
    3. Fallback: quiz_url khud.
    """
    # 1) Absolute submit URLs
    abs_pattern = r"https?://[^\s\"'<>]*submit[^\s\"'<>]*"
    matches_abs = re.findall(abs_pattern, text)
    if matches_abs:
        print(f"[SUBMIT-URL] Found absolute submit URL: {matches_abs[0]}")
        return matches_abs[0]

    # 2) Relative '/submit' path
    rel_match = re.search(r"/submit[^\s\"'<>]*", text)
    if rel_match:
        rel_path = rel_match.group(0)   # e.g. "/submit" or "/submit?x=1"
        full_url = urljoin(quiz_url, rel_path)
        print(f"[SUBMIT-URL] Found relative submit URL, resolved to: {full_url}")
        return full_url

    # 3) Fallbacks
    urls = extract_urls_from_text(text)
    if len(urls) == 1:
        print(f"[SUBMIT-URL] Only one URL found on page, using it: {urls[0]}")
        return urls[0]

    print(f"[SUBMIT-URL] Falling back to quiz URL itself: {quiz_url}")
    return quiz_url


# =========================
# LLM-related helpers
# =========================

def call_llm_for_code(page_text: str, quiz_url: str, urls_on_page):
    """
    LLM se Python code mangwana jo ANSWER variable set kare.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    # Strong, corrected system prompt
    system_prompt = dedent("""
        You are an expert data engineer and data scientist.

        Your ONLY job is:
        → Read the quiz text and instructions.
        → Write Python code to calculate the FINAL ANSWER.
        → Store ONLY the final answer in a variable named ANSWER.

        DO NOT:
        - Construct JSON payloads for submission.
        - Add fields like email, secret, url inside ANSWER.
        - Create dictionaries like {"email": ..., "secret": ..., "url": ..., "answer": ...}.
        - Print anything.
        - Use input().
        - Write explanations or comments.

        ANSWER MUST BE ONLY:
        - int
        - float
        - bool
        - string
        - list
        - dict (ONLY if the quiz explicitly wants a structured answer)

        The submission JSON will be constructed by another part of the system.
        YOU ARE RESPONSIBLE ONLY FOR THE VALUE THAT GOES INTO "answer".

        Imports already available:
            import pandas as pd
            from pathlib import Path
            from quiz_solver import download, read_text, read_csv, read_excel
    """)

    urls_str = "\n".join(urls_on_page) if urls_on_page else "(no extra URLs found)"

    user_prompt = dedent(f"""
        You are solving a quiz hosted at: {quiz_url}

        Here is the visible text of the page:

        --- PAGE TEXT START ---
        {page_text[:12000]}
        --- PAGE TEXT END ---

        URLs mentioned on the page (might include data files or APIs):
        {urls_str}

        Write ONLY Python code (no comments, no backticks) that:
        - Follows the instructions in the text.
        - Downloads/parses any needed files or APIs.
        - Computes the exact final answer required by the question.
        - Stores it in a variable named ANSWER (with ONLY the final answer value).
    """)

    print(f"[LLM] Calling model: {OPENAI_MODEL}")

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        temperature=0,
    )

    code = resp.choices[0].message.content.strip()

    # Agar LLM ne ```python ...``` diya ho to strip karein
    if code.startswith("```"):
        code = re.sub(r"^```(python)?", "", code).strip()
        code = re.sub(r"```$", "", code).strip()

    return code


def normalize_answer(raw):
    """
    ANSWER ko saf bana do:
    - Agar LLM galti se {"email":..., "secret":..., "url":..., "answer": X} deta hai,
      to sirf X (inner answer) ko return karo.
    """
    if isinstance(raw, dict):
        keys = set(k.lower() for k in raw.keys())
        if {"email", "secret", "url", "answer"}.issubset(keys):
            print("[NORMALIZE] Detected wrapper dict with email/secret/url/answer, extracting inner 'answer'.")
            for k in raw.keys():
                if k.lower() == "answer":
                    return raw[k]
    return raw


def run_llm_code(code: str):
    """
    LLM se aaya hua Python code execute karo ek controlled namespace mein.
    """
    exec_globals = {
        "pd": pd,
        "Path": Path,
        "download": download,
        "read_text": read_text,
        "read_csv": read_csv,
        "read_excel": read_excel,
        "ANSWER": None,
    }
    exec_locals = {}

    try:
        exec(code, exec_globals, exec_locals)
    except Exception as e:
        print("[ERROR] LLM code execution failed:", repr(e))
        return None

    answer = exec_locals.get("ANSWER", exec_globals.get("ANSWER"))
    answer = normalize_answer(answer)
    return answer


# =========================
# Core quiz solving logic
# =========================

def solve_single_quiz(email: str, secret: str, quiz_url: str):
    """
    Ek single quiz URL solve karo:
    - Page render
    - LLM se code
    - Code se ANSWER
    - Submit answer
    - Response se (correct, next_url) return
    """
    print(f"[INFO] Solving quiz URL: {quiz_url}")

    # 1. Page load via Playwright (JS execute hoga)
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(quiz_url, wait_until="networkidle", timeout=120000)
        html = page.content()
        page_text = extract_visible_text(html)
        browser.close()

    # 2. Submit URL nikaalo
    submit_url = extract_submit_url_from_text(page_text, quiz_url)
    print("[INFO] Using submit URL:", submit_url)

    # 3. LLM se code lo
    urls_on_page = extract_urls_from_text(page_text)
    code = call_llm_for_code(page_text=page_text, quiz_url=quiz_url, urls_on_page=urls_on_page)
    print("[DEBUG] Generated code from LLM (first 500 chars):\n", code[:500], "...\n")

    # 4. Code execute karo, ANSWER nikaalo
    answer = run_llm_code(code)
    print("[INFO] Computed ANSWER:", answer)

    # 5. Submit answer
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    print(f"[INFO] Submitting answer to {submit_url}")
    resp = requests.post(submit_url, json=payload, timeout=120)
    resp.raise_for_status()

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[WARN] Non-JSON response from submit endpoint")
        return False, None

    correct = data.get("correct", False)
    next_url = data.get("url")
    reason = data.get("reason")

    print(f"[INFO] Submit result: correct={correct}, next_url={next_url}, reason={reason}")

    return correct, next_url


def solve_quiz_chain(email: str, secret: str, initial_url: str, start_time: float, max_duration_sec: int = 180):
    """
    Pura chain solve karo jab tak:
    - Time 3 minute se zyada na ho
    - Ya next_url na mile
    """
    url = initial_url
    attempt_count = 0

    while url and (time.time() - start_time) < max_duration_sec:
        attempt_count += 1
        print(f"[CHAIN] Attempt {attempt_count}, URL = {url}")

        correct, next_url = solve_single_quiz(email, secret, url)

        # Agar incorrect aur next_url nahi hai, aur time bacha hai, same URL retry
        if not correct and not next_url:
            elapsed = time.time() - start_time
            if elapsed < max_duration_sec - 10:
                print("[CHAIN] Incorrect answer, retrying same URL...")
                continue
            else:
                print("[CHAIN] Time almost up, stopping.")
                break

        # Agar next_url mila to uspar jao
        if next_url:
            url = next_url
        else:
            print("[CHAIN] No next URL, quiz seems over.")
            break

    print("[CHAIN] Finished quiz chain.")
