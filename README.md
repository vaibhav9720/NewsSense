# NewsSense


import os
import sys
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from openai import OpenAI
from langchain.tools.tavily_search import TavilySearchResults

# ====== paths for your project utils ======
sys.path.append("/Users/vaibhav/Langchain/Project Preparation/NewsSense")
from utils.excel_io import read_alerts, write_alerts

# ====== config ======
INPUT_PATH  = "/Users/vaibhav/Langchain/Project Preparation/NewsSense/data/dummy_alerts.xlsx"
OUTPUT_PATH = "/Users/vaibhav/Langchain/Project Preparation/NewsSense/data/alerts_output.xlsx"

# Optional synonyms mapping file (JSON): {"crude oil":["WTI","Brent"], "gold":["XAU","bullion"]}
ONTOLOGY_SYNONYMS_PATH = "/Users/vaibhav/Langchain/Project Preparation/NewsSense/data/ontology_synonyms.json"

# Toggle: use LLM to re-check relevance after filters (higher precision, slightly more cost)
USE_LLM_RERANK = True

# ====== clients ======
tavily = TavilySearchResults(max_results=12)
llm = OpenAI()

# ====== preferred sources ======
SOURCES_TIER1 = [
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "apnews.com", "cnbc.com", "trading.com"
]
SOURCES_TIER2 = [
    "marketwatch.com", "moneycontrol.com", "economictimes.indiatimes.com",
    "seekingalpha.com", "nasdaq.com", "investing.com", "tradingview.com",
    "cnn.com", "bbc.com"
]

# ====== generic market-moving keywords (require at least one) ======
KEYWORDS = (
    "volatility|surge|spike|jump|plunge|sell-off|rally|production cut|sanction|"
    "strike|shutdown|OPEC|rate hike|inflation|CPI|war|conflict|embargo|"
    "inventory|draw|build|downgrade|ban|export|import|tariff|supply|demand|"
    "price|futures|options|hedge|flows|positions|positioning|output"
)

# ====== date patterns to extract pub date from snippet/URL ======
DATE_PATTERNS = [
    r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',            # 2025-08-07
    r'([A-Z][a-z]{2,9})\s+(\d{1,2}),\s+(\d{4})',     # August 7, 2025
    r'(\d{1,2})\s+([A-Z][a-z]{2,9})\s+(\d{4})',      # 07 August 2025
]

# =========================
# helpers
# =========================
def _load_synonyms(path: str) -> Dict[str, List[str]]:
    try:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                # lowercase keys for robust lookup
                return {k.lower(): v for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}

ONTOLOGY_SYNONYMS = _load_synonyms(ONTOLOGY_SYNONYMS_PATH)

def _site_filter(sites: List[str]) -> str:
    return " OR ".join([f"site:{s}" for s in sites])

def _mk_queries(ontology: str, alert_date_str: str, sites=None, strict=True) -> List[str]:
    dt = datetime.strptime(alert_date_str, "%d %B %Y")
    prev = dt - timedelta(days=1)
    d1 = dt.strftime("%d %B %Y")
    d0 = prev.strftime("%d %B %Y")
    base = f'{ontology} market news' if strict else f'{ontology}'
    if sites:
        base += f" ({_site_filter(sites)})"
    return [f'{base} "{d1}"', f'{base} "{d0}"']

def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in items:
        url = (r.get("url") or "").strip()
        if url and url not in seen:
            seen.add(url)
            out.append(r)
    return out

def _score(item: Dict[str, Any], ontology: str) -> int:
    text = f'{item.get("title","")} {item.get("content","")}'
    url  = (item.get("url") or "").lower()
    s = 0
    if ontology.lower() in text.lower(): s += 2
    if re.search(KEYWORDS, text, flags=re.I): s += 1
    if any(d in url for d in SOURCES_TIER1): s += 1
    s += min(len(text) // 300, 1)
    return s

def _try_parse_date(text: str):
    if not text: return None
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                if pat == DATE_PATTERNS[0]:
                    y, mo, d = map(int, m.groups()); return datetime(y, mo, d)
                elif pat == DATE_PATTERNS[1]:
                    mon, d, y = m.groups(); return datetime.strptime(f"{mon} {d} {y}", "%B %d %Y")
                else:
                    d, mon, y = m.groups(); return datetime.strptime(f"{d} {mon} {y}", "%d %B %Y")
            except Exception:
                continue
    return None

def _is_in_window(item_dt: datetime, target_dt: datetime) -> bool:
    if not item_dt: return False
    return item_dt.date() in {target_dt.date(), (target_dt - timedelta(days=1)).date()}

def _allow_patterns_for(ontology: str) -> List[re.Pattern]:
    """
    Build allow-patterns automatically from ontology + optional synonyms.
    If ontology is multi-word (e.g., "natural gas"), include the full phrase and each token.
    """
    terms = set()
    o = ontology.strip()
    if o: terms.add(o)
    syns = ONTOLOGY_SYNONYMS.get(o.lower(), [])
    print("check",syns)
    for s in syns:
        if s: terms.add(s)

    # add tokens for multi-word ontologies
    more = set()
    for t in list(terms):
        toks = [w for w in re.split(r"\s+", t) if len(w) >= 3]
        more.update(toks)
    terms.update(more)

    # compile to regex with word boundaries, case-insensitive
    patterns = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in terms if t]
    # if no terms (shouldn’t happen), fallback to plain ontology word boundary
    if not patterns and o:
        patterns = [re.compile(rf"\b{re.escape(o)}\b", re.I)]
    return patterns

def _required_patterns() -> List[re.Pattern]:
    return [re.compile(KEYWORDS, re.I)]

def _filter_relevant(items: List[Dict[str, Any]], ontology: str, target_dt: datetime) -> List[Dict[str, Any]]:
    allow_patterns   = _allow_patterns_for(ontology)
    require_patterns = _required_patterns()

    out = []
    for it in items:
        title   = (it.get("title") or "").strip()
        snippet = (it.get("content") or "").strip()
        combined = f"{title} {snippet}".strip()
        url = it.get("url") or ""

        pub_dt = _try_parse_date(combined + " " + url)
        it["_pub_dt"] = pub_dt
        if not _is_in_window(pub_dt, target_dt):
            continue

        # Must match ontology (allow) in title/snippet
        if allow_patterns and not any(p.search(combined) for p in allow_patterns):
            continue

        # Must match at least one market-moving keyword
        if require_patterns and not any(p.search(combined) for p in require_patterns):
            continue

        out.append(it)
    return out

def _search_stage(ontology: str, alert_date: str, sites: List[str], tier_label: str, strict: bool) -> List[Dict[str, Any]]:
    raw = []
    for q in _mk_queries(ontology, alert_date, sites, strict=strict):
        out = tavily.run(q)  # List[{"content","url", maybe "title"}]
        if isinstance(out, list):
            for item in out:
                item["_tier"] = tier_label
            raw.extend(out)
    return raw

def _llm_rerank(items: List[Dict[str, Any]], ontology: str) -> List[Dict[str, Any]]:
    if not items:
        return items
    # compact list for LLM
    lines = []
    for i, it in enumerate(items, 1):
        title = it.get("title") or ""
        snippet = (it.get("content") or "").replace("\n", " ").strip()
        url = it.get("url") or ""
        if len(snippet) > 220: snippet = snippet[:217] + "…"
        lines.append(f"{i}. {title} — {snippet} ({url})")
    prompt = f"""
You are filtering market news for "{ontology}".
Keep only items that are clearly about this ontology AND describe market-moving factors
(volatility, price/volume moves, supply/demand, policy/geopolitics, OPEC/Fed, sanctions, inventories, safe-haven flows, etc.).
Return the kept item indices as a comma-separated list (e.g., "1,3,7"). If none are relevant, return "none".

News:
{chr(10).join(lines)}
"""
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=50,
        messages=[
            {"role": "system", "content": "You precisely filter market-relevant news about a given ontology."},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content.strip().lower()

    if "none" in resp:
        return []
    kept = []
    for tok in re.split(r"[^\d]+", resp):
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(items):
                kept.append(items[idx])
    return kept

# =========================
# NEWS AGENT
# =========================
def run_news_agent(ontology: str, alert_date: str, k: int = 7) -> str:
    target_dt = datetime.strptime(alert_date, "%d %B %Y")

    # Pass 1: Tier1 strict
    raw = _search_stage(ontology, alert_date, SOURCES_TIER1, "Tier1-Strict", strict=True)
    items = _filter_relevant(_dedupe(raw), ontology, target_dt)

    # Pass 2: Tier1 loose
    if not items:
        raw += _search_stage(ontology, alert_date, SOURCES_TIER1, "Tier1-Loose", strict=False)
        items = _filter_relevant(_dedupe(raw), ontology, target_dt)

    # Pass 3: Tier2 fallback
    if not items:
        raw += _search_stage(ontology, alert_date, SOURCES_TIER2, "Tier2-Fallback", strict=False)
        items = _filter_relevant(_dedupe(raw), ontology, target_dt)

    if not items:
        return "No relevant D/D-1 news found for the specified ontology and date window."

    if USE_LLM_RERANK:
        items = _llm_rerank(items, ontology)
        if not items:
            return "No relevant D/D-1 news found for the specified ontology and date window (after LLM filter)."

    # rank & select top k
    items.sort(key=lambda x: (_score(x, ontology), x["_pub_dt"] or datetime.min), reverse=True)
    top = items[:k]

    # format output
    lines = []
    for i, it in enumerate(top, 1):
        title = (it.get("title") or "").strip()
        snippet = (it.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 220: snippet = snippet[:217] + "…"
        url = it.get("url") or ""
        domain = url.split("/")[2] if "://" in url else url
        tier = it.get("_tier", "Unknown")
        dt_str = it["_pub_dt"].strftime("%d %b %Y") if it.get("_pub_dt") else "Unknown date"
        header = f"{i}. [{dt_str}] [{domain}] [{tier}]"
        lines.append(f"{header} {title} — {snippet} ({url})" if title else f"{header} {snippet} ({url})")
    return "\n".join(lines)

# =========================
# REASONER
# =========================
REASONING_PROMPT = """
You are a market surveillance analyst.

Analyze the following news and determine if it could explain a spike in trading alerts for {ontology} on {alert_date}.

Instructions:
1) ONLY use the information present in the news items; do not invent facts.
2) If the news clearly indicates volatility, price/volume shock, major policy/geopolitical event, or supply/demand shock, explain briefly how that could increase alerts.
3) If the news is neutral or does not suggest abnormal activity, explicitly state that it likely does NOT explain a spike.
4) Prefer precision over generic language. 1–2 sentences max.

News:
{news_summary}

Output:
"""

def run_reasoning_agent(news_summary: str, ontology: str, alert_date: str) -> str:
    if news_summary.startswith("No relevant"):
        return "No D/D-1 news available; insufficient evidence to link this alert to external news."
    prompt = REASONING_PROMPT.format(
        news_summary=news_summary,
        ontology=ontology,
        alert_date=alert_date
    )
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=120,
        messages=[
            {"role": "system", "content": "You are a precise, cautious market surveillance analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# =========================
# MAIN
# =========================
def main():
    df = read_alerts(INPUT_PATH)
    print("Input shape:", df.shape)
    print(df.head())

    news_list, reasoning_list = [], []
    for _, row in df.iterrows():
        row['Date'] = pd.to_datetime('2025-08-21', errors='coerce')
        row['cdh_asset_ontology_name'] = 'Natural Gas'
        ontology   = str(row["cdh_asset_ontology_name"])
        alert_date = row["Date"].strftime("%d %B %Y")


        news_summary = run_news_agent(ontology, alert_date)
        reasoning    = run_reasoning_agent(news_summary, ontology, alert_date)

        print(f"\n--- {ontology} | {alert_date} ---")
        print("NEWS:\n", news_summary)
        print("REASONING:\n", reasoning)

        news_list.append(news_summary)
        reasoning_list.append(reasoning)

    df["News_Summary"]  = news_list
    df["LLM_Reasoning"] = reasoning_list

    write_alerts(df, OUTPUT_PATH)
    print("✅ Alert reasoning written to", OUTPUT_PATH)

if __name__ == "__main__":
    main()


news.py

from openai import OpenAI

from datetime import datetime, timedelta
from langchain.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults(max_results=6) 
llm = OpenAI()



KEYWORDS = (
    "volatility|surge|spike|jump|plunge|sell-off|rally|production cut|sanction|"
    "strike|shutdown|OPEC|rate hike|inflation|CPI|war|conflict|embargo|"
    "inventory|draw|build|downgrade|ban|export|import|tariff"
)

# def load_prompt(file_path):
#     with open(file_path, "r") as f:
#         return f.read()

# news_prompt = load_prompt("/Users/vaibhav/Langchain/Project Preparation/NewsSense/prompts/news_prompt.txt")

# def run_news_agent(ontology, alert_date):
#     prompt = news_prompt.format(ontology=ontology, date=alert_date)
#     response = llm.chat.completions.create(
#         model="gpt-4",
#         #model = "gpt-3.5-turbo",
#         max_tokens =300,
#         messages=[
#     {"role": "system", "content": "You are a financial market assistant that retrieves relevant news for a specific asset and date."},
#     {"role": "user", "content": prompt}
# ],
#         temperature=0.7,
#     )
#     return response.choices[0].message.content.strip()


def _mk_queries(ontology: str, alert_date_str: str):
    """Build 2 date-focused queries: on the date and the day before."""
    dt = datetime.strptime(alert_date_str, "%d %B %Y")
    prev = dt - timedelta(days=1)
    date_q = dt.strftime("%d %B %Y")
    prev_q = prev.strftime("%d %B %Y")
    # steer to reputable finance outlets via query text (simple but effective)
    base = f'{ontology} market news (Reuters OR FT OR "Wall Street Journal" OR AP OR CNBC)'
    return [f'{base} "{date_q}"', f'{base} "{prev_q}"']

def _dedupe(results):
    seen = set()
    out = []
    for r in results:
        t = (r.get("url") or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(r)
    return out

def _score(item: dict, ontology: str):
    """Tiny heuristic to push clearly relevant/impactful snippets up."""
    text = (item.get("content") or "").lower()
    score = 0
    if ontology.lower() in text:
        score += 2
    # keyword bump
    import re
    if re.search(KEYWORDS, text, flags=re.I):
        score += 1
    # prefer Reuters/AP/WSJ/FT if visible in URL
    url = (item.get("url") or "").lower()
    if any(d in url for d in ["reuters.com", "apnews.com", "wsj.com", "ft.com", "bloomberg.com", "cnbc.com"]):
        score += 1
    # longer snippet gets tiny bump
    score += min(len(text) // 300, 1)
    return score

def run_news_agent(ontology: str, alert_date: str, k: int = 4) -> str:
    """
    Returns a compact string of top-k headlines/snippets for ontology & date window (D and D-1).
    Designed to feed your reasoning LLM.
    """
    queries = _mk_queries(ontology, alert_date)

    # Call Tavily twice (date and previous day)
    raw = []
    for q in queries:
        # returns List[{"content": "...", "url": "..."}]
        out = tavily.run(q)
        if isinstance(out, list):
            raw.extend(out)

    # de-dupe & rank
    items = _dedupe(raw)
    items.sort(key=lambda x: _score(x, ontology), reverse=True)
    top = items[:k]

    # format compactly; Tavily tool doesn’t always include titles/dates, so use snippet+url
    lines = []
    for i, it in enumerate(top, 1):
        snippet = (it.get("content") or "").strip().replace("\n", " ")
        url = it.get("url") or ""
        # keep each item short to control LLM cost downstream
        if len(snippet) > 260:
            snippet = snippet[:257] + "…"
        lines.append(f"{i}. {snippet} ({url})")

    if not lines:
        return "No clear, reputable news found for the date window. Consider deprioritizing this alert."

    return "\n".join(lines)

news_list, reasoning_list = [], []

for _, row in df.iterrows():
    ontology = str(row["cdh_asset_ontology_name"])
    alert_date = row["Date"].strftime("%d %B %Y")  # keep your original per-row date

    news_summary = run_news_agent(ontology, alert_date)
    reasoning = run_reasoning_agent(news_summary, ontology, alert_date)  # <-- pass date

    news_list.append(news_summary)
    reasoning_list.append(reasoning)

df["News_Summary"] = news_list
df["LLM_Reasoning"] = reasoning_list



reasoning.py

from openai import OpenAI

llm = OpenAI()

# def load_prompt(file_path):
#     with open(file_path, "r") as f:
#         return f.read()

# reasoning_prompt = load_prompt("/Users/vaibhav/Langchain/Project Preparation/NewsSense/prompts/reasoning_prompt.txt")

# def run_reasoning_agent(news_summary, ontology, alert_date):
#     prompt = reasoning_prompt.format(news_summary=news_summary, ontology=ontology,alert_date=alert_date)
#     response = llm.chat.completions.create(
#         #model="gpt-4",
#         model = "gpt-4",
#         max_tokens=300,
#         messages=[
#     {"role": "system", "content": "You are a market surveillance analyst that explains whether news could justify a spike in trading alerts."},
#     {"role": "user", "content": prompt}
# ],
#         temperature=0.7,
#     )
#     return response.choices[0].message.content.strip()

from openai import OpenAI
llm = OpenAI()

REASONING_PROMPT = """
You are a market surveillance analyst.

Analyze the following news and determine if it could explain a spike in trading alerts for {ontology} on {alert_date}.

Instructions:
1) ONLY use the information present in the news items; do not invent facts.
2) If the news clearly indicates volatility, price/volume shock, major policy/geopolitical event, or supply/demand shock, explain briefly how that could increase alerts.
3) If the news is neutral or does not suggest abnormal activity, explicitly state that it likely does NOT explain a spike.
4) Prefer precision over generic language. 1–2 sentences max.

News:
{news_summary}

Output:
"""

def run_reasoning_agent(news_summary: str, ontology: str, alert_date: str) -> str:
    prompt = REASONING_PROMPT.format(
        news_summary=news_summary,
        ontology=ontology,
        alert_date=alert_date
    )
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",  # cost-efficient; switch to gpt-4o if needed
        temperature=0.2,
        max_tokens=120,
        messages=[
            {"role": "system", "content": "You are a precise, cautious market surveillance analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()





