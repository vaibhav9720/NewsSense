import os
import sys
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

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
USE_LLM_RERANK = True  # set False while debugging

# ====== clients ======
tavily = TavilySearchResults(max_results=12)  # needs TAVILY_API_KEY in env
llm = OpenAI()  # needs OPENAI_API_KEY in env if you use rerank/reasoning

# ====== preferred sources ======
SOURCES_TIER1 = [
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "apnews.com", "cnbc.com"
]
SOURCES_TIER2 = [
    "marketwatch.com", "moneycontrol.com", "economictimes.indiatimes.com",
    "seekingalpha.com", "nasdaq.com", "investing.com", "tradingview.com",
    "cnn.com", "bbc.com",
    "oilprice.com", "fxempire.com", "barchart.com", "fxstreet.com"
]

# ====== generic market-moving keywords (used for scoring, not mandatory) ======
KEYWORDS = (
    "volatility|surge|spike|jump|plunge|sell-off|rally|production cut|sanction|"
    "strike|shutdown|OPEC|rate hike|inflation|CPI|war|conflict|embargo|"
    "inventory|draw|build|downgrade|ban|export|import|tariff|supply|demand|"
    "price|futures|options|hedge|flows|positions|positioning|output|cut|hike"
)

# ====== date patterns to extract pub date from snippet/URL ======
DATE_PATTERNS = [
    r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',            # 2025-08-07 or 2025/08/07
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
                return {k.lower(): v for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}

ONTOLOGY_SYNONYMS = _load_synonyms(ONTOLOGY_SYNONYMS_PATH)

def _site_filter(sites: List[str]) -> str:
    return " OR ".join([f"site:{s}" for s in sites])

def _mk_queries(ontology: str, alert_date_str: str, sites=None, strict=True) -> List[str]:
    """
    Build a set of queries covering D and D-1 in multiple literal formats, plus non-literal fallbacks.
    """
    dt = datetime.strptime(alert_date_str, "%d %B %Y")
    prev = dt - timedelta(days=1)
    d1_a = dt.strftime("%d %B %Y")      # 22 August 2025
    d1_b = dt.strftime("%B %d, %Y")     # August 22, 2025
    d0_a = prev.strftime("%d %B %Y")
    d0_b = prev.strftime("%B %d, %Y")

    site_clause = f" ({_site_filter(sites)})" if sites else ""
    if strict:
        base = f'"{ontology}" (price OR prices OR market OR news)'
    else:
        base = f'"{ontology}"'

    queries = [
        f'{base}{site_clause} "{d1_a}"',
        f'{base}{site_clause} "{d1_b}"',
        f'{base}{site_clause} "{d0_a}"',
        f'{base}{site_clause} "{d0_b}"',
        # non-literal helpers (search engine can interpret recency)
        f"{base}{site_clause} past day",
        f"{base}{site_clause} last 24 hours",
        f"{base}{site_clause} latest news",
    ]
    seen, uniq = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq

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

def _try_parse_date(text: str) -> Optional[datetime]:
    if not text: 
        return None
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if not m:
            continue
        try:
            if pat == DATE_PATTERNS[0]:
                y, mo, d = map(int, m.groups())
                return datetime(y, mo, d)
            elif pat == DATE_PATTERNS[1]:
                mon, d, y = m.groups()
                return datetime.strptime(f"{mon} {d} {y}", "%B %d %Y")
            else:
                d, mon, y = m.groups()
                return datetime.strptime(f"{d} {mon} {y}", "%d %B %Y")
        except Exception:
            continue
    return None

def _is_in_window(item_dt: Optional[datetime], target_dt: datetime) -> bool:
    """
    If no date can be parsed from snippet/URL, DO NOT hard-drop; keep for later scoring/rerank.
    """
    if not item_dt:
        return True
    return item_dt.date() in {target_dt.date(), (target_dt - timedelta(days=1)).date()}

def _allow_patterns_for(ontology: str) -> List[re.Pattern]:
    """
    Build allow-patterns automatically from ontology + optional synonyms.
    If ontology is multi-word (e.g., "natural gas"), include the full phrase and each token.
    """
    terms = set()
    o = ontology.strip()
    if o:
        terms.add(o)
    syns = ONTOLOGY_SYNONYMS.get(o.lower(), [])
    for s in syns:
        if s:
            terms.add(s)

    # add tokens for multi-word ontologies
    more = set()
    for t in list(terms):
        toks = [w for w in re.split(r"\s+", t) if len(w) >= 3]
        more.update(toks)
    terms.update(more)

    patterns = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in terms if t]
    if not patterns and o:
        patterns = [re.compile(rf"\b{re.escape(o)}\b", re.I)]
    return patterns

def _required_patterns() -> List[re.Pattern]:
    """
    Previously enforced market-moving keywords as a hard filter. Now relaxed to improve recall.
    Keep empty to disable mandatory keyword gating; keywords influence _score() instead.
    """
    return []

def _filter_relevant(items: List[Dict[str, Any]], ontology: str, target_dt: datetime) -> List[Dict[str, Any]]:
    allow_patterns   = _allow_patterns_for(ontology)
    require_patterns = _required_patterns()

    out = []
    for it in items:
        title   = (it.get("title") or "").strip()
        snippet = (it.get("content") or "").strip()
        combined = f"{title} {snippet}".strip()
        url = (it.get("url") or "").strip()

        pub_dt = _try_parse_date(" ".join([combined, url]))
        it["_pub_dt"] = pub_dt

        # Date window is soft: if no date, keep; if parsed date, require D or D-1
        if not _is_in_window(pub_dt, target_dt):
            continue

        # Must match ontology allow-patterns somewhere in text
        if allow_patterns and not any(p.search(combined) for p in allow_patterns):
            continue

        # Market-moving keywords are not mandatory anymore (rely on scoring)
        if require_patterns and not any(p.search(combined) for p in require_patterns):
            continue

        out.append(it)
    return out

def _search_stage(ontology: str, alert_date: str, sites: List[str], tier_label: str, strict: bool) -> List[Dict[str, Any]]:
    raw = []
    for q in _mk_queries(ontology, alert_date, sites, strict=strict):
        out = tavily.run(q)  # List[{"content","url","title",...}]
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
        if len(snippet) > 220:
            snippet = snippet[:217] + "…"
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
    items.sort(key=lambda x: (_score(x, ontology), x.get("_pub_dt") or datetime.min), reverse=True)
    top = items[:k]

    # format output
    lines = []
    for i, it in enumerate(top, 1):
        title = (it.get("title") or "").strip()
        snippet = (it.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "…"
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
def _normalize_date_to_display(d: Any) -> Optional[str]:
    """
    Convert various date inputs to 'DD Month YYYY' string; return None if unparseable.
    """
    try:
        dt = pd.to_datetime(d, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%d %B %Y")
    except Exception:
        return None

def main():
    df = read_alerts(INPUT_PATH)
    print("Input shape:", df.shape)
    print(df.head())

    news_list, reasoning_list = [], []
    # try to detect columns
    onto_col = None
    date_col = None
    for c in df.columns:
        lc = c.strip().lower()
        if onto_col is None and ("ontology" in lc or "cdh_asset_ontology_name" in lc):
            onto_col = c
        if date_col is None and lc in {"date", "alert_date", "trade_date"}:
            date_col = c

    if onto_col is None or date_col is None:
        raise ValueError("Could not find ontology and date columns. Ensure your sheet has 'cdh_asset_ontology_name' and 'Date' columns (or similar).")

    for _, row in df.iterrows():
        ontology_raw = str(row[onto_col]).strip() if pd.notna(row[onto_col]) else ""
        date_disp = _normalize_date_to_display(row[date_col])

        if not ontology_raw or not date_disp:
            news_list.append("No ontology/date in row.")
            reasoning_list.append("Insufficient inputs for reasoning.")
            continue

        news_summary = run_news_agent(ontology_raw, date_disp)
        reasoning    = run_reasoning_agent(news_summary, ontology_raw, date_disp)

        print(f"\n--- {ontology_raw} | {date_disp} ---")
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
