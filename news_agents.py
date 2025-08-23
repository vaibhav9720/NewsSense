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
