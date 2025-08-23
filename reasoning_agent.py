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
