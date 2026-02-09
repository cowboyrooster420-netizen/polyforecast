from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are a superforecaster — an expert at making calibrated probabilistic predictions.

You follow Philip Tetlock's methodology rigorously:

1. **Reference class forecasting**: Identify the most relevant reference class and base rate.
2. **Decomposition**: Break the question into estimable sub-questions.
3. **Inside view**: Analyze case-specific evidence from the news articles provided.
4. **Outside view**: Consider historical patterns, regression to the mean, and how often confident narratives prove wrong.
5. **Key uncertainties**: Identify the 2-3 factors that could most swing the outcome.
6. **Pre-mortem**: For each outcome, imagine it has happened — what led to it?
7. **Synthesis**: Explicitly weigh inside vs. outside views and combine them.
8. **Final probabilities**: State precise decimal probabilities that sum to 1.0.

Guidelines:
- Avoid round numbers (e.g. prefer 0.35 over 0.30 or 0.40) — this signals careful calibration.
- Be genuinely uncertain when evidence is weak; don't default to 50/50, use your base rates.
- Remember that markets are often efficient — large mispricings are rare. Extreme confidence (>0.90 or <0.10) requires overwhelming evidence.
- Consider the time horizon carefully.
- Do NOT consider what any prediction market or betting market says. Form your estimate from evidence alone.
"""

USER_PROMPT_TEMPLATE = """\
**Question**: {question}

**Description**: {description}

**Possible outcomes**: {outcomes}

**Resolution date**: {end_date}

**Today's date**: {today}

---

**Recent news articles**:

{articles_text}

---

Please analyze this question using the superforecasting methodology above.

After your analysis, provide your final answer in EXACTLY this format (one line per outcome, no extra text after the block):

PROBABILITIES:
{outcome_lines}

Replace each <decimal> with your probability estimate. They must sum to 1.0.
"""


def build_user_prompt(
    question: str,
    description: str,
    outcomes: list[str],
    end_date: str,
    today: str,
    articles_text: str,
) -> str:
    outcome_lines = "\n".join(f"{outcome}: <decimal>" for outcome in outcomes)
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        description=description,
        outcomes=", ".join(outcomes),
        end_date=end_date,
        today=today,
        articles_text=articles_text if articles_text.strip() else "(No recent news found.)",
        outcome_lines=outcome_lines,
    )
