"""
StockReason v0.7 - Module: News Event Classifier
Rule-based keyword and regular expression classifier for financial headlines.
Categorizes news events into 8 distinct financial event types:
- earnings
- guidance
- management_change
- regulatory
- merger_acquisition
- analyst_rating
- macro
- other
"""
import re
from typing import List, Dict, Any, Tuple, Optional

EVENT_TYPES = (
    "earnings",
    "guidance",
    "management_change",
    "regulatory",
    "merger_acquisition",
    "analyst_rating",
    "macro",
    "other",
)

# Ordered compiled regex patterns with priority weighting
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # 1. Macroeconomic & Monetary Policy (Checked before regulatory to separate MPC/forex/repo from RBI enforcement)
    (
        "macro",
        re.compile(
            r"\b(inflation|retail\s*inflation|cpi|wpi|wholesale\s*inflation|gdp\s*(growth|data|numbers)?|"
            r"repo\s*rate|reverse\s*repo|rbi\s*mpc|monetary\s*policy(\s*committee)?|"
            r"interest\s*rate\s*(hike|cut|pause|decision)|"
            r"central\s*bank(\s*policy|\s*rate|\s*announcement)?|"
            r"union\s*budget|fiscal\s*deficit|crude\s*oil|brent\s*crude|crude\s*prices?|"
            r"rupee\s*(slips|gains|falls|rises|all-time\s*low|strengthens|depreciates)|"
            r"us\s*fed|federal\s*reserve|rate\s*cut|rate\s*hike|forex\s*reserves?)\b",
            re.IGNORECASE,
        ),
    ),
    # 2. Guidance, Outlook & Forward-Looking Projections (Checked before earnings so forward guidance takes priority)
    (
        "guidance",
        re.compile(
            r"\b(guidance|forecasts?|forecasting|outlook|projections?|"
            r"sees\s+(\w+\s+)?(revenue|profit|growth|sales|margin|ebitda)|"
            r"expects?\s+(\w+\s+)*(revenue|profit|growth|sales|margin|festive)|"
            r"raises\s+(guidance|outlook|revenue\s*target|growth\s*target|order\s*book\s*guidance)|"
            r"lowers\s+(guidance|outlook|forecast)|cuts\s+(outlook|guidance|forecast)|"
            r"fy\d+\s*(guidance|outlook|target|growth|projection)|growth\s*projection)\b",
            re.IGNORECASE,
        ),
    ),
    # 3. Regulatory, Legal & Compliance
    (
        "regulatory",
        re.compile(
            r"\b(sebi(\s*slaps|\s*bars|\s*probes|\s*notices?|\s*penalty)?|"
            r"rbi\s+(penalty|fines|penalizes|slaps|bars|imposes|directs|probe|order|curbs|action|mandate)|"
            r"cci(\s*initiates|\s*probe|\s*approves)?|incometax|income\s*tax|"
            r"enforcement\s*directorate|\bed\b\s+(summons|raids|probes)|"
            r"penalty|penalties|fined?|prob(es?|ing)|investigat(es?|ion)|"
            r"show\s*cause|scams?|frauds?|money\s*laundering|raids?|searches|"
            r"nclt|insolvency|bankruptcy|tribunal|supreme\s*court|high\s*court|lawsuits?|litigation|"
            r"usfda(\s*issues)?|warning\s*letter|form\s*483|import\s*alert|inspection\s*observations?|"
            r"sanctions?|debarred|barred|non-compliance|regulatory\s*action|audit\s*lapse)\b",
            re.IGNORECASE,
        ),
    ),
    # 4. Management & Leadership Changes
    (
        "management_change",
        re.compile(
            r"\b(appoints?|appointed|appointing|appointments?|resigns?|resigned|resignations?|"
            r"steps?\s*down|stepping\s*down|quits?|ousted|fired|replaces?|replaced|"
            r"names\s+(.+?\s+)?as\s+(new\s+)?(ceo|cfo|coo|md|managing\s*director|director|chairman|chief|leadership)|"
            r"names\s+new\s+(ceo|cfo|coo|md|managing\s*director|director|chairman|chief|leadership)|"
            r"(ceo|cfo|coo|managing\s*director|executive\s*director|chairman|vice\s*chairman)\s+(exit|transition|tenure|joins|retires|resigns|leaves|steps\s*down)|"
            r"board\s*(reshuffle|shakeup|changes|approves\s*appointment)|succession\s*plan|"
            r"term\s*extension|leadership\s*(shuffle|transition))\b",
            re.IGNORECASE,
        ),
    ),
    # 5. Mergers, Acquisitions, JVs & Divestments
    (
        "merger_acquisition",
        re.compile(
            r"\b(acqui(re|res|red|ring|sition|sitions)|mergers?|merging|amalgamations?|takeovers?|takes\s*over|"
            r"(buy|buys|buying|bought|sell|sells|selling|sold|offload|offloads|picks?\s*up)\s+(.+?\s+)?stake|"
            r"stake\s*(sale|purchase|buyout)|enters\s*pact\s*to\s*buy|signs\s*pact\s*to\s*(buy|acquire)|"
            r"joint\s*venture|\bjv\b|strategic\s*alliance|divest(ment|ments|s|ing)|"
            r"open\s*offer|buyouts?|consortium\s*bids?|bids\s*for\s*(.+?\s+)?assets|m&a|deal\s*worth)\b",
            re.IGNORECASE,
        ),
    ),
    # 6. Analyst Ratings & Target Prices
    (
        "analyst_rating",
        re.compile(
            r"\b(upgrades?|upgraded|downgrades?|downgraded|"
            r"price\s*target|target\s*price|target\s*raised|target\s*cut|fair\s*value|"
            r"buy\s*rating|sell\s*rating|hold\s*rating|neutral\s*rating|outperform|underperform|overweight|underweight|"
            r"initiates?\s*coverage|reiterates?\s*(buy|sell|hold|outperform)|rates\s+(.+?\s+)?a\s*(buy|sell|hold)|"
            r"brokerage\s*(view|radar|calls|target)|(clsa|jefferies|morgan\s*stanley|goldman\s*sachs|nomura|macquarie|hsbc|motilal\s*oswal|bernstein|citi|jpmorgan|nuvama|kotak\s*institutional)\s*(retains|raises|cuts|recommends|rates|maintains))\b",
            re.IGNORECASE,
        ),
    ),
    # 7. Earnings, Quarterly Results & Corporate Actions (Dividends/Splits)
    (
        "earnings",
        re.compile(
            r"\b(q[1-4]\s*(results|profit|net|revenue|pat|ebitda|loss|earnings|numbers)|"
            r"quarterly\s*(results|profit|earnings|performance|numbers|net\s*profit)|"
            r"net\s*profit|operating\s*profit|\bpat\b|ebitda\s*(margin|grows|drops)?|"
            r"revenue\s*(growth|up|down|rises|falls)|profit\s*(after\s*tax|jumps|surges|falls|slumps|rises|drops|doubles)|"
            r"losses?\s*(widen|narrow|spikes?)|loss\s*of\s*rs|profit\s*of\s*rs|net\s*loss|"
            r"dividends?|interim\s*dividend|special\s*dividend|bonus\s*issue|stock\s*split|financial\s*results|"
            r"posts\s+(rs\s*)?[\d,\.]+\s*(crore|cr)?\s*pat)\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_headline(title: str, summary: Optional[str] = None) -> str:
    """
    Classifies a headline into one of the 8 event types.
    Checks regex patterns sequentially according to priority rules.
    If no pattern matches, returns 'other'.
    """
    text = (title or "").strip()
    if summary:
        text = f"{text} {summary.strip()}"

    if not text:
        return "other"

    for event_type, pattern in _PATTERNS:
        if pattern.search(text):
            return event_type

    return "other"


def classify_batch(titles: List[str], summaries: Optional[List[Optional[str]]] = None) -> List[str]:
    """
    Classifies a batch of headlines into event types.
    """
    if summaries is None:
        summaries = [None] * len(titles)
    return [classify_headline(t, s) for t, s in zip(titles, summaries)]


def get_classification_details(title: str, summary: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the event type, matched pattern (if any), and confidence estimate.
    """
    text = (title or "").strip()
    if summary:
        text = f"{text} {summary.strip()}"

    matched_type = "other"
    matched_span = None
    matched_text = ""

    for event_type, pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            matched_type = event_type
            matched_span = match.span()
            matched_text = match.group(0)
            break

    confidence = 0.92 if matched_type != "other" else 0.40
    return {
        "event_type": matched_type,
        "matched_text": matched_text,
        "matched_span": matched_span,
        "confidence": confidence,
    }
