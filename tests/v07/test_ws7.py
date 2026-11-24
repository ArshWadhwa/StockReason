"""
StockReason v0.7 - Workstream 7 Test Suite: Point-in-Time News Event Store
Tests calendar timing, market close rollover, deduplication, 100+ headline event classification,
strict point-in-time feature immutability, and missing day handling.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.v07.news.store import (
    assign_trading_date,
    build_event_record,
    NewsEventStore,
    EVENT_SCHEMA_COLUMNS,
    IST_TZ,
)
from src.v07.news.dedup import (
    normalize_title,
    title_similarity,
    cluster_headlines,
    deduplicate_dataframe,
)
from src.v07.news.classifier import (
    classify_headline,
    classify_batch,
    get_classification_details,
    EVENT_TYPES,
)
from src.v07.news.features import (
    compute_ticker_daily_features,
    build_daily_news_features,
)
from src.v07.news.sources import (
    GoogleNewsRSSSource,
    GDELTSource,
    ingest_articles_to_store,
)


# ══════════════════════════════════════════════════════════════════════
# 1. Market Close & Calendar Rollover Tests
# ══════════════════════════════════════════════════════════════════════

def test_after_close_rolls_to_next_trading_day():
    """
    Article published after 15:30 IST on a Monday trading day must roll to Tuesday.
    Article published at or before 15:30 IST on a Monday must stay on Monday.
    """
    # Monday 2026-09-07 15:20 IST (Before close)
    dt_before = datetime(2026, 9, 7, 15, 20, tzinfo=IST_TZ)
    assigned_before = assign_trading_date(dt_before)
    assert assigned_before == "2026-09-07", "Before 15:30 IST must remain on the current trading day"

    # Monday 2026-09-07 15:35 IST (After close)
    dt_after = datetime(2026, 9, 7, 15, 35, tzinfo=IST_TZ)
    assigned_after = assign_trading_date(dt_after)
    assert assigned_after == "2026-09-08", "After 15:30 IST must roll to next trading day (Tuesday)"


def test_weekend_rolls_to_monday():
    """
    Articles published on Saturday or Sunday must roll to Monday.
    """
    # Saturday 2026-09-05 11:00 IST
    dt_sat = datetime(2026, 9, 5, 11, 0, tzinfo=IST_TZ)
    assert assign_trading_date(dt_sat) == "2026-09-07"

    # Sunday 2026-09-06 22:30 IST
    dt_sun = datetime(2026, 9, 6, 22, 30, tzinfo=IST_TZ)
    assert assign_trading_date(dt_sun) == "2026-09-07"

    # Friday after-hours (18:00 IST) must roll past the weekend to Monday
    dt_fri_after = datetime(2026, 9, 4, 18, 0, tzinfo=IST_TZ)
    assert assign_trading_date(dt_fri_after) == "2026-09-07"


def test_nse_holiday_rolls_forward():
    """
    Articles published on an official NSE holiday must roll to the next valid trading day.
    Example: 2026-01-26 is Republic Day (Monday). Must roll to 2026-01-27 (Tuesday).
    """
    dt_holiday = datetime(2026, 1, 26, 10, 0, tzinfo=IST_TZ)
    assigned = assign_trading_date(dt_holiday)
    assert assigned == "2026-01-27", "Republic Day must roll forward to Tuesday Jan 27"


# ══════════════════════════════════════════════════════════════════════
# 2. Near-Duplicate Clustering Tests
# ══════════════════════════════════════════════════════════════════════

def test_duplicate_clustering():
    """
    Syndicated near-duplicate headlines should receive the same dedup_cluster_id.
    """
    headlines = [
        {
            "event_id": "ev_1",
            "title": "Reliance Industries Q1 net profit jumps 15% to Rs 18,000 crore - Economic Times",
            "published_at_utc": "2026-09-07T08:00:00Z",
            "confidence": 0.85,
        },
        {
            "event_id": "ev_2",
            "title": "Reliance Q1 profit surges 15% to Rs 18000 cr | Livemint",
            "published_at_utc": "2026-09-07T08:05:00Z",
            "confidence": 0.90,
        },
        {
            "event_id": "ev_3",
            "title": "TCS bags $1 billion digital transformation deal from European bank",
            "published_at_utc": "2026-09-07T09:00:00Z",
            "confidence": 0.88,
        },
    ]

    clustered = cluster_headlines(headlines, similarity_threshold=0.60)
    assert len(clustered) == 3

    # ev_1 and ev_2 are syndicated duplicates of the Reliance earnings announcement
    assert clustered[0]["dedup_cluster_id"] == clustered[1]["dedup_cluster_id"]
    # ev_3 is completely distinct
    assert clustered[2]["dedup_cluster_id"] != clustered[0]["dedup_cluster_id"]

    # Test deduplication
    df = pd.DataFrame(clustered)
    deduped = deduplicate_dataframe(df, keep="highest_confidence")
    assert len(deduped) == 2
    # Highest confidence for cluster 1 was ev_2 (0.90)
    assert "ev_2" in deduped["event_id"].values
    assert "ev_3" in deduped["event_id"].values


# ══════════════════════════════════════════════════════════════════════
# 3. Rule-Based Event Classifier (100+ Sample Headlines)
# ══════════════════════════════════════════════════════════════════════

SAMPLE_100_HEADLINES = [
    # ── Earnings (15 headlines) ──
    ("Reliance Industries Q1 net profit surges 19% to Rs 19,500 crore", "earnings"),
    ("TCS Q4 results: Operating margin expands 150 bps, dividend of Rs 28 declared", "earnings"),
    ("Infosys quarterly performance: Revenue growth beats estimates, PAT up 7%", "earnings"),
    ("HDFC Bank net profit jumps 33% YoY on healthy net interest income", "earnings"),
    ("ICICI Bank quarterly profit rises 14.5% to Rs 11,059 crore", "earnings"),
    ("Tata Motors posts Rs 3,100 crore PAT in Q2 vs net loss last year", "earnings"),
    ("State Bank of India Q3 net profit rises 22% to record high", "earnings"),
    ("Bharti Airtel EBITDA grows 12% to Rs 19,800 crore as ARPU rises", "earnings"),
    ("Wipro quarterly profit falls 8% YoY to Rs 2,870 crore", "earnings"),
    ("L&T Q1 results: Order inflow up 20%, declares special dividend of Rs 10", "earnings"),
    ("ITC reports 10% rise in quarterly net profit, interim dividend approved", "earnings"),
    ("Kotak Mahindra Bank Q2 profit after tax grows 18% to Rs 3,190 crore", "earnings"),
    ("Axis Bank quarterly results: Net profit of Rs 5,800 crore, asset quality improves", "earnings"),
    ("IndusInd Bank Q4 losses widen as provisions spike", "earnings"),
    ("AU Small Finance Bank approves Rs 1 dividend and financial results", "earnings"),

    # ── Guidance & Outlook (13 headlines) ──
    ("Infosys raises FY26 revenue guidance to 4-5% in constant currency", "guidance"),
    ("TCS management sees double-digit revenue growth in BFSI over FY26", "guidance"),
    ("HCL Tech cuts revenue growth outlook for FY25 to 3-4%", "guidance"),
    ("Larsen & Toubro raises order book guidance by 15% for current fiscal", "guidance"),
    ("Wipro expects flat to 1% revenue growth in Q3 constant currency", "guidance"),
    ("Tech Mahindra lowers FY25 profit margin forecast amid furloughs", "guidance"),
    ("Maruti Suzuki sees vehicle sales growth projection at 8% for FY26", "guidance"),
    ("Tata Steel management forecasts stronger European margins in H2", "guidance"),
    ("Sun Pharma provides double-digit growth projection for global specialty portfolio", "guidance"),
    ("Bajaj Finance raises credit growth guidance to 25% for FY26", "guidance"),
    ("Titan Company expects festive season sales to surge 20%", "guidance"),
    ("Cipla raises FY26 EBITDA margin outlook to 24-25%", "guidance"),
    ("Zomato management outlook sees quick commerce reaching breakeven by Q4", "guidance"),

    # ── Management Change (14 headlines) ──
    ("HDFC Bank appoints Kaizad Bharucha as Deputy Managing Director", "management_change"),
    ("Infosys CFO Nilanjan Roy resigns, Jayesh Sanghrajka named new CFO", "management_change"),
    ("TCS CEO Rajesh Gopinathan steps down, K Krithivasan appointed CEO", "management_change"),
    ("Wipro CEO Thierry Delaporte resigns, board appoints Srinivas Pallia as new CEO", "management_change"),
    ("Kotak Mahindra Bank names Ashok Vaswani as new MD and CEO", "management_change"),
    ("Tata Sons chairman N Chandrasekaran gets board approval for term extension", "management_change"),
    ("IndusInd Bank executive director quits ahead of tenure completion", "management_change"),
    ("Bandhan Bank founder and MD Chandra Shekhar Ghosh steps down", "management_change"),
    ("ICICI Prudential AMC names new chief investment officer after leadership transition", "management_change"),
    ("L&T board shakeup: S N Subrahmanyan takes charge as Chairman and MD", "management_change"),
    ("Axis Bank appoints Amitabh Chaudhry as MD & CEO for another three-year term", "management_change"),
    ("State Bank of India names new managing director following government notification", "management_change"),
    ("Paytm COO Bhavesh Gupta resigns amid executive leadership shuffle", "management_change"),
    ("Tata Motors board approves appointment of independent director", "management_change"),

    # ── Regulatory, Legal & Compliance (15 headlines) ──
    ("SEBI slaps Rs 25 crore penalty on promoter group for disclosure lapses", "regulatory"),
    ("RBI imposes monetary penalty of Rs 1.5 crore on private lender for KYC violations", "regulatory"),
    ("Enforcement Directorate raids corporate office in money laundering probe", "regulatory"),
    ("CCI initiates antitrust investigation into e-commerce discount practices", "regulatory"),
    ("USFDA issues Form 483 with five inspection observations to formulation facility", "regulatory"),
    ("Supreme Court dismisses review petition in telecom AGR dues case", "regulatory"),
    ("NCLT admits insolvency plea against infrastructure developer over debt default", "regulatory"),
    ("Income tax department conducts searches at pharmaceutical company premises", "regulatory"),
    ("USFDA issues warning letter for active pharmaceutical ingredient plant", "regulatory"),
    ("SEBI bars former fund manager from securities market in front-running scam", "regulatory"),
    ("RBI directs financial institution to cease onboarding new co-branded credit cards", "regulatory"),
    ("High Court issues notice to company on environmental non-compliance litigation", "regulatory"),
    ("CCI approves acquisition subject to voluntary structural remedies", "regulatory"),
    ("ED summons executives in connection with cross-border remittance investigation", "regulatory"),
    ("Securities tribunal stays regulator's debarment order against brokerage", "regulatory"),

    # ── Mergers, Acquisitions & Divestments (14 headlines) ──
    ("Reliance Retail acquires 51% stake in premium apparel brand for Rs 800 crore", "merger_acquisition"),
    ("Tata Consumer Products enters pact to buy Capital Foods for Rs 5,100 crore", "merger_acquisition"),
    ("Adani Ports acquires 95% stake in Gopalpur Port for Rs 3,080 crore", "merger_acquisition"),
    ("HDFC Life approves merger of Exide Life Insurance into itself", "merger_acquisition"),
    ("Bharti Airtel offloads 1% stake in Indus Towers via open market transaction", "merger_acquisition"),
    ("JSW Steel forms joint venture with JFE Steel for electrical steel manufacturing", "merger_acquisition"),
    ("Cipla promoters in talks to sell 33% stake to private equity consortium", "merger_acquisition"),
    ("UltraTech Cement announces acquisition of Kesoram cement assets for Rs 7,600 cr", "merger_acquisition"),
    ("Larsen & Toubro divestment of non-core electrical unit completed for Rs 14,000 crore", "merger_acquisition"),
    ("State Bank of India consortium bids for stressed toll road assets under IBC", "merger_acquisition"),
    ("ITC signs pact to acquire 100% stake in organic health foods company", "merger_acquisition"),
    ("PVR and Inox Leisure amalgamation receives final shareholder approval", "merger_acquisition"),
    ("Vedanta plans divestment of steel business to deleverage parent balance sheet", "merger_acquisition"),
    ("Zomato board approves acquisition of Paytm movie ticketing business for Rs 2,048 crore", "merger_acquisition"),

    # ── Analyst Ratings & Target Prices (14 headlines) ──
    ("Jefferies upgrades ICICI Bank to Buy, raises price target to Rs 1,400", "analyst_rating"),
    ("Morgan Stanley cuts Tata Motors target price to Rs 950 on slowdown concerns", "analyst_rating"),
    ("CLSA reiterates Outperform on Reliance Industries with target of Rs 3,300", "analyst_rating"),
    ("Goldman Sachs initiates coverage on Indian banks with Buy rating on HDFC Bank", "analyst_rating"),
    ("Nomura downgrades Wipro to Neutral, cuts target price to Rs 480", "analyst_rating"),
    ("Macquarie maintains Underperform on One97 Communications with Rs 325 target", "analyst_rating"),
    ("Motilal Oswal rates Larsen & Toubro a Buy, expects 22% upside", "analyst_rating"),
    ("Bernstein raises target price on Bharti Airtel to Rs 1,750, maintains Overweight", "analyst_rating"),
    ("HSBC upgrades State Bank of India to Buy following strong asset quality", "analyst_rating"),
    ("Brokerage calls on IT sector: Citi retains Sell rating on midcap tier", "analyst_rating"),
    ("JPMorgan maintains Overweight rating on Maruti Suzuki, fair value pegged at Rs 14,500", "analyst_rating"),
    ("Nuvama upgrades Sun Pharma to Buy, cites specialty pipeline momentum", "analyst_rating"),
    ("Kotak Institutional Equities cuts target price on Tata Steel to Rs 145", "analyst_rating"),
    ("CLSA retains Buy call on ITC, cites attractive valuation and dividend yield", "analyst_rating"),

    # ── Macro, Monetary & Commodities (15 headlines) ──
    ("RBI MPC keeps repo rate unchanged at 6.5%, maintains withdrawal of accommodation", "macro"),
    ("India retail inflation CPI eases to five-month low of 4.85% in March", "macro"),
    ("India GDP growth surges to 8.2% in FY24 beating all economist estimates", "macro"),
    ("Brent crude oil surges above $90 a barrel as Middle East tensions escalate", "macro"),
    ("Indian rupee falls to all-time low of 83.55 against US dollar on foreign outflows", "macro"),
    ("US Fed signals interest rate cut may be delayed amid persistent inflation data", "macro"),
    ("WPI wholesale inflation rises to 1.26% driven by manufactured goods", "macro"),
    ("Union Budget targets fiscal deficit reduction to 4.9% of GDP", "macro"),
    ("India forex reserves hit historic high of $651 billion, RBI data shows", "macro"),
    ("Central bank policy announcement: Governor warns against financial exuberance", "macro"),
    ("Crude oil prices tumble 3% as US crude inventories build up", "macro"),
    ("US Federal Reserve delivers 50 bps rate hike to tackle stubborn price pressures", "macro"),
    ("Retail inflation cools as food prices moderate across urban centres", "macro"),
    ("Monetary policy committee members highlight inflation risks from volatile weather", "macro"),
    ("Rupee gains 15 paise against dollar supported by softening crude prices", "macro"),

    # ── Other (Corporate General & CSR) (15 headlines) ──
    ("Tata Consultancy Services inaugurates new innovation hub in London", "other"),
    ("Reliance Foundation launches nationwide digital literacy initiative for rural schools", "other"),
    ("Infosys employees organize annual sustainability marathon in Bengaluru", "other"),
    ("Mahindra & Mahindra showcases next-generation electric vehicle design concepts", "other"),
    ("HDFC Bank hosts annual sports tournament for corporate employees", "other"),
    ("State Bank of India celebrates 69th foundation day with customer outreach program", "other"),
    ("Wipro collaborates with premier engineering institute on quantum computing research", "other"),
    ("Bharti Airtel rolls out solar power installations across 5,000 telecom towers", "other"),
    ("Larsen & Toubro organises technical symposium on green hydrogen engineering", "other"),
    ("Tata Motors conducts road safety awareness campaign across major highways", "other"),
    ("Sun Pharma inaugurates modern warehouse logistics center in Maharashtra", "other"),
    ("Bajaj Auto showcases premium touring motorcycle at international auto expo", "other"),
    ("ICICI Bank launches special savings account variant for young professionals", "other"),
    ("Titan Company opens 500th retail store milestone in southern region", "other"),
    ("Cipla foundation partners with non-profit to provide clean drinking water", "other"),
]


def test_event_classifier_100_headlines():
    """
    Tests the rule-based event classifier on 115 realistic financial headlines across all 8 classes.
    Verifies that classification accuracy exceeds 95%.
    """
    assert len(SAMPLE_100_HEADLINES) >= 100, "Must test at least 100 sample headlines"

    correct = 0
    mismatches = []

    for title, expected_type in SAMPLE_100_HEADLINES:
        predicted_type = classify_headline(title)
        if predicted_type == expected_type:
            correct += 1
        else:
            mismatches.append({
                "title": title,
                "expected": expected_type,
                "predicted": predicted_type,
            })

    accuracy = correct / len(SAMPLE_100_HEADLINES)
    msg = f"Accuracy {accuracy:.2%} ({correct}/{len(SAMPLE_100_HEADLINES)}). Mismatches: {mismatches}"
    assert accuracy >= 0.95, msg


# ══════════════════════════════════════════════════════════════════════
# 4. Strict Point-in-Time Immutability Test
# ══════════════════════════════════════════════════════════════════════

def test_point_in_time_feature_immutability():
    """
    Strict Leakage Test:
    Features computed for date T must be 100% identical before and after
    adding new future articles with available_at / trading_date_assigned > T.
    """
    trading_dates = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"]

    # Initial historical events up to 2026-09-03
    events_t1 = [
        build_event_record(
            title="Reliance Q1 profit jumps 20%",
            ticker="RELIANCE.NS",
            source="Moneycontrol",
            published_at="2026-09-01T10:00:00Z",
            sentiment_data={"finbert_pos": 0.8, "finbert_neu": 0.1, "finbert_neg": 0.1, "sentiment_score": 0.7, "confidence": 0.8},
            event_type="earnings",
        ),
        build_event_record(
            title="Reliance signs retail partnership",
            ticker="RELIANCE.NS",
            source="Livemint",
            published_at="2026-09-02T12:00:00Z",
            sentiment_data={"finbert_pos": 0.6, "finbert_neu": 0.3, "finbert_neg": 0.1, "sentiment_score": 0.5, "confidence": 0.6},
            event_type="merger_acquisition",
        ),
    ]
    df_events_t1 = pd.DataFrame(events_t1)

    # Compute features for date T = 2026-09-02
    feat_before = compute_ticker_daily_features(
        events_df=df_events_t1,
        ticker="RELIANCE.NS",
        trading_dates=trading_dates[:2],  # Up to 2026-09-02
        format_mode="tree",
    )

    # Now add FUTURE articles on 2026-09-04 and 2026-09-07 (including negative earnings and regulatory probes)
    events_future = [
        build_event_record(
            title="Reliance refinery unit hit by unexpected shutdown",
            ticker="RELIANCE.NS",
            source="Reuters",
            published_at="2026-09-04T10:00:00Z",
            sentiment_data={"finbert_pos": 0.05, "finbert_neu": 0.15, "finbert_neg": 0.80, "sentiment_score": -0.75, "confidence": 0.8},
            event_type="regulatory",
        ),
        build_event_record(
            title="SEBI issues notice to Reliance over disclosure",
            ticker="RELIANCE.NS",
            source="ET",
            published_at="2026-09-07T09:00:00Z",
            sentiment_data={"finbert_pos": 0.0, "finbert_neu": 0.1, "finbert_neg": 0.9, "sentiment_score": -0.9, "confidence": 0.9},
            event_type="regulatory",
        ),
    ]
    df_events_all = pd.DataFrame(events_t1 + events_future)

    # Re-compute features for date T = 2026-09-02 using full dataset
    feat_after = compute_ticker_daily_features(
        events_df=df_events_all,
        ticker="RELIANCE.NS",
        trading_dates=trading_dates[:2],  # Still looking at T = 2026-09-02
        format_mode="tree",
    )

    # Assert exact bitwise / equality
    pd.testing.assert_frame_equal(feat_before, feat_after, check_dtype=True)


# ══════════════════════════════════════════════════════════════════════
# 5. Missing Days & Tree vs Neural Representation Tests
# ══════════════════════════════════════════════════════════════════════

def test_missing_days_yield_news_available_zero_and_nan_sentiment():
    """
    On a trading day with zero news:
    - news_available must be 0
    - news_count_1d must be 0
    - sent_mean_1d must be NaN (NOT 0.0)
    """
    trading_dates = ["2026-09-01", "2026-09-02"]
    # 08:00 UTC is 13:30 IST (before 15:30 close, so assigned to 2026-09-01)
    events = [
        build_event_record(
            title="TCS bags cloud deal",
            ticker="TCS.NS",
            source="Mint",
            published_at="2026-09-01T08:00:00Z",
            sentiment_data={"finbert_pos": 0.9, "finbert_neu": 0.1, "finbert_neg": 0.0, "sentiment_score": 0.9, "confidence": 0.9},
            event_type="merger_acquisition",
        )
    ]
    df_events = pd.DataFrame(events)

    df_feats = compute_ticker_daily_features(
        events_df=df_events,
        ticker="TCS.NS",
        trading_dates=trading_dates,
        format_mode="tree",
    )

    day1 = df_feats[df_feats["date"] == "2026-09-01"].iloc[0]
    day2 = df_feats[df_feats["date"] == "2026-09-02"].iloc[0]

    # Day 1 has news
    assert day1["news_available"] == 1
    assert day1["news_count_1d"] == 1
    assert day1["sent_mean_1d"] == 0.9

    # Day 2 has no news
    assert day2["news_available"] == 0
    assert day2["news_count_1d"] == 0
    assert np.isnan(day2["sent_mean_1d"]), "Missing news must have NaN sentiment, not 0.0"

    # Neural format: missing sentiment filled with 0.0 with news_available=0 mask
    df_neural = compute_ticker_daily_features(
        events_df=df_events,
        ticker="TCS.NS",
        trading_dates=trading_dates,
        format_mode="neural",
    )
    day2_neural = df_neural[df_neural["date"] == "2026-09-02"].iloc[0]
    assert day2_neural["news_available"] == 0
    assert day2_neural["sent_mean_1d"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# 6. Event Store Persistence & Deduplication
# ══════════════════════════════════════════════════════════════════════

def test_event_store_save_load(tmp_path):
    """
    Verifies Parquet persistence, schema conformance, and idempotency on duplicate event_ids.
    """
    store_file = tmp_path / "test_events.parquet"
    store = NewsEventStore(store_path=store_file)

    # 08:00 UTC is 13:30 IST (before 15:30 close, so assigned to 2026-09-01)
    rec1 = build_event_record(
        title="Infosys announces share buyback program of Rs 9,300 crore",
        ticker="INFY.NS",
        source="BSE Filing",
        published_at="2026-09-01T08:00:00Z",
        sentiment_data={"finbert_pos": 0.85, "finbert_neu": 0.10, "finbert_neg": 0.05, "sentiment_score": 0.80, "confidence": 0.85},
        event_type="earnings",
    )

    # Save record
    store.save_events([rec1])
    assert store.count() == 1

    # Save exact duplicate again -> should update/deduplicate to 1
    store.save_events([rec1])
    assert store.count() == 1

    # Load and check schema
    df_loaded = store.load_events(ticker="INFY.NS")
    assert len(df_loaded) == 1
    for col in EVENT_SCHEMA_COLUMNS:
        assert col in df_loaded.columns
    assert df_loaded.iloc[0]["event_type"] == "earnings"
    assert df_loaded.iloc[0]["trading_date_assigned"] == "2026-09-01"


# ══════════════════════════════════════════════════════════════════════
# 7. Graceful Degradation on Source Failures
# ══════════════════════════════════════════════════════════════════════

def test_sources_graceful_failure(monkeypatch):
    """
    External API network errors or rate-limits must be caught gracefully and return empty lists.
    """
    rss = GoogleNewsRSSSource(timeout_sec=1)
    gdelt = GDELTSource(timeout_sec=1)

    import requests

    def mock_get_fail(*args, **kwargs):
        raise requests.exceptions.ConnectTimeout("Connection timed out")

    monkeypatch.setattr(requests, "get", mock_get_fail)

    # Must return empty list, not throw uncaught exception
    res_rss = rss.fetch(ticker="RELIANCE.NS")
    assert res_rss == []

    res_gdelt = gdelt.fetch(ticker="RELIANCE.NS")
    assert res_gdelt == []


# ══════════════════════════════════════════════════════════════════════
# 8. Exponential Decay Half-Life & Ablation Pipeline Execution Tests
# ══════════════════════════════════════════════════════════════════════

def test_exponential_decay_half_life():
    """
    Verifies that sent_decay discounts older sentiment by 50% every 3 trading days.
    """
    trading_dates = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    # Single article on day 0 (2026-09-01) with sentiment 1.0
    events = [
        build_event_record(
            title="TCS announces massive expansion",
            ticker="TCS.NS",
            source="BSE",
            published_at="2026-09-01T08:00:00Z",
            sentiment_data={"finbert_pos": 1.0, "finbert_neu": 0.0, "finbert_neg": 0.0, "sentiment_score": 1.0, "confidence": 1.0},
            event_type="earnings",
        )
    ]
    df_events = pd.DataFrame(events)

    df_feats = compute_ticker_daily_features(
        events_df=df_events,
        ticker="TCS.NS",
        trading_dates=trading_dates,
        half_life_trading_days=3.0,
        format_mode="tree",
    )

    # Day 0 (2026-09-01): decay sentiment = 1.0
    d0 = df_feats[df_feats["date"] == "2026-09-01"].iloc[0]
    assert np.isclose(d0["sent_decay"], 1.0)

    # Day 3 (2026-09-04): 3 trading days later, single event decayed, normalized weighted avg remains 1.0 if only 1 event,
    # but age is 3 trading days
    d3 = df_feats[df_feats["date"] == "2026-09-04"].iloc[0]
    assert d3["news_age_days"] == 3.0
    assert d3["news_available"] == 0
    assert np.isnan(d3["sent_mean_1d"])

    # Now add a second contrasting event on day 3 (2026-09-04) with sentiment -1.0
    events.append(
        build_event_record(
            title="TCS faces unexpected slowdown",
            ticker="TCS.NS",
            source="BSE",
            published_at="2026-09-04T08:00:00Z",
            sentiment_data={"finbert_pos": 0.0, "finbert_neu": 0.0, "finbert_neg": 1.0, "sentiment_score": -1.0, "confidence": 1.0},
            event_type="guidance",
        )
    )
    df_events_2 = pd.DataFrame(events)
    df_feats_2 = compute_ticker_daily_features(
        events_df=df_events_2,
        ticker="TCS.NS",
        trading_dates=trading_dates,
        half_life_trading_days=3.0,
        format_mode="tree",
    )

    # On day 3: weight for day 0 is 0.5 (one half-life = 3 days). Weight for day 3 is 1.0.
    # Weighted sentiment sum = 0.5 * 1.0 + 1.0 * (-1.0) = -0.5
    # Total weight = 0.5 + 1.0 = 1.5
    # Expected decayed sentiment = -0.5 / 1.5 = -0.3333
    d3_2 = df_feats_2[df_feats_2["date"] == "2026-09-04"].iloc[0]
    assert np.isclose(d3_2["sent_decay"], -1.0 / 3.0, atol=1e-3)


def test_ablation_pipeline_execution():
    """
    Verifies that the ablation evaluation pipeline executes and returns well-formed metrics.
    """
    from src.v07.news.ablation import run_ablation_experiment

    # Run quick 2-fold ablation on Target_1D
    res = run_ablation_experiment(target_col="Target_1D", n_splits=2)
    assert "overall_base_rank_ic" in res
    assert "overall_news_rank_ic" in res
    assert "overall_delta_rank_ic" in res
    assert "recommendation" in res
    assert len(res["fold_breakdown"]) == 2

