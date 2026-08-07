#!/usr/bin/env python3
"""
================================================================================
mutual_fund_report.py
Top Mutual Funds Performance Report — India (single-file edition)
================================================================================

A production-grade, single-file Python script that fetches, analyses and
ranks Indian mutual funds by 1Y/3Y/5Y performance, risk-adjusted return,
expense ratio and AUM, and generates a detailed, interactive HTML report
(plus optional CSV/JSON/Markdown export and a live console dashboard).

WHAT'S REAL VS. WHAT NEEDS A DATA PROVIDER
--------------------------------------------------------------------------
  - NAV history, latest NAV, and every metric derived from it (1Y return,
    3Y/5Y CAGR, volatility, Sharpe Ratio, SIP XIRR) are LIVE, REAL data,
    fetched from mfapi.in (a free, no-auth mirror of AMFI's daily NAV feed).
  - AUM, Expense Ratio, Fund Manager, Sector Allocation and Stock Holdings
    are NOT available from any free public API in India. Value Research,
    Morningstar and Moneycontrol don't publish open APIs, and scraping them
    violates their Terms of Service, so this script does not do that.
    Instead, plug a licensed data source into the `SupplementaryDataProvider`
    interface below (a working `CSVProvider` example is included). Without
    one, those fields report as "N/A" and the composite score automatically
    re-normalises its weights across whatever metrics ARE available.

QUICK START
--------------------------------------------------------------------------
    pip install -r requirements.txt        # or: pip install pandas numpy requests rich

    python mutual_fund_report.py --demo --export html
        -> generates outputs/report.html using synthetic offline demo data
           (no internet required) -- open it and click any fund row.

    python mutual_fund_report.py --export html
        -> generates outputs/report.html using LIVE NAV data from mfapi.in

    python mutual_fund_report.py --demo --provider-csv sample_supplementary_data.csv --export html
        -> demo run with sector allocation / stock holdings populated, so
           the click-to-expand drill-down has real content to show.

    python mutual_fund_report.py --help    for the full list of options

See the bottom of this file for the CLI / orchestration logic, and the
README section at the very end of this docstring for full documentation.
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

logger = logging.getLogger("mf_report")


# ==============================================================================
# CONFIGURATION
# Edit these values directly to change data sources, ranking weights, or
# output behaviour. (In the multi-file edition of this project this lived in
# config/config.yaml -- it's embedded here to keep this a single file.)
# ==============================================================================
DEFAULT_CONFIG = {
    "api": {
        # mfapi.in -- free, no-auth, community-maintained mirror of AMFI NAV data.
        # Provides: scheme list + full historical NAV series only.
        # Does NOT provide category, AUM, expense ratio, holdings, or manager data.
        "mfapi_base_url": "https://api.mfapi.in",
        "scheme_list_endpoint": "/mf",
        "scheme_nav_endpoint": "/mf/{scheme_code}",
        "request_timeout_seconds": 30,  # mfapi.in can be slow under load -- give it real time before giving up
        "max_retries": 4,
        "backoff_factor": 2.0,          # seconds, exponential: 2, 4, 6, 8 ...
        "max_parallel_requests": 5,     # thread pool size for concurrent NAV fetches -- kept modest so we
                                         # don't open more concurrent connections than a free, no-SLA API
                                         # (and some home/office networks) can comfortably sustain
    },
    # Risk-free rate used for Sharpe Ratio calculation. Default approximates
    # the 10-year Indian G-Sec yield -- override with --risk-free-rate.
    "risk_free_rate_annual": 0.069,
    "min_history_days": {
        "one_year": 300,
        "three_year": 1000,
        "five_year": 1700,
    },
    "sip": {
        "monthly_investment": 10000,    # INR, notional, used only to compute % XIRR
        "lookback_months": 12,
    },
    # Category inference keywords (fallback used ONLY when a data provider
    # does not supply an explicit category). Case-insensitive match against
    # the scheme name returned by mfapi.in.
    "category_keywords": {
        "Large Cap": ["large cap", "bluechip", "large-cap"],
        "Mid Cap": ["mid cap", "midcap", "mid-cap"],
        "Small Cap": ["small cap", "smallcap", "small-cap"],
        "Flexi Cap": ["flexi cap", "flexicap", "multi cap", "multicap"],
        "ELSS": ["elss", "tax saver", "tax saving"],
        "Hybrid": ["hybrid", "balanced advantage", "balanced fund", "equity savings"],
        "International FoF": ["overseas", "global", "international", "fof", "us equity",
                               "emerging market", "china", "taiwan", "nasdaq", "world"],
        "Debt": ["debt", "gilt", "bond", "liquid", "money market", "credit risk",
                 "corporate bond", "banking and psu"],
        "Sectoral/Thematic": ["banking", "psu", "infrastructure", "pharma", "technology",
                               "energy", "consumption", "commodities", "manufacturing"],
        "Index/ETF": ["index fund", "etf", "nifty", "sensex"],
    },
    # Ranking weights -- must sum to 1.0 (validated by validate_weights()).
    #   1Y Return 25% | 3Y CAGR 25% | 5Y CAGR 20% | Risk-Adjusted 15%
    #   Expense Ratio 10% | AUM Stability 5%
    "ranking_weights": {
        "one_year_return": 0.25,
        "three_year_cagr": 0.25,
        "five_year_cagr": 0.20,
        "risk_adjusted_score": 0.15,
        "expense_ratio": 0.10,          # lower is better -> inverted before scoring
        "aum_stability": 0.05,
    },
    "output": {
        "outputs_dir": "outputs",
        "top_n_default": 10,
    },
}


# ==============================================================================
# SECTION 1 of 5 — SUPPLEMENTARY DATA PROVIDERS
# (AUM, Expense Ratio, Fund Manager, Sector Allocation, Stock Holdings)
# ==============================================================================


@dataclass
class SupplementaryFundData:
    """Container for the fields a premium provider can supply."""
    category: Optional[str] = None
    fund_house: Optional[str] = None
    expense_ratio_pct: Optional[float] = None
    aum_cr: Optional[float] = None
    fund_manager: Optional[str] = None
    manager_tenure_years: Optional[float] = None
    equity_allocation_pct: Optional[float] = None
    debt_allocation_pct: Optional[float] = None
    cash_allocation_pct: Optional[float] = None
    # Stock-level holdings: list of {"name": str, "percent": float} -- percent
    # is that stock's weight as a % of the fund's total portfolio.
    top_holdings: list = field(default_factory=list)
    # Sector-level allocation: list of {"sector": str, "percent": float}
    sector_allocation: list = field(default_factory=list)
    benchmark_name: Optional[str] = None


class SupplementaryDataProvider(ABC):
    """Abstract base class every data provider plugin must implement."""

    @abstractmethod
    def get_fund_details(self, scheme_code: str, scheme_name: str) -> SupplementaryFundData:
        """Return whatever supplementary fields are available for a scheme.

        Implementations MUST NOT raise on a missing field -- return None for
        that field instead so the pipeline can degrade gracefully.
        """
        raise NotImplementedError


class NullProvider(SupplementaryDataProvider):
    """
    Default no-op provider used when no premium data source is configured.
    Returns an empty SupplementaryFundData for every fund. The ranking engine
    treats these as missing and re-weights the scoring model accordingly.
    """

    def __init__(self):
        self._warned = False

    def get_fund_details(self, scheme_code: str, scheme_name: str) -> SupplementaryFundData:
        if not self._warned:
            logger.warning(
                "No SupplementaryDataProvider configured -- AUM, Expense Ratio, "
                "Fund Manager, Holdings and Portfolio Allocation will be reported "
                "as 'N/A'. Implement a provider class (see SupplementaryDataProvider) in this script, against a "
                "licensed data source to populate these fields. (This warning "
                "is shown once per run.)"
            )
            self._warned = True
        return SupplementaryFundData()


class CSVProvider(SupplementaryDataProvider):
    """
    Example concrete provider: reads supplementary data from a CSV you
    maintain (e.g. exported weekly from your research terminal / internal
    data warehouse). Expected columns:

        scheme_code, category, fund_house, expense_ratio_pct, aum_cr,
        fund_manager, manager_tenure_years, equity_allocation_pct,
        debt_allocation_pct, cash_allocation_pct, benchmark_name,
        top_holdings, sector_allocation

    `top_holdings` and `sector_allocation` use "name:percent" pairs
    separated by ';', e.g.:
        top_holdings      = "HDFC Bank:8.2;Infosys:6.5;ICICI Bank:5.9"
        sector_allocation = "Financial Services:32.1;Technology:18.4;Energy:9.7"

    Percent values are that stock's / sector's weight as a % of the fund's
    total portfolio -- multiply by aum_cr to get an approximate ₹ Cr amount
    (this is what the HTML report's per-fund drill-down does automatically).

    See sample_supplementary_data.csv in the project root for a filled-in example.
    """

    def __init__(self, csv_path: str):
        import pandas as pd  # local import to keep base module dependency-light
        self._df = None
        try:
            df = pd.read_csv(csv_path, dtype={"scheme_code": str})
            self._df = df.set_index("scheme_code")
            logger.info(f"CSVProvider loaded {len(df)} supplementary records from {csv_path}")
        except FileNotFoundError:
            logger.error(f"CSVProvider: file not found at {csv_path}. Falling back to empty data.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"CSVProvider: failed to load {csv_path}: {exc}")

    @staticmethod
    def _parse_weighted_pairs(raw: str) -> list:
        """Parse 'Name:12.3;Other:4.5' -> [{"name": "Name", "percent": 12.3}, ...]"""
        if not raw or str(raw).lower() == "nan":
            return []
        pairs = []
        for chunk in str(raw).split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            name, _, pct = chunk.rpartition(":")
            pct_val = _safe_float(pct)
            if name.strip() and pct_val is not None:
                pairs.append({"name": name.strip(), "percent": pct_val})
        return sorted(pairs, key=lambda x: x["percent"], reverse=True)

    def get_fund_details(self, scheme_code: str, scheme_name: str) -> SupplementaryFundData:
        if self._df is None or scheme_code not in self._df.index:
            return SupplementaryFundData()
        row = self._df.loc[scheme_code]
        holdings = self._parse_weighted_pairs(row.get("top_holdings", ""))
        sectors_raw = self._parse_weighted_pairs(row.get("sector_allocation", ""))
        # sector_allocation uses "sector" as the key name rather than "name"
        sectors = [{"sector": s["name"], "percent": s["percent"]} for s in sectors_raw]
        return SupplementaryFundData(
            category=row.get("category") or None,
            fund_house=row.get("fund_house") or None,
            expense_ratio_pct=_safe_float(row.get("expense_ratio_pct")),
            aum_cr=_safe_float(row.get("aum_cr")),
            fund_manager=row.get("fund_manager") or None,
            manager_tenure_years=_safe_float(row.get("manager_tenure_years")),
            equity_allocation_pct=_safe_float(row.get("equity_allocation_pct")),
            debt_allocation_pct=_safe_float(row.get("debt_allocation_pct")),
            cash_allocation_pct=_safe_float(row.get("cash_allocation_pct")),
            top_holdings=holdings,
            sector_allocation=sectors,
            benchmark_name=row.get("benchmark_name") or None,
        )


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or str(val).strip() == "" or str(val).lower() == "nan":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None






# ==============================================================================
# SECTION 2 of 5 — DATA ACQUISITION (live mfapi.in fetch + synthetic demo data)
# ==============================================================================

class MFAPIFetcher:
    """Thin, resilient client around the mfapi.in REST API."""

    def __init__(self, config: dict):
        api_cfg = config["api"]
        self.base_url = api_cfg["mfapi_base_url"].rstrip("/")
        self.scheme_list_endpoint = api_cfg["scheme_list_endpoint"]
        self.scheme_nav_endpoint = api_cfg["scheme_nav_endpoint"]
        self.timeout = api_cfg.get("request_timeout_seconds", 10)
        self.max_retries = api_cfg.get("max_retries", 3)
        self.backoff_factor = api_cfg.get("backoff_factor", 1.5)
        self.max_workers = api_cfg.get("max_parallel_requests", 12)
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    # Internal: resilient GET with retry + exponential backoff
    # ------------------------------------------------------------------ #
    def _get(self, url: str) -> Optional[dict]:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                wait = self.backoff_factor * attempt
                logger.warning(
                    f"GET {url} failed (attempt {attempt}/{self.max_retries}): {exc}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
        logger.error(f"GET {url} failed after {self.max_retries} attempts: {last_exc}")
        return None

    # ------------------------------------------------------------------ #
    def get_scheme_list(self) -> List[dict]:
        """Fetch the full list of {schemeCode, schemeName} from mfapi.in."""
        url = f"{self.base_url}{self.scheme_list_endpoint}"
        logger.info(f"Fetching master scheme list from {url} ...")
        data = self._get(url)
        if not data:
            raise RuntimeError(
                "Could not retrieve the scheme master list. Check network "
                "connectivity / that api.mfapi.in is reachable from this host."
            )
        logger.info(f"Retrieved {len(data)} schemes from mfapi.in")
        return data

    def filter_schemes_by_keywords(
        self, schemes: List[dict], category_keywords: Dict[str, List[str]], limit: Optional[int] = None
    ) -> List[dict]:
        """
        mfapi.in has no category field, so we pre-filter the (very large,
        ~30k scheme) universe down to schemes whose *name* matches one of the
        configured category keyword sets. This keeps downstream NAV fetching
        (one HTTP call per scheme) to a sane volume.
        """
        all_keywords = [kw.lower() for kws in category_keywords.values() for kw in kws]
        matched = [
            s for s in schemes
            if any(kw in s.get("schemeName", "").lower() for kw in all_keywords)
        ]
        logger.info(f"{len(matched)} schemes matched configured category keywords")
        if limit:
            matched = matched[:limit]
            logger.info(f"Capped scheme universe to {limit} schemes for this run")
        return matched

    def get_nav_history(self, scheme_code: str) -> Optional[pd.DataFrame]:
        """Fetch full NAV history for one scheme -> DataFrame[date, nav]."""
        url = f"{self.base_url}{self.scheme_nav_endpoint.format(scheme_code=scheme_code)}"
        data = self._get(url)
        if not data or "data" not in data or not data["data"]:
            logger.warning(f"No NAV data returned for scheme {scheme_code}")
            return None
        df = pd.DataFrame(data["data"])  # columns: date, nav
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        meta = data.get("meta", {})
        df.attrs["fund_house"] = meta.get("fund_house")
        df.attrs["scheme_category"] = meta.get("scheme_category")
        df.attrs["scheme_name"] = meta.get("scheme_name")
        return df

    def fetch_all_nav_histories(self, schemes: List[dict]) -> Dict[str, pd.DataFrame]:
        """Concurrently fetch NAV histories for a list of schemes."""
        results: Dict[str, pd.DataFrame] = {}
        logger.info(f"Fetching NAV history for {len(schemes)} schemes "
                     f"({self.max_workers} parallel workers)...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.get_nav_history, s["schemeCode"]): s for s in schemes
            }
            done = 0
            for fut in as_completed(futures):
                scheme = futures[fut]
                code = str(scheme["schemeCode"])
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        results[code] = df
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error fetching NAV history for {code}: {exc}")
                done += 1
                if done % 25 == 0:
                    logger.info(f"  ...{done}/{len(schemes)} schemes fetched")
        logger.info(f"Successfully fetched NAV history for {len(results)}/{len(schemes)} schemes")
        return results


# ============================================================================
# DEMO / OFFLINE MODE
# ============================================================================
_DEMO_CATEGORIES = [
    "Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Hybrid", "International FoF",
]

_DEMO_FUND_HOUSES = [
    "Nippon India", "Kotak Mahindra", "DSP", "Bank of India", "HDFC", "SBI",
    "ICICI Prudential", "Axis", "Parag Parikh", "Mirae Asset", "UTI", "Tata",
]


def generate_synthetic_demo_data(n_funds: int = 30, seed: int = 42) -> Dict[str, dict]:
    """
    Generates a small, CLEARLY LABELLED synthetic universe of funds with
    5 years of daily NAV history via geometric Brownian motion, so the
    cleaning / ranking / reporting stages can be demonstrated end-to-end
    without network access.

    THIS IS NOT REAL MARKET DATA. Every record is tagged is_synthetic=True
    and the report generator prints a prominent banner whenever this data
    is used.
    """
    rng = random.Random(seed)
    records = {}
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365 + 30)
    dates = pd.date_range(start_date, end_date, freq="B")  # business days

    for i in range(n_funds):
        category = _DEMO_CATEGORIES[i % len(_DEMO_CATEGORIES)]
        fund_house = _DEMO_FUND_HOUSES[i % len(_DEMO_FUND_HOUSES)]
        scheme_code = f"DEMO{1000 + i}"
        scheme_name = f"{fund_house} {category} Fund - Growth (Demo)"

        # Category-flavoured drift/volatility so rankings look plausible.
        drift_by_cat = {
            "Large Cap": 0.11, "Mid Cap": 0.15, "Small Cap": 0.18,
            "Flexi Cap": 0.13, "Hybrid": 0.09, "International FoF": 0.14,
        }
        vol_by_cat = {
            "Large Cap": 0.14, "Mid Cap": 0.19, "Small Cap": 0.24,
            "Flexi Cap": 0.16, "Hybrid": 0.10, "International FoF": 0.20,
        }
        mu = drift_by_cat[category] / 252
        sigma = vol_by_cat[category] / (252 ** 0.5)

        nav = [rng.uniform(20, 100)]
        for _ in range(len(dates) - 1):
            shock = rng.gauss(mu, sigma)
            nav.append(max(0.5, nav[-1] * (1 + shock)))

        df = pd.DataFrame({"date": dates, "nav": nav})
        df.attrs["fund_house"] = fund_house
        df.attrs["scheme_category"] = category
        df.attrs["scheme_name"] = scheme_name

        records[scheme_code] = {
            "scheme_code": scheme_code,
            "scheme_name": scheme_name,
            "nav_df": df,
            "is_synthetic": True,
        }

    logger.warning(
        f"DEMO MODE: generated {n_funds} SYNTHETIC fund records with simulated "
        f"NAV histories. This is NOT real market data -- do not use for "
        f"investment decisions."
    )
    return records






# ==============================================================================
# SECTION 3 of 5 — DATA CLEANING & VALIDATION
# ==============================================================================

def clean_nav_series(df: pd.DataFrame, scheme_code: str) -> Optional[pd.DataFrame]:
    """
    Clean a raw NAV DataFrame:
      - drop rows with null date/nav
      - de-duplicate on date (keep last)
      - sort ascending by date
      - drop non-positive NAV values (data errors)
      - forward-fill isolated single-day gaps (holidays reported inconsistently)

    Returns None if the resulting series has too few points to be useful.
    """
    if df is None or df.empty:
        return None

    original_len = len(df)
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0]
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    dropped = original_len - len(df)
    if dropped > 0:
        logger.debug(f"[{scheme_code}] cleaned NAV series: dropped {dropped} invalid/duplicate rows")

    if len(df) < 30:
        logger.warning(f"[{scheme_code}] insufficient NAV history after cleaning "
                        f"({len(df)} points) -- excluding from ranking")
        return None

    # Carry attrs (fund_house, scheme_category, scheme_name) through the copy
    df.attrs = getattr(df, "attrs", {})
    return df


def validate_fund_record(record: dict) -> bool:
    """
    Sanity-check computed metrics before they're used for ranking.
    Flags (but does not silently discard) implausible values so the
    ranking engine can decide whether to exclude the metric.

    Returns True if the record is broadly usable (has at least a valid
    1-year return), False if it should be dropped entirely.
    """
    code = record.get("scheme_code", "UNKNOWN")
    one_yr = record.get("one_year_return")

    if one_yr is None:
        logger.warning(f"[{code}] missing 1-year return -- excluding fund from this run")
        return False

    # Plausibility bounds -- flags rather than hard-fails, since genuine
    # extreme years do happen (e.g. narrow international/thematic funds).
    checks = {
        "one_year_return": (-90, 400),
        "three_year_cagr": (-60, 150),
        "five_year_cagr": (-40, 100),
        "expense_ratio_pct": (0, 3.0),
        "annualized_volatility_pct": (0, 120),
    }
    for field, (lo, hi) in checks.items():
        val = record.get(field)
        if val is not None and not (lo <= val <= hi):
            logger.warning(
                f"[{code}] {field}={val} is outside plausible range [{lo}, {hi}] "
                f"-- likely a NAV data glitch (e.g. a stale/bad price point or a "
                f"face-value change mfapi.in didn't adjust for). Excluding this "
                f"metric from the composite score for this fund (weights are "
                f"re-normalised across the remaining metrics); value is retained "
                f"in the export for manual review."
            )
            record[f"{field}_flagged"] = True
            record[f"{field}_raw"] = val   # preserved for manual review/export
            record[field] = None           # excluded from composite scoring
            if field == "annualized_volatility_pct":
                # Sharpe Ratio is derived directly from volatility, so a bad
                # volatility reading makes the Sharpe Ratio unreliable too.
                if record.get("sharpe_ratio") is not None:
                    logger.warning(
                        f"[{code}] also excluding sharpe_ratio from the composite "
                        f"score, since it was computed from the flagged volatility."
                    )
                record["sharpe_ratio"] = None

    return True


def build_master_dataframe(fund_records: Dict[str, dict]) -> pd.DataFrame:
    """Assemble the per-fund metric dictionaries into one clean pandas DataFrame."""
    if not fund_records:
        logger.error("No valid fund records to assemble -- returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(fund_records, orient="index")
    df = df.reset_index(drop=True)

    numeric_cols = [
        "one_year_return", "three_year_cagr", "five_year_cagr",
        "sip_return_1yr", "annualized_volatility_pct", "sharpe_ratio",
        "expense_ratio_pct", "aum_cr", "latest_nav",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Master DataFrame assembled: {len(df)} funds, {df.shape[1]} fields")
    return df




# ============================================================================
# METRIC COMPUTATION (per fund, from a cleaned NAV DataFrame)
# ============================================================================



# ==============================================================================
# SECTION 4 of 5 — RANKING ENGINE (CAGR, SIP-XIRR, volatility, Sharpe, composite score)
# ==============================================================================

def _nav_on_or_before(df: pd.DataFrame, target_date) -> Optional[float]:
    """Latest available NAV on or before target_date (handles weekends/holidays)."""
    subset = df[df["date"] <= target_date]
    if subset.empty:
        return None
    return float(subset.iloc[-1]["nav"])


def compute_point_to_point_return(df: pd.DataFrame, years: float) -> Optional[float]:
    """
    Point-to-point return over `years`. Returns CAGR (%) for years >= 1,
    simple absolute return (%) for shorter periods.
    """
    if df is None or df.empty:
        return None
    latest_date = df.iloc[-1]["date"]
    latest_nav = float(df.iloc[-1]["nav"])
    target_date = latest_date - timedelta(days=int(years * 365.25))
    past_nav = _nav_on_or_before(df, target_date)

    if past_nav is None or past_nav <= 0:
        return None

    ratio = latest_nav / past_nav
    if years >= 1:
        cagr = (ratio ** (1 / years) - 1) * 100
        return round(cagr, 2)
    else:
        return round((ratio - 1) * 100, 2)


def compute_annualized_volatility(df: pd.DataFrame, window_days: int = 252) -> Optional[float]:
    """Annualised standard deviation (%) of daily returns over the last `window_days`."""
    if df is None or len(df) < 30:
        return None
    recent = df.tail(window_days).copy()
    recent["daily_return"] = recent["nav"].pct_change()
    daily_std = recent["daily_return"].std(skipna=True)
    if daily_std is None or math.isnan(daily_std):
        return None
    annualized = daily_std * math.sqrt(252) * 100
    return round(annualized, 2)


def compute_sharpe_ratio(
    df: pd.DataFrame, risk_free_rate_annual: float, window_days: int = 252
) -> Optional[float]:
    """Sharpe Ratio = (annualised return - risk-free rate) / annualised volatility."""
    ann_return = compute_point_to_point_return(df, years=1)
    ann_vol_pct = compute_annualized_volatility(df, window_days=window_days)
    if ann_return is None or ann_vol_pct is None or ann_vol_pct == 0:
        return None
    excess_return = (ann_return / 100) - risk_free_rate_annual
    sharpe = excess_return / (ann_vol_pct / 100)
    return round(sharpe, 2)


def _xirr(cashflows: list) -> Optional[float]:
    """
    Compute XIRR from a list of (date, amount) tuples via bisection on NPV.
    Robust, dependency-free alternative to scipy.optimize for this use case.
    """
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]

    def npv(rate):
        total = 0.0
        for date, amount in cashflows:
            days = (date - t0).days
            total += amount / ((1 + rate) ** (days / 365.0))
        return total

    lo, hi = -0.99, 10.0
    npv_lo, npv_hi = npv(lo), npv(hi)
    if npv_lo * npv_hi > 0:
        return None  # no sign change -> bisection won't converge reliably

    for _ in range(100):
        mid = (lo + hi) / 2
        npv_mid = npv(mid)
        if abs(npv_mid) < 1e-6:
            return round(mid * 100, 2)
        if npv_lo * npv_mid < 0:
            hi = mid
        else:
            lo = mid
            npv_lo = npv_mid
    return round(((lo + hi) / 2) * 100, 2)


def compute_sip_return(df: pd.DataFrame, monthly_amount: float = 10000, months: int = 12) -> Optional[float]:
    """
    Simulate a monthly SIP of `monthly_amount` over the last `months` months
    and return the annualised XIRR (%) of that investment.
    """
    if df is None or df.empty:
        return None
    latest_date = df.iloc[-1]["date"]
    cashflows = []
    units_accumulated = 0.0

    for m in range(months, 0, -1):
        install_date = latest_date - pd.DateOffset(months=m)
        nav = _nav_on_or_before(df, install_date)
        if nav is None or nav <= 0:
            continue
        units = monthly_amount / nav
        units_accumulated += units
        cashflows.append((install_date.to_pydatetime(), -monthly_amount))

    if not cashflows or units_accumulated == 0:
        return None

    latest_nav = float(df.iloc[-1]["nav"])
    redemption_value = units_accumulated * latest_nav
    cashflows.append((latest_date.to_pydatetime(), redemption_value))

    return _xirr(cashflows)


def infer_category(scheme_name: str, category_keywords: dict, provider_category: Optional[str]) -> str:
    """Use provider category if available, else infer from scheme name keywords."""
    if provider_category:
        return provider_category
    name_lower = scheme_name.lower()
    for category, keywords in category_keywords.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "Uncategorised"


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def _minmax_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Scale a numeric column to 0-100. NaNs are preserved (left as NaN) so the
    caller can decide how to handle missing data during weight re-normalisation.
    If invert=True, lower raw values score higher (used for expense ratio).
    """
    valid = series.dropna()
    if valid.empty or valid.max() == valid.min():
        return pd.Series([np.nan] * len(series), index=series.index)
    scaled = (series - valid.min()) / (valid.max() - valid.min()) * 100
    if invert:
        scaled = 100 - scaled
    return scaled


def compute_composite_score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Weighted composite score per the ranking spec:
      1Y Return 25% | 3Y CAGR 25% | 5Y CAGR 20% | Risk-Adjusted 15%
      Expense Ratio 10% (lower=better) | AUM Stability 5%

    Missing metrics are handled by RE-NORMALISING weights across only the
    metrics actually present for each fund (rather than silently zero-filling,
    which would unfairly punish funds simply because premium data -- e.g.
    expense ratio -- wasn't available). Funds missing >50% of weighted
    metrics are flagged as low-confidence scores.
    """
    df = df.copy()

    df["_score_1y"] = _minmax_normalize(df.get("one_year_return", pd.Series(dtype=float)))
    df["_score_3y"] = _minmax_normalize(df.get("three_year_cagr", pd.Series(dtype=float)))
    df["_score_5y"] = _minmax_normalize(df.get("five_year_cagr", pd.Series(dtype=float)))
    df["_score_risk"] = _minmax_normalize(df.get("sharpe_ratio", pd.Series(dtype=float)))
    df["_score_expense"] = _minmax_normalize(df.get("expense_ratio_pct", pd.Series(dtype=float)), invert=True)
    df["_score_aum"] = _minmax_normalize(df.get("aum_cr", pd.Series(dtype=float)))

    metric_weight_map = {
        "_score_1y": weights["one_year_return"],
        "_score_3y": weights["three_year_cagr"],
        "_score_5y": weights["five_year_cagr"],
        "_score_risk": weights["risk_adjusted_score"],
        "_score_expense": weights["expense_ratio"],
        "_score_aum": weights["aum_stability"],
    }

    composite_scores = []
    confidence_flags = []
    for _, row in df.iterrows():
        available = {k: v for k, v in metric_weight_map.items() if pd.notna(row[k])}
        if not available:
            composite_scores.append(np.nan)
            confidence_flags.append("No Data")
            continue
        total_weight = sum(available.values())
        score = sum(row[k] * (w / total_weight) for k, w in available.items())
        composite_scores.append(round(score, 2))
        coverage = len(available) / len(metric_weight_map)
        confidence_flags.append("High" if coverage >= 0.8 else "Medium" if coverage >= 0.5 else "Low")

    df["composite_score"] = composite_scores
    df["score_confidence"] = confidence_flags
    df = df.drop(columns=[c for c in df.columns if c.startswith("_score_")])
    return df


def validate_weights(weights: dict) -> None:
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=0.01):
        raise ValueError(f"Ranking weights must sum to 1.0, got {total:.3f}. Check config.yaml.")


# ============================================================================
# LEADERBOARDS
# ============================================================================

def rank_top_n(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.dropna(subset=["composite_score"]).sort_values(
        "composite_score", ascending=False
    ).head(n).reset_index(drop=True)


def category_winners(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["composite_score"])
    if valid.empty:
        return valid
    idx = valid.groupby("category")["composite_score"].idxmax()
    return valid.loc[idx].sort_values("composite_score", ascending=False).reset_index(drop=True)


def best_small_cap(df: pd.DataFrame) -> Optional[pd.Series]:
    subset = df[(df["category"] == "Small Cap")].dropna(subset=["composite_score"])
    return subset.sort_values("composite_score", ascending=False).iloc[0] if not subset.empty else None


def best_sip_performer(df: pd.DataFrame) -> Optional[pd.Series]:
    subset = df.dropna(subset=["sip_return_1yr"])
    return subset.sort_values("sip_return_1yr", ascending=False).iloc[0] if not subset.empty else None


def best_risk_adjusted(df: pd.DataFrame) -> Optional[pd.Series]:
    subset = df.dropna(subset=["sharpe_ratio"])
    return subset.sort_values("sharpe_ratio", ascending=False).iloc[0] if not subset.empty else None


def best_long_term_wealth_creator(df: pd.DataFrame) -> Optional[pd.Series]:
    subset = df.dropna(subset=["five_year_cagr"])
    return subset.sort_values("five_year_cagr", ascending=False).iloc[0] if not subset.empty else None


def best_overall(df: pd.DataFrame) -> Optional[pd.Series]:
    subset = df.dropna(subset=["composite_score"])
    return subset.sort_values("composite_score", ascending=False).iloc[0] if not subset.empty else None


def international_funds_performance(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["category"] == "International FoF"].dropna(subset=["composite_score"])
    return subset.sort_values("composite_score", ascending=False).reset_index(drop=True)


console = Console(width=150)




# ==============================================================================
# SECTION 5 of 5 — REPORT GENERATOR (console dashboard + CSV/JSON/Markdown/HTML export)
# ==============================================================================

def _fmt(val, suffix="", decimals=2, na="N/A"):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return na
    return f"{val:.{decimals}f}{suffix}"


def print_banner(is_synthetic: bool, fund_count: int, run_date: str):
    if is_synthetic:
        console.print(Panel(
            "[bold yellow]⚠ DEMO MODE — SYNTHETIC DATA[/bold yellow]\n"
            "All figures below are randomly generated for demonstration purposes "
            "only and DO NOT reflect real market performance. Run without "
            "--demo (with internet access) for live mfapi.in data.",
            style="yellow", box=box.DOUBLE
        ))
    title = Text(f"TOP MUTUAL FUNDS PERFORMANCE REPORT — INDIA", style="bold white on dark_blue")
    subtitle = Text(f"Generated {run_date}  |  Universe: {fund_count} funds analysed", style="dim")
    console.print(Panel.fit(title, box=box.HEAVY))
    console.print(subtitle, justify="center")
    console.print()


def print_executive_summary(top_n_df: pd.DataFrame, category_df: pd.DataFrame, specialty: dict):
    console.rule("[bold cyan]Executive Summary")
    if top_n_df.empty:
        console.print("[red]No funds could be ranked in this run — see logs for data issues.[/red]")
        return

    best = top_n_df.iloc[0]
    lines = [
        f"• [bold]{best['scheme_name']}[/bold] ({best['category']}) leads the overall rankings this "
        f"period with a composite score of [bold]{_fmt(best['composite_score'])}/100[/bold] "
        f"(confidence: {best.get('score_confidence', 'N/A')}), on the back of a "
        f"{_fmt(best.get('one_year_return'), '%')} 1-year return and "
        f"{_fmt(best.get('three_year_cagr'), '%')} 3-year CAGR.",
        f"• {len(category_df)} categories were represented in this run's leaderboard "
        f"({', '.join(category_df['category'].tolist())}).",
    ]
    if specialty.get("best_small_cap") is not None:
        f = specialty["best_small_cap"]
        lines.append(f"• Small Cap leadership: [bold]{f['scheme_name']}[/bold] "
                      f"({_fmt(f.get('one_year_return'), '%')} 1Y return) — small caps remain the "
                      f"highest-return, highest-volatility segment of the market.")
    if specialty.get("best_sip") is not None:
        f = specialty["best_sip"]
        lines.append(f"• Best SIP performer: [bold]{f['scheme_name']}[/bold] with a trailing "
                      f"12-month SIP XIRR of {_fmt(f.get('sip_return_1yr'), '%')}.")
    if specialty.get("best_risk_adjusted") is not None:
        f = specialty["best_risk_adjusted"]
        lines.append(f"• Best risk-adjusted performer: [bold]{f['scheme_name']}[/bold] "
                      f"(Sharpe Ratio: {_fmt(f.get('sharpe_ratio'))}).")

    for line in lines:
        console.print(line)
    console.print()


def print_top_n_table(df: pd.DataFrame, title: str = "Top 10 Mutual Funds of the Year"):
    console.rule(f"[bold cyan]{title}")
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Scheme Name", style="bold", no_wrap=True, overflow="ellipsis", max_width=42)
    table.add_column("Category", no_wrap=True)
    table.add_column("1Y Ret%", justify="right")
    table.add_column("3Y CAGR%", justify="right")
    table.add_column("5Y CAGR%", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Exp.%", justify="right")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("Conf.", justify="center", no_wrap=True)

    for i, row in df.iterrows():
        table.add_row(
            str(i + 1),
            str(row.get("scheme_name", "N/A")),
            str(row.get("category", "N/A")),
            _fmt(row.get("one_year_return")),
            _fmt(row.get("three_year_cagr")),
            _fmt(row.get("five_year_cagr")),
            _fmt(row.get("sharpe_ratio")),
            _fmt(row.get("expense_ratio_pct")),
            _fmt(row.get("composite_score")),
            str(row.get("score_confidence", "N/A")),
        )
    console.print(table)
    console.print()


def print_category_winners(df: pd.DataFrame):
    console.rule("[bold cyan]Category-wise Winners")
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Category", style="bold", no_wrap=True)
    table.add_column("Winning Scheme", no_wrap=True, overflow="ellipsis", max_width=48)
    table.add_column("1Y Ret%", justify="right")
    table.add_column("Score", justify="right", style="bold green")
    for _, row in df.iterrows():
        table.add_row(
            str(row.get("category")),
            str(row.get("scheme_name", "N/A")),
            _fmt(row.get("one_year_return")),
            _fmt(row.get("composite_score")),
        )
    console.print(table)
    console.print()


def print_specialty_leaders(specialty: dict):
    console.rule("[bold cyan]Specialty Leaderboards")
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Leaderboard", style="bold", no_wrap=True)
    table.add_column("Fund", no_wrap=True, overflow="ellipsis", max_width=42)
    table.add_column("Key Metric", justify="right")

    label_metric = {
        "best_overall": ("Best Overall Fund", "composite_score", "", " score"),
        "best_small_cap": ("Best Small Cap Performer", "one_year_return", "%", " 1Y return"),
        "best_sip": ("Best SIP Performer", "sip_return_1yr", "%", " SIP XIRR"),
        "best_risk_adjusted": ("Best Risk-Adjusted Performer", "sharpe_ratio", "", " Sharpe"),
        "best_long_term": ("Best Long-Term Wealth Creator", "five_year_cagr", "%", " 5Y CAGR"),
    }
    for key, (label, metric_col, suffix, _) in label_metric.items():
        fund = specialty.get(key)
        if fund is None:
            table.add_row(label, "[dim]No qualifying fund this run[/dim]", "")
        else:
            table.add_row(label, str(fund.get("scheme_name", "N/A")),
                           _fmt(fund.get(metric_col), suffix))
    console.print(table)
    console.print()


def print_international_funds(df: pd.DataFrame):
    console.rule("[bold cyan]International / FoF Funds Performance")
    if df.empty:
        console.print("[dim]No international/FoF category funds in this run's universe.[/dim]\n")
        return
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Scheme Name", style="bold", no_wrap=True, overflow="ellipsis", max_width=48)
    table.add_column("1Y Ret%", justify="right")
    table.add_column("3Y CAGR%", justify="right")
    table.add_column("Score", justify="right", style="bold green")
    for _, row in df.iterrows():
        table.add_row(str(row.get("scheme_name", "N/A")),
                       _fmt(row.get("one_year_return")),
                       _fmt(row.get("three_year_cagr")),
                       _fmt(row.get("composite_score")))
    console.print(table)
    console.print()


def print_market_context_and_recommendations():
    console.rule("[bold cyan]Risk Commentary & Market Context")
    console.print(
        "Category returns in any given year are heavily influenced by where we sit in the market "
        "cycle — small/mid-cap and thematic/international segments tend to lead in risk-on years "
        "and lag sharply in drawdowns. A single year of outperformance is not, on its own, evidence "
        "of manager skill; 3-year and 5-year CAGR, and risk-adjusted metrics like the Sharpe Ratio, "
        "are more reliable indicators of consistency.\n"
    )
    console.rule("[bold cyan]Final Recommendations (Generic)")
    console.print(
        "• Anchor a core equity allocation in Large Cap / Flexi Cap funds with a long, consistent "
        "track record before allocating to higher-volatility Small Cap or thematic/international funds.\n"
        "• Evaluate funds on rolling 3-5 year returns and risk-adjusted metrics rather than the latest "
        "12-month number alone.\n"
        "• Expense ratio compounds over holding periods of 10+ years — prefer lower-cost options among "
        "funds that are otherwise comparable on returns and risk.\n"
        "• This report is for informational/research purposes only and is not personalised investment "
        "advice. Consult a SEBI-registered investment adviser before making investment decisions.\n"
    )


# ---------------------------------------------------------------------- #
# EXPORT
# ---------------------------------------------------------------------- #

def export_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logger.info(f"Exported CSV report to {path}")


def _sanitize_for_json(obj):
    """Recursively replace NaN/NaT/Timestamp values so output is strictly valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def export_json(data: dict, path: str):
    clean = _sanitize_for_json(data)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Exported JSON report to {path}")


def export_markdown_summary(top_n_df: pd.DataFrame, category_df: pd.DataFrame, path: str, run_date: str):
    lines = [f"# Top Mutual Funds Performance Report — India\n", f"_Generated {run_date}_\n",
             "## Top Funds\n",
             "| Rank | Scheme | Category | 1Y % | 3Y CAGR % | 5Y CAGR % | Score |",
             "|---|---|---|---|---|---|---|"]
    for i, row in top_n_df.iterrows():
        lines.append(f"| {i+1} | {row.get('scheme_name')} | {row.get('category')} | "
                      f"{_fmt(row.get('one_year_return'))} | {_fmt(row.get('three_year_cagr'))} | "
                      f"{_fmt(row.get('five_year_cagr'))} | {_fmt(row.get('composite_score'))} |")
    lines.append("\n## Category Winners\n")
    lines.append("| Category | Scheme | Score |")
    lines.append("|---|---|---|")
    for _, row in category_df.iterrows():
        lines.append(f"| {row.get('category')} | {row.get('scheme_name')} | {_fmt(row.get('composite_score'))} |")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Exported Markdown summary to {path}")


# ---------------------------------------------------------------------- #
# HTML EXPORT — full detailed, self-contained report
# ---------------------------------------------------------------------- #

_HTML_CSS = """
:root {
  --navy: #0b1f3a; --navy-light: #13294b; --gold: #c9a227; --green: #1b7a3d;
  --red: #b3261e; --grey: #6b7280; --bg: #f5f6f8; --card: #ffffff; --border: #e3e6eb;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; background: var(--bg);
       color: #1a1a1a; margin: 0; padding: 0 0 60px 0; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px; }
header.masthead { background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
       color: #fff; padding: 36px 24px; }
header.masthead h1 { margin: 0 0 6px 0; font-size: 26px; letter-spacing: 0.3px; }
header.masthead .sub { color: #cfd8e6; font-size: 13px; }
.demo-banner { background: #fff3cd; border: 1px solid #f0c94a; color: #6b5200; padding: 14px 20px;
       margin: 18px 0; border-radius: 6px; font-size: 14px; }
.demo-banner strong { display:block; font-size:15px; margin-bottom:4px; }
section { margin-top: 34px; }
h2.section-title { font-size: 19px; color: var(--navy); border-bottom: 3px solid var(--gold);
       padding-bottom: 8px; margin-bottom: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px 22px;
       box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.summary-list { list-style: none; padding: 0; margin: 0; }
.summary-list li { padding: 8px 0; border-bottom: 1px dashed var(--border); font-size: 14.5px; line-height: 1.6; }
.summary-list li:last-child { border-bottom: none; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th { text-align: left; background: var(--navy); color: #fff; padding: 10px 12px; font-weight: 600;
     position: sticky; top: 0; }
td { padding: 9px 12px; border-bottom: 1px solid var(--border); }
tr:nth-child(even) td { background: #fafbfc; }
tr:hover td { background: #eef2f8; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.rank { text-align: right; color: var(--grey); width: 34px; }
.score { text-align: right; font-weight: 700; color: var(--green); }
.pos { color: var(--green); } .neg { color: var(--red); }
.badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.badge-high { background: #e3f5e8; color: var(--green); }
.badge-medium { background: #fff3cd; color: #8a6d00; }
.badge-low { background: #fde3e1; color: var(--red); }
.badge-nodata { background: #eee; color: var(--grey); }
.leader-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.leader-card { background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--gold);
       border-radius: 6px; padding: 14px 16px; }
.leader-card .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--grey); }
.leader-card .fund { font-size: 14.5px; font-weight: 600; margin: 4px 0; color: var(--navy); }
.leader-card .metric { font-size: 18px; font-weight: 700; color: var(--green); }
.leader-card .none { color: var(--grey); font-style: italic; font-size: 13px; }
.fund-row { cursor: pointer; }
.fund-row td:first-child { position: relative; }
.expand-icon { display: inline-block; width: 16px; color: var(--gold); font-size: 11px;
       transition: transform 0.15s ease; }
.fund-row.open .expand-icon { transform: rotate(90deg); }
.detail-row { display: none; }
.detail-row.open { display: table-row; }
.detail-row td { background: #f7f8fb !important; padding: 0; border-bottom: 2px solid var(--border); }
.detail-panel { padding: 18px 24px 22px 46px; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 820px) { .detail-grid { grid-template-columns: 1fr; } }
.detail-block h4 { font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--grey);
       margin: 0 0 10px 0; }
.alloc-bar { display: flex; height: 22px; border-radius: 4px; overflow: hidden; margin-bottom: 6px;
       border: 1px solid var(--border); }
.alloc-bar span { display: flex; align-items: center; justify-content: center; font-size: 10.5px;
       color: #fff; font-weight: 600; }
.alloc-legend { display: flex; gap: 14px; font-size: 12px; color: var(--grey); flex-wrap: wrap; }
.alloc-legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }
.mini-table { width: 100%; font-size: 12.5px; border-collapse: collapse; }
.mini-table th { background: none; color: var(--grey); font-weight: 600; font-size: 11px;
       text-transform: uppercase; padding: 4px 6px; border-bottom: 1px solid var(--border);
       position: static; }
.mini-table td { padding: 6px 6px; border-bottom: 1px solid #eef0f3; }
.weight-bar-wrap { display: flex; align-items: center; gap: 8px; }
.weight-bar-track { flex: 1; height: 6px; background: #e8eaee; border-radius: 3px; overflow: hidden; }
.weight-bar-fill { height: 100%; background: var(--gold); border-radius: 3px; }
.no-provider-note { font-size: 12.5px; color: var(--grey); background: #f0f1f4; border: 1px dashed var(--border);
       border-radius: 6px; padding: 12px 14px; line-height: 1.6; }
.fund-meta-row { display: flex; flex-wrap: wrap; gap: 22px; margin-bottom: 16px; font-size: 12.5px; }
.fund-meta-row .item .label { color: var(--grey); display: block; font-size: 10.5px; text-transform: uppercase; }
.fund-meta-row .item .value { font-weight: 600; color: var(--navy); font-size: 13.5px; }
.hint-banner { font-size: 12px; color: var(--grey); text-align: center; margin: 6px 0 18px 0; }
.prose { font-size: 14.5px; line-height: 1.7; color: #33383f; }
.prose ul { padding-left: 20px; }
.prose li { margin-bottom: 8px; }
footer.disclaimer { margin-top: 44px; padding: 18px 22px; background: #fff; border: 1px solid var(--border);
       border-radius: 8px; font-size: 12px; color: var(--grey); line-height: 1.6; }
.no-data { color: var(--grey); font-style: italic; padding: 14px 0; }

/* --- Metric chips --- */
.chip { display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 9px; border-radius: 10px;
       font-variant-numeric: tabular-nums; white-space: nowrap; }
.chip-green { background: #e3f5e8; color: var(--green); }
.chip-blue  { background: #e5eefc; color: #1a56b0; }
.chip-grey  { background: #eef0f3; color: var(--grey); }
.chip-yellow{ background: #fff3cd; color: #8a6d00; }
.chip-red   { background: #fde3e1; color: var(--red); }

/* --- Fixed filter bar --- */
.filter-bar { position: sticky; top: 0; z-index: 50; background: #fff; border-bottom: 1px solid var(--border);
       box-shadow: 0 2px 6px rgba(0,0,0,0.05); padding: 10px 24px; }
.filter-bar .wrap { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 0; }
.filter-bar label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; color: var(--grey);
       margin-right: 4px; }
.filter-bar select { font-size: 13px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
       background: #fff; color: #1a1a1a; }
.filter-bar .filter-group { display: flex; align-items: center; gap: 4px; }
.filter-bar .filter-count { margin-left: auto; font-size: 12.5px; color: var(--grey); }
.filter-bar .reset-btn { font-size: 12px; border: 1px solid var(--border); background: #f5f6f8; color: var(--navy);
       padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }
.filter-bar .reset-btn:hover { background: #eef2f8; }
tr.row-hidden { display: none !important; }

/* --- Weighting summary panel --- */
.weight-panel-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.weight-panel-row .wlabel { width: 150px; font-size: 13px; color: #33383f; flex-shrink: 0; }
.weight-panel-row .wtrack { flex: 1; height: 10px; background: #e8eaee; border-radius: 5px; overflow: hidden; }
.weight-panel-row .wfill { height: 100%; background: linear-gradient(90deg, var(--gold), #e0bb4a); border-radius: 5px; }
.weight-panel-row .wpct { width: 42px; text-align: right; font-size: 12.5px; font-weight: 700; color: var(--navy);
       font-variant-numeric: tabular-nums; }
.weight-panel-note { margin-top: 12px; font-size: 12.5px; color: var(--grey); background: #f7f8fb;
       border: 1px dashed var(--border); border-radius: 6px; padding: 10px 12px; line-height: 1.6; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
@media (max-width: 820px) { .two-col { grid-template-columns: 1fr; } }

/* --- Collapsible category panels --- */
.category-panel { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 12px; overflow: hidden;
       background: #fff; }
.category-panel summary { list-style: none; cursor: pointer; padding: 14px 18px; display: flex;
       align-items: center; gap: 12px; font-weight: 700; color: var(--navy); background: #fafbfc; }
.category-panel summary::-webkit-details-marker { display: none; }
.category-panel summary .cat-arrow { color: var(--gold); transition: transform 0.15s ease; font-size: 12px; }
.category-panel[open] summary .cat-arrow { transform: rotate(90deg); }
.category-panel summary .cat-count { font-weight: 500; color: var(--grey); font-size: 12.5px; }
.category-panel-body { padding: 14px 18px 18px 18px; }
.mini-card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
.mini-card { border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; background: #fafbfc; }
.mini-card .mc-rank { font-size: 10.5px; color: var(--grey); text-transform: uppercase; }
.mini-card .mc-name { font-size: 13px; font-weight: 600; color: var(--navy); margin: 3px 0 6px 0;
       line-height: 1.3; min-height: 34px; }
.mini-card .mc-metrics { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.view-full-cat-btn { margin-top: 12px; font-size: 12px; border: 1px solid var(--gold); background: #fff;
       color: #8a6d00; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }
.view-full-cat-btn:hover { background: #fff8e6; }
"""


def _score_badge(confidence: Optional[str]) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(confidence, "badge-nodata")
    label = confidence or "No Data"
    return f'<span class="badge {cls}">{label}</span>'


def _ret_class(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return "pos" if val >= 0 else "neg"


def _is_na(val) -> bool:
    return val is None or (isinstance(val, float) and pd.isna(val))


def _return_chip(val) -> str:
    """1Y-return style chip: >=20% green, 10-20% blue, <10% grey."""
    if _is_na(val):
        return '<span class="chip chip-grey">N/A</span>'
    cls = "chip-green" if val >= 20 else "chip-blue" if val >= 10 else "chip-grey"
    return f'<span class="chip {cls}">{val:.1f}%</span>'


def _sharpe_chip(val) -> str:
    """Sharpe Ratio chip: >=2 green, 1-2 yellow, <1 red."""
    if _is_na(val):
        return '<span class="chip chip-grey">N/A</span>'
    cls = "chip-green" if val >= 2 else "chip-yellow" if val >= 1 else "chip-red"
    return f'<span class="chip {cls}">{val:.2f}</span>'


def _risk_level(vol) -> str:
    """Bucket annualised volatility into a coarse risk label for filtering/display."""
    if _is_na(vol):
        return "Unknown"
    return "Low" if vol < 10 else "Medium" if vol < 20 else "High"


def _risk_chip(vol) -> str:
    label = _risk_level(vol)
    cls = {"Low": "chip-green", "Medium": "chip-yellow", "High": "chip-red"}.get(label, "chip-grey")
    return f'<span class="chip {cls}">{label}</span>'


_ALLOC_COLORS = {"equity": "#0b1f3a", "debt": "#c9a227", "cash": "#8a9099"}
_SECTOR_PALETTE = ["#0b1f3a", "#13294b", "#c9a227", "#1b7a3d", "#7c5cbf",
                    "#b3261e", "#2a7f9e", "#8a9099", "#d4823f", "#5a7d3a"]


def _amount_cr(pct, aum_cr) -> str:
    """₹ Cr amount implied by a % weight and total AUM, or 'N/A' if AUM unknown."""
    if pct is None or aum_cr is None or (isinstance(aum_cr, float) and pd.isna(aum_cr)):
        return "N/A"
    return f"₹{pct/100*aum_cr:,.1f} Cr"


def _html_allocation_bar(equity, debt, cash) -> str:
    parts = [("Equity", equity, _ALLOC_COLORS["equity"]),
             ("Debt", debt, _ALLOC_COLORS["debt"]),
             ("Cash & Others", cash, _ALLOC_COLORS["cash"])]
    valid = [(l, v, c) for l, v, c in parts if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not valid:
        return '<p class="no-data" style="padding:4px 0;">No portfolio allocation data available.</p>'
    total = sum(v for _, v, _ in valid) or 1
    bar = "".join(
        f'<span style="width:{v/total*100:.2f}%; background:{c};" title="{l}: {v:.1f}%">{v:.0f}%</span>'
        for l, v, c in valid if v > 0
    )
    legend = "".join(
        f'<span><span class="dot" style="background:{c};"></span>{l}: {v:.1f}%</span>'
        for l, v, c in valid
    )
    return f'<div class="alloc-bar">{bar}</div><div class="alloc-legend">{legend}</div>'


def _html_weighted_list(items: list, key: str, aum_cr, max_rows: int = 10) -> str:
    """Render a stock- or sector-level holdings table with % weight bars and ₹ Cr amounts."""
    if not items:
        return ""
    rows = []
    for item in items[:max_rows]:
        name = item.get(key, "N/A")
        pct = item.get("percent")
        pct_disp = f"{pct:.1f}%" if pct is not None else "N/A"
        bar_width = min(pct, 100) if pct is not None else 0
        rows.append(f"""<tr>
          <td>{name}</td>
          <td style="width:130px;">
            <div class="weight-bar-wrap">
              <div class="weight-bar-track"><div class="weight-bar-fill" style="width:{bar_width}%;"></div></div>
              <span style="width:38px;text-align:right;">{pct_disp}</span>
            </div>
          </td>
          <td class="num">{_amount_cr(pct, aum_cr)}</td>
        </tr>""")
    return f"""<table class="mini-table">
      <thead><tr><th>Name</th><th>% of Portfolio</th><th class="num">Approx. Amount</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _html_detail_panel(row, row_id: str) -> str:
    """
    Expandable per-fund panel: fund overview, equity/debt/cash allocation bar,
    sector allocation, and top stock holdings with % weight and implied ₹ Cr amount.
    Shows a clear explanatory note in place of any section a data provider didn't supply.
    """
    aum_cr = row.get("aum_cr")
    holdings = row.get("top_holdings") or []
    sectors = row.get("sector_allocation") or []
    has_alloc = any(row.get(k) is not None and not pd.isna(row.get(k))
                     for k in ("equity_allocation_pct", "debt_allocation_pct", "cash_allocation_pct"))

    meta_items = [
        ("Fund House", row.get("fund_house", "N/A")),
        ("Category", row.get("category", "N/A")),
        ("AUM", f"₹{aum_cr:,.0f} Cr" if aum_cr and not pd.isna(aum_cr) else "N/A"),
        ("Expense Ratio", _fmt(row.get("expense_ratio_pct"), "%")),
        ("Fund Manager", row.get("fund_manager", "N/A")),
        ("Manager Tenure", _fmt(row.get("manager_tenure_years"), " yrs")),
        ("Benchmark", row.get("benchmark_name", "N/A")),
        ("Latest NAV", f"₹{row.get('latest_nav'):.2f}" if row.get("latest_nav") is not None
                        and not pd.isna(row.get("latest_nav")) else "N/A"),
    ]
    meta_html = "".join(
        f'<div class="item"><span class="label">{label}</span><span class="value">{val}</span></div>'
        for label, val in meta_items
    )

    if not holdings and not sectors and not has_alloc:
        body = """<div class="no-provider-note">
          Sector allocation and stock-level holdings are not available for this fund because no
          supplementary data provider is configured for this run. AMFI/mfapi.in do not publish this
          data for free — plug in a licensed source (Morningstar Direct, Value Research Premium, or
          your own research CSV) by adding a provider class in this script and re-run with
          <code>--provider-csv your_data.csv</code> to populate this section. See
          <code>sample_supplementary_data.csv</code> in the project root for the exact format.
        </div>"""
    else:
        alloc_section = f"""<div class="detail-block">
          <h4>Portfolio Allocation</h4>
          {_html_allocation_bar(row.get('equity_allocation_pct'), row.get('debt_allocation_pct'), row.get('cash_allocation_pct'))}
        </div>""" if has_alloc else ""
        sector_section = f"""<div class="detail-block">
          <h4>Sector Allocation</h4>
          {_html_weighted_list(sectors, 'sector', aum_cr) if sectors else '<p class="no-data" style="padding:4px 0;">Not available for this fund.</p>'}
        </div>"""
        holdings_section = f"""<div class="detail-block">
          <h4>Top Holdings</h4>
          {_html_weighted_list(holdings, 'name', aum_cr) if holdings else '<p class="no-data" style="padding:4px 0;">Not available for this fund.</p>'}
        </div>"""
        body = f'<div class="detail-grid">{sector_section}{holdings_section}</div>{alloc_section}'

    return f"""<tr class="detail-row" id="{row_id}">
      <td colspan="12"><div class="detail-panel">
        <div class="fund-meta-row">{meta_html}</div>
        {body}
      </div></td>
    </tr>"""


def _html_top_table(df: pd.DataFrame, clickable: bool = True, table_id: Optional[str] = None) -> str:
    if df.empty:
        return '<p class="no-data">No funds could be ranked in this run.</p>'
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        row_id = f"detail-{row.get('scheme_code', i)}"
        row_attrs = f'onclick="toggleDetail(this, \'{row_id}\')"' if clickable else ""
        icon = '<span class="expand-icon">&#9656;</span> ' if clickable else ""
        data_attrs = (
            f'data-category="{row.get("category","N/A")}" '
            f'data-amc="{row.get("fund_house","N/A")}" '
            f'data-confidence="{row.get("score_confidence","No Data")}" '
            f'data-risk="{_risk_level(row.get("annualized_volatility_pct"))}" '
            f'data-score="{row.get("composite_score") if not _is_na(row.get("composite_score")) else ""}" '
            f'data-1y="{row.get("one_year_return") if not _is_na(row.get("one_year_return")) else ""}" '
            f'data-sharpe="{row.get("sharpe_ratio") if not _is_na(row.get("sharpe_ratio")) else ""}" '
            f'data-sip="{row.get("sip_return_1yr") if not _is_na(row.get("sip_return_1yr")) else ""}"'
        )
        rows.append(f"""<tr class="fund-row" {row_attrs} {data_attrs}>
          <td class="rank">{i+1}</td>
          <td>{icon}<strong>{row.get('scheme_name','N/A')}</strong><br>
              <span style="color:#888;font-size:11.5px;padding-left:{'18px' if clickable else '0'};">{row.get('fund_house','N/A')} &middot; {row.get('benchmark_name','N/A')}</span></td>
          <td>{row.get('category','N/A')}</td>
          <td class="num">{_return_chip(row.get('one_year_return'))}</td>
          <td class="num {_ret_class(row.get('three_year_cagr'))}">{_fmt(row.get('three_year_cagr'),'%')}</td>
          <td class="num {_ret_class(row.get('five_year_cagr'))}">{_fmt(row.get('five_year_cagr'),'%')}</td>
          <td class="num">{_fmt(row.get('sip_return_1yr'),'%')}</td>
          <td class="num">{_sharpe_chip(row.get('sharpe_ratio'))}</td>
          <td class="num">{_risk_chip(row.get('annualized_volatility_pct'))}</td>
          <td class="num">{_fmt(row.get('expense_ratio_pct'),'%')}</td>
          <td class="score">{_fmt(row.get('composite_score'))}</td>
          <td>{_score_badge(row.get('score_confidence'))}</td>
        </tr>""")
        if clickable:
            rows.append(_html_detail_panel(row, row_id))
    hint = '<div class="hint-banner">Click any fund row to see sector allocation, top holdings, and portfolio detail.</div>' if clickable else ""
    id_attr = f' id="{table_id}"' if table_id else ""
    return f"""{hint}<div style="overflow-x:auto;"><table{id_attr}>
      <thead><tr>
        <th class="rank">#</th><th>Scheme</th><th>Category</th>
        <th class="num">1Y Ret</th><th class="num">3Y CAGR</th><th class="num">5Y CAGR</th>
        <th class="num">SIP XIRR</th><th class="num">Sharpe</th><th class="num">Risk</th>
        <th class="num">Expense</th><th class="num">Score</th><th>Confidence</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>"""


def _html_category_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="no-data">No category winners could be determined.</p>'
    rows = []
    for _, row in df.iterrows():
        rows.append(f"""<tr>
          <td><strong>{row.get('category','N/A')}</strong></td>
          <td>{row.get('scheme_name','N/A')}</td>
          <td class="num">{_fmt(row.get('one_year_return'),'%')}</td>
          <td class="num">{_fmt(row.get('three_year_cagr'),'%')}</td>
          <td class="score">{_fmt(row.get('composite_score'))}</td>
          <td>{_score_badge(row.get('score_confidence'))}</td>
        </tr>""")
    return f"""<table>
      <thead><tr><th>Category</th><th>Winning Scheme</th><th class="num">1Y Ret</th>
      <th class="num">3Y CAGR</th><th class="num">Score</th><th>Confidence</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _html_leader_card(label: str, fund, metric_col: str, suffix: str = "") -> str:
    if fund is None:
        return f"""<div class="leader-card"><div class="label">{label}</div>
                   <div class="none">No qualifying fund this run</div></div>"""
    return f"""<div class="leader-card">
      <div class="label">{label}</div>
      <div class="fund">{fund.get('scheme_name','N/A')}</div>
      <div class="metric">{_fmt(fund.get(metric_col), suffix)}</div>
    </div>"""


def _html_intl_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="no-data">No international/FoF category funds in this run\'s universe.</p>'
    rows = []
    for _, row in df.iterrows():
        rows.append(f"""<tr>
          <td><strong>{row.get('scheme_name','N/A')}</strong></td>
          <td class="num">{_fmt(row.get('one_year_return'),'%')}</td>
          <td class="num">{_fmt(row.get('three_year_cagr'),'%')}</td>
          <td class="num">{_fmt(row.get('five_year_cagr'),'%')}</td>
          <td class="score">{_fmt(row.get('composite_score'))}</td>
        </tr>""")
    return f"""<table>
      <thead><tr><th>Scheme</th><th class="num">1Y Ret</th><th class="num">3Y CAGR</th>
      <th class="num">5Y CAGR</th><th class="num">Score</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _html_exec_summary(top_n_df: pd.DataFrame, category_df: pd.DataFrame, specialty: dict) -> str:
    if top_n_df.empty:
        return '<p class="no-data">No funds could be ranked in this run — see logs for data issues.</p>'
    best = top_n_df.iloc[0]
    items = [
        f"<strong>{best.get('scheme_name')}</strong> ({best.get('category')}) leads the overall rankings "
        f"this period with a composite score of <strong>{_fmt(best.get('composite_score'))}/100</strong> "
        f"(confidence: {best.get('score_confidence','N/A')}), on the back of a "
        f"{_fmt(best.get('one_year_return'),'%')} 1-year return and {_fmt(best.get('three_year_cagr'),'%')} 3-year CAGR.",
        f"{len(category_df)} categories were represented in this run's leaderboard "
        f"({', '.join(category_df['category'].tolist())}).",
    ]
    if specialty.get("best_small_cap") is not None:
        f_ = specialty["best_small_cap"]
        items.append(f"Small Cap leadership: <strong>{f_.get('scheme_name')}</strong> "
                      f"({_fmt(f_.get('one_year_return'),'%')} 1Y return) — small caps remain the highest-return, "
                      f"highest-volatility segment of the market.")
    if specialty.get("best_sip") is not None:
        f_ = specialty["best_sip"]
        items.append(f"Best SIP performer: <strong>{f_.get('scheme_name')}</strong> with a trailing 12-month "
                      f"SIP XIRR of {_fmt(f_.get('sip_return_1yr'),'%')}.")
    if specialty.get("best_risk_adjusted") is not None:
        f_ = specialty["best_risk_adjusted"]
        items.append(f"Best risk-adjusted performer: <strong>{f_.get('scheme_name')}</strong> "
                      f"(Sharpe Ratio: {_fmt(f_.get('sharpe_ratio'))}).")
    return '<ul class="summary-list">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


_WEIGHT_LABELS = {
    "one_year_return": "1Y Return",
    "three_year_cagr": "3Y CAGR",
    "five_year_cagr": "5Y CAGR",
    "risk_adjusted_score": "Risk-Adjusted (Sharpe)",
    "expense_ratio": "Expense Ratio",
    "aum_stability": "AUM Stability",
}


def _html_weight_panel(weights: dict, full_df: pd.DataFrame) -> str:
    """Horizontal bar breakdown of the composite-score ranking weights, plus a
    plain-language note on how confidence is derived, to build trust in the model."""
    rows = []
    for key, w in weights.items():
        label = _WEIGHT_LABELS.get(key, key)
        pct = w * 100
        rows.append(f"""<div class="weight-panel-row" title="{label} contributes {pct:.0f}% of the composite score">
          <span class="wlabel">{label}</span>
          <div class="wtrack"><div class="wfill" style="width:{pct:.0f}%;"></div></div>
          <span class="wpct">{pct:.0f}%</span>
        </div>""")

    low_conf_n = int((full_df["score_confidence"].isin(["Low", "No Data"])).sum()) if "score_confidence" in full_df else 0
    note = (
        "Scores are computed by re-normalising these weights across whichever metrics are actually "
        "available for a given fund (rather than penalising it for missing data). A fund's Confidence "
        "badge reflects how much of the weighted model it was actually scored on: "
        "<strong>High</strong> \u2265 80% of metrics present, <strong>Medium</strong> \u2265 50%, "
        "<strong>Low</strong> below that \u2014 usually because AUM or Expense Ratio wasn't supplied by "
        "a data provider."
    )
    if low_conf_n:
        note += f" In this run, <strong>{low_conf_n}</strong> fund(s) carry a Low or No-Data confidence flag."

    return f"""<div class="two-col">
      <div>{''.join(rows)}</div>
      <div class="weight-panel-note">{note}</div>
    </div>"""


def _html_category_collapsible(full_df: pd.DataFrame, category_df: pd.DataFrame, top_n: int = 5) -> str:
    """Collapsible per-category panels: each opens to show mini-cards for its
    top N funds (by composite score) with a button to jump to the full,
    filterable table for that category."""
    if category_df.empty:
        return '<p class="no-data">No category winners could be determined.</p>'

    valid = full_df.dropna(subset=["composite_score"])
    panels = []
    # Order categories by their winning fund's composite score (best category first).
    for _, winner in category_df.iterrows():
        cat = winner.get("category", "N/A")
        cat_funds = valid[valid["category"] == cat].sort_values("composite_score", ascending=False).head(top_n)
        cards = []
        for i, (_, f) in enumerate(cat_funds.iterrows()):
            cards.append(f"""<div class="mini-card">
              <div class="mc-rank">#{i+1} in {cat}</div>
              <div class="mc-name">{f.get('scheme_name','N/A')}</div>
              <div class="mc-metrics">
                {_return_chip(f.get('one_year_return'))}
                {_sharpe_chip(f.get('sharpe_ratio'))}
                {_score_badge(f.get('score_confidence'))}
              </div>
            </div>""")
        n_in_cat = int((valid["category"] == cat).sum())
        panels.append(f"""<details class="category-panel">
          <summary><span class="cat-arrow">&#9656;</span> {cat}
            <span class="cat-count">&middot; {n_in_cat} fund{'s' if n_in_cat != 1 else ''} &middot; winner: {winner.get('scheme_name','N/A')}</span>
          </summary>
          <div class="category-panel-body">
            <div class="mini-card-grid">{''.join(cards)}</div>
            <button class="view-full-cat-btn" onclick="viewFullCategory('{cat}')">View full category in ranked table &rarr;</button>
          </div>
        </details>""")
    return "".join(panels)


def _html_filter_bar(full_df: pd.DataFrame) -> str:
    """Fixed top filter bar for the full ranked universe: category, risk, AMC,
    confidence and a sort-by control, all applied client-side via JS."""
    categories = sorted(x for x in full_df.get("category", pd.Series(dtype=str)).dropna().unique())
    amcs = sorted(x for x in full_df.get("fund_house", pd.Series(dtype=str)).dropna().unique() if x != "N/A")
    cat_opts = "".join(f'<option value="{c}">{c}</option>' for c in categories)
    amc_opts = "".join(f'<option value="{a}">{a}</option>' for a in amcs)
    return f"""<div class="filter-bar">
      <div class="wrap">
        <div class="filter-group"><label for="f-category">Category</label>
          <select id="f-category" onchange="applyFilters()"><option value="">All</option>{cat_opts}</select></div>
        <div class="filter-group"><label for="f-risk">Risk</label>
          <select id="f-risk" onchange="applyFilters()"><option value="">All</option>
            <option value="Low">Low</option><option value="Medium">Medium</option><option value="High">High</option></select></div>
        <div class="filter-group"><label for="f-amc">AMC</label>
          <select id="f-amc" onchange="applyFilters()"><option value="">All</option>{amc_opts}</select></div>
        <div class="filter-group"><label for="f-confidence">Confidence</label>
          <select id="f-confidence" onchange="applyFilters()"><option value="">All</option>
            <option value="High">High</option><option value="Medium">Medium</option>
            <option value="Low">Low</option><option value="No Data">No Data</option></select></div>
        <div class="filter-group"><label for="f-sort">Sort by</label>
          <select id="f-sort" onchange="applyFilters()">
            <option value="data-score">Score</option>
            <option value="data-1y">1Y Return</option>
            <option value="data-sharpe">Sharpe</option>
            <option value="data-sip">SIP XIRR</option></select></div>
        <button class="reset-btn" onclick="resetFilters()">Reset</button>
        <span class="filter-count" id="f-count"></span>
      </div>
    </div>"""


def export_html_report(
    top_n_df: pd.DataFrame,
    category_df: pd.DataFrame,
    specialty: dict,
    intl_df: pd.DataFrame,
    full_df: pd.DataFrame,
    path: str,
    run_date: str,
    fund_count: int,
    is_synthetic: bool = False,
    ranking_weights: Optional[dict] = None,
):
    """
    Generate a detailed, self-contained (single-file, no external assets)
    HTML report covering every section of the console dashboard: executive
    summary, top-N leaderboard, category winners, specialty leaderboards,
    international funds, full ranked universe, risk commentary and
    recommendations.
    """
    demo_banner = ""
    if is_synthetic:
        demo_banner = """<div class="demo-banner">
          <strong>&#9888; DEMO MODE &mdash; SYNTHETIC DATA</strong>
          All figures in this report are randomly generated for demonstration purposes only and
          DO NOT reflect real market performance. Re-run without --demo (with internet access) for
          live mfapi.in data.
        </div>"""

    top_n = len(top_n_df)
    weights = ranking_weights or DEFAULT_CONFIG["ranking_weights"]
    sorted_full = full_df.sort_values("composite_score", ascending=False) if not full_df.empty else full_df
    full_table_html = _html_top_table(sorted_full, table_id="universe-table") if not full_df.empty else ""
    filter_bar_html = _html_filter_bar(full_df) if not full_df.empty else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top Mutual Funds Performance Report — India | {run_date}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<header class="masthead">
  <div class="wrap">
    <h1>Top Mutual Funds Performance Report &mdash; India</h1>
    <div class="sub">Generated {run_date} &nbsp;|&nbsp; Universe: {fund_count} funds analysed &nbsp;|&nbsp;
    Ranking model: 1Y 25% &middot; 3Y CAGR 25% &middot; 5Y CAGR 20% &middot; Risk-Adjusted 15% &middot;
    Expense Ratio 10% &middot; AUM Stability 5%</div>
  </div>
</header>

{filter_bar_html}

<div class="wrap">
  {demo_banner}

  <section>
    <h2 class="section-title">Executive Summary</h2>
    <div class="card">{_html_exec_summary(top_n_df, category_df, specialty)}</div>
  </section>

  <section>
    <h2 class="section-title">Ranking Model &mdash; Metric Weights</h2>
    <div class="card">{_html_weight_panel(weights, full_df)}</div>
  </section>

  <section>
    <h2 class="section-title">Top {top_n} Mutual Funds of the Year</h2>
    <div class="card">{_html_top_table(top_n_df)}</div>
  </section>

  <section>
    <h2 class="section-title">Category-wise Winners</h2>
    {_html_category_collapsible(full_df, category_df)}
  </section>

  <section>
    <h2 class="section-title">Specialty Leaderboards</h2>
    <div class="leader-grid">
      {_html_leader_card("Best Overall Fund", specialty.get("best_overall"), "composite_score")}
      {_html_leader_card("Best Small Cap Performer", specialty.get("best_small_cap"), "one_year_return", "%")}
      {_html_leader_card("Best SIP Performer", specialty.get("best_sip"), "sip_return_1yr", "%")}
      {_html_leader_card("Best Risk-Adjusted Performer", specialty.get("best_risk_adjusted"), "sharpe_ratio")}
      {_html_leader_card("Best Long-Term Wealth Creator", specialty.get("best_long_term"), "five_year_cagr", "%")}
    </div>
  </section>

  <section>
    <h2 class="section-title">International / FoF Funds Performance</h2>
    <div class="card">{_html_intl_table(intl_df)}</div>
  </section>

  <section>
    <h2 class="section-title">Risk Commentary &amp; Market Context</h2>
    <div class="card prose">
      <p>Category returns in any given year are heavily influenced by where we sit in the market cycle —
      small/mid-cap and thematic/international segments tend to lead in risk-on years and lag sharply in
      drawdowns. A single year of outperformance is not, on its own, evidence of manager skill; 3-year and
      5-year CAGR, and risk-adjusted metrics like the Sharpe Ratio, are more reliable indicators of
      consistency.</p>
      <p>Where a fund's <span class="badge badge-medium">Medium</span> or <span class="badge badge-low">Low</span>
      confidence badge appears above, its composite score is based on a reduced set of metrics (typically
      because AUM, Expense Ratio or another premium data field was not available) — weight the score
      accordingly relative to <span class="badge badge-high">High</span>-confidence entries.</p>
    </div>
  </section>

  <section>
    <h2 class="section-title">Final Recommendations (Generic)</h2>
    <div class="card prose">
      <ul>
        <li>Anchor a core equity allocation in Large Cap / Flexi Cap funds with a long, consistent track
        record before allocating to higher-volatility Small Cap or thematic/international funds.</li>
        <li>Evaluate funds on rolling 3&ndash;5 year returns and risk-adjusted metrics rather than the
        latest 12-month number alone.</li>
        <li>Expense ratio compounds over holding periods of 10+ years — prefer lower-cost options among
        funds that are otherwise comparable on returns and risk.</li>
      </ul>
    </div>
  </section>

  <section>
    <h2 class="section-title">Full Ranked Universe ({len(full_df)} funds)</h2>
    <div class="card">{full_table_html}</div>
  </section>

  <footer class="disclaimer">
    This report is for informational and research purposes only. It is not personalised investment advice.
    Past performance is not indicative of future returns. Mutual fund investments are subject to market
    risk; please read all scheme-related documents carefully. Consult a SEBI-registered investment adviser
    before making investment decisions. NAV data sourced from mfapi.in (a mirror of AMFI data); AUM,
    Expense Ratio, Fund Manager and Holdings fields require a licensed supplementary data provider and
    show as N/A when none is configured — see the SupplementaryDataProvider class in this script.
  </footer>
</div>
<script>
  function toggleDetail(rowEl, detailId) {{
    var detail = document.getElementById(detailId);
    if (!detail) return;
    var isOpen = detail.classList.contains('open');
    detail.classList.toggle('open', !isOpen);
    rowEl.classList.toggle('open', !isOpen);
  }}

  function applyFilters() {{
    var table = document.getElementById('universe-table');
    if (!table) return;
    var tbody = table.tBodies[0];
    var cat = document.getElementById('f-category').value;
    var risk = document.getElementById('f-risk').value;
    var amc = document.getElementById('f-amc').value;
    var conf = document.getElementById('f-confidence').value;
    var sortKey = document.getElementById('f-sort').value;

    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr.fund-row'));
    var visibleCount = 0;

    rows.forEach(function(row) {{
      var detail = row.nextElementSibling && row.nextElementSibling.classList.contains('detail-row')
        ? row.nextElementSibling : null;
      var match = (!cat || row.dataset.category === cat)
        && (!risk || row.dataset.risk === risk)
        && (!amc || row.dataset.amc === amc)
        && (!conf || row.dataset.confidence === conf);
      row.classList.toggle('row-hidden', !match);
      if (detail) detail.classList.toggle('row-hidden', !match);
      if (match) visibleCount++;
    }});

    // Re-sort visible + hidden rows together by the chosen metric (desc),
    // keeping each fund's detail-row glued immediately after it.
    var pairs = rows.map(function(row) {{
      var detail = row.nextElementSibling && row.nextElementSibling.classList.contains('detail-row')
        ? row.nextElementSibling : null;
      var raw = row.dataset[sortKey.replace('data-', '')];
      var val = raw === '' || raw === undefined ? -Infinity : parseFloat(raw);
      return {{ row: row, detail: detail, val: val }};
    }});
    pairs.sort(function(a, b) {{ return b.val - a.val; }});
    pairs.forEach(function(p) {{
      tbody.appendChild(p.row);
      if (p.detail) tbody.appendChild(p.detail);
    }});

    var countEl = document.getElementById('f-count');
    if (countEl) countEl.textContent = visibleCount + ' of ' + rows.length + ' funds shown';
  }}

  function resetFilters() {{
    ['f-category', 'f-risk', 'f-amc', 'f-confidence'].forEach(function(id) {{
      document.getElementById(id).value = '';
    }});
    document.getElementById('f-sort').value = 'data-score';
    applyFilters();
  }}

  function viewFullCategory(category) {{
    document.getElementById('f-category').value = category;
    applyFilters();
    var table = document.getElementById('universe-table');
    if (table) table.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }}

  document.addEventListener('DOMContentLoaded', applyFilters);
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Exported HTML report to {path}")


# ==============================================================================
# ORCHESTRATOR — CLI, logging setup, and the fetch -> clean -> rank -> report pipeline
# ==============================================================================
def setup_logging(log_level: str, log_to_file: bool, output_dir: str) -> logging.Logger:
    """
    Console-only logging by default -- creates NO folders or files. Pass
    --log-file to also write a timestamped log into `output_dir` (the same
    single folder used for report exports, so a run never creates more than
    one directory).
    """
    logger = logging.getLogger("mf_report")
    logger.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.WARNING)  # keep console clean; use --log-file for full detail
    logger.addHandler(ch)

    if log_to_file:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"mf_report_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"File logging enabled -> {log_path}")

    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top Mutual Funds Performance Report (India)")
    p.add_argument("--demo", action="store_true", help="Run with synthetic offline data")
    p.add_argument("--demo-funds", type=int, default=30, help="Number of synthetic funds in demo mode")
    p.add_argument("--limit", type=int, default=200,
                    help="Max number of live schemes to fetch NAV history for (keeps runtime sane)")
    p.add_argument("--top-n", type=int, default=None, help="Override top-N size (default from config)")
    p.add_argument("--export", choices=["csv", "json", "markdown", "html", "all", "none"], default="html",
                    help="Export format for the ranked results (default: html)")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--provider-csv", default=None,
                    help="Path to a CSV of supplementary data (AUM, expense ratio, etc.) — see the SupplementaryDataProvider classes above")
    p.add_argument("--risk-free-rate", type=float, default=None, help="Override annual risk-free rate, e.g. 0.069")
    p.add_argument("--timeout", type=float, default=None,
                    help="Override per-request timeout in seconds (default 30) -- raise this if you see "
                         "'Read timed out' errors against api.mfapi.in on your network")
    p.add_argument("--workers", type=int, default=None,
                    help="Override number of concurrent NAV-fetch requests (default 5) -- lower this if "
                         "your network/API keeps timing out under load")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", action="store_true",
                    help="Also write a timestamped log file into the output folder "
                         "(off by default -- console-only logging creates no files/folders)")
    return p.parse_args()


def build_fund_record(scheme_code, scheme_name, nav_df, category_keywords, provider, is_synthetic, config):
    """Compute all metrics for a single fund and return a flat dict record."""
    supp = provider.get_fund_details(scheme_code, scheme_name)
    rf = config["risk_free_rate_annual"]

    record = {
        "scheme_code": scheme_code,
        "scheme_name": scheme_name,
        "fund_house": supp.fund_house or nav_df.attrs.get("fund_house") or "N/A",
        "category": infer_category(
            scheme_name, config["category_keywords"], supp.category or nav_df.attrs.get("scheme_category")
        ),
        "latest_nav": float(nav_df.iloc[-1]["nav"]),
        "one_year_return": compute_point_to_point_return(nav_df, 1),
        "three_year_cagr": compute_point_to_point_return(nav_df, 3),
        "five_year_cagr": compute_point_to_point_return(nav_df, 5),
        "sip_return_1yr": compute_sip_return(
            nav_df, config["sip"]["monthly_investment"], config["sip"]["lookback_months"]
        ),
        "annualized_volatility_pct": compute_annualized_volatility(nav_df),
        "sharpe_ratio": compute_sharpe_ratio(nav_df, rf),
        "expense_ratio_pct": supp.expense_ratio_pct,
        "aum_cr": supp.aum_cr,
        "fund_manager": supp.fund_manager or "N/A",
        "manager_tenure_years": supp.manager_tenure_years,
        "equity_allocation_pct": supp.equity_allocation_pct,
        "debt_allocation_pct": supp.debt_allocation_pct,
        "cash_allocation_pct": supp.cash_allocation_pct,
        "top_holdings": supp.top_holdings,
        "sector_allocation": supp.sector_allocation,
        "benchmark_name": supp.benchmark_name or "N/A",
        "is_synthetic": is_synthetic,
    }
    return record


def run_pipeline(args: argparse.Namespace) -> int:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.risk_free_rate is not None:
        config["risk_free_rate_annual"] = args.risk_free_rate
    if args.timeout is not None:
        config["api"]["request_timeout_seconds"] = args.timeout
    if args.workers is not None:
        config["api"]["max_parallel_requests"] = args.workers
    top_n = args.top_n or config["output"]["top_n_default"]
    # Default output folder is anchored to THIS SCRIPT'S location (not the
    # shell's current working directory), so `outputs/report.html` always
    # ends up next to the script regardless of which folder you run it from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, config["output"]["outputs_dir"])
    output_dir = args.output_dir or default_output_dir
    # NOTE: output_dir is intentionally NOT created here. It's created lazily,
    # only if/when a file actually needs to be written into it (an export or
    # --log-file) -- so a run with no --export produces zero folders.

    logger = setup_logging(args.log_level, args.log_file, output_dir)
    validate_weights(config["ranking_weights"])

    provider = CSVProvider(args.provider_csv) if args.provider_csv else NullProvider()

    fund_records = {}

    try:
        if args.demo:
            logger.warning("Running in --demo mode: using SYNTHETIC data, not live market data.")
            demo_data = generate_synthetic_demo_data(n_funds=args.demo_funds)
            for code, d in demo_data.items():
                cleaned = clean_nav_series(d["nav_df"], code)
                if cleaned is None:
                    continue
                cleaned.attrs = d["nav_df"].attrs
                rec = build_fund_record(code, d["scheme_name"], cleaned,
                                         config["category_keywords"], provider, True, config)
                if validate_fund_record(rec):
                    fund_records[code] = rec
        else:
            fetcher = MFAPIFetcher(config)
            all_schemes = fetcher.get_scheme_list()
            candidates = fetcher.filter_schemes_by_keywords(
                all_schemes, config["category_keywords"], limit=args.limit
            )
            nav_histories = fetcher.fetch_all_nav_histories(candidates)

            for code, raw_df in nav_histories.items():
                scheme_name = raw_df.attrs.get("scheme_name") or f"Scheme {code}"
                cleaned = clean_nav_series(raw_df, code)
                if cleaned is None:
                    continue
                cleaned.attrs = raw_df.attrs
                rec = build_fund_record(code, scheme_name, cleaned,
                                         config["category_keywords"], provider, False, config)
                if validate_fund_record(rec):
                    fund_records[code] = rec

    except RuntimeError as exc:
        logger.error(f"Fatal data acquisition error: {exc}")
        console.print(
            f"[bold red]Fatal error:[/bold red] {exc}\n"
            f"[dim]Tip: run with --demo to see the report using synthetic offline data.[/dim]"
        )
        return 1

    if not fund_records:
        logger.error("No fund records survived fetching/cleaning/validation. Aborting.")
        console.print("[bold red]No usable fund data was produced. Check logs.[/bold red]")
        return 1

    master_df = build_master_dataframe(fund_records)
    master_df = compute_composite_score(master_df, config["ranking_weights"])

    top_df = rank_top_n(master_df, n=top_n)
    cat_df = category_winners(master_df)
    specialty = {
        "best_overall": best_overall(master_df),
        "best_small_cap": best_small_cap(master_df),
        "best_sip": best_sip_performer(master_df),
        "best_risk_adjusted": best_risk_adjusted(master_df),
        "best_long_term": best_long_term_wealth_creator(master_df),
    }
    intl_df = international_funds_performance(master_df)

    run_date = datetime.now().strftime("%d %b %Y, %H:%M IST")
    print_banner(bool(args.demo), len(master_df), run_date)
    print_executive_summary(top_df, cat_df, specialty)
    print_top_n_table(top_df, title=f"Top {top_n} Mutual Funds of the Year")
    print_category_winners(cat_df)
    print_specialty_leaders(specialty)
    print_international_funds(intl_df)
    print_market_context_and_recommendations()

    written_files = []
    if args.export != "none":
        os.makedirs(output_dir, exist_ok=True)  # lazy: only create the folder if we're writing into it

    if args.export in ("csv", "all"):
        path = os.path.join(output_dir, "ranked_funds.csv")
        export_csv(master_df, path)
        written_files.append(path)
    if args.export in ("json", "all"):
        path = os.path.join(output_dir, "ranked_funds.json")
        export_json(
            {"generated_at": run_date, "top_funds": top_df.to_dict(orient="records"),
             "category_winners": cat_df.to_dict(orient="records")},
            path,
        )
        written_files.append(path)
    if args.export in ("markdown", "all"):
        path = os.path.join(output_dir, "report_summary.md")
        export_markdown_summary(top_df, cat_df, path, run_date)
        written_files.append(path)
    if args.export in ("html", "all"):
        path = os.path.join(output_dir, "report.html")
        export_html_report(
            top_df, cat_df, specialty, intl_df, master_df,
            path, run_date, len(master_df), is_synthetic=bool(args.demo),
            ranking_weights=config["ranking_weights"],
        )
        written_files.append(path)

    if written_files:
        console.rule("[bold cyan]Files Saved")
        for p in written_files:
            console.print(f"  [bold green]\u2713[/bold green] {os.path.abspath(p)}")
        console.print()
    else:
        console.print("[dim]--export none was set -- results shown above only, no file written. "
                       "Omit --export (defaults to html) or pass csv/json/markdown/all to save a file.[/dim]")

    logger.info(f"Pipeline complete. {len(master_df)} funds ranked, top {top_n} reported.")
    return 0


if __name__ == "__main__":
    exit_code = run_pipeline(parse_args())
    sys.exit(exit_code)
