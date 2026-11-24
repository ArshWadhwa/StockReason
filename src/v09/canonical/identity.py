"""
StockReason v0.9 — Canonical Date-Aware Security Identity System.
================================================================
Tracks securities across time using exchange identifiers, ISIN, historical
ticker mappings, mergers, and delistings without lookahead bias.
"""

import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

DEFAULT_SYMBOL_HISTORY_PATH = "data/indian-markets/symbol_history/nse.parquet"

class SecurityIdentityRegistry:
    """
    Date-aware security identity registry.
    Maps tickers and ISINs across historical corporate events, renames, and delistings.
    """
    def __init__(self, history_path: str = DEFAULT_SYMBOL_HISTORY_PATH):
        self.history_path = history_path
        self._load_registry()

    def _load_registry(self):
        if not os.path.exists(self.history_path):
            raise FileNotFoundError(f"Symbol history not found at {self.history_path}")
        self.df_sh = pd.read_parquet(self.history_path)
        # Ensure valid_from and valid_to are python dates
        self.df_sh["valid_from"] = pd.to_datetime(self.df_sh["valid_from"]).dt.date
        self.df_sh["valid_to"] = pd.to_datetime(self.df_sh["valid_to"]).dt.date
        
        # Build index maps
        self.isin_to_records = {}
        for _, row in self.df_sh.iterrows():
            isin = row["isin"]
            if isin not in self.isin_to_records:
                self.isin_to_records[isin] = []
            self.isin_to_records[isin].append(row.to_dict())

        # Symbol to records
        self.sym_to_records = {}
        for _, row in self.df_sh.iterrows():
            sym = row["symbol"].upper()
            if sym not in self.sym_to_records:
                self.sym_to_records[sym] = []
            self.sym_to_records[sym].append(row.to_dict())

    def resolve(self, identifier: str, as_of_date: Optional[Any] = None) -> Dict[str, Any]:
        """
        Resolves a symbol (e.g. 'INFY', 'INFOSYSTCH', 'RELIANCE.NS') or ISIN to its canonical record.
        If as_of_date is provided, resolves the point-in-time ticker and active status.
        """
        clean_id = identifier.upper().replace(".NS", "").replace(".BO", "").strip()
        
        ref_date = None
        if as_of_date is not None:
            if isinstance(as_of_date, str):
                ref_date = pd.to_datetime(as_of_date).date()
            elif isinstance(as_of_date, pd.Timestamp):
                ref_date = as_of_date.date()
            elif isinstance(as_of_date, (datetime, date)):
                ref_date = as_of_date if isinstance(as_of_date, date) else as_of_date.date()

        # Check if identifier is an ISIN
        if clean_id.startswith("IN") and len(clean_id) == 12:
            records = self.isin_to_records.get(clean_id, [])
        else:
            records = self.sym_to_records.get(clean_id, [])

        if not records:
            # Fallback for unknown/unmapped ticker
            return {
                "identifier": identifier,
                "isin": None,
                "canonical_symbol": clean_id,
                "symbol_at_date": clean_id,
                "is_active": True,
                "status": "UNMAPPED",
                "aliases": []
            }

        # If as_of_date is specified, find matching validity slice
        if ref_date:
            for r in records:
                if r["valid_from"] <= ref_date <= r["valid_to"]:
                    # Found point-in-time record
                    all_syms = [rec["symbol"] for rec in self.isin_to_records.get(r["isin"], [])]
                    return {
                        "identifier": identifier,
                        "isin": r["isin"],
                        "canonical_symbol": all_syms[-1],  # Latest symbol is canonical
                        "symbol_at_date": r["symbol"],
                        "valid_from": r["valid_from"],
                        "valid_to": r["valid_to"],
                        "is_active": True,
                        "status": "ACTIVE_AT_DATE",
                        "aliases": list(set(all_syms) - {all_syms[-1]})
                    }
            # If date is past valid_to of all records, it's delisted/ceased
            latest_rec = sorted(records, key=lambda x: x["valid_to"])[-1]
            all_syms = [rec["symbol"] for rec in self.isin_to_records.get(latest_rec["isin"], [])]
            is_active = (ref_date <= latest_rec["valid_to"])
            return {
                "identifier": identifier,
                "isin": latest_rec["isin"],
                "canonical_symbol": all_syms[-1],
                "symbol_at_date": latest_rec["symbol"],
                "valid_from": latest_rec["valid_from"],
                "valid_to": latest_rec["valid_to"],
                "is_active": is_active,
                "status": "DELISTED_OR_CEASED" if not is_active else "ACTIVE",
                "aliases": list(set(all_syms) - {all_syms[-1]})
            }

        # No date provided: return latest known state
        latest_rec = sorted(records, key=lambda x: x["valid_to"])[-1]
        all_syms = [rec["symbol"] for rec in self.isin_to_records.get(latest_rec["isin"], [])]
        is_currently_active = latest_rec["valid_to"] >= date(2026, 1, 1)
        return {
            "identifier": identifier,
            "isin": latest_rec["isin"],
            "canonical_symbol": all_syms[-1],
            "symbol_at_date": latest_rec["symbol"],
            "valid_from": latest_rec["valid_from"],
            "valid_to": latest_rec["valid_to"],
            "is_active": is_currently_active,
            "status": "ACTIVE" if is_currently_active else "DELISTED",
            "aliases": list(set(all_syms) - {all_syms[-1]})
        }

    def get_delisted_equities(self, cutoff_date: date = date(2026, 1, 1)) -> List[Dict[str, Any]]:
        """Returns list of distinct equities that completely ceased trading before cutoff."""
        results = []
        for isin, recs in self.isin_to_records.items():
            max_to = max(r["valid_to"] for r in recs)
            if max_to < cutoff_date:
                latest = sorted(recs, key=lambda x: x["valid_to"])[-1]
                results.append({
                    "isin": isin,
                    "symbol": latest["symbol"],
                    "valid_from": min(r["valid_from"] for r in recs),
                    "delisting_date": max_to,
                    "trading_days": sum(r["trading_days"] for r in recs)
                })
        return results
