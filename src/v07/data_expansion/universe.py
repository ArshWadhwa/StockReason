"""
Tiered NSE Stock Universes for StockReason v0.7.

Defines:
- U16: Current pilot universe (16 stocks across banking, IT, energy, FMCG)
- U100: Nifty 100 (top 100 large-cap stocks)
- U200: Nifty 200 (top 200 large and mid-cap stocks)
- U300: Nifty 300 (top 300 liquid NSE stocks)
- Benchmark and Macro instruments

Survivorship Bias Notice:
-------------------------
NOTE: These constituent lists are constructed based on current/recent index compositions.
Historical backtesting on static current constituents introduces survivorship bias,
as failed, acquired, or demoted companies from past decades are not included in the
historical panel.
Metadata explicitly sets .
"""

from typing import Dict, List, Optional
import pandas as pd

SURVIVORSHIP_SAFE = False
SURVIVORSHIP_BIAS_NOTE = (
    "Constructed from recent NSE index constituents without point-in-time constituent "
    "reconstruction. Subject to survivorship bias when evaluated on historical periods."
)

BENCHMARK_INDICES: List[str] = ["^NSEI", "^NSEBANK", "^INDIAVIX"]
MACRO_TICKERS: List[str] = ["USDINR=X", "CL=F"]

# Base pilot 16 stocks from config/tickers.yaml (exactly 16)
_U16_CONSTITUENTS = [
    {
        "ticker": "RELIANCE.NS",
        "company": "Reliance Industries Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "TCS.NS",
        "company": "Tata Consultancy Services Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "INFY.NS",
        "company": "Infosys Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "HDFCBANK.NS",
        "company": "HDFC Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ICICIBANK.NS",
        "company": "ICICI Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "BHARTIARTL.NS",
        "company": "Bharti Airtel Ltd",
        "sector": "Telecommunication"
    },
    {
        "ticker": "ITC.NS",
        "company": "ITC Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "LT.NS",
        "company": "Larsen & Toubro Ltd",
        "sector": "Construction"
    },
    {
        "ticker": "SBIN.NS",
        "company": "State Bank of India",
        "sector": "Financial Services"
    },
    {
        "ticker": "AXISBANK.NS",
        "company": "Axis Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "KOTAKBANK.NS",
        "company": "Kotak Mahindra Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "INDUSINDBK.NS",
        "company": "IndusInd Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "BANKBARODA.NS",
        "company": "Bank of Baroda",
        "sector": "Financial Services"
    },
    {
        "ticker": "PNB.NS",
        "company": "Punjab National Bank",
        "sector": "Financial Services"
    },
    {
        "ticker": "AUBANK.NS",
        "company": "AU Small Finance Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "IDFCFIRSTB.NS",
        "company": "IDFC First Bank Ltd",
        "sector": "Financial Services"
    }
]

# Additional 84 constituents to form U100 (exactly 100 total)
_ADDITIONAL_U100_CONSTITUENTS = [
    {
        "ticker": "HINDUNILVR.NS",
        "company": "Hindustan Unilever Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "BAJFINANCE.NS",
        "company": "Bajaj Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "BAJAJFINSV.NS",
        "company": "Bajaj Finserv Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "MARUTI.NS",
        "company": "Maruti Suzuki India Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "TATAMOTORS.NS",
        "company": "Tata Motors Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "SUNPHARMA.NS",
        "company": "Sun Pharmaceutical Industries Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "TITAN.NS",
        "company": "Titan Company Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "HCLTECH.NS",
        "company": "HCL Technologies Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "WIPRO.NS",
        "company": "Wipro Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "NTPC.NS",
        "company": "NTPC Ltd",
        "sector": "Power"
    },
    {
        "ticker": "POWERGRID.NS",
        "company": "Power Grid Corporation of India Ltd",
        "sector": "Power"
    },
    {
        "ticker": "ONGC.NS",
        "company": "Oil & Natural Gas Corporation Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "COALINDIA.NS",
        "company": "Coal India Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "TATASTEEL.NS",
        "company": "Tata Steel Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "JSWSTEEL.NS",
        "company": "JSW Steel Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "ADANIENT.NS",
        "company": "Adani Enterprises Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "ADANIPORTS.NS",
        "company": "Adani Ports and Special Economic Zone Ltd",
        "sector": "Services"
    },
    {
        "ticker": "ULTRACEMCO.NS",
        "company": "UltraTech Cement Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "ASIANPAINT.NS",
        "company": "Asian Paints Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "NESTLEIND.NS",
        "company": "Nestle India Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "GRASIM.NS",
        "company": "Grasim Industries Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "BAJAJ-AUTO.NS",
        "company": "Bajaj Auto Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "M&M.NS",
        "company": "Mahindra & Mahindra Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "TECHM.NS",
        "company": "Tech Mahindra Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "CIPLA.NS",
        "company": "Cipla Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "DRREDDY.NS",
        "company": "Dr. Reddy's Laboratories Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "APOLLOHOSP.NS",
        "company": "Apollo Hospitals Enterprise Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "DIVISLAB.NS",
        "company": "Divi's Laboratories Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "EICHERMOT.NS",
        "company": "Eicher Motors Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "HEROMOTOCO.NS",
        "company": "Hero MotoCorp Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "BRITANNIA.NS",
        "company": "Britannia Industries Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "HINDALCO.NS",
        "company": "Hindalco Industries Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "BPCL.NS",
        "company": "Bharat Petroleum Corporation Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "SHRIRAMFIN.NS",
        "company": "Shriram Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "BEL.NS",
        "company": "Bharat Electronics Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "TRENT.NS",
        "company": "Trent Ltd",
        "sector": "Services"
    },
    {
        "ticker": "HAL.NS",
        "company": "Hindustan Aeronautics Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "VBL.NS",
        "company": "Varun Beverages Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "ZOMATO.NS",
        "company": "Zomato Ltd",
        "sector": "Services"
    },
    {
        "ticker": "JIOFIN.NS",
        "company": "Jio Financial Services Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "CHOLAFIN.NS",
        "company": "Cholamandalam Investment and Finance Co Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "VEDL.NS",
        "company": "Vedanta Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "DLF.NS",
        "company": "DLF Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "GODREJCP.NS",
        "company": "Godrej Consumer Products Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "DABUR.NS",
        "company": "Dabur India Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "PIDILITIND.NS",
        "company": "Pidilite Industries Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "SIEMENS.NS",
        "company": "Siemens Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "ABB.NS",
        "company": "ABB India Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "HAVELLS.NS",
        "company": "Havells India Ltd",
        "sector": "Consumer Durables"
    },
    {
        "ticker": "GAIL.NS",
        "company": "GAIL (India) Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "IOC.NS",
        "company": "Indian Oil Corporation Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "AMBUJACEM.NS",
        "company": "Ambuja Cements Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "INDIGO.NS",
        "company": "InterGlobe Aviation Ltd",
        "sector": "Services"
    },
    {
        "ticker": "CANBK.NS",
        "company": "Canara Bank",
        "sector": "Financial Services"
    },
    {
        "ticker": "UNIONBANK.NS",
        "company": "Union Bank of India",
        "sector": "Financial Services"
    },
    {
        "ticker": "SRF.NS",
        "company": "SRF Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "TVSMOTOR.NS",
        "company": "TVS Motor Company Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "MOTHERSON.NS",
        "company": "Samvardhana Motherson International Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "BOSCHLTD.NS",
        "company": "Bosch Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "LODHA.NS",
        "company": "Macrotech Developers Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "POLYCAB.NS",
        "company": "Polycab India Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "PERSISTENT.NS",
        "company": "Persistent Systems Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "LTIM.NS",
        "company": "LTIMindtree Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "MPHASIS.NS",
        "company": "Mphasis Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "COFORGE.NS",
        "company": "Coforge Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "TATAELXSI.NS",
        "company": "Tata Elxsi Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "KPITTECH.NS",
        "company": "KPIT Technologies Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "MAXHEALTH.NS",
        "company": "Max Healthcare Institute Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "MANKIND.NS",
        "company": "Mankind Pharma Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "TORNTPHARM.NS",
        "company": "Torrent Pharmaceuticals Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "LUPIN.NS",
        "company": "Lupin Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "AUROPHARMA.NS",
        "company": "Aurobindo Pharma Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "ALKEM.NS",
        "company": "Alkem Laboratories Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "COLPAL.NS",
        "company": "Colgate-Palmolive (India) Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "MARICO.NS",
        "company": "Marico Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "BERGEPAINT.NS",
        "company": "Berger Paints India Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "PGHH.NS",
        "company": "Procter & Gamble Hygiene and Health Care Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "TATACONSUM.NS",
        "company": "Tata Consumer Products Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "ICICIPRULI.NS",
        "company": "ICICI Prudential Life Insurance Co Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "HDFCLIFE.NS",
        "company": "HDFC Life Insurance Co Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "SBILIFE.NS",
        "company": "SBI Life Insurance Co Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ICICIGI.NS",
        "company": "ICICI Lombard General Insurance Co Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "MUTHOOTFIN.NS",
        "company": "Muthoot Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "PFC.NS",
        "company": "Power Finance Corporation Ltd",
        "sector": "Financial Services"
    }
]

# Additional 100 constituents to form U200 (exactly 200 total)
_ADDITIONAL_U200_CONSTITUENTS = [
    {
        "ticker": "RECLTD.NS",
        "company": "REC Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ADANIGREEN.NS",
        "company": "Adani Green Energy Ltd",
        "sector": "Power"
    },
    {
        "ticker": "ADANIPOWER.NS",
        "company": "Adani Power Ltd",
        "sector": "Power"
    },
    {
        "ticker": "ATGL.NS",
        "company": "Adani Total Gas Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "APLAPOLLO.NS",
        "company": "APL Apollo Tubes Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "ASHOKLEY.NS",
        "company": "Ashok Leyland Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "ASTRAL.NS",
        "company": "Astral Ltd",
        "sector": "Building Materials"
    },
    {
        "ticker": "BALKRISIND.NS",
        "company": "Balkrishna Industries Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "BATAINDIA.NS",
        "company": "Bata India Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "BHARATFORG.NS",
        "company": "Bharat Forge Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "BHEL.NS",
        "company": "Bharat Heavy Electricals Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "BIOCON.NS",
        "company": "Biocon Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "CGPOWER.NS",
        "company": "CG Power and Industrial Solutions Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "CONCOR.NS",
        "company": "Container Corporation of India Ltd",
        "sector": "Services"
    },
    {
        "ticker": "COROMANDEL.NS",
        "company": "Coromandel International Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "CROMPTON.NS",
        "company": "Crompton Greaves Consumer Electricals Ltd",
        "sector": "Consumer Durables"
    },
    {
        "ticker": "CUMMINSIND.NS",
        "company": "Cummins India Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "DEEPAKNTR.NS",
        "company": "Deepak Nitrite Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "DELHIVERY.NS",
        "company": "Delhivery Ltd",
        "sector": "Services"
    },
    {
        "ticker": "ESCORTS.NS",
        "company": "Escorts Kubota Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "EXIDEIND.NS",
        "company": "Exide Industries Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "FEDERALBNK.NS",
        "company": "The Federal Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "FORTIS.NS",
        "company": "Fortis Healthcare Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "GLENMARK.NS",
        "company": "Glenmark Pharmaceuticals Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "GMRINFRA.NS",
        "company": "GMR Airports Infrastructure Ltd",
        "sector": "Services"
    },
    {
        "ticker": "GODREJPROP.NS",
        "company": "Godrej Properties Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "GUJGASLTD.NS",
        "company": "Gujarat Gas Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "HDFCAMC.NS",
        "company": "HDFC Asset Management Company Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "HINDPETRO.NS",
        "company": "Hindustan Petroleum Corporation Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "HINDZINC.NS",
        "company": "Hindustan Zinc Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "IDBI.NS",
        "company": "IDBI Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "INDIANB.NS",
        "company": "Indian Bank",
        "sector": "Financial Services"
    },
    {
        "ticker": "INDHOTEL.NS",
        "company": "The Indian Hotels Company Ltd",
        "sector": "Services"
    },
    {
        "ticker": "IPCALAB.NS",
        "company": "IPCA Laboratories Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "JINDALSTEL.NS",
        "company": "Jindal Steel & Power Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "JSWENERGY.NS",
        "company": "JSW Energy Ltd",
        "sector": "Power"
    },
    {
        "ticker": "JUBLFOOD.NS",
        "company": "Jubilant FoodWorks Ltd",
        "sector": "Services"
    },
    {
        "ticker": "KEI.NS",
        "company": "KEI Industries Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "KIMS.NS",
        "company": "Krishna Institute of Medical Sciences Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "LALPATHLAB.NS",
        "company": "Dr. Lal PathLabs Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "LAURUSLABS.NS",
        "company": "Laurus Labs Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "LICHSGFIN.NS",
        "company": "LIC Housing Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "LTTS.NS",
        "company": "L&T Technology Services Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "MAXFIN.NS",
        "company": "Max Financial Services Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "MFSL.NS",
        "company": "Max Financial Services Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "MRF.NS",
        "company": "MRF Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "NATCOPHARM.NS",
        "company": "Natco Pharma Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "NAVINFLUOR.NS",
        "company": "Navin Fluorine International Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "NHPC.NS",
        "company": "NHPC Ltd",
        "sector": "Power"
    },
    {
        "ticker": "NMDC.NS",
        "company": "NMDC Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "OBEROIRLTY.NS",
        "company": "Oberoi Realty Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "OFSS.NS",
        "company": "Oracle Financial Services Software Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "PAGEIND.NS",
        "company": "Page Industries Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "PETRONET.NS",
        "company": "Petronet LNG Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "PHOENIXLTD.NS",
        "company": "The Phoenix Mills Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "PIIND.NS",
        "company": "PI Industries Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "POONAWALLA.NS",
        "company": "Poonawalla Fincorp Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "PRESTIGE.NS",
        "company": "Prestige Estates Projects Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "RAMCOCEM.NS",
        "company": "The Ramco Cements Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "SAIL.NS",
        "company": "Steel Authority of India Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "SBICARD.NS",
        "company": "SBI Cards and Payment Services Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "SCHAEFFLER.NS",
        "company": "Schaeffler India Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "SOLARINDS.NS",
        "company": "Solar Industries India Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "SONACOMS.NS",
        "company": "Sona BLW Precision Forgings Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "STARHEALTH.NS",
        "company": "Star Health and Allied Insurance Co Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "SUNDARMFIN.NS",
        "company": "Sundaram Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "SUNTV.NS",
        "company": "Sun TV Network Ltd",
        "sector": "Media"
    },
    {
        "ticker": "SUPREMEIND.NS",
        "company": "Supreme Industries Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "SYNGENE.NS",
        "company": "Syngene International Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "TATACHEM.NS",
        "company": "Tata Chemicals Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "TATACOMM.NS",
        "company": "Tata Communications Ltd",
        "sector": "Telecommunication"
    },
    {
        "ticker": "TATAPOWER.NS",
        "company": "Tata Power Company Ltd",
        "sector": "Power"
    },
    {
        "ticker": "TORNTPOWER.NS",
        "company": "Torrent Power Ltd",
        "sector": "Power"
    },
    {
        "ticker": "UBL.NS",
        "company": "United Breweries Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "UPL.NS",
        "company": "UPL Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "VOLTAS.NS",
        "company": "Voltas Ltd",
        "sector": "Consumer Durables"
    },
    {
        "ticker": "WHIRLPOOL.NS",
        "company": "Whirlpool of India Ltd",
        "sector": "Consumer Durables"
    },
    {
        "ticker": "YESBANK.NS",
        "company": "Yes Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ZEEL.NS",
        "company": "Zee Entertainment Enterprises Ltd",
        "sector": "Media"
    },
    {
        "ticker": "ZYDUSLIFE.NS",
        "company": "Zydus Lifesciences Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "CENTRALBK.NS",
        "company": "Central Bank of India",
        "sector": "Financial Services"
    },
    {
        "ticker": "BANKINDIA.NS",
        "company": "Bank of India",
        "sector": "Financial Services"
    },
    {
        "ticker": "IOB.NS",
        "company": "Indian Overseas Bank",
        "sector": "Financial Services"
    },
    {
        "ticker": "UCOBANK.NS",
        "company": "UCO Bank",
        "sector": "Financial Services"
    },
    {
        "ticker": "MAHABANK.NS",
        "company": "Bank of Maharashtra",
        "sector": "Financial Services"
    },
    {
        "ticker": "HUDCO.NS",
        "company": "Housing and Urban Development Corporation Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "IRFC.NS",
        "company": "Indian Railway Finance Corporation Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "RVNL.NS",
        "company": "Rail Vikas Nigam Ltd",
        "sector": "Construction"
    },
    {
        "ticker": "IRCTC.NS",
        "company": "Indian Railway Catering and Tourism Corp Ltd",
        "sector": "Services"
    },
    {
        "ticker": "SJVN.NS",
        "company": "SJVN Ltd",
        "sector": "Power"
    },
    {
        "ticker": "IREDA.NS",
        "company": "Indian Renewable Energy Development Agency Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "SUZLON.NS",
        "company": "Suzlon Energy Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "NBCC.NS",
        "company": "NBCC (India) Ltd",
        "sector": "Construction"
    },
    {
        "ticker": "MAZDOCK.NS",
        "company": "Mazagon Dock Shipbuilders Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "COCHINSHIP.NS",
        "company": "Cochin Shipyard Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "BDL.NS",
        "company": "Bharat Dynamics Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "FACT.NS",
        "company": "Fertilisers and Chemicals Travancore Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "KALYANKJIL.NS",
        "company": "Kalyan Jewellers India Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "LICI.NS",
        "company": "Life Insurance Corporation of India",
        "sector": "Financial Services"
    },
    {
        "ticker": "MOTILALOFS.NS",
        "company": "Motilal Oswal Financial Services Ltd",
        "sector": "Financial Services"
    }
]

# Additional 100 constituents to form U300 (exactly 300 total)
_ADDITIONAL_U300_CONSTITUENTS = [
    {
        "ticker": "ABCAPITAL.NS",
        "company": "Aditya Birla Capital Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ABFRL.NS",
        "company": "Aditya Birla Fashion and Retail Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "AJANTPHARM.NS",
        "company": "Ajanta Pharma Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "ALOKINDS.NS",
        "company": "Alok Industries Ltd",
        "sector": "Textiles"
    },
    {
        "ticker": "AMARAJABAT.NS",
        "company": "Amara Raja Energy & Mobility Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "ANGELONE.NS",
        "company": "Angel One Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ANURAS.NS",
        "company": "Anupam Rasayan India Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "APARINDS.NS",
        "company": "Apar Industries Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "ASIANENE.NS",
        "company": "Asian Energy Services Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "ASTERDM.NS",
        "company": "Aster DM Healthcare Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "ATUL.NS",
        "company": "Atul Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "AVANTIFEED.NS",
        "company": "Avanti Feeds Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "BASF.NS",
        "company": "BASF India Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "BEML.NS",
        "company": "BEML Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "BLUEDART.NS",
        "company": "Blue Dart Express Ltd",
        "sector": "Services"
    },
    {
        "ticker": "BLUESTARCO.NS",
        "company": "Blue Star Ltd",
        "sector": "Consumer Durables"
    },
    {
        "ticker": "BRIGADE.NS",
        "company": "Brigade Enterprises Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "BSOFT.NS",
        "company": "Birlasoft Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "CAMPUS.NS",
        "company": "Campus Activewear Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "CANFINHOME.NS",
        "company": "Can Fin Homes Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "CASTROLIND.NS",
        "company": "Castrol India Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "CDSL.NS",
        "company": "Central Depository Services (India) Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "CENTURYTEX.NS",
        "company": "Century Textiles & Industries Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "CESC.NS",
        "company": "CESC Ltd",
        "sector": "Power"
    },
    {
        "ticker": "CHAMBLFERT.NS",
        "company": "Chambal Fertilisers and Chemicals Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "CLEAN.NS",
        "company": "Clean Science and Technology Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "CREDITACC.NS",
        "company": "CreditAccess Grameen Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "CYIENT.NS",
        "company": "Cyient Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "DATAPATTNS.NS",
        "company": "Data Patterns (India) Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "DCMSHRIRAM.NS",
        "company": "DCM Shriram Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "DEEPAKFERT.NS",
        "company": "Deepak Fertilisers and Petrochemicals Corp Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "DEVYANI.NS",
        "company": "Devyani International Ltd",
        "sector": "Services"
    },
    {
        "ticker": "EIDPARRY.NS",
        "company": "E.I.D. - Parry (India) Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "EIHOTEL.NS",
        "company": "EIH Ltd",
        "sector": "Services"
    },
    {
        "ticker": "ELGIEQUIP.NS",
        "company": "Elgi Equipments Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "EMAMILTD.NS",
        "company": "Emami Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "ENDURANCE.NS",
        "company": "Endurance Technologies Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "ENGINERSIN.NS",
        "company": "Engineers India Ltd",
        "sector": "Construction"
    },
    {
        "ticker": "EQUITASBNK.NS",
        "company": "Equitas Small Finance Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "ERIS.NS",
        "company": "Eris Lifesciences Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "FINCABLES.NS",
        "company": "Finolex Cables Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "FINPIPE.NS",
        "company": "Finolex Industries Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "FSL.NS",
        "company": "Firstsource Solutions Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "GLS.NS",
        "company": "Glenmark Life Sciences Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "GNFC.NS",
        "company": "Gujarat Narmada Valley Fertilizers & Chem Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "GODFRYPHLP.NS",
        "company": "Godfrey Phillips India Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "GRANULES.NS",
        "company": "Granules India Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "GRAPHITE.NS",
        "company": "Graphite India Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "GRINDWELL.NS",
        "company": "Grindwell Norton Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "GSFC.NS",
        "company": "Gujarat State Fertilizers & Chemicals Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "GSPL.NS",
        "company": "Gujarat State Petronet Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "HAPPSTMNDS.NS",
        "company": "Happiest Minds Technologies Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "HEG.NS",
        "company": "HEG Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "HFCL.NS",
        "company": "HFCL Ltd",
        "sector": "Telecommunication"
    },
    {
        "ticker": "HIKAL.NS",
        "company": "Hikal Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "HINDCOPPER.NS",
        "company": "Hindustan Copper Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "HOMESFY.NS",
        "company": "Homesfy Realty Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "HONAUT.NS",
        "company": "Honeywell Automation India Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "IBREALEST.NS",
        "company": "Indiabulls Real Estate Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "IEX.NS",
        "company": "Indian Energy Exchange Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "IIFL.NS",
        "company": "IIFL Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "INDIGOPNTS.NS",
        "company": "Indigo Paints Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "INTELLECT.NS",
        "company": "Intellect Design Arena Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "JBCHEPHARM.NS",
        "company": "J.B. Chemicals & Pharmaceuticals Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "JKCEMENT.NS",
        "company": "JK Cement Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "JKLAKSHMI.NS",
        "company": "JK Lakshmi Cement Ltd",
        "sector": "Construction Materials"
    },
    {
        "ticker": "JKTYRE.NS",
        "company": "JK Tyre & Industries Ltd",
        "sector": "Automobile"
    },
    {
        "ticker": "JSL.NS",
        "company": "Jindal Stainless Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "JUSTDIAL.NS",
        "company": "Just Dial Ltd",
        "sector": "Services"
    },
    {
        "ticker": "JYOTHYLAB.NS",
        "company": "Jyothy Labs Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "KAJARIACER.NS",
        "company": "Kajaria Ceramics Ltd",
        "sector": "Building Materials"
    },
    {
        "ticker": "KARURVYSYA.NS",
        "company": "The Karur Vysya Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "KEC.NS",
        "company": "KEC International Ltd",
        "sector": "Construction"
    },
    {
        "ticker": "KFINTECH.NS",
        "company": "KFin Technologies Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "KRBL.NS",
        "company": "KRBL Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "KSB.NS",
        "company": "KSB Ltd",
        "sector": "Capital Goods"
    },
    {
        "ticker": "LATENTVIEW.NS",
        "company": "Latent View Analytics Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "LEMONTREE.NS",
        "company": "Lemon Tree Hotels Ltd",
        "sector": "Services"
    },
    {
        "ticker": "MANAPPURAM.NS",
        "company": "Manappuram Finance Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "MAPMYINDIA.NS",
        "company": "C.E. Info Systems Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "METROPOLIS.NS",
        "company": "Metropolis Healthcare Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "MGL.NS",
        "company": "Mahanagar Gas Ltd",
        "sector": "Energy"
    },
    {
        "ticker": "NATIONALUM.NS",
        "company": "National Aluminium Co Ltd",
        "sector": "Metals & Mining"
    },
    {
        "ticker": "NAZARA.NS",
        "company": "Nazara Technologies Ltd",
        "sector": "Services"
    },
    {
        "ticker": "NLCINDIA.NS",
        "company": "NLC India Ltd",
        "sector": "Power"
    },
    {
        "ticker": "PVRINOX.NS",
        "company": "PVR INOX Ltd",
        "sector": "Media"
    },
    {
        "ticker": "RADICO.NS",
        "company": "Radico Khaitan Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "RAILTEL.NS",
        "company": "RailTel Corporation of India Ltd",
        "sector": "Telecommunication"
    },
    {
        "ticker": "RBLBANK.NS",
        "company": "RBL Bank Ltd",
        "sector": "Financial Services"
    },
    {
        "ticker": "RITES.NS",
        "company": "RITES Ltd",
        "sector": "Services"
    },
    {
        "ticker": "ROUTE.NS",
        "company": "Route Mobile Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "SAFARI.NS",
        "company": "Safari Industries (India) Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "SHOPERSTOP.NS",
        "company": "Shoppers Stop Ltd",
        "sector": "Services"
    },
    {
        "ticker": "SOBHA.NS",
        "company": "Sobha Ltd",
        "sector": "Realty"
    },
    {
        "ticker": "STAR.NS",
        "company": "Strides Pharma Science Ltd",
        "sector": "Healthcare"
    },
    {
        "ticker": "SUMICHEM.NS",
        "company": "Sumitomo Chemical India Ltd",
        "sector": "Chemicals"
    },
    {
        "ticker": "TANLA.NS",
        "company": "Tanla Platforms Ltd",
        "sector": "Information Technology"
    },
    {
        "ticker": "TRIDENT.NS",
        "company": "Trident Ltd",
        "sector": "Textiles"
    },
    {
        "ticker": "VIPIND.NS",
        "company": "VIP Industries Ltd",
        "sector": "Consumer Goods"
    },
    {
        "ticker": "MAHINDCIE.NS",
        "company": "CIE Automotive India Ltd",
        "sector": "Automobile"
    }
]


def _build_tier_registry() -> Dict[str, List[Dict[str, str]]]:
    """Builds the constituent registry for all tiers ensuring hierarchical nesting."""
    u16 = [{**item, "tier": "U16"} for item in _U16_CONSTITUENTS]
    u100 = u16 + [{**item, "tier": "U100"} for item in _ADDITIONAL_U100_CONSTITUENTS]
    u200 = u100 + [{**item, "tier": "U200"} for item in _ADDITIONAL_U200_CONSTITUENTS]
    u300 = u200 + [{**item, "tier": "U300"} for item in _ADDITIONAL_U300_CONSTITUENTS]

    return {
        "U16": u16,
        "U100": u100,
        "U200": u200,
        "U300": u300,
    }


_REGISTRY = _build_tier_registry()
UNIVERSE_TIERS: List[str] = list(_REGISTRY.keys())


def get_universe(tier: str = "U16") -> List[str]:
    """
    Returns list of stock tickers (e.g. ['RELIANCE.NS', 'TCS.NS', ...]) for a given tier.

    Parameters
    ----------
    tier : str
        One of 'U16', 'U100', 'U200', 'U300'.

    Returns
    -------
    List[str]
        Tickers in the requested tier.
    """
    tier_upper = tier.upper()
    if tier_upper not in _REGISTRY:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {UNIVERSE_TIERS}")
    return [c["ticker"] for c in _REGISTRY[tier_upper]]


def get_universe_details(tier: str = "U16") -> pd.DataFrame:
    """
    Returns DataFrame of constituents for a given tier with ticker, company, sector, and tier.

    Parameters
    ----------
    tier : str
        One of 'U16', 'U100', 'U200', 'U300'.

    Returns
    -------
    pd.DataFrame
        Columns: ['ticker', 'company', 'sector', 'tier']
    """
    tier_upper = tier.upper()
    if tier_upper not in _REGISTRY:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {UNIVERSE_TIERS}")
    df = pd.DataFrame(_REGISTRY[tier_upper])
    df.attrs["survivorship_safe"] = SURVIVORSHIP_SAFE
    df.attrs["survivorship_bias_note"] = SURVIVORSHIP_BIAS_NOTE
    return df


def get_universe_metadata(tier: str = "U16") -> Dict:
    """
    Returns metadata dictionary for a given tier, including survivorship bias warnings.
    """
    tier_upper = tier.upper()
    if tier_upper not in _REGISTRY:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {UNIVERSE_TIERS}")

    constituents = _REGISTRY[tier_upper]
    sectors = sorted(list({c["sector"] for c in constituents}))
    return {
        "tier": tier_upper,
        "count": len(constituents),
        "survivorship_safe": SURVIVORSHIP_SAFE,
        "survivorship_bias_note": SURVIVORSHIP_BIAS_NOTE,
        "sectors": sectors,
        "num_sectors": len(sectors),
        "benchmarks": BENCHMARK_INDICES,
        "macro_tickers": MACRO_TICKERS,
    }


def get_all_tickers(include_benchmarks: bool = True, include_macro: bool = True) -> List[str]:
    """
    Returns all unique tickers in U300 plus optional benchmark indices and macro instruments.
    """
    tickers = get_universe("U300")
    if include_benchmarks:
        tickers = tickers + [b for b in BENCHMARK_INDICES if b not in tickers]
    if include_macro:
        tickers = tickers + [m for m in MACRO_TICKERS if m not in tickers]
    return tickers
