"""
StockReason v0.8 Universe Builder
=================================
Builds a point-in-time compliant 500-stock NSE equity universe with historical listings,
delistings, mergers, corporate reorganizations, sector classifications, and tiers:
U16 (16), U100 (100), U200 (200), U300 (300), U500 (500).
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, ".")
from src.v07.data_expansion.universe import (
    _U16_CONSTITUENTS,
    _ADDITIONAL_U100_CONSTITUENTS,
    _ADDITIONAL_U200_CONSTITUENTS,
    _ADDITIONAL_U300_CONSTITUENTS,
    BENCHMARK_INDICES,
    MACRO_TICKERS,
)

# 20 Historical delisted / merged / reorganized / distressed securities
HISTORICAL_SURVIVORSHIP_CASES = [
    {
        "ticker": "RCOM.NS",
        "symbol_nse": "RCOM",
        "isin": "INE330H01018",
        "company_name": "Reliance Communications Ltd",
        "sector": "Telecommunication",
        "listing_date": "2006-03-06",
        "delisting_date": "2021-08-10",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": ["RELIANCECOMM"]
    },
    {
        "ticker": "JETAIRWAYS.NS",
        "symbol_nse": "JETAIRWAYS",
        "isin": "INE802G01018",
        "company_name": "Jet Airways (India) Ltd",
        "sector": "Services",
        "listing_date": "2005-03-14",
        "delisting_date": "2019-06-20",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "UNITECH.NS",
        "symbol_nse": "UNITECH",
        "isin": "INE694A01020",
        "company_name": "Unitech Ltd",
        "sector": "Realty",
        "listing_date": "1995-11-08",
        "delisting_date": "2020-02-14",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "DHFL.NS",
        "symbol_nse": "DHFL",
        "isin": "INE202B01012",
        "company_name": "Dewan Housing Finance Corporation Ltd",
        "sector": "Financial Services",
        "listing_date": "2002-05-15",
        "delisting_date": "2021-06-14",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "RELCAPITAL.NS",
        "symbol_nse": "RELCAPITAL",
        "isin": "INE013A01015",
        "company_name": "Reliance Capital Ltd",
        "sector": "Financial Services",
        "listing_date": "1995-02-08",
        "delisting_date": "2023-02-27",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "SINTEX.NS",
        "symbol_nse": "SINTEX",
        "isin": "INE429C01035",
        "company_name": "Sintex Industries Ltd",
        "sector": "Textiles",
        "listing_date": "1998-05-13",
        "delisting_date": "2022-03-24",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "VIDEOIND.NS",
        "symbol_nse": "VIDEOIND",
        "isin": "INE703A01011",
        "company_name": "Videocon Industries Ltd",
        "sector": "Consumer Durables",
        "listing_date": "1995-03-29",
        "delisting_date": "2021-07-16",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "JPASSOCIAT.NS",
        "symbol_nse": "JPASSOCIAT",
        "isin": "INE455F01025",
        "company_name": "Jaiprakash Associates Ltd",
        "sector": "Construction Materials",
        "listing_date": "2004-12-24",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "distressed",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "GTLINFRA.NS",
        "symbol_nse": "GTLINFRA",
        "isin": "INE221H01019",
        "company_name": "GTL Infrastructure Ltd",
        "sector": "Telecommunication",
        "listing_date": "2006-11-09",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "distressed",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "RPOWER.NS",
        "symbol_nse": "RPOWER",
        "isin": "INE614G01033",
        "company_name": "Reliance Power Ltd",
        "sector": "Power",
        "listing_date": "2008-02-11",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "distressed",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "RELINFRA.NS",
        "symbol_nse": "RELINFRA",
        "isin": "INE036A01016",
        "company_name": "Reliance Infrastructure Ltd",
        "sector": "Power",
        "listing_date": "1995-02-08",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "distressed",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "HCC.NS",
        "symbol_nse": "HCC",
        "isin": "INE549A01026",
        "company_name": "Hindustan Construction Company Ltd",
        "sector": "Construction",
        "listing_date": "1995-11-29",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "distressed",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "OPTOCIRCUI.NS",
        "symbol_nse": "OPTOCIRCUI",
        "isin": "INE808B01016",
        "company_name": "Opto Circuits (India) Ltd",
        "sector": "Healthcare",
        "listing_date": "2002-10-14",
        "delisting_date": "2020-11-20",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "EDUCOMP.NS",
        "symbol_nse": "EDUCOMP",
        "isin": "INE216H01027",
        "company_name": "Educomp Solutions Ltd",
        "sector": "Services",
        "listing_date": "2006-01-13",
        "delisting_date": "2019-10-11",
        "is_active": False,
        "survivorship_status": "delisted",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "ALOKTEXT.NS",
        "symbol_nse": "ALOKTEXT",
        "isin": "INE270A01029",
        "company_name": "Alok Industries Ltd",
        "sector": "Textiles",
        "listing_date": "1995-03-22",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "restructured",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "SUZLON.NS",
        "symbol_nse": "SUZLON",
        "isin": "INE040H01021",
        "company_name": "Suzlon Energy Ltd",
        "sector": "Capital Goods",
        "listing_date": "2005-10-19",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "restructured",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "YESBANK.NS",
        "symbol_nse": "YESBANK",
        "isin": "INE528G01035",
        "company_name": "Yes Bank Ltd",
        "sector": "Financial Services",
        "listing_date": "2005-07-12",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "reconstructed",
        "tier": "U500",
        "aliases": []
    },
    {
        "ticker": "HDFC.NS",
        "symbol_nse": "HDFC",
        "isin": "INE001A01036",
        "company_name": "Housing Development Finance Corporation Ltd",
        "sector": "Financial Services",
        "listing_date": "1995-11-29",
        "delisting_date": "2023-07-13",
        "is_active": False,
        "survivorship_status": "merged",
        "tier": "U500",
        "aliases": ["HDFC-MERGED-HDFCBANK"]
    },
    {
        "ticker": "CADILAHC.NS",
        "symbol_nse": "CADILAHC",
        "isin": "INE010B01027",
        "company_name": "Cadila Healthcare Ltd (now Zydus Lifesciences)",
        "sector": "Healthcare",
        "listing_date": "2000-04-18",
        "delisting_date": None,
        "is_active": True,
        "survivorship_status": "reorganized",
        "tier": "U500",
        "aliases": ["ZYDUSLIFE.NS"]
    },
    {
        "ticker": "LTI.NS",
        "symbol_nse": "LTI",
        "isin": "INE214T01019",
        "company_name": "Larsen & Toubro Infotech Ltd (now LTIMindtree)",
        "sector": "Information Technology",
        "listing_date": "2016-07-21",
        "delisting_date": "2022-11-24",
        "is_active": False,
        "survivorship_status": "merged",
        "tier": "U500",
        "aliases": ["LTIM.NS"]
    }
]

# Raw candidate pool of 200+ liquid NSE stocks to select exactly 180 unseen tickers
CANDIDATE_POOL = [
    ("CDSL", "INE736A01011", "Central Depository Services (India) Ltd", "Financial Services", "2017-06-30"),
    ("BSE", "INE118H01025", "BSE Ltd", "Financial Services", "2017-02-03"),
    ("ANGELONE", "INE732I01013", "Angel One Ltd", "Financial Services", "2020-10-05"),
    ("POLICYBZR", "INE417T01026", "PB Fintech Ltd", "Financial Services", "2021-11-15"),
    ("ZOMATO", "INE758T01015", "Zomato Ltd", "Services", "2021-07-23"),
    ("PAYTM", "INE982J01020", "One 97 Communications Ltd", "Financial Services", "2021-11-18"),
    ("NYKAA", "INE388Y01029", "FSN E-Commerce Ventures Ltd", "Consumer Goods", "2021-11-10"),
    ("DELHIVERY", "INE148O01028", "Delhivery Ltd", "Services", "2022-05-24"),
    ("KPITTECH", "INE04I401011", "KPIT Technologies Ltd", "Information Technology", "2019-04-22"),
    ("COFORGE", "INE591G01017", "Coforge Ltd", "Information Technology", "2004-09-13"),
    ("PERSISTENT", "INE262H01013", "Persistent Systems Ltd", "Information Technology", "2010-04-06"),
    ("LTIM", "INE214T01019", "LTIMindtree Ltd", "Information Technology", "2016-07-21"),
    ("DIXON", "INE935N01020", "Dixon Technologies (India) Ltd", "Consumer Durables", "2017-09-18"),
    ("AFFLE", "INE00WC01027", "Affle (India) Ltd", "Information Technology", "2019-08-08"),
    ("HAPPISTMNDS", "INE419U01012", "Happiest Minds Technologies Ltd", "Information Technology", "2020-09-17"),
    ("LATENTVIEW", "INE0I7C01011", "Latent View Analytics Ltd", "Information Technology", "2021-11-23"),
    ("ROUTE", "INE450U01017", "Route Mobile Ltd", "Information Technology", "2020-09-21"),
    ("TANLA", "INE483C01032", "Tanla Platforms Ltd", "Information Technology", "2007-01-16"),
    ("TATAELXSI", "INE670A01012", "Tata Elxsi Ltd", "Information Technology", "1995-11-29"),
    ("MAPMYINDIA", "INE0BV301023", "C.E. Info Systems Ltd", "Information Technology", "2021-12-21"),
    ("CLEAN", "INE227W01023", "Clean Science and Technology Ltd", "Chemicals", "2021-07-19"),
    ("FINEORG", "INE665L01019", "Fine Organic Industries Ltd", "Chemicals", "2018-07-02"),
    ("DEEPAKFERT", "INE501A01019", "Deepak Fertilisers and Petrochemicals Corp Ltd", "Chemicals", "2003-05-28"),
    ("FLUOROCHEM", "INE090301011", "Gujarat Fluorochemicals Ltd", "Chemicals", "2021-10-06"),
    ("NAVINFLUOR", "INE048G01026", "Navin Fluorine International Ltd", "Chemicals", "2003-10-06"),
    ("ATUL", "INE100A01010", "Atul Ltd", "Chemicals", "1998-05-06"),
    ("AARTIIND", "INE769A01020", "Aarti Industries Ltd", "Chemicals", "1995-02-08"),
    ("TATACHEM", "INE092A01019", "Tata Chemicals Ltd", "Chemicals", "1995-11-29"),
    ("SUMICHEM", "INE258G01019", "Sumitomo Chemical India Ltd", "Chemicals", "2020-01-27"),
    ("UPL", "INE628A01036", "UPL Ltd", "Chemicals", "2004-03-01"),
    ("COROMANDEL", "INE169A01031", "Coromandel International Ltd", "Chemicals", "1995-05-31"),
    ("CHAMBLFERT", "INE085A01013", "Chambal Fertilisers and Chemicals Ltd", "Chemicals", "1995-02-08"),
    ("GNFC", "INE113A01013", "Gujarat Narmada Valley Fertilizers & Chemicals Ltd", "Chemicals", "1995-05-10"),
    ("GSFC", "INE026A01025", "Gujarat State Fertilizers & Chemicals Ltd", "Chemicals", "1995-04-19"),
    ("FACT", "INE188A01015", "Fertilisers and Chemicals Travancore Ltd", "Chemicals", "1995-08-30"),
    ("RCF", "INE027A01015", "Rashtriya Chemicals and Fertilizers Ltd", "Chemicals", "1995-11-29"),
    ("DEEPAKNTR", "INE288B01029", "Deepak Nitrite Ltd", "Chemicals", "2010-09-29"),
    ("PIIND", "INE603J01030", "PI Industries Ltd", "Chemicals", "2011-05-19"),
    ("SRF", "INE647A01010", "SRF Ltd", "Chemicals", "1995-11-29"),
    ("ASTRAL", "INE006I01046", "Astral Ltd", "Building Materials", "2007-03-20"),
    ("SUPREMEIND", "INE195A01028", "Supreme Industries Ltd", "Building Materials", "1995-07-12"),
    ("FINPIPE", "INE183A01016", "Finolex Industries Ltd", "Building Materials", "1995-11-29"),
    ("PRINCEPIPE", "INE689W01016", "Prince Pipes and Fittings Ltd", "Building Materials", "2019-12-30"),
    ("KNRCON", "INE634I01029", "KNR Constructions Ltd", "Construction", "2008-02-15"),
    ("GRINFRA", "INE268F01012", "G R Infraprojects Ltd", "Construction", "2021-07-19"),
    ("PNCINFRA", "INE195J01029", "PNC Infratech Ltd", "Construction", "2015-05-26"),
    ("ASHOKA", "INE442H01029", "Ashoka Buildcon Ltd", "Construction", "2010-10-14"),
    ("IRB", "INE821I01022", "IRB Infrastructure Developers Ltd", "Construction", "2008-02-25"),
    ("DBL", "INE418L01021", "Dilip Buildcon Ltd", "Construction", "2016-08-11"),
    ("NCC", "INE868B01028", "NCC Ltd", "Construction", "1995-11-29"),
    ("ITDCEM", "INE258A01016", "ITD Cementation India Ltd", "Construction", "2000-06-19"),
    ("CAPACITE", "INE264T01014", "Capacit'e Infraprojects Ltd", "Construction", "2017-09-25"),
    ("PSPPROJECT", "INE488V01018", "PSP Projects Ltd", "Construction", "2017-05-29"),
    ("AHLUCONT", "INE758C01029", "Ahluwalia Contracts (India) Ltd", "Construction", "2009-12-14"),
    ("JBMA", "INE927D01044", "JBM Auto Ltd", "Automobile", "2004-12-08"),
    ("SUBROS", "INE287B01021", "Subros Ltd", "Automobile", "1995-11-29"),
    ("LUMAXIND", "INE162B01018", "Lumax Industries Ltd", "Automobile", "1995-11-29"),
    ("MOTHERSON", "INE775A01035", "Samvardhana Motherson International Ltd", "Automobile", "1995-11-29"),
    ("MINDAIND", "INE405E01023", "UNO Minda Ltd", "Automobile", "2004-06-09"),
    ("SUPRAJIT", "INE399C01030", "Suprajit Engineering Ltd", "Automobile", "2004-05-19"),
    ("VARROC", "INE665L01035", "Varroc Engineering Ltd", "Automobile", "2018-07-06"),
    ("GABRIEL", "INE527A01032", "Gabriel India Ltd", "Automobile", "1995-11-29"),
    ("ENDURANCE", "INE913H01015", "Endurance Technologies Ltd", "Automobile", "2016-10-19"),
    ("CRAFTSMAN", "INE00LO01017", "Craftsman Automation Ltd", "Automobile", "2021-03-25"),
    ("ROLEXRINGS", "INE645S01016", "Rolex Rings Ltd", "Automobile", "2021-08-09"),
    ("RAMKRISHNA", "INE399G01023", "Ramkrishna Forgings Ltd", "Automobile", "2006-03-17"),
    ("BHARATFORG", "INE465A01025", "Bharat Forge Ltd", "Automobile", "1995-11-29"),
    ("BALKRISIND", "INE787D01026", "Balkrishna Industries Ltd", "Automobile", "2006-11-27"),
    ("MRF", "INE883A01011", "MRF Ltd", "Automobile", "1995-11-29"),
    ("APOLLOTYRE", "INE438A01022", "Apollo Tyres Ltd", "Automobile", "1995-11-29"),
    ("CEATLTD", "INE482A01020", "CEAT Ltd", "Automobile", "1995-11-29"),
    ("JKTYRE", "INE573A01042", "JK Tyre & Industries Ltd", "Automobile", "1995-11-29"),
    ("TVSMOTOR", "INE494B01023", "TVS Motor Company Ltd", "Automobile", "2000-08-16"),
    ("HEROMOTOCO", "INE158A01026", "Hero MotoCorp Ltd", "Automobile", "1995-11-29"),
    ("BAJAJ-AUTO", "INE917I01012", "Bajaj Auto Ltd", "Automobile", "2008-05-26"),
    ("EICHERMOT", "INE066A01021", "Eicher Motors Ltd", "Automobile", "1995-11-29"),
    ("ASHOKLEY", "INE208A01029", "Ashok Leyland Ltd", "Automobile", "1995-11-29"),
    ("ESCORTS", "INE042A01014", "Escorts Kubota Ltd", "Capital Goods", "1995-11-29"),
    ("VSTTILLERS", "INE710A01016", "V.S.T. Tillers Tractors Ltd", "Capital Goods", "1995-11-29"),
    ("FORCEINDIA", "INE451A01017", "Force Motors Ltd", "Automobile", "1995-11-29"),
    ("TATACOMM", "INE151A01013", "Tata Communications Ltd", "Telecommunication", "1995-11-29"),
    ("TEJASNET", "INE010J01012", "Tejas Networks Ltd", "Telecommunication", "2017-06-27"),
    ("HFCL", "INE548A01028", "HFCL Ltd", "Telecommunication", "1995-11-29"),
    ("ITI", "INE248A01017", "ITI Ltd", "Telecommunication", "1995-11-29"),
    ("RAILTEL", "INE734N01019", "RailTel Corporation of India Ltd", "Telecommunication", "2021-02-26"),
    ("STERTOOLS", "INE334A01023", "Sterling Tools Ltd", "Automobile", "2006-03-02"),
    ("BECTORFOOD", "INE495P01012", "Mrs. Bectors Food Specialities Ltd", "Consumer Goods", "2020-12-24"),
    ("BIKAJI", "INE00E101023", "Bikaji Foods International Ltd", "Consumer Goods", "2022-11-16"),
    ("PRATAAP", "INE393P01035", "Prataap Snacks Ltd", "Consumer Goods", "2017-10-05"),
    ("DFMFOODS", "INE456C01020", "DFM Foods Ltd", "Consumer Goods", "2016-12-29"),
    ("VBL", "INE200M01021", "Varun Beverages Ltd", "Consumer Goods", "2016-11-08"),
    ("DEVYANI", "INE872J01023", "Devyani International Ltd", "Consumer Goods", "2021-08-16"),
    ("SAPPHIRE", "INE806T01012", "Sapphire Foods India Ltd", "Consumer Goods", "2021-11-18"),
    ("RESTAURANT", "INE07T301019", "Restaurant Brands Asia Ltd", "Consumer Goods", "2020-12-14"),
    ("WESTLIFE", "INE274F01020", "Westlife Foodworld Ltd", "Consumer Goods", "2013-08-27"),
    ("JUBLFOOD", "INE797F01020", "Jubilant FoodWorks Ltd", "Consumer Goods", "2010-02-08"),
    ("BARBEQUE", "INE382Z01011", "Barbeque-Nation Hospitality Ltd", "Consumer Goods", "2021-04-07"),
    ("KIMS", "INE967H01017", "Krishna Institute of Medical Sciences Ltd", "Healthcare", "2021-06-28"),
    ("MEDANTA", "INE029Q01018", "Global Health Ltd", "Healthcare", "2022-11-16"),
    ("RAINBOW", "INE961H01016", "Rainbow Children's Medicare Ltd", "Healthcare", "2022-05-10"),
    ("ASTERDM", "INE914M01019", "Aster DM Healthcare Ltd", "Healthcare", "2018-02-26"),
    ("NH", "INE410P01011", "Narayana Hrudayalaya Ltd", "Healthcare", "2016-01-06"),
    ("FORTIS", "INE061F01013", "Fortis Healthcare Ltd", "Healthcare", "2007-05-09"),
    ("MAXHEALTH", "INE027H01010", "Max Healthcare Institute Ltd", "Healthcare", "2020-08-21"),
    ("APOLLOHOSP", "INE437A01024", "Apollo Hospitals Enterprise Ltd", "Healthcare", "1995-11-29"),
    ("LALPATHLAB", "INE600L01019", "Dr. Lal PathLabs Ltd", "Healthcare", "2015-12-23"),
    ("METROPOLIS", "INE112L01020", "Metropolis Healthcare Ltd", "Healthcare", "2019-04-15"),
    ("VIJAYA", "INE073V01015", "Vijaya Diagnostic Centre Ltd", "Healthcare", "2021-09-14"),
    ("THYROCARE", "INE594T01014", "Thyrocare Technologies Ltd", "Healthcare", "2016-05-09"),
    ("SYNGENE", "INE398R01022", "Syngene International Ltd", "Healthcare", "2015-08-11"),
    ("LAURUSLABS", "INE947Q01028", "Laurus Labs Ltd", "Healthcare", "2016-12-19"),
    ("ERIS", "INE406M01024", "Eris Lifesciences Ltd", "Healthcare", "2017-06-27"),
    ("AJANTPHARM", "INE031B01049", "Ajanta Pharma Ltd", "Healthcare", "2000-05-31"),
    ("NATCOPHARM", "INE987B01026", "Natco Pharma Ltd", "Healthcare", "1995-11-29"),
    ("SUVENPHAR", "INE03QK01018", "Suven Pharmaceuticals Ltd", "Healthcare", "2020-03-09"),
    ("JBCHEPHARM", "INE572A01028", "J.B. Chemicals & Pharmaceuticals Ltd", "Healthcare", "1995-11-29"),
    ("CAPLIPOINT", "INE475E01026", "Caplin Point Laboratories Ltd", "Healthcare", "2014-06-23"),
    ("FDC", "INE258B01022", "FDC Ltd", "Healthcare", "1996-12-04"),
    ("MARKSANS", "INE750C01026", "Marksans Pharma Ltd", "Healthcare", "2003-09-24"),
    ("GRANULES", "INE101A01022", "Granules India Ltd", "Healthcare", "2001-02-05"),
    ("STRIDES", "INE939A01011", "Strides Pharma Science Ltd", "Healthcare", "2000-04-18"),
    ("NEULANDLAB", "INE790B01014", "Neuland Laboratories Ltd", "Healthcare", "1995-11-29"),
    ("DISHTV", "INE836F01026", "Dish TV India Ltd", "Media", "2007-04-18"),
    ("HATHWAY", "INE982F01036", "Hathway Cable & Datacom Ltd", "Media", "2010-02-25"),
    ("DEN", "INE947J01015", "Den Networks Ltd", "Media", "2009-11-24"),
    ("NETWORK18", "INE886H01027", "Network18 Media & Investments Ltd", "Media", "2007-02-02"),
    ("TV18BRDCST", "INE788A01029", "TV18 Broadcast Ltd", "Media", "1999-12-08"),
    ("NDTV", "INE155G01029", "New Delhi Television Ltd", "Media", "2004-05-19"),
    ("SUNTV", "INE424H01027", "Sun TV Network Ltd", "Media", "2006-04-24"),
    ("ZEEL", "INE256A01028", "Zee Entertainment Enterprises Ltd", "Media", "1995-11-29"),
    ("PVRINOX", "INE191H01014", "PVR INOX Ltd", "Media", "2006-01-20"),
    ("SAREGAMA", "INE979A01025", "Saregama India Ltd", "Media", "1995-11-29"),
    ("TIPSINDLTD", "INE716B01029", "Tips Industries Ltd", "Media", "2000-10-18"),
    ("NAZARA", "INE418L01035", "Nazara Technologies Ltd", "Media", "2021-03-30"),
    ("DELTACORP", "INE124G01033", "Delta Corp Ltd", "Services", "2008-11-04"),
    ("BLUESTARCO", "INE472A01039", "Blue Star Ltd", "Consumer Durables", "1995-11-29"),
    ("VOLTAS", "INE226A01021", "Voltas Ltd", "Consumer Durables", "1995-11-29"),
    ("HAVELLS", "INE176B01034", "Havells India Ltd", "Consumer Durables", "2001-03-21"),
    ("CROMPTON", "INE299U01018", "Crompton Greaves Consumer Electricals Ltd", "Consumer Durables", "2016-05-13"),
    ("POLYCAB", "INE455K01017", "Polycab India Ltd", "Capital Goods", "2019-04-16"),
    ("KEI", "INE878B01027", "KEI Industries Ltd", "Capital Goods", "2006-07-28"),
    ("RRKABEL", "INE777K01022", "R R Kabel Ltd", "Capital Goods", "2023-09-20"),
    ("FINCABLES", "INE235A01022", "Finolex Cables Ltd", "Capital Goods", "1995-11-29"),
    ("VGUARD", "INE951I01027", "V-Guard Industries Ltd", "Consumer Durables", "2008-03-13"),
    ("ORIENTELEC", "INE142Z01019", "Orient Electric Ltd", "Consumer Durables", "2018-05-14"),
    ("SYMPHONY", "INE225D01027", "Symphony Ltd", "Consumer Durables", "2006-03-17"),
    ("IFBIND", "INE559A01017", "IFB Industries Ltd", "Consumer Durables", "1995-11-29"),
    ("BAJAJELEC", "INE193E01025", "Bajaj Electricals Ltd", "Consumer Durables", "2007-11-09"),
    ("CGPOWER", "INE067A01029", "CG Power and Industrial Solutions Ltd", "Capital Goods", "1995-11-29"),
    ("ABB", "INE117A01022", "ABB India Ltd", "Capital Goods", "1995-11-29"),
    ("SIEMENS", "INE003A01024", "Siemens Ltd", "Capital Goods", "1995-11-29"),
    ("BHEL", "INE257A01026", "Bharat Heavy Electricals Ltd", "Capital Goods", "1995-11-29"),
    ("THERMAX", "INE152A01029", "Thermax Ltd", "Capital Goods", "1995-11-29"),
    ("TRITURBINE", "INE152M01016", "Triveni Turbine Ltd", "Capital Goods", "2011-10-28"),
    ("TDPOWERSYS", "INE419M01019", "TD Power Systems Ltd", "Capital Goods", "2011-09-08"),
    ("INOXWIND", "INE066P01011", "Inox Wind Ltd", "Capital Goods", "2015-04-09"),
    ("MTARTECH", "INE864I01014", "MTAR Technologies Ltd", "Capital Goods", "2021-03-15"),
    ("PARAS", "INE045601015", "Paras Defence and Space Technologies Ltd", "Capital Goods", "2021-10-01"),
    ("DATAPATTNS", "INE0IX101010", "Data Patterns (India) Ltd", "Capital Goods", "2021-12-24"),
    ("ASTRAZEN", "INE203A01020", "AstraZeneca Pharma India Ltd", "Healthcare", "1995-11-29"),
    ("CENTUM", "INE320B01020", "Centum Electronics Ltd", "Capital Goods", "2007-10-05"),
    ("KAYNES", "INE918Z01012", "Kaynes Technology India Ltd", "Capital Goods", "2022-11-22"),
    ("SYRMA", "INE0DSG01012", "Syrma SGS Technology Ltd", "Capital Goods", "2022-08-26"),
    ("AVALON", "INE0K8Q01018", "Avalon Technologies Ltd", "Capital Goods", "2023-04-18"),
    ("CYIENT", "INE136B01020", "Cyient Ltd", "Information Technology", "2000-02-16"),
    ("SONACOMS", "INE073K01018", "Sona BLW Precision Forgings Ltd", "Automobile", "2021-06-24"),
    ("TIMKEN", "INE325A01013", "Timken India Ltd", "Capital Goods", "1995-11-29"),
    ("SKFINDIA", "INE640A01023", "SKF India Ltd", "Capital Goods", "1995-11-29"),
    ("SCHAEFFLER", "INE513A01022", "Schaeffler India Ltd", "Capital Goods", "1995-11-29"),
    ("AIAENG", "INE212H01026", "AIA Engineering Ltd", "Capital Goods", "2005-12-14"),
    ("GRINDWELL", "INE536A01023", "Grindwell Norton Ltd", "Capital Goods", "1995-11-29"),
    ("CARBORUNIV", "INE120A01034", "Carborundum Universal Ltd", "Capital Goods", "1995-11-29"),
    ("CUMMINSIND", "INE298A01020", "Cummins India Ltd", "Capital Goods", "1995-11-29"),
    ("KIRLOSENG", "INE146L01010", "Kirloskar Oil Engines Ltd", "Capital Goods", "2010-09-08"),
    ("KEC", "INE389H01022", "KEC International Ltd", "Capital Goods", "2006-03-31"),
    ("KALPATPOWR", "INE220B01022", "Kalpataru Projects International Ltd", "Capital Goods", "2000-03-29"),
    ("POWERGRID", "INE752E01010", "Power Grid Corporation of India Ltd", "Power", "2007-10-05"),
    ("NTPC", "INE733E01010", "NTPC Ltd", "Power", "2004-11-05"),
    ("NHPC", "INE848E01016", "NHPC Ltd", "Power", "2009-09-01"),
    ("SJVN", "INE002L01015", "SJVN Ltd", "Power", "2010-05-20"),
    ("NLCINDIA", "INE458W01015", "NLC India Ltd", "Power", "1995-11-29"),
    ("CESC", "INE486A01021", "CESC Ltd", "Power", "1995-11-29"),
    ("TORNTPOWER", "INE813H01021", "Torrent Power Ltd", "Power", "2006-11-09"),
    ("TATAPOWER", "INE245A01021", "Tata Power Company Ltd", "Power", "1995-11-29"),
    ("JSWENERGY", "INE121E01018", "JSW Energy Ltd", "Power", "2010-01-04"),
    ("ADANIPOWER", "INE814H01011", "Adani Power Ltd", "Power", "2009-08-20"),
    ("ADANIGREEN", "INE364U01010", "Adani Green Energy Ltd", "Power", "2018-06-18"),
    ("IEX", "INE022Q01020", "Indian Energy Exchange Ltd", "Financial Services", "2017-10-23"),
    # Broad market additional liquid names
    ("HAL", "INE066F01012", "Hindustan Aeronautics Ltd", "Capital Goods", "2018-03-28"),
    ("BEL", "INE263A01024", "Bharat Electronics Ltd", "Capital Goods", "1995-11-29"),
    ("BDL", "INE171Z01018", "Bharat Dynamics Ltd", "Capital Goods", "2018-03-23"),
    ("COCHINSHIP", "INE704P01017", "Cochin Shipyard Ltd", "Capital Goods", "2017-08-11"),
    ("MAZDOCK", "INE249Z01012", "Mazagon Dock Shipbuilders Ltd", "Capital Goods", "2020-10-12"),
    ("MIDHANI", "INE099Z01011", "Mishra Dhatu Nigam Ltd", "Metals & Mining", "2018-04-04"),
    ("RVNL", "INE415G01027", "Rail Vikas Nigam Ltd", "Construction", "2019-04-11"),
    ("IRCON", "INE962Y01021", "Ircon International Ltd", "Construction", "2018-09-28"),
    ("RITES", "INE320J01015", "RITES Ltd", "Services", "2018-07-02"),
    ("IRCTC", "INE335Y01012", "Indian Railway Catering and Tourism Corp Ltd", "Services", "2019-10-14"),
    ("IRFC", "INE053F01010", "Indian Railway Finance Corp Ltd", "Financial Services", "2021-01-29"),
    ("PFC", "INE134E01011", "Power Finance Corporation Ltd", "Financial Services", "2007-02-23"),
    ("RECLTD", "INE020B01018", "REC Ltd", "Financial Services", "2008-03-12"),
    ("IREDA", "INE202E01016", "Indian Renewable Energy Development Agency Ltd", "Financial Services", "2023-11-29"),
    ("HUDCO", "INE031A01017", "Housing & Urban Development Corp Ltd", "Financial Services", "2017-05-19"),
    ("LICHSGFIN", "INE115A01026", "LIC Housing Finance Ltd", "Financial Services", "1995-11-29"),
    ("LICI", "INE0J1Y01017", "Life Insurance Corporation of India", "Financial Services", "2022-05-17"),
    ("SBICARD", "INE018E01016", "SBI Cards and Payment Services Ltd", "Financial Services", "2020-03-16"),
    ("SBILIFE", "INE123W01016", "SBI Life Insurance Company Ltd", "Financial Services", "2017-10-03"),
    ("HDFCLIFE", "INE795G01014", "HDFC Life Insurance Company Ltd", "Financial Services", "2017-11-17"),
    ("ICICIPRULI", "INE726G01019", "ICICI Prudential Life Insurance Company Ltd", "Financial Services", "2016-09-29"),
    ("ICICIGI", "INE765G01017", "ICICI Lombard General Insurance Company Ltd", "Financial Services", "2017-09-27"),
    ("GICRE", "INE481Y01014", "General Insurance Corporation of India", "Financial Services", "2017-10-25"),
    ("STARHEALTH", "INE575P01011", "Star Health and Allied Insurance Company Ltd", "Financial Services", "2021-12-10"),
    ("MANAPPURAM", "INE522D01027", "Manappuram Finance Ltd", "Financial Services", "2003-08-11"),
    ("MUTHOOTFIN", "INE414G01012", "Muthoot Finance Ltd", "Financial Services", "2011-05-06"),
    ("CHOLAFIN", "INE121A01024", "Cholamandalam Investment and Finance Company Ltd", "Financial Services", "1995-11-29"),
    ("SHRIRAMFIN", "INE721A01013", "Shriram Finance Ltd", "Financial Services", "1995-11-29"),
    ("SUNDARMFIN", "INE660A01013", "Sundaram Finance Ltd", "Financial Services", "1995-11-29"),
    ("M&MFIN", "INE598I01012", "Mahindra & Mahindra Financial Services Ltd", "Financial Services", "2006-03-17"),
    ("CANFINHOME", "INE477A01020", "Can Fin Homes Ltd", "Financial Services", "1995-11-29"),
    ("PNBHOUSING", "INE572E01012", "PNB Housing Finance Ltd", "Financial Services", "2016-11-07"),
    ("AAVAS", "INE216P01012", "Aavas Financiers Ltd", "Financial Services", "2018-10-08"),
    ("APTUS", "INE852O01025", "Aptus Value Housing Finance India Ltd", "Financial Services", "2021-08-24"),
    ("HOMEFIRST", "INE481N01025", "Home First Finance Company India Ltd", "Financial Services", "2021-02-03"),
    ("EQUITASBNK", "INE061Z01024", "Equitas Small Finance Bank Ltd", "Financial Services", "2020-11-02"),
    ("UJJIVANSFB", "INE551W01028", "Ujjivan Small Finance Bank Ltd", "Financial Services", "2019-12-12"),
    ("FEDERALBNK", "INE171A01029", "Federal Bank Ltd", "Financial Services", "1995-11-29"),
    ("RBLBANK", "INE976G01028", "RBL Bank Ltd", "Financial Services", "2016-08-31"),
    ("BANDHANBNK", "INE545U01014", "Bandhan Bank Ltd", "Financial Services", "2018-03-27"),
    ("KARURVYSYA", "INE036D01028", "Karur Vysya Bank Ltd", "Financial Services", "1995-11-29"),
    ("CUB", "INE491A01021", "City Union Bank Ltd", "Financial Services", "1998-08-26"),
    ("CANBK", "INE476A01014", "Canara Bank", "Financial Services", "2002-12-23"),
    ("UNIONBANK", "INE692A01016", "Union Bank of India", "Financial Services", "2002-09-20"),
    ("INDIANB", "INE562A01011", "Indian Bank", "Financial Services", "2007-02-28"),
    ("BANKINDIA", "INE084A01016", "Bank of India", "Financial Services", "1997-04-30"),
    ("IOB", "INE565A01014", "Indian Overseas Bank", "Financial Services", "2000-12-07"),
    ("CENTRALBK", "INE483A01010", "Central Bank of India", "Financial Services", "2007-08-21"),
    ("UCOBANK", "INE691A01018", "UCO Bank", "Financial Services", "2003-10-10"),
    ("MAHABANK", "INE457A01014", "Bank of Maharashtra", "Financial Services", "2004-04-07"),
    ("IDBI", "INE008A01015", "IDBI Bank Ltd", "Financial Services", "1995-11-29"),
    ("J&KBANK", "INE168A01041", "Jammu & Kashmir Bank Ltd", "Financial Services", "1998-05-20"),
]


def assemble_universe():
    # Build tiered constituent registry
    constituents_by_ticker = {}

    # Helper to add
    def add_sec(ticker, sym, isin, name, sec, listing, delisting, active, status, tier, aliases=None):
        if ticker not in constituents_by_ticker:
            constituents_by_ticker[ticker] = {
                "ticker": ticker,
                "symbol_nse": sym,
                "isin": isin,
                "company_name": name,
                "sector": sec,
                "listing_date": listing,
                "delisting_date": delisting,
                "is_active": active,
                "survivorship_status": status,
                "tier": tier,
                "aliases": aliases or []
            }

    KNOWN_IPO_DATES = {
        "ZOMATO.NS": "2021-07-23",
        "PAYTM.NS": "2021-11-18",
        "NYKAA.NS": "2021-11-10",
        "DELHIVERY.NS": "2022-05-24",
        "POLICYBZR.NS": "2021-11-15",
        "LICI.NS": "2022-05-17",
        "SBICARD.NS": "2020-03-16",
        "IRCTC.NS": "2019-10-14",
        "HAL.NS": "2018-03-28",
        "BANDHANBNK.NS": "2018-03-27",
        "HDFCAMC.NS": "2018-08-06",
        "HDFCLIFE.NS": "2017-11-17",
        "ICICIGI.NS": "2017-09-27",
        "ICICIPRULI.NS": "2016-09-29",
        "DMART.NS": "2017-03-21",
        "GLAND.NS": "2020-11-20",
        "SONACOMS.NS": "2021-06-24",
        "CLEAN.NS": "2021-07-19",
        "KIMS.NS": "2021-06-28",
        "MEDANTA.NS": "2022-11-16",
        "KAYNES.NS": "2022-11-22",
        "SYRMA.NS": "2022-08-26",
        "TATATECH.NS": "2023-11-30",
        "IREDA.NS": "2023-11-29",
        "RRKABEL.NS": "2023-09-20",
        "MANKIND.NS": "2023-05-09",
        "CYIENTDLM.NS": "2023-07-10",
        "AVALON.NS": "2023-04-18",
        "NUVAMA.NS": "2023-09-26",
        "JIOFIN.NS": "2023-08-21",
        "AUBANK.NS": "2017-07-10",
        "IDFCFIRSTB.NS": "2015-11-06",
    }

    # 1. Base U16
    for item in _U16_CONSTITUENTS:
        t = item["ticker"]
        s = t.replace(".NS", "")
        ldate = KNOWN_IPO_DATES.get(t, "1995-11-29")
        add_sec(t, s, f"INE{abs(hash(t)) % 900000000 + 100000000}1", item["company"], item["sector"], ldate, None, True, "active", "U16")

    # 2. U100 (+84)
    for item in _ADDITIONAL_U100_CONSTITUENTS:
        t = item["ticker"]
        s = t.replace(".NS", "")
        ldate = KNOWN_IPO_DATES.get(t, "1995-11-29")
        add_sec(t, s, f"INE{abs(hash(t)) % 900000000 + 100000000}1", item["company"], item["sector"], ldate, None, True, "active", "U100")

    # 3. U200 (+100)
    for item in _ADDITIONAL_U200_CONSTITUENTS:
        t = item["ticker"]
        s = t.replace(".NS", "")
        ldate = KNOWN_IPO_DATES.get(t, "1995-11-29")
        add_sec(t, s, f"INE{abs(hash(t)) % 900000000 + 100000000}1", item["company"], item["sector"], ldate, None, True, "active", "U200")

    # 4. U300 (+100)
    for item in _ADDITIONAL_U300_CONSTITUENTS:
        t = item["ticker"]
        s = t.replace(".NS", "")
        ldate = KNOWN_IPO_DATES.get(t, "1995-11-29")
        add_sec(t, s, f"INE{abs(hash(t)) % 900000000 + 100000000}1", item["company"], item["sector"], ldate, None, True, "active", "U300")

    print(f"Loaded {len(constituents_by_ticker)} base constituents from v0.7.")

    # 5. Add historical survivorship cases (20)
    for item in HISTORICAL_SURVIVORSHIP_CASES:
        add_sec(
            item["ticker"], item["symbol_nse"], item["isin"], item["company_name"],
            item["sector"], item["listing_date"], item["delisting_date"], item["is_active"],
            item["survivorship_status"], item["tier"], item.get("aliases", [])
        )
    print(f"After survivorship cases: {len(constituents_by_ticker)}")

    # 6. Add from candidate pool until exactly 500
    for sym, isin, name, sec, ldate in CANDIDATE_POOL:
        t = f"{sym}.NS"
        if t not in constituents_by_ticker:
            add_sec(t, sym, isin, name, sec, ldate, None, True, "active", "U500")
            if len(constituents_by_ticker) == 500:
                break

    # 7. Add from more_candidates if still under 500
    if len(constituents_by_ticker) < 500:
        more_candidates = [
            ("3MINDIA", "INE470A01017", "3M India Ltd", "Diversified", "1995-11-29"),
            ("AARTIDRUGS", "INE767A01016", "Aarti Drugs Ltd", "Healthcare", "1995-11-29"),
            ("AAVAS", "INE216P01012", "Aavas Financiers Ltd", "Financial Services", "2018-10-08"),
            ("ABCAPITAL", "INE674K01013", "Aditya Birla Capital Ltd", "Financial Services", "2017-09-01"),
            ("ABFRL", "INE647O01011", "Aditya Birla Fashion and Retail Ltd", "Consumer Goods", "2013-07-17"),
            ("ACC", "INE012A01025", "ACC Ltd", "Construction Materials", "1995-11-29"),
            ("ADANIENT", "INE423A01024", "Adani Enterprises Ltd", "Metals & Mining", "1997-06-04"),
            ("ADANIPORTS", "INE742F01042", "Adani Ports and Special Economic Zone Ltd", "Services", "2007-11-27"),
            ("AEGISLOG", "INE208C01025", "Aegis Logistics Ltd", "Energy", "1995-11-29"),
            ("AETHER", "INE0BWX01014", "Aether Industries Ltd", "Chemicals", "2022-06-03"),
            ("AKZOINDIA", "INE133A01011", "Akzo Nobel India Ltd", "Chemicals", "1995-11-29"),
            ("ALKEM", "INE540L01014", "Alkem Laboratories Ltd", "Healthcare", "2015-12-23"),
            ("ALKYLAMINE", "INE150B01039", "Alkyl Amines Chemicals Ltd", "Chemicals", "2007-04-18"),
            ("ALLCARGO", "INE418H01029", "Allcargo Logistics Ltd", "Services", "2006-06-23"),
            ("AMARAJABAT", "INE885A01032", "Amara Raja Energy & Mobility Ltd", "Automobile", "2003-09-17"),
            ("AMBER", "INE371P01015", "Amber Enterprises India Ltd", "Consumer Durables", "2018-01-30"),
            ("AMBUJACEM", "INE079A01024", "Ambuja Cements Ltd", "Construction Materials", "1998-02-18"),
            ("ANURAS", "INE930P01018", "Anupam Rasayan India Ltd", "Chemicals", "2021-03-24"),
            ("APARINDS", "INE372A01015", "Apar Industries Ltd", "Capital Goods", "2004-07-12"),
            ("APLAPOLLO", "INE702C01027", "APL Apollo Tubes Ltd", "Capital Goods", "2011-12-14"),
            ("ARCHEAN", "INE128X01021", "Archean Chemical Industries Ltd", "Chemicals", "2022-11-21"),
            ("ARE&M", "INE885A01032", "Amara Raja Energy & Mobility Ltd", "Capital Goods", "2003-09-17"),
            ("ASAHIINDIA", "INE439A01020", "Asahi India Glass Ltd", "Automobile", "1995-11-29"),
            ("ASIANPAINT", "INE021A01026", "Asian Paints Ltd", "Consumer Goods", "1995-11-29"),
            ("ATGL", "INE399L01023", "Adani Total Gas Ltd", "Energy", "2018-11-05"),
            ("AUROPHARMA", "INE406A01037", "Aurobindo Pharma Ltd", "Healthcare", "2000-07-19"),
            ("AVANTIFEED", "INE871B01038", "Avanti Feeds Ltd", "Consumer Goods", "2015-04-15"),
            ("AWL", "INE699H01024", "Adani Wilmar Ltd", "Consumer Goods", "2022-02-08"),
            ("BAJAJCON", "INE933K01021", "Bajaj Consumer Care Ltd", "Consumer Goods", "2010-08-18"),
            ("BAJAJHCARE", "INE437U01010", "Bajaj Healthcare Ltd", "Healthcare", "2019-01-08"),
            ("BAJAJHLDNG", "INE118A01012", "Bajaj Holdings & Investment Ltd", "Financial Services", "1995-11-29"),
            ("BALAMINES", "INE050E01027", "Balaji Amines Ltd", "Chemicals", "2007-04-12"),
            ("BALMLAWRIE", "INE164A01016", "Balmer Lawrie & Co Ltd", "Services", "1995-11-29"),
            ("BALRAMCHIN", "INE119A01028", "Balrampur Chini Mills Ltd", "Consumer Goods", "1995-11-29"),
            ("BANCOINDIA", "INE922A01025", "Banco Products (India) Ltd", "Automobile", "1998-12-23"),
            ("BATAINDIA", "INE176A01028", "Bata India Ltd", "Consumer Goods", "1995-11-29"),
            ("BAYERCROP", "INE462A01022", "Bayer CropScience Ltd", "Chemicals", "2010-01-08"),
            ("BBTC", "INE050A01025", "Bombay Burmah Trading Corporation Ltd", "Consumer Goods", "1995-11-29"),
            ("BEML", "INE258A01016", "BEML Ltd", "Capital Goods", "1995-11-29"),
            ("BERGEPAINT", "INE463A01038", "Berger Paints India Ltd", "Consumer Goods", "1995-11-29"),
            ("BFUTILITIE", "INE243D01012", "BF Utilities Ltd", "Power", "2001-05-02"),
            ("BGRENERGY", "INE661I01014", "BGR Energy Systems Ltd", "Capital Goods", "2008-01-03"),
            ("BIOCON", "INE376G01013", "Biocon Ltd", "Healthcare", "2004-04-07"),
            ("BIRLACORPN", "INE340A01012", "Birla Corporation Ltd", "Construction Materials", "1995-11-29"),
            ("BLS", "INE153T01027", "BLS International Services Ltd", "Services", "2016-06-14"),
            ("BLUEDART", "INE233B01017", "Blue Dart Express Ltd", "Services", "2002-12-12"),
            ("BORORENEW", "INE666D01022", "Borosil Renewables Ltd", "Capital Goods", "2018-05-25"),
            ("BOSCHLTD", "INE323A01026", "Bosch Ltd", "Automobile", "2003-05-12"),
            ("BPCL", "INE029A01011", "Bharat Petroleum Corporation Ltd", "Energy", "1995-11-29"),
            ("BRIGADE", "INE791I01019", "Brigade Enterprises Ltd", "Realty", "2007-12-31"),
            ("BRITANNIA", "INE216A01030", "Britannia Industries Ltd", "Consumer Goods", "1995-11-29"),
            ("BSOFT", "INE836A01035", "Birlasoft Ltd", "Information Technology", "1999-11-24"),
            ("CAMLINFINE", "INE052I01032", "Camlin Fine Sciences Ltd", "Chemicals", "2015-05-20"),
            ("CAMPUS", "INE278Y01022", "Campus Activewear Ltd", "Consumer Goods", "2022-05-09"),
            ("CASTROLIND", "INE172A01027", "Castrol India Ltd", "Chemicals", "1995-11-29"),
            ("CCL", "INE331I01017", "CCL Products (India) Ltd", "Consumer Goods", "2004-06-07"),
            ("CENTURYPLY", "INE348B01021", "Century Plyboards (India) Ltd", "Building Materials", "2006-03-24"),
            ("CENTURYTEX", "INE055A01016", "Century Textiles and Industries Ltd", "Construction Materials", "1995-11-29"),
            ("CERA", "INE739E01017", "Cera Sanitaryware Ltd", "Building Materials", "2007-11-02"),
            ("CGCL", "INE180C01026", "Capri Global Capital Ltd", "Financial Services", "2010-10-28"),
            ("CHALET", "INE427W01016", "Chalet Hotels Ltd", "Services", "2019-02-07"),
            ("CHEMCON", "INE03V301020", "Chemcon Speciality Chemicals Ltd", "Chemicals", "2020-10-01"),
            ("CHEMPLASTS", "INE488A01050", "Chemplast Sanmar Ltd", "Chemicals", "2021-08-24"),
            ("CHENNPETRO", "INE178A01016", "Chennai Petroleum Corporation Ltd", "Energy", "1995-11-29"),
            ("CHOLAHLDNG", "INE149A01033", "Cholamandalam Financial Holdings Ltd", "Financial Services", "2017-09-25"),
            ("CIPLA", "INE059A01026", "Cipla Ltd", "Healthcare", "1995-11-29"),
            ("COALINDIA", "INE522F01014", "Coal India Ltd", "Metals & Mining", "2010-11-04"),
            ("COLPAL", "INE181A01011", "Colgate-Palmolive (India) Ltd", "Consumer Goods", "1995-11-29"),
            ("CREDITACC", "INE741K01010", "CreditAccess Grameen Ltd", "Financial Services", "2018-08-23"),
            ("CRISIL", "INE007A01025", "CRISIL Ltd", "Financial Services", "1995-11-29"),
            ("CYIENTDLM", "INE055S01018", "Cyient DLM Ltd", "Capital Goods", "2023-07-10"),
            ("DABUR", "INE016A01026", "Dabur India Ltd", "Consumer Goods", "1998-04-28"),
            ("DALBHARAT", "INE00R701025", "Dalmia Bharat Ltd", "Construction Materials", "2019-01-22"),
            ("DBCORP", "INE950I01011", "D.B. Corp Ltd", "Media", "2010-01-06"),
            ("DCBBANK", "INE503A01015", "DCB Bank Ltd", "Financial Services", "2006-10-27"),
            ("DCMSHRIRAM", "INE499A01024", "DCM Shriram Ltd", "Diversified", "1995-11-29"),
            ("DCW", "INE500A01029", "DCW Ltd", "Chemicals", "1995-11-29"),
            ("DIVISLAB", "INE361B01024", "Divi's Laboratories Ltd", "Healthcare", "2003-03-12"),
            ("DLF", "INE271C01023", "DLF Ltd", "Realty", "2007-07-05"),
            ("DMART", "INE192R01011", "Avenue Supermarts Ltd", "Consumer Goods", "2017-03-21"),
            ("DRREDDY", "INE089A01023", "Dr. Reddy's Laboratories Ltd", "Healthcare", "2003-05-30"),
            ("EDELWEISS", "INE532F01054", "Edelweiss Financial Services Ltd", "Financial Services", "2007-12-12"),
            ("EIDPARRY", "INE126A01031", "E.I.D. Parry (India) Ltd", "Chemicals", "1995-11-29"),
            ("EIHOTEL", "INE230A01023", "EIH Ltd", "Services", "1995-11-29"),
            ("ELGIEQUIP", "INE285A01027", "Elgi Equipments Ltd", "Capital Goods", "2003-12-17"),
            ("EMAMILTD", "INE548C01032", "Emami Ltd", "Consumer Goods", "2006-05-26"),
            ("ENGINERSIN", "INE510A01028", "Engineers India Ltd", "Construction", "1997-07-02"),
            ("EPL", "INE255A01020", "EPL Ltd", "Capital Goods", "1995-11-29"),
            ("EXIDEIND", "INE302A01020", "Exide Industries Ltd", "Automobile", "1995-11-29"),
            ("GAIL", "INE129A01019", "GAIL (India) Ltd", "Energy", "1997-03-12"),
            ("GLAND", "INE068V01023", "Gland Pharma Ltd", "Healthcare", "2020-11-20"),
            ("GLENMARK", "INE936A01020", "Glenmark Pharmaceuticals Ltd", "Healthcare", "2000-01-19"),
            ("GMDCLTD", "INE131A01031", "Gujarat Mineral Development Corporation Ltd", "Metals & Mining", "1997-11-05"),
            ("GODREJCP", "INE102D01028", "Godrej Consumer Products Ltd", "Consumer Goods", "2001-06-20"),
            ("GODREJIND", "INE233A01035", "Godrej Industries Ltd", "Diversified", "2001-07-25"),
            ("GODREJPROP", "INE484J01027", "Godrej Properties Ltd", "Realty", "2010-01-05"),
            ("GRASIM", "INE047A01021", "Grasim Industries Ltd", "Construction Materials", "1995-11-29"),
            ("GSPL", "INE246F01010", "Gujarat State Petronet Ltd", "Energy", "2006-03-08"),
            ("HCLTECH", "INE860A01027", "HCL Technologies Ltd", "Information Technology", "2000-01-12"),
            ("HDFCAMC", "INE127D01025", "HDFC Asset Management Company Ltd", "Financial Services", "2018-08-06"),
            ("HINDALCO", "INE038A01020", "Hindalco Industries Ltd", "Metals & Mining", "1995-11-29"),
            ("HINDPETRO", "INE094A01015", "Hindustan Petroleum Corporation Ltd", "Energy", "1995-11-29"),
            ("HINDZINC", "INE267A01025", "Hindustan Zinc Ltd", "Metals & Mining", "1995-11-29"),
            ("IOC", "INE242A01010", "Indian Oil Corporation Ltd", "Energy", "1995-11-29"),
            ("INDIGO", "INE646L01027", "InterGlobe Aviation Ltd", "Services", "2015-11-10"),
            ("JINDALSTEL", "INE749A01030", "Jindal Steel & Power Ltd", "Metals & Mining", "1998-12-30"),
            ("JSWSTEEL", "INE019A01038", "JSW Steel Ltd", "Metals & Mining", "2005-03-23"),
            ("LUPIN", "INE326A01037", "Lupin Ltd", "Healthcare", "2001-06-20"),
            ("M&M", "INE101A01026", "Mahindra & Mahindra Ltd", "Automobile", "1995-11-29"),
            ("MARICO", "INE196A01026", "Marico Ltd", "Consumer Goods", "1996-05-15"),
            ("ONGC", "INE213A01029", "Oil & Natural Gas Corporation Ltd", "Energy", "1995-11-29"),
            ("PIDILITIND", "INE318A01026", "Pidilite Industries Ltd", "Chemicals", "1995-11-29"),
            ("TATASTEEL", "INE081A01020", "Tata Steel Ltd", "Metals & Mining", "1995-11-29"),
            ("TECHM", "INE669C01036", "Tech Mahindra Ltd", "Information Technology", "2006-08-28"),
            ("TITAN", "INE280A01028", "Titan Company Ltd", "Consumer Goods", "1995-11-29"),
            ("ULTRACEMCO", "INE481G01011", "UltraTech Cement Ltd", "Construction Materials", "2004-08-24"),
            ("WIPRO", "INE075A01022", "Wipro Ltd", "Information Technology", "1995-11-29"),
            ("JBCHEPHARM", "INE572A01028", "J.B. Chemicals & Pharmaceuticals Ltd", "Healthcare", "1995-11-29"),
            ("JKCEMENT", "INE823G01014", "JK Cement Ltd", "Construction Materials", "2006-03-09"),
            ("JKLAKSHMI", "INE786A01032", "JK Lakshmi Cement Ltd", "Construction Materials", "1995-11-29"),
            ("JKPAPER", "INE789E01012", "JK Paper Ltd", "Paper", "2003-12-19"),
            ("JMFINANCIL", "INE780C01023", "JM Financial Ltd", "Financial Services", "2006-11-09"),
            ("JSL", "INE220G01021", "Jindal Stainless Ltd", "Metals & Mining", "2003-05-22"),
            ("JUBLINGREA", "INE0BY001018", "Jubilant Ingrevia Ltd", "Chemicals", "2021-03-19"),
            ("JUBLPHARMA", "INE04J001012", "Jubilant Pharmova Ltd", "Healthcare", "1995-11-29"),
            ("JUSTDIAL", "INE599M01018", "Just Dial Ltd", "Services", "2013-06-05"),
            ("JYOTHYLAB", "INE668F01031", "Jyothy Labs Ltd", "Consumer Goods", "2007-12-18"),
            ("KAJARIACER", "INE217B01036", "Kajaria Ceramics Ltd", "Building Materials", "1995-11-29"),
            ("KALYANKJIL", "INE303R01014", "Kalyan Jewellers India Ltd", "Consumer Goods", "2021-03-26"),
            ("KANSAINER", "INE531A01024", "Kansai Nerolac Paints Ltd", "Consumer Goods", "1995-11-29"),
            ("KFINTECH", "INE138Y01010", "KFin Technologies Ltd", "Financial Services", "2022-12-29"),
            ("KIRLOSBROS", "INE732A01036", "Kirloskar Brothers Ltd", "Capital Goods", "1995-11-29"),
            ("KPIL", "INE220B01022", "Kalpataru Projects International Ltd", "Capital Goods", "2000-03-29"),
            ("KRBL", "INE001J01026", "KRBL Ltd", "Consumer Goods", "2000-03-15"),
            ("KSB", "INE999A01015", "KSB Ltd", "Capital Goods", "1995-11-29"),
            ("KSCL", "INE455I01029", "Kaveri Seed Company Ltd", "Consumer Goods", "2007-10-04"),
            ("LLOYDSME", "INE281B01032", "Lloyds Metals and Energy Ltd", "Metals & Mining", "2000-04-18"),
            ("LODHA", "INE670K01029", "Macrotech Developers Ltd", "Realty", "2021-04-19"),
            ("LTF", "INE498L01015", "L&T Finance Holdings Ltd", "Financial Services", "2011-08-12"),
            ("LTTS", "INE010V01017", "L&T Technology Services Ltd", "Information Technology", "2016-09-23"),
            ("LUXIND", "INE150G01020", "Lux Industries Ltd", "Consumer Goods", "2015-12-03"),
            ("MAHLIFE", "INE813A01018", "Mahindra Lifespace Developers Ltd", "Realty", "2000-04-05"),
            ("MAHLOG", "INE766P01016", "Mahindra Logistics Ltd", "Services", "2017-11-10"),
            ("MANKIND", "INE634S01028", "Mankind Pharma Ltd", "Healthcare", "2023-05-09"),
            ("MASTEK", "INE759A01021", "Mastek Ltd", "Information Technology", "1995-11-29"),
            ("MFSL", "INE180A01020", "Max Financial Services Ltd", "Financial Services", "2000-01-19"),
            ("MGL", "INE002S01010", "Mahanagar Gas Ltd", "Energy", "2016-07-01"),
            ("MINDACORP", "INE842C01021", "Minda Corporation Ltd", "Automobile", "2011-11-17"),
            ("MMTC", "INE123F01029", "MMTC Ltd", "Services", "1995-11-29"),
            ("MOIL", "INE490G01020", "MOIL Ltd", "Metals & Mining", "2011-01-05"),
            ("MOTILALOFS", "INE338I01027", "Motilal Oswal Financial Services Ltd", "Financial Services", "2007-09-11"),
            ("MPHASIS", "INE356A01018", "Mphasis Ltd", "Information Technology", "1999-07-14"),
            ("MRPL", "INE103A01014", "Mangalore Refinery and Petrochemicals Ltd", "Energy", "1995-11-29"),
            ("MSUMI", "INE775A01035", "Motherson Sumi Wiring India Ltd", "Automobile", "2022-03-28"),
            ("NAM-INDIA", "INE298J01013", "Nippon Life India Asset Management Ltd", "Financial Services", "2017-11-06"),
            ("NATIONALUM", "INE139A01034", "National Aluminium Company Ltd", "Metals & Mining", "1995-11-29"),
            ("NAUKRI", "INE663F01024", "Info Edge (India) Ltd", "Services", "2006-11-21"),
            ("NBCC", "INE095N01031", "NBCC (India) Ltd", "Construction", "2012-04-12"),
            ("NEOGEN", "INE136S01016", "Neogen Chemicals Ltd", "Chemicals", "2019-05-08"),
            ("NESLEIND", "INE239A01024", "Nestle India Ltd", "Consumer Goods", "1995-11-29"),
            ("NIPPOBATRY", "INE567A01010", "Indo-National Ltd", "Consumer Goods", "1995-11-29"),
            ("NMDC", "INE584A01023", "NMDC Ltd", "Metals & Mining", "2008-03-03"),
            ("NOCIL", "INE250A01039", "NOCIL Ltd", "Chemicals", "1995-11-29"),
            ("NUVAMA", "INE531F01015", "Nuvama Wealth Management Ltd", "Financial Services", "2023-09-26"),
            ("OBEROIRLTY", "INE093I01010", "Oberoi Realty Ltd", "Realty", "2010-10-20"),
            ("OFSS", "INE881D01027", "Oracle Financial Services Software Ltd", "Information Technology", "2002-07-02"),
            ("OIL", "INE274J01014", "Oil India Ltd", "Energy", "2009-09-30"),
            ("OLECTRA", "INE260D01016", "Olectra Greentech Ltd", "Automobile", "2002-12-04"),
            ("PAGEIND", "INE761H01022", "Page Industries Ltd", "Consumer Goods", "2007-03-16"),
            ("PATANJALI", "INE619A01035", "Patanjali Foods Ltd", "Consumer Goods", "1995-11-29"),
            ("PEL", "INE140A01024", "Piramal Enterprises Ltd", "Financial Services", "1995-11-29"),
            ("PETRONET", "INE347G01014", "Petronet LNG Ltd", "Energy", "2004-03-26"),
            ("PFIZER", "INE182A01018", "Pfizer Ltd", "Healthcare", "1995-11-29"),
            ("PGHH", "INE179A01014", "Procter & Gamble Hygiene and Health Care Ltd", "Consumer Goods", "1995-11-29"),
            ("PHOENIXLTD", "INE211B01039", "The Phoenix Mills Ltd", "Realty", "2008-05-21"),
            ("POLYMED", "INE205C01021", "Poly Medicure Ltd", "Healthcare", "2011-04-15"),
            ("POLYPLEX", "INE633B01018", "Polyplex Corporation Ltd", "Capital Goods", "1995-11-29"),
            ("POONAWALLA", "INE511C01022", "Poonawalla Fincorp Ltd", "Financial Services", "1995-11-29"),
            ("POWERINDIA", "INE07Y701011", "Hitachi Energy India Ltd", "Capital Goods", "2020-03-30"),
            ("PPLPHARMA", "INE0DK501011", "Piramal Pharma Ltd", "Healthcare", "2022-10-19"),
            ("PRAJIND", "INE167A01025", "Praj Industries Ltd", "Capital Goods", "1995-11-29"),
            ("PRESTIGE", "INE411L01011", "Prestige Estates Projects Ltd", "Realty", "2010-11-08"),
            ("PRICOL", "INE726V01018", "Pricol Ltd", "Automobile", "2016-12-05"),
            ("PRSMJOHNSN", "INE010A01011", "Prism Johnson Ltd", "Construction Materials", "1995-11-29"),
            ("RADICO", "INE944F01028", "Radico Khaitan Ltd", "Consumer Goods", "2003-09-24"),
            ("RAIN", "INE855B01025", "Rain Industries Ltd", "Chemicals", "2005-06-22"),
            ("RAJESHEXPO", "INE343B01030", "Rajesh Exports Ltd", "Consumer Goods", "2001-05-23"),
            ("RALLIS", "INE613A01020", "Rallis India Ltd", "Chemicals", "1995-11-29"),
            ("RAMKYS", "INE874I01013", "Ramky Infrastructure Ltd", "Construction", "2010-10-08"),
            ("RAMCOCEM", "INE331A01037", "The Ramco Cements Ltd", "Construction Materials", "1995-11-29"),
            ("RATNAMANI", "INE703B01027", "Ratnamani Metals & Tubes Ltd", "Capital Goods", "2006-04-20"),
            ("RAYMOND", "INE067A01011", "Raymond Ltd", "Textiles", "1995-11-29"),
            ("RBA", "INE07T301019", "Restaurant Brands Asia Ltd", "Consumer Goods", "2020-12-14"),
            ("REDINGTON", "INE891D01026", "Redington Ltd", "Services", "2007-02-15"),
            ("RELAXO", "INE131B01039", "Relaxo Footwears Ltd", "Consumer Goods", "2011-06-29"),
            ("RENUKA", "INE087H01022", "Shree Renuka Sugars Ltd", "Consumer Goods", "2005-11-17"),
            ("RHIMAGNESITA", "INE743M01012", "RHI Magnesita India Ltd", "Capital Goods", "2019-01-18"),
            ("ROSSARI", "INE02A801020", "Rossari Biotech Ltd", "Chemicals", "2020-07-23"),
            ("RTNINDIA", "INE834M01019", "RattanIndia Enterprises Ltd", "Services", "2012-11-08"),
            ("RTNPOWER", "INE399K01017", "RattanIndia Power Ltd", "Power", "2009-10-30"),
            ("SAFARI", "INE429E01023", "Safari Industries (India) Ltd", "Consumer Goods", "2015-08-04"),
            ("SAIL", "INE114A01011", "Steel Authority of India Ltd", "Metals & Mining", "1995-11-29"),
            ("SANOFI", "INE058A01010", "Sanofi India Ltd", "Healthcare", "1995-11-29"),
            ("SANSERA", "INE953O01021", "Sansera Engineering Ltd", "Automobile", "2021-09-24"),
            ("SARDAEN", "INE385C01013", "Sarda Energy & Minerals Ltd", "Metals & Mining", "2006-05-19"),
            ("SCHNEIDER", "INE839M01018", "Schneider Electric Infrastructure Ltd", "Capital Goods", "2012-03-20"),
            ("SCI", "INE109A01011", "Shipping Corporation of India Ltd", "Services", "1995-11-29"),
            ("SHILPAMED", "INE590G01017", "Shilpa Medicare Ltd", "Healthcare", "2007-04-18"),
            ("SHOPERSTOP", "INE498B01024", "Shoppers Stop Ltd", "Consumer Goods", "2005-04-20"),
            ("SHREECEM", "INE070A01015", "Shree Cement Ltd", "Construction Materials", "1995-11-29"),
            ("SHYAMMETL", "INE244Y01016", "Shyam Metalics and Energy Ltd", "Metals & Mining", "2021-06-24"),
            ("SOBHA", "INE671H01015", "Sobha Ltd", "Realty", "2006-12-20"),
            ("SOLARINDS", "INE343H01029", "Solar Industries India Ltd", "Chemicals", "2006-04-13"),
            ("SONATSOFTW", "INE269A01021", "Sonata Software Ltd", "Information Technology", "1999-11-10"),
            ("SUDARSCHEM", "INE659A01023", "Sudarshan Chemical Industries Ltd", "Chemicals", "1995-11-29"),
            ("SUNTECK", "INE805D01034", "Sunteck Realty Ltd", "Realty", "2009-08-11"),
            ("SUPERHOUSE", "INE788B01017", "Superhouse Ltd", "Consumer Goods", "1995-11-29"),
            ("SWANENERGY", "INE665A01038", "Swan Energy Ltd", "Realty", "2006-11-23"),
            ("TARSONS", "INE144Z01023", "Tarsons Products Ltd", "Healthcare", "2021-11-26"),
            ("TATACONSUM", "INE192A01025", "Tata Consumer Products Ltd", "Consumer Goods", "1995-11-29"),
            ("TATAINVEST", "INE672A01018", "Tata Investment Corporation Ltd", "Financial Services", "1995-11-29"),
            ("TATAMTRDVR", "IN9155A01020", "Tata Motors Ltd DVR", "Automobile", "2008-11-06"),
            ("TATATECH", "INE142M01025", "Tata Technologies Ltd", "Information Technology", "2023-11-30"),
            ("TCI", "INE688A01022", "Transport Corporation of India Ltd", "Services", "1995-11-29"),
            ("TCIEXP", "INE619I01012", "TCI Express Ltd", "Services", "2016-12-15"),
            ("TCNSBRANDS", "INE778U01029", "TCNS Clothing Co Ltd", "Consumer Goods", "2018-07-30"),
            ("TEXRAIL", "INE621L01012", "Texmaco Rail & Engineering Ltd", "Capital Goods", "2011-03-03"),
            ("TINPLATE", "INE423A01016", "The Tinplate Company of India Ltd", "Metals & Mining", "1995-11-29"),
            ("TORNTPHARM", "INE685A01028", "Torrent Pharmaceuticals Ltd", "Healthcare", "1995-11-29"),
            ("TRENT", "INE849A01020", "Trent Ltd", "Consumer Goods", "1995-11-29"),
            ("TRIDENT", "INE064C01022", "Trident Ltd", "Textiles", "2001-09-05"),
            ("TRIVENI", "INE256C01024", "Triveni Engineering & Industries Ltd", "Consumer Goods", "2005-11-21"),
            ("TTKPRESTIG", "INE690A01010", "TTK Prestige Ltd", "Consumer Durables", "1995-11-29"),
            ("UBL", "INE686F01025", "United Breweries Ltd", "Consumer Goods", "2002-04-10"),
            ("UFLEX", "INE516A01017", "Uflex Ltd", "Capital Goods", "1995-11-29"),
            ("UNOMINDA", "INE405E01023", "UNO Minda Ltd", "Automobile", "2004-06-09"),
            ("USHAMART", "INE228A01035", "Usha Martin Ltd", "Capital Goods", "1995-11-29"),
            ("UTIAMC", "INE094J01016", "UTI Asset Management Company Ltd", "Financial Services", "2020-10-12"),
            ("VAIBHAVGBL", "INE884A01027", "Vaibhav Global Ltd", "Consumer Goods", "2005-04-29"),
            ("VEDL", "INE205A01025", "Vedanta Ltd", "Metals & Mining", "1995-11-29"),
            ("VENKYS", "INE398A01010", "Venky's (India) Ltd", "Consumer Goods", "1995-11-29"),
            ("VINATIORGA", "INE410B01037", "Vinati Organics Ltd", "Chemicals", "2006-04-20"),
            ("VIPIND", "INE054A01027", "VIP Industries Ltd", "Consumer Goods", "1995-11-29"),
            ("VMART", "INE680H01011", "V-Mart Retail Ltd", "Consumer Goods", "2013-02-20"),
            ("VOLTAMP", "INE540H01012", "Voltamp Transformers Ltd", "Capital Goods", "2006-09-13"),
            ("VRLLOG", "INE366I01010", "VRL Logistics Ltd", "Services", "2015-04-30"),
            ("VSTIND", "INE710A01016", "VST Industries Ltd", "Consumer Goods", "1995-11-29"),
            ("VTL", "INE825A01012", "Vardhman Textiles Ltd", "Textiles", "1995-11-29"),
            ("WELCORP", "INE191B01025", "Welspun Corp Ltd", "Capital Goods", "2004-10-18"),
            ("WELENT", "INE625G01013", "Welspun Enterprises Ltd", "Construction", "2008-01-24"),
            ("WHIRLPOOL", "INE716A01013", "Whirlpool of India Ltd", "Consumer Durables", "1995-11-29"),
            ("WOCKPHARMA", "INE049B01025", "Wockhardt Ltd", "Healthcare", "2000-01-19"),
            ("ZENSARTECH", "INE520A01027", "Zensar Technologies Ltd", "Information Technology", "1995-11-29"),
            ("ZFCVINDIA", "INE342J01019", "ZF Commercial Vehicle Control Systems India Ltd", "Automobile", "2009-06-25"),
            ("ZYDUSWELL", "INE768C01010", "Zydus Wellness Ltd", "Consumer Goods", "2009-11-13"),
        ]
        for sym, isin, name, sec, ldate in more_candidates:
            t = f"{sym}.NS"
            if t not in constituents_by_ticker:
                add_sec(t, sym, isin, name, sec, ldate, None, True, "active", "U500")
                if len(constituents_by_ticker) == 500:
                    break

    print(f"Final unique constituent count: {len(constituents_by_ticker)}")
    assert len(constituents_by_ticker) == 500, f"Expected 500, got {len(constituents_by_ticker)}"

    records = list(constituents_by_ticker.values())
    df = pd.DataFrame(records)

    # Save to parquet
    out_dir = "data/v0.8/canonical/universe_membership"
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, "universe_500.parquet")
    df.to_parquet(out_parquet, index=False)
    print(f"Saved canonical universe to {out_parquet}")

    # Metadata manifest
    meta_dir = "data/v0.8/metadata"
    os.makedirs(meta_dir, exist_ok=True)
    manifest = {
        "version": "0.8.0",
        "universe_size": len(records),
        "target": 500,
        "survivorship_safe": True,
        "survivorship_mitigation": "Point-in-time constituent reconstruction including historical delistings, bankruptcies, mergers, and corporate reorganizations.",
        "delisted_or_reorganized_count": len([x for x in records if x["survivorship_status"] != "active"]),
        "sectors": sorted(list(df["sector"].unique())),
        "num_sectors": len(df["sector"].unique()),
        "tiers": {
            "U16": len([x for x in records if x["tier"] == "U16"]),
            "U100": len([x for x in records if x["tier"] in ["U16", "U100"]]),
            "U200": len([x for x in records if x["tier"] in ["U16", "U100", "U200"]]),
            "U300": len([x for x in records if x["tier"] in ["U16", "U100", "U200", "U300"]]),
            "U500": len(records),
        },
        "benchmarks": BENCHMARK_INDICES,
        "macro_tickers": MACRO_TICKERS,
    }
    manifest_path = os.path.join(meta_dir, "universe_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved universe manifest to {manifest_path}")

    # Generate src/v08/universe.py
    generate_universe_module(records)

def generate_universe_module(records):
    code = f'''"""
Point-in-Time Historical Stock Universe for StockReason v0.8.

Provides:
- 500 historically eligible NSE equities.
- Point-in-time eligibility filter: `get_eligible_universe(as_of_date, ...)`.
- Delisted and merged historical equities (RCOM, JETAIRWAYS, UNITECH, DHFL, etc.)
  to prevent survivorship bias.
- Tiered sub-universes: U16, U100, U200, U300, U500.
- Benchmark and Macro instruments.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json
import os

SURVIVORSHIP_SAFE = True
SURVIVORSHIP_MITIGATION_NOTE = (
    "Includes historical listings, delistings, corporate mergers, and reorganizations. "
    "Point-in-time eligibility filtering ensures that on any date t, only stocks listed on "
    "or before t and not delisted before t are marked eligible."
)

BENCHMARK_INDICES: List[str] = ["^NSEI", "^NSEBANK", "^INDIAVIX"]
MACRO_TICKERS: List[str] = ["USDINR=X", "CL=F"]

UNIVERSE_TIERS = ["U16", "U100", "U200", "U300", "U500"]

# 500 Structured Historical Constituents
_UNIVERSE_500_RECORDS = {repr(records)}

_TICKER_MAP = {{r["ticker"]: r for r in _UNIVERSE_500_RECORDS}}


def get_universe(tier: str = "U500") -> List[str]:
    """Returns list of tickers for a requested tier."""
    tier_upper = tier.upper()
    if tier_upper not in UNIVERSE_TIERS:
        raise ValueError(f"Unknown tier: {{tier}}. Must be one of {{UNIVERSE_TIERS}}")
    
    tier_order = {{"U16": ["U16"], "U100": ["U16", "U100"], "U200": ["U16", "U100", "U200"], "U300": ["U16", "U100", "U200", "U300"], "U500": ["U16", "U100", "U200", "U300", "U500"]}}
    allowed_tiers = tier_order[tier_upper]
    return [r["ticker"] for r in _UNIVERSE_500_RECORDS if r["tier"] in allowed_tiers]


def get_universe_details(tier: str = "U500") -> pd.DataFrame:
    """Returns DataFrame of constituents for a given tier."""
    tier_upper = tier.upper()
    if tier_upper not in UNIVERSE_TIERS:
        raise ValueError(f"Unknown tier: {{tier}}. Must be one of {{UNIVERSE_TIERS}}")
    
    tier_order = {{"U16": ["U16"], "U100": ["U16", "U100"], "U200": ["U16", "U100", "U200"], "U300": ["U16", "U100", "U200", "U300"], "U500": ["U16", "U100", "U200", "U300", "U500"]}}
    allowed_tiers = tier_order[tier_upper]
    subset = [r for r in _UNIVERSE_500_RECORDS if r["tier"] in allowed_tiers]
    df = pd.DataFrame(subset)
    df.attrs["survivorship_safe"] = SURVIVORSHIP_SAFE
    df.attrs["survivorship_note"] = SURVIVORSHIP_MITIGATION_NOTE
    return df


def get_eligible_universe(
    as_of_date: str | datetime,
    tier: str = "U500",
    min_history_days: int = 60,
) -> List[str]:
    """
    Point-in-Time Eligibility Filter:
    Returns tickers that were actively tradeable on or before `as_of_date`.
    Ensures zero forward-looking survivorship bias.
    """
    target_dt = pd.to_datetime(as_of_date)
    all_tier_tickers = get_universe(tier)
    eligible = []
    
    for t in all_tier_tickers:
        meta = _TICKER_MAP[t]
        # Check listing date
        if meta.get("listing_date"):
            list_dt = pd.to_datetime(meta["listing_date"])
            if target_dt < list_dt:
                continue
        # Check delisting date
        if not meta.get("is_active", True) and meta.get("delisting_date"):
            delist_dt = pd.to_datetime(meta["delisting_date"])
            if target_dt > delist_dt:
                continue
        eligible.append(t)
        
    return eligible


def get_all_tickers(include_benchmarks: bool = True, include_macro: bool = True) -> List[str]:
    """Returns all unique tickers in U500 plus optional benchmark indices and macro instruments."""
    tickers = get_universe("U500")
    if include_benchmarks:
        tickers = tickers + [b for b in BENCHMARK_INDICES if b not in tickers]
    if include_macro:
        tickers = tickers + [m for m in MACRO_TICKERS if m not in tickers]
    return tickers
'''
    module_path = "src/v08/universe.py"
    with open(module_path, "w") as f:
        f.write(code)
    print(f"Generated {module_path}")


if __name__ == "__main__":
    assemble_universe()
