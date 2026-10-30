"""Market data layer - free Yahoo Finance data for NSE (symbols end with .NS)."""
from __future__ import annotations

import pandas as pd

import engine

NAMES = {
    'RELIANCE': 'Reliance Industries',
    'TCS': 'Tata Consultancy Services',
    'HDFCBANK': 'HDFC Bank',
    'ICICIBANK': 'ICICI Bank',
    'INFY': 'Infosys',
    'BHARTIARTL': 'Bharti Airtel',
    'SBIN': 'State Bank of India',
    'ITC': 'ITC',
    'HINDUNILVR': 'Hindustan Unilever',
    'LT': 'Larsen & Toubro',
    'KOTAKBANK': 'Kotak Mahindra Bank',
    'AXISBANK': 'Axis Bank',
    'BAJFINANCE': 'Bajaj Finance',
    'BAJAJFINSV': 'Bajaj Finserv',
    'ASIANPAINT': 'Asian Paints',
    'MARUTI': 'Maruti Suzuki',
    'M&M': 'Mahindra & Mahindra',
    'SUNPHARMA': 'Sun Pharma',
    'TITAN': 'Titan',
    'ULTRACEMCO': 'UltraTech Cement',
    'NTPC': 'NTPC',
    'POWERGRID': 'Power Grid',
    'ONGC': 'ONGC',
    'COALINDIA': 'Coal India',
    'TATASTEEL': 'Tata Steel',
    'JSWSTEEL': 'JSW Steel',
    'HINDALCO': 'Hindalco',
    'WIPRO': 'Wipro',
    'HCLTECH': 'HCL Tech',
    'TECHM': 'Tech Mahindra',
    'ADANIENT': 'Adani Enterprises',
    'ADANIPORTS': 'Adani Ports',
    'APOLLOHOSP': 'Apollo Hospitals',
    'BAJAJ-AUTO': 'Bajaj Auto',
    'BEL': 'Bharat Electronics',
    'CIPLA': 'Cipla',
    'DRREDDY': "Dr Reddy's",
    'EICHERMOT': 'Eicher Motors',
    'ETERNAL': 'Eternal (Zomato)',
    'GRASIM': 'Grasim',
    'HDFCLIFE': 'HDFC Life',
    'HEROMOTOCO': 'Hero MotoCorp',
    'INDUSINDBK': 'IndusInd Bank',
    'JIOFIN': 'Jio Financial',
    'NESTLEIND': 'Nestle India',
    'SBILIFE': 'SBI Life',
    'SHRIRAMFIN': 'Shriram Finance',
    'TATACONSUM': 'Tata Consumer',
    'TMPV': 'Tata Motors (PV)',
    'TRENT': 'Trent',
    'DMART': 'Avenue Supermarts (DMart)',
    'PIDILITIND': 'Pidilite',
    'HAVELLS': 'Havells',
    'DLF': 'DLF',
    'SIEMENS': 'Siemens',
    'VEDL': 'Vedanta',
    'TVSMOTOR': 'TVS Motor',
    'BANKBARODA': 'Bank of Baroda',
    'PNB': 'Punjab National Bank',
    'IRFC': 'IRFC',
    'HAL': 'Hindustan Aeronautics',
    'GODREJCP': 'Godrej Consumer',
    'DIVISLAB': "Divi's Labs",
    'LTIM': 'LTIMindtree',
    'CHOLAFIN': 'Cholamandalam Finance',
    'ZYDUSLIFE': 'Zydus Lifesciences',
    'TORNTPHARM': 'Torrent Pharma',
    'ABB': 'ABB India',
    'INDIGO': 'IndiGo (InterGlobe)',
    'BPCL': 'BPCL',
    'IOC': 'Indian Oil',
    'GAIL': 'GAIL',
    'TATAPOWER': 'Tata Power',
    'AMBUJACEM': 'Ambuja Cements',
    'LODHA': 'Lodha (Macrotech)',
    'NAUKRI': 'Info Edge (Naukri)',
    'ICICIGI': 'ICICI Lombard',
    'CANBK': 'Canara Bank',
    'UNIONBANK': 'Union Bank',
    'RECLTD': 'REC',
    'PFC': 'Power Finance Corp',
    'MOTHERSON': 'Samvardhana Motherson',
    'VBL': 'Varun Beverages',
    'ADANIPOWER': 'Adani Power',
    'ADANIGREEN': 'Adani Green',
    'BOSCHLTD': 'Bosch',
    'SHREECEM': 'Shree Cement',
    'HINDZINC': 'Hindustan Zinc',
    'POLYCAB': 'Polycab',
    'PERSISTENT': 'Persistent Systems',
    'COFORGE': 'Coforge',
    'DIXON': 'Dixon Technologies',
    'BSE': 'BSE Ltd',
    'MAZDOCK': 'Mazagon Dock',
    'CUMMINSIND': 'Cummins India',
    'TATAELXSI': 'Tata Elxsi',
    'BHEL': 'BHEL',
    'IRCTC': 'IRCTC',
    'SOLARINDS': 'Solar Industries',
    'MUTHOOTFIN': 'Muthoot Finance',
    'ICICIPRULI': 'ICICI Pru Life',
    'SBICARD': 'SBI Card',
    'BAJAJHLDNG': 'Bajaj Holdings',
    'HDFCAMC': 'HDFC AMC',
    'INDHOTEL': 'Indian Hotels',
    'JINDALSTEL': 'Jindal Steel',
    'SAIL': 'SAIL',
    'NMDC': 'NMDC',
    'IDFCFIRSTB': 'IDFC First Bank',
    'FEDERALBNK': 'Federal Bank',
    'AUBANK': 'AU Small Finance Bank',
    'BANDHANBNK': 'Bandhan Bank',
    'YESBANK': 'Yes Bank',
    'IDEA': 'Vodafone Idea',
    'SUZLON': 'Suzlon Energy',
    'NHPC': 'NHPC',
    'SJVN': 'SJVN',
    'IREDA': 'IREDA',
    'RVNL': 'Rail Vikas Nigam',
    'IRCON': 'IRCON',
    'BDL': 'Bharat Dynamics',
    'COCHINSHIP': 'Cochin Shipyard',
    'HINDPETRO': 'HPCL',
    'PETRONET': 'Petronet LNG',
    'OIL': 'Oil India',
    'ASHOKLEY': 'Ashok Leyland',
    'BHARATFORG': 'Bharat Forge',
    'MRF': 'MRF',
    'APOLLOTYRE': 'Apollo Tyres',
    'BALKRISIND': 'Balkrishna Ind',
    'EXIDEIND': 'Exide',
    'TIINDIA': 'Tube Investments',
    'SONACOMS': 'Sona BLW',
    'LUPIN': 'Lupin',
    'AUROPHARMA': 'Aurobindo Pharma',
    'ALKEM': 'Alkem Labs',
    'MANKIND': 'Mankind Pharma',
    'BIOCON': 'Biocon',
    'GLENMARK': 'Glenmark',
    'MAXHEALTH': 'Max Healthcare',
    'FORTIS': 'Fortis Healthcare',
    'LALPATHLAB': 'Dr Lal PathLabs',
    'COLPAL': 'Colgate Palmolive',
    'DABUR': 'Dabur',
    'MARICO': 'Marico',
    'BRITANNIA': 'Britannia',
    'TATACHEM': 'Tata Chemicals',
    'UPL': 'UPL',
    'PIIND': 'PI Industries',
    'SRF': 'SRF',
    'DEEPAKNTR': 'Deepak Nitrite',
    'BERGEPAINT': 'Berger Paints',
    'ASTRAL': 'Astral',
    'SUPREMEIND': 'Supreme Industries',
    'KEI': 'KEI Industries',
    'VOLTAS': 'Voltas',
    'BLUESTARCO': 'Blue Star',
    'CROMPTON': 'Crompton Greaves CE',
    'KALYANKJIL': 'Kalyan Jewellers',
    'PAGEIND': 'Page Industries',
    'JUBLFOOD': 'Jubilant FoodWorks',
    'ZEEL': 'Zee Entertainment',
    'PVRINOX': 'PVR INOX',
    'NYKAA': 'Nykaa (FSN)',
    'PAYTM': 'Paytm (One97)',
    'POLICYBZR': 'PB Fintech (Policybazaar)',
    'DELHIVERY': 'Delhivery',
    'LICI': 'LIC',
    'GODREJPROP': 'Godrej Properties',
    'OBEROIRLTY': 'Oberoi Realty',
    'PRESTIGE': 'Prestige Estates',
    'PHOENIXLTD': 'Phoenix Mills',
    'MPHASIS': 'Mphasis',
    'OFSS': 'Oracle Fin Services',
    'KPITTECH': 'KPIT Tech',
    'LTTS': 'L&T Technology',
    'TATACOMM': 'Tata Communications',
    'INDUSTOWER': 'Indus Towers',
    'CGPOWER': 'CG Power',
    'THERMAX': 'Thermax',
    'ABCAPITAL': 'Aditya Birla Capital',
    'LICHSGFIN': 'LIC Housing Finance',
    'M&MFIN': 'M&M Financial',
    'POONAWALLA': 'Poonawalla Fincorp',
    'MCX': 'MCX',
    'CDSL': 'CDSL',
    'ANGELONE': 'Angel One',
    'CAMS': 'CAMS',
    'HUDCO': 'HUDCO',
    'CONCOR': 'Container Corp',
    'GMRAIRPORT': 'GMR Airports',
    'JSWENERGY': 'JSW Energy',
    'TORNTPOWER': 'Torrent Power',
    'ADANIENSOL': 'Adani Energy Solutions',
    'ATGL': 'Adani Total Gas',
    'IGL': 'Indraprastha Gas',
    'DALBHARAT': 'Dalmia Bharat',
    'ACC': 'ACC',
    'JKCEMENT': 'JK Cement',
    'UNITDSPR': 'United Spirits',
    'UBL': 'United Breweries',
    'PATANJALI': 'Patanjali Foods',
    'HONAUL': 'Honeywell Automation',
    'WAAREEENER': 'Waaree Energies',
    'PREMIERENE': 'Premier Energies',
    'HYUNDAI': 'Hyundai Motor India',
    'SWIGGY': 'Swiggy',
    'BAJAJHFL': 'Bajaj Housing Finance',
}
NIFTY50 = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'BHARTIARTL', 'SBIN', 'ITC', 'HINDUNILVR', 'LT', 'KOTAKBANK', 'AXISBANK', 'BAJFINANCE', 'BAJAJFINSV', 'ASIANPAINT', 'MARUTI', 'M&M', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'NTPC', 'POWERGRID', 'ONGC', 'COALINDIA', 'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'WIPRO', 'HCLTECH', 'TECHM', 'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'BAJAJ-AUTO', 'BEL', 'CIPLA', 'DRREDDY', 'EICHERMOT', 'ETERNAL', 'GRASIM', 'HDFCLIFE', 'HEROMOTOCO', 'INDUSINDBK', 'JIOFIN', 'NESTLEIND', 'SBILIFE', 'SHRIRAMFIN', 'TATACONSUM', 'TMPV', 'TRENT']
ALL_STOCKS = list(NAMES.keys())

# --------------------------------------------------------------------- sectors
# Sector tags for the curated large/mid-caps. Anything not listed shows as
# "Other" – good enough to group today's picks and see where strength is.
SECTORS = {
    "Banking & Finance": [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'BAJFINANCE', 'BAJAJFINSV',
        'INDUSINDBK', 'BANKBARODA', 'PNB', 'CANBK', 'UNIONBANK', 'IDFCFIRSTB', 'FEDERALBNK',
        'AUBANK', 'BANDHANBNK', 'YESBANK', 'CHOLAFIN', 'SHRIRAMFIN', 'JIOFIN', 'SBICARD',
        'HDFCAMC', 'MUTHOOTFIN', 'RECLTD', 'PFC', 'IRFC', 'BAJAJHLDNG', 'ABCAPITAL', 'LICHSGFIN',
        'M&MFIN', 'POONAWALLA', 'MCX', 'CDSL', 'ANGELONE', 'CAMS', 'HUDCO', 'BSE', 'BAJAJHFL',
    ],
    "Insurance": ['HDFCLIFE', 'SBILIFE', 'ICICIGI', 'ICICIPRULI', 'LICI', 'POLICYBZR'],
    "IT & Tech": [
        'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'PERSISTENT', 'COFORGE', 'MPHASIS',
        'OFSS', 'KPITTECH', 'LTTS', 'TATAELXSI', 'TATACOMM',
    ],
    "Pharma & Healthcare": [
        'SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'ZYDUSLIFE', 'TORNTPHARM', 'LUPIN',
        'AUROPHARMA', 'ALKEM', 'MANKIND', 'BIOCON', 'GLENMARK', 'APOLLOHOSP', 'MAXHEALTH',
        'FORTIS', 'LALPATHLAB',
    ],
    "Auto": [
        'MARUTI', 'M&M', 'TMPV', 'EICHERMOT', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTOR', 'ASHOKLEY',
        'BHARATFORG', 'MRF', 'APOLLOTYRE', 'BALKRISIND', 'EXIDEIND', 'TIINDIA', 'SONACOMS', 'MOTHERSON',
    ],
    "FMCG & Consumer": [
        'HINDUNILVR', 'ITC', 'NESTLEIND', 'TATACONSUM', 'GODREJCP', 'COLPAL', 'DABUR', 'MARICO',
        'BRITANNIA', 'VBL', 'UNITDSPR', 'UBL', 'PATANJALI', 'JUBLFOOD', 'DMART', 'PAGEIND',
        'KALYANKJIL', 'TITAN', 'TRENT', 'NYKAA', 'ETERNAL', 'SWIGGY',
    ],
    "Oil, Gas & Energy": [
        'RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'HINDPETRO', 'PETRONET', 'OIL', 'IGL', 'ATGL',
    ],
    "Power & Utilities": [
        'NTPC', 'POWERGRID', 'TATAPOWER', 'ADANIPOWER', 'ADANIGREEN', 'JSWENERGY', 'TORNTPOWER',
        'ADANIENSOL', 'NHPC', 'SJVN', 'IREDA', 'WAAREEENER', 'PREMIERENE',
    ],
    "Metals & Mining": [
        'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL', 'SAIL', 'NMDC', 'HINDZINC', 'COALINDIA',
    ],
    "Cement & Building": [
        'ULTRACEMCO', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALBHARAT', 'JKCEMENT', 'GRASIM',
        'ASTRAL', 'SUPREMEIND', 'PIDILITIND', 'HAVELLS', 'POLYCAB', 'KEI', 'VOLTAS', 'BLUESTARCO',
        'CROMPTON', 'BERGEPAINT', 'ASIANPAINT',
    ],
    "Infra & Capital Goods": [
        'LT', 'SIEMENS', 'ABB', 'BEL', 'HAL', 'BHEL', 'CUMMINSIND', 'THERMAX', 'CGPOWER', 'HONAUL',
        'BDL', 'MAZDOCK', 'COCHINSHIP', 'SOLARINDS', 'DIXON',
    ],
    "Realty": ['DLF', 'LODHA', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'PHOENIXLTD'],
    "Chemicals": ['TATACHEM', 'UPL', 'PIIND', 'SRF', 'DEEPAKNTR'],
    "Telecom & Media": ['BHARTIARTL', 'IDEA', 'INDUSTOWER', 'ZEEL', 'PVRINOX'],
    "Infra Services & Logistics": [
        'ADANIPORTS', 'ADANIENT', 'INDIGO', 'CONCOR', 'GMRAIRPORT', 'DELHIVERY', 'IRCTC',
        'RVNL', 'IRCON', 'INDHOTEL', 'NAUKRI', 'PAYTM', 'HYUNDAI',
    ],
}
# reverse lookup: symbol -> sector
SYMBOL_SECTOR = {sym: sec for sec, syms in SECTORS.items() for sym in syms}


def sector_of(symbol: str) -> str:
    return SYMBOL_SECTOR.get(symbol, "Other")
UNIVERSES = {
    f"Sab stocks ({len(ALL_STOCKS)})": ALL_STOCKS,
    "Sirf Nifty 50 (fast)": NIFTY50,
}
NIFTY_INDEX = "^NSEI"


def _yf():
    import yfinance as yf  # imported lazily so demo mode works without it
    return yf


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


# ----------------------------------------------------------- robust Yahoo fetch
# yfinance's crumb/cookie handshake (fc.yahoo.com) often fails on office
# networks even when Yahoo's chart API itself is reachable - that silently
# pushes the app into DEMO (fake) data. We hit the public chart endpoint
# directly (browser User-Agent, no crumb), which is far more reliable, and keep
# yfinance only as a fallback.
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"}
_CHART_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")


def _chart_json(symbol_ns: str, rng: str, interval: str):
    import requests
    params = {"range": rng, "interval": interval, "includeAdjustedClose": "true"}
    for host in _CHART_HOSTS:
        try:
            r = requests.get(f"{host}/v8/finance/chart/{symbol_ns}", params=params,
                             headers=_UA, timeout=8)
            r.raise_for_status()
            res = r.json().get("chart", {}).get("result")
            if res:
                return res[0]
        except Exception:
            continue
    return None


def _chart(symbol_ns: str, rng: str, interval: str) -> pd.DataFrame:
    """Download OHLCV via Yahoo's public chart API. Returns a clean df or empty."""
    js = _chart_json(symbol_ns, rng, interval)
    if not js:
        return pd.DataFrame()
    ts = js.get("timestamp")
    if not ts:
        return pd.DataFrame()
    try:
        q = js["indicators"]["quote"][0]
        df = pd.DataFrame(
            {"Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
             "Close": q.get("close"), "Volume": q.get("volume")},
            index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata"),
        )
        adj = (js.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        if adj is not None:
            a = pd.Series(adj, index=df.index)
            factor = (a / df["Close"]).where(df["Close"] > 0, 1.0)
            for col in ("Open", "High", "Low"):
                df[col] = df[col] * factor
            df["Close"] = a
        return _clean(df)
    except Exception:
        return pd.DataFrame()


def download_daily(symbols: list[str], period: str = "3y") -> dict[str, pd.DataFrame]:
    """Bulk download daily bars. Returns {SYMBOL: DataFrame} for those that worked.

    Tries yfinance's fast bulk download first, then fills any gaps with the
    direct chart API so a crumb failure doesn't wipe out the whole scan.
    """
    symbols = list(dict.fromkeys(symbols))
    out: dict[str, pd.DataFrame] = {}
    try:
        yf = _yf()
        # download in chunks - a single huge ticker list is unreliable / rejected
        for i in range(0, len(symbols), 120):
            chunk = symbols[i:i + 120]
            tickers = [f"{s}.NS" for s in chunk]
            try:
                raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                                  auto_adjust=True, threads=True, progress=False)
            except Exception:
                continue
            for s, t in zip(chunk, tickers):
                try:
                    df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                    df = _clean(df)
                    if len(df) >= 260:
                        out[s] = df
                except Exception:
                    continue
    except Exception:
        pass
    # fill whatever bulk missed (or everything, if bulk failed) via the chart API
    missing = [s for s in symbols if s not in out]
    if missing:
        from concurrent.futures import ThreadPoolExecutor
        def one(s):
            return s, _chart(f"{s}.NS", period, "1d")
        with ThreadPoolExecutor(max_workers=12) as ex:
            for s, df in ex.map(one, missing):
                if len(df) >= 260:
                    out[s] = df
    return out


def download_index(period: str = "3y") -> pd.DataFrame:
    df = _chart(NIFTY_INDEX, period, "1d")
    if not df.empty:
        return df
    try:
        yf = _yf()
        df = yf.download(NIFTY_INDEX, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _clean(df)
    except Exception:
        return pd.DataFrame()


def download_intraday(symbol: str) -> pd.DataFrame:
    df = _chart(f"{symbol}.NS", "5d", "5m")
    if not df.empty:
        return df
    try:
        yf = _yf()
        df = yf.download(f"{symbol}.NS", period="5d", interval="5m", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _clean(df)
    except Exception:
        return pd.DataFrame()


def key_stats(symbol: str) -> dict:
    """Raw fundamental numbers used for the quality score. Returns {} if Yahoo
    doesn't give them (the app then falls back to technicals only)."""
    try:
        info = _yf().Ticker(f"{symbol}.NS").info
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}

    def g(k):
        v = info.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    de = g("debtToEquity")
    if de is not None and de > 5:   # yfinance gives D/E as a percent (e.g. 45.0 = 0.45x)
        de = de / 100.0
    peg = g("pegRatio") or g("trailingPegRatio")
    return dict(
        roe=g("returnOnEquity"), de=de,
        rev_growth=g("revenueGrowth"), earnings_growth=g("earningsGrowth"),
        profit_margin=g("profitMargins"),
        pe=g("trailingPE"), pb=g("priceToBook"), peg=peg,
        market_cap=g("marketCap"),
        sector=info.get("sector"), reco=info.get("recommendationKey"),
    )


def fundamentals(symbol: str) -> dict:
    try:
        info = _yf().Ticker(f"{symbol}.NS").info
    except Exception:
        return {}
    keys = {
        "longName": "Company", "sector": "Sector", "marketCap": "Market cap (₹ Cr)",
        "trailingPE": "P/E", "priceToBook": "P/B", "returnOnEquity": "ROE %",
        "debtToEquity": "Debt/Equity", "dividendYield": "Dividend yield %",
        "earningsGrowth": "Earnings growth %", "recommendationKey": "Analyst consensus",
    }
    out = {}
    for k, label in keys.items():
        v = info.get(k)
        if v is None:
            continue
        if k == "marketCap":
            v = f"{v / 1e7:,.0f}"
        elif k in ("returnOnEquity", "earningsGrowth"):
            v = f"{v * 100:.1f}"
        elif isinstance(v, float):
            v = f"{v:.2f}"
        out[label] = v
    return out


# ---------------------------------------------------------------------- demo
def demo_daily(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {s: engine.make_sample_data(seed=i, start_price=200 + 37 * i) for i, s in enumerate(symbols)}


def demo_index() -> pd.DataFrame:
    return engine.make_sample_data(seed=999, start_price=24000)


def demo_intraday(symbol: str, base: float = 1000.0) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(abs(hash(symbol)) % 10000)
    idx = pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min")
    close = base * np.exp(np.cumsum(rng.normal(0.0002, 0.002, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, len(idx))))
    return pd.DataFrame(dict(Open=open_, High=high, Low=low, Close=close,
                             Volume=rng.lognormal(10, 0.5, len(idx))), index=idx)


# ----------------------------------------------------------- full NSE universe
# The curated NAMES above is a fast default. For "all stocks" we pull the
# official NSE equity master list (and the Nifty 500 list) at runtime. If NSE
# blocks the request we quietly fall back to the curated list, so the app always
# works.
_NSE_EQUITY_URLS = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
)
_NSE_500_URLS = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
)


def _fetch_csv(urls):
    import io
    import requests
    headers = dict(_UA)
    headers["Referer"] = "https://www.nseindia.com/"
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            if len(df):
                return df
        except Exception:
            continue
    return None


def nse_all_equities() -> dict[str, str]:
    """Full NSE equity list as {SYMBOL: company name}. Falls back to curated NAMES."""
    df = _fetch_csv(_NSE_EQUITY_URLS)
    if df is not None and "SYMBOL" in df.columns:
        series_col = next((c for c in df.columns if c.upper() == "SERIES"), None)
        if series_col:
            df = df[df[series_col].astype(str).str.strip() == "EQ"]
        name_col = next((c for c in df.columns if "NAME" in c.upper()), None)
        out = {}
        for i, s in enumerate(df["SYMBOL"]):
            sym = str(s).strip().upper()
            nm = str(df[name_col].iloc[i]).strip() if name_col else sym
            if sym:
                out[sym] = NAMES.get(sym, nm)
        # keep curated nice names too (merge), curated wins for display
        merged = dict(out)
        merged.update(NAMES)
        if len(out) > 100:
            return merged
    return dict(NAMES)


def nse_500() -> list[str]:
    """Nifty 500 constituents. Falls back to the curated large-cap list."""
    df = _fetch_csv(_NSE_500_URLS)
    if df is not None:
        col = next((c for c in df.columns if c.strip().lower() in ("symbol", "symbol ")), None)
        if col:
            syms = [str(s).strip().upper() for s in df[col] if str(s).strip()]
            if len(syms) > 100:
                return syms
    return ALL_STOCKS


def live_price(symbol: str) -> dict | None:
    """Latest traded price (Yahoo, may lag 1-15 min). Chart-meta first, then yfinance."""
    js = _chart_json(f"{symbol}.NS", "1d", "1m")
    if js:
        try:
            m = js.get("meta", {})
            last = m.get("regularMarketPrice")
            prev = m.get("previousClose") or m.get("chartPreviousClose")
            if last is not None:
                last = float(last)
                prev = float(prev) if prev is not None else None
                return dict(price=last, prev=prev,
                            chg=(last / prev - 1) * 100 if prev else None,
                            high=float(m.get("regularMarketDayHigh") or last),
                            low=float(m.get("regularMarketDayLow") or last))
        except Exception:
            pass
    try:
        yf = _yf()
        fi = yf.Ticker(f"{symbol}.NS").fast_info
        last = float(fi["last_price"])
        prev = float(fi["previous_close"])
        return dict(price=last, prev=prev, chg=(last / prev - 1) * 100,
                    high=float(fi.get("day_high") or last), low=float(fi.get("day_low") or last))
    except Exception:
        return None
