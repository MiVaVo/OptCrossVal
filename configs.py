from datetime import datetime

import numpy as np

TECH = ['AAPL',  # AAPLE
        'MSFT',  # MICROSOFT
        "FB",  # FB
        "NVDA",  # NVDA
        "BABA",  # BABA
        "NFLX",
        "AMD",
        "AMZN"

        ]
OIL_GAZ = ["RDS-B",  # Royal Dutch Shell
           "TOT",  # TOTAL S.A.
           "BP",  # BP
           "GRA"  # W. R. Grace & Co.
           ]
AIRLINES = [
    "DAL",  # Delta Air Lines
    # "UAL", #United Airlines Holdings,
    "LUV",  # Southwest Airlines,
    "SAVE",  # Spirit Airlines
    "AAL",  # American Airlines Group
    "RYAAY"
]
CLOTHING = ["GPS",
            "COLM",  # Columbia Sportswear Company
            ]
ENTERAINMENT = ["CCL",  # Carnival
                "DIS",  # The Walt Disney Company
                "RCL",  # Royal Caribbean Cruises Ltd.
                ]
BIOPHARM = ["DLA",  # Delta Apparel
            "ABC",  # AmerisourceBergen Corporation (ABC)
            ]

SP = ["UPS", "SPG", "EQR", "NVDA", "ABC", "ZBH", "ZION", "FISV", "CTAS", "SYK", "INTU", "RHI", "EOG", "DVN", "TIF", "A",
      "CTXS", "XLNX", "ADI", "VMC", "BBY", "NTAP", "AFL", "CMS", "AGN", "CTL", "CCL", "AES", "RF", "COF", "CINF", "TFC",
      "YUM", "PGR", "APA", "EFX", "SCHW", "CAH", "ADBE", "AZO", "AON", "CMA", "ALL", "BK", "AMAT", "BSX", "MU", "LUV",
      "UNH", "MSFT", "KEY", "UNM", "EMN", "CSCO", "COST", "IPG", "LIN", "AMGN", "AEE", "MRO", "ADSK", "ORCL", "GL",
      "NWL", "ECL", "NKE", "C", "PNC", "HD", "AVY", "MMC", "SYY", "HRB", "MDT", "GPS", "JWN", "ITW", "PH", "DOV", "TJX",
      "CNP", "NOC", "PKI", "APD", "NUE", "BLL", "HAS", "LMT", "HES", "PHM", "LOW", "T", "VZ", "LB", "CAG", "OXY",
      "AAPL", "BF", "SNA", "SWK", "WMT", "ADM", "GWW", "MAS", "ADP", "FDX", "PCAR", "AIG", "WBA", "VFC", "TXT", "INTC",
      "TGT", "MMM", "AXP", "BAC", "CI", "DUK", "LNC", "TAP", "NEE", "DIS", "WFC", "IFF", "JPM", "WMB", "HPQ", "GPC",
      "JNJ", "BAX", "BDX", "LLY", "MCD", "NEM", "GIS", "CLX", "CSX", "CMI", "EMR", "SLB", "SHW", "ABT", "HON", "HWM"]
INDEXES = ['^GSPC']
DJIA = ["MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "DD", "XOM", "GE", "GS", "HD", "IBM", "INTC",
        "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "RTX", "UNH", "VZ", "V", "WMT"]
# OTHERS=["000001.SS",'BIDU',"BABA","NVDA","FB"]

# COMPANIES_LIST=OIL_GAZ+CLOTHING+ENTERAINMENT+AIRLINES+TECH+OTHERS+SP
COMPANIES_LIST = AIRLINES + OIL_GAZ + CLOTHING + ENTERAINMENT + TECH
COMPANIES_LIST = np.unique(COMPANIES_LIST).tolist()
start_date = datetime.strptime('2005-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2019-10-20', '%Y-%m-%d')

import os

os.listdir("C:/PycharmProjects/finance/")
