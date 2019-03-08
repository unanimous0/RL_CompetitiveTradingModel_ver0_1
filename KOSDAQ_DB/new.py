# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from KOSDAQ_DB.Webreader2 import Webreader2
from KOSDAQ_DB.Kiwoom import Kiwoom
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame

con = sqlite3.connect("/Users/unanimous0/RL_CompetitiveTradingModel_ver0_1/KOSDAQ_DB/pyquery.db")
cursor = con.cursor()
