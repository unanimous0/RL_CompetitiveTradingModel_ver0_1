import os
import sqlite3

os.chdir("/Users/unanimous0/RL_CompetitiveTradingModel_ver0_1/Project")


lastdate = []
con2 = sqlite3.connect("kospi.db")
cursor2 = con2.cursor()
# cursor2.execute("SELECT open from '000440' limit 20")
cursor2.execute("SELECT date, open, close from '000440' where date like '2015%'")
# cursor2.execute("SELECT open from '000440' where Date(date) BETWEEN Date('20150201') AND Date('20150210')")

lastdate.append(cursor2.fetchall())

lastdate



