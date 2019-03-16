# -*- coding: utf-8 -*-

# Author: EunHwan Koh

### 1. fix_yahoo_finance 사용 ###

from pandas_datareader import data
import fix_yahoo_finance

def get_data_from_FYF():

    fix_yahoo_finance.pdr_override()    # <== that's all it takes: 데이터 획득 방식을 크롤릴으로 변

    chart_data = data.get_data_yahoo('005930.KS', '2015-01-01', '2017-12-31')

    # print(chart_data.head())

    return chart_data



### 2. 증권사 API 사용 ###

import sys
from PyQt5.QAxContainer import *

class GetDataWithAPI():

    def __init__(self):
        super().__init__()

        self.kw = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        self.kw.dynamicCall("CommConnect()")
        self.kw.OnReceiveTrData.connect(self.receive_data)

        self.kw.dynamicCall("SetInputValue('종목코드', '00593')")

        self.kw.dynamicCall("CommRqData('opt10001_req', 'opt10001', 0, '0101')")

    def receive_data(self, screen_no, rqname, trcode, recordname, prev_next, data_len, err_code, msg1, msg2):
        if rqname == "opt10001_req":
            name = self.kw.dynamicCall("CommGetData('trcode', '', 'rqname', 0, '종목명')")
            volume = self.kw.dynamicCall("CommGetData('trcode', '', 'rqname', 0, '거래량')")

            print(name)
            print(volume)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     app.exec_()