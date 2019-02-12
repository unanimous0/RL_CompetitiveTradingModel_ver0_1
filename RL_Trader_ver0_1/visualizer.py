# Author: EunHwan Koh

# 가시화기 모듈(visualizer.py)은 정책 신경망을 학습하는 과정에서 에이전트의 투자 상황, 정책 신경망의 투자 결정 상황, 포트폴리오 가치의 상황을
# 시간에 따라 연속적으로 보여주기 위해 시각화 기능을 담당하는 가시화기 클래스(Visualizer)를 가진다.
"""
<Attributes>
fig: 캔퍼스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
axes: 차트를 그리기 위한 Matplotlib의 Axes의 클래스 객체

<Methods>
prepare(): Figure를 초기화하고 일봉 차트를 출력
plot(): 일봉 차트를 제외한 나머지 차트들을 출력
save(): Figure를 그림 파일로 저장
clear(): 일봉 차트를 제외한 나머지 차트들을 초기화

<Visualizer 모듈이 만들어내는 결과가 나타내는 정보>
Figure 제목 : 에포크 및 탐험률(by E-Greedy)
Axes 1: 종목의 일봉 차트
Axes 2: 보유 주식 수 및 에이전트 행동 차트
Axes 3: 정책 신경망 출력 및 탐험 차트
Axes 4: 포트폴리오 가치 차트

<일봉 차트를 그리기 위해 Matplotlib에 있던 mpl_finance 모듈을 사용한다.
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class Visualizer:
    def __init__(self):
        self.fig = None
        self.axes = None


    def prepare(self, chart_data):
        # 캔버스를 초기화하고 4개의 차트를 그릴 준비
        # subplots() 메서드는 2개의 변수를 튜플로 반환한다. 첫 번째는 Figure 객체이고, 두 번째는 이 Figure 클래스 객체에 포함된 Axes 객체의 배열이다.
        # subplots 메서드의 sharex, sharey 파라미터: Share the x or y axis with sharex and/or sharey.
        # The axis will have the same limits, ticks, and scale as the axis of the shared axes.
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
        for ax in self.axes:        # axex는 Figure 클래스 객체에 포함된 Axes 객체의 배열이므로 for문을 통해 하나씩 처리한다.
            # 보기 어려운 과학적 표기 (단위) 비활성화
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)

        # 차트 1. 일봉 차트 (일봉 차트는 전체 학습 과정에서 동일하기 때문에 한 번만 호출되는 prepare() 함수에서 그려준다.)
        #                  --> 그 외의 남은 3가지의 차트는 에포크마다 다르기 때문에 plot() 함수에서 그려준다.
        self.axes[0].set_ylabel("Env.")

        # 거래량 가시화
        x = np.arange(len(chart_data))      # chart_data의 길이에 맞춘 인덱스 배열
        volume = np.array(chart_data)[:, -1].tolist()       # tolist(): Return the array as a (possibly nested) list
        self.axes[0].bar(x, volume, color='b', alpha=0.3)   # alpha - transparency

        ax = self.axes[0].twinx()       # 2개의 y축 사용: y축을 volume(좌), stock_price(우) 2개로 만든다.

        # horizen-stack이므로, 인덱스 배열(x.reshape(-1,1) + 차트 데이터 일부인 OHLC(np.array(chart_data)[:, 1:-1]) - hstack을 쓰려면 행의 수가 같아야한다.
        ohlc = np.hstack((x.reshape(-1,1), np.array(chart_data)[:, 1:-1]))
        # self.axes[0]에 봉 차트 출력
        # 양봉은 빨간색, 음봉은 파란색으로 출력
        candlestick_ohlc(ax, ohlc, colorup='r', colordown='b')      # 이 메서드의 입력은 Axes 객체와 ohlc 데이터이다. 이때 두 줄 위의 ax는 Axes 객체 배열 중
                                                                    # 첫 번째 객체에 twinx 속성을 추가해준 것일 뿐, Axes 객체 중 하나임에는 변함이 없다.


    # 일봉 차트를 제외한 나머지 3개의 차트는 에포크마다 다르기 때문에 plot() 함수에서 그려준다.
    def plot(self, epoch_str=None, num_epoches=None, epsilon=None, action_list=None, actions=None, num_stocks=None,
             outvals=None, exps=None, learning=None, initial_balance=None, pvs=None):
        """
        <plot() 함수의 Parameters>
        :param epoch_str: Figure 제목으로 표시할 에포크
        :param num_epoches: 총 수행할 에포크 수
        :param epsilon: 탐험률
        :param action_list: 에이전트가 수행할 수 있는 전체 행동 리스트
        :param actions: 에이전트가 수행한 행동 배열
        :param num_stocks: 주식 보유 수 배열
        :param outvals: 정책 신경망의 출력 배열
        :param exps: 탐험 여부 배열
        :param initial_balance: 초기 자본금
        :param pvs: 포트폴리오 가치 배열
        """

        x = np.arange(len(actions))     # 모든 차트가 공유할 x축 데이터 (actions의 길이에 맞춘 인덱스 배열) --> 위에서 subplots() 사용시 sharex를 True로 해주었었다.
        # 이 x축 데이터는 모든 차트가 공유한다. 이때 actions, num_stocks, outvals, exps, pvs의 모든 크기가 같기 때문에
        # 이들 중 하나인 actions의 크기만큼 배열을 생성하여 x축으로 사용한다. (즉 모든 변수들의 크기(길이)가 같다는 뜻이다.)

        actions = np.array(actions)     # 에이전트의 행동 배열 --> actions는 리스트이고 Matplotlib는 Numpy-배열을 입력으로 받기 때문에 배열로 변환한다.
        outvals = np.array(outvals)     # 정책 신경망의 출력 배열 --> outvals는 리스트이고 Matplotlib는 Numpy-배열을 입력으로 받기 때문에 배열로 변환한다.
        pvs_base = np.zeros(len(actions)) + initial_balance     # 초기 자본금 배열 (초기 자본금 직선): 포트폴리오 가치 차트에서 초기 자본금에 직선을 그어서,
        # 포트폴리오 가치와 초기 자본금을 쉽게 비교할 수 있도록 배열(pvs_base)로 준비하여 직선이 되도록 한다. --> 예를 들어, np.zeros(10)은 0으로 구성된 10개의 1차원 배열이고,
        # 여기에 5000을 더하면 5000으로 구성된 10개의 1차원 배열이 된다. 즉 x축 길이만큼의 배열에 초기자본금을 더하면 해당 그래프에 초기 자본금을 처음부터 끝까지 직선으로 나타낼 수 있는 것이다.


        # 차트 2. 에이전트 상태 (행동, 보유 주식 수) : 에이전트가 수행한 행동을 배경의 색으로 표시하고, 보유 주식 수를 라인 차트로 그린다.
        # TODO --> 두 번째 for문에서 첫 반복에서는 '매수-빨강'으로, 두 번째 반복에서는 '매도-파랑'으로 "x에 있는 행동 중" 매수, 매도에 맞게 해당 색으로 axes[1]을 칠하는게 맞는가? --> 체크 필수!
        # TODO      --> 왜냐하면 x는 단순 범위(or 숫자/인덱스의 모임)에 불과한데 x[actions==actiontype]으로 매수 또는 매도가 어떤 인덱스(위치)에 있는지를 파악할 수 있는지를 모르겠슴. (enumerate도 아니고)
        colors = ['r', 'b']
        for actiontype, color in zip(action_list, colors):      # zip()은 두 개의 배열에서 같은 인덱스의 요소를 순서대로 묶어주므로, 매수가 red, 매도가 blue로 나타나도록 순서를 맞춰준다.
            for i in x[actions == actiontype]:                  # 예를 들어 zip([1,2,3], [4,5,6])은 [(1,4),(2,5),(3,6)]이 된다.
                self.axes[1].axvline(i, color=color, alpha=0.1)     # axvline(): 인지로 들어온 x(여기선 i)축 위치에 세로로 선을 긋는 함수로써 배경 색으로 행동을 표시한다.
        self.axes[1].plot(x, num_stocks, '-k')      # 보유 주식 수 검은 실선으로 그리기


        # 차트 3. 정책 신경망의 출력 및 탐험
        # action이 Exploit이 아닌 Explore인 경우
        for exp_idx in exps:        # exps: 탐험 여부 배열로써, 이 배열이 탐험을 수행한 x축 인덱스를 가지고 있다.
            # 탐험을 한 x축 인덱스에 대해 노란색 배경으로 표시하기
            self.axes[2].axvline(exp_idx, color='y')

        # TODO --> outvals는 2차원 배열인데, [[매수,매도], [매도,매수], [매도,매수], [매수,매도]...] 이런 식의 모양인지 확인 (그래서 outval.argmax() ==0 또는 1인 것 같은데) --> 체크 필수!
        # TODO      --> outvals가 2차원은 맞지만 [[매수,매도], [매도,매수]...]가 아니라 element가 숫자(확률값)인 것 같은데... (0의 자리가 가장 큰 확률이면 매수고, 1의 자리가 가장 큰 확률이면 매도이도록)
        # action이 Explore가 아닌 Exploit인 경우: 탐험을 하지 않은 매수/매도 지점에서는 빨간색 및 파란색으로 표시한다.
        for idx, outval in zip(x, outvals):
            color = 'white'
            if outval.argmax() == 0:
                color = 'r'     # 매수면 빨간색
            elif outval.argmax() == 1:
                color = 'b'     # 매도면 파란색
            # 행동을 빨간색 또는 파란색 배경으로 그리기
            self.axes[2].axvline(idx, color=color, alpha=0.1)

        styles = ['.r', '.b']       # 빨간점, 파란점
        for action, style in zip(action_list, styles):
            # 정책 신경망의 출력을 빨간색 또는 파란색 점으로 그리기 --> x축 전체 중 각 하나씩 위치에 대해서(x[0], x[1], ...) 매수와 매도를 전부 표현하고, 두 점의 높낮이 차이를 비교하면 매수와 매도 중 어떤 행동을 선택했는지 알 수 있다.
            # TODO --> outvals가 위의 TODO에 쓴 것처럼 [[0.9,0.1], [0.3,0.7] ...] 이런 형태라면, 아래의 outvlas[:, action]은 어느 때나 매수와 매도의 확률이 같은 값으로 정해져 있는 것 아닌가? --> 체크 필수!
            # TODO      --> 그리고 애초에 outvals가 이런 모양이라면, 그리고 outvals[:, action]은 예를 들면 outvals[:,'매수'] 이런 모양일텐데, 이러면 element가 전자는 확률값이고, 후자는 str(행동)이라 안맞지않나?
            # TODO          --> outvals[:, action]은 첫 반복에서 매수인 것들을, 두 번째 반복에서 매도인 것들을 모아서 한 번에 나타내는 것 같음 --> 그래도 행동(str)으로 확률값(숫자)을 어떻게 나오게 하는가?
            self.axes[2].plot(x, outvals[:, action], style)     # 매수에 대한 정책 신경망의 출력 값은 빨간색 점으로, 매도에 대한 정책 신경망의 출력 값은 파란색 점으로 그린다.
                                                                # 빨간색 점이 파란색 점보다 위에 위치하면 매수하고, 그 반대의 경우에는 매도를 수행한다.
                                                                # action은 매수와 매도 값들을 가져 2가지의 값을 가지고 있으므로, 빨간점과 파란점을 동시에 표현 수 있도록 style이 .r과 .b로 구성되어있다.


        # 차트 4. 포트폴리오 가치
        # fill_between()은 x축 배열과 두 개의 y축 배열을 입력으로 받는다. 두 개의 y축 배열의 같은 인덱스 위치의 값 사이에 색을 칠한다. (where 옵션으로 조건 지정 가능)
        self.axes[3].axhline(initial_balance, linestyle='-', color='gray')      # 초기 자본금을 가로로 그음으로써 손익을 쉽게 파악할 수 있게 한다.
        self.axes[3].fill_between(x, pvs, pvs_base, where=pvs>pvs_base, facecolor='r', alpha=0.1)       # pvs: 포트폴리오 가치 배열
        self.axes[3].fill_between(x, pvs, pvs_base, where=pvs<pvs_base, facecolor='b', alpha=0.1)       # pvs_base: 초기 자본금 배열
        self.axes[3].plot(x, pvs, '-k')

        # TODO --> learning이 무엇인가?
        for learning_idx, delayed_reward in learning:       # 학습을 수행한 위치를 표시한다.
            # 학습 위치 표시
            if delayed_reward > 0:
                self.axes[3].axvline(learning_idx, color='r', alpha=0.1)
            else:
                self.axes[3].axvline(learning_idx, color='b', alpha=0.1)

        # epoch 및 탐험 비율
        self.fig.suptitle("Epoch %s/%s (e=%.2f)" % (epoch_str, num_epoches, epsilon))
        # self.fig.suptitle("Epoch {es}/{ne} (e=%{ep})".format(es=epoch_str, ne=num_epoches, ep=epsilon))

        # 캔버스 레이아웃 조정
        plt.tight_layout()      # Figure의 크기에 알맞게 내부 차트들의 크기를 조정
        plt.subplots_adjust(top=.9)


    # Figure 초기화
    def clear(self, xlim):      # (입력으로 받는 xlim: 모든 차트의 x축 값 범위를 설정해 줄 튜플)
        for ax in self.axes[1:]:
            ax.cla()        # matplotlib.pyplot.cla(): clear the current axes -->  학습 과정에서 변하지 않는 환경에 관한 차트(1번 차트)를 제외하고, 이 전에 그린 그 외의 차트들을 초기화한다.
            ax.relim()      # limit을 초기화한다. --> 이 전에 설정한 차트의 x축과 y축 값의 범위를 초기화한다.
            ax.autoscale()  # 스케일 재설정 (자동 크기 조정 기능 활성화)

        # y축 레이블 재설정
        self.axes[1].set_ylabel('Agent')    # Agent: Agent's actions & Number of holded stocks
        self.axes[2].set_ylabel('PN_Exp')   # PN_Exp: Outputs of Policy Network and Exploration
        self.axes[3].set_ylabel('PV')       # PV: The value of ortfolio
        for ax in self.axes:
            ax.set_xlim(xlim)       # x축 lim 재설정 (x축 값의 범위 설정)
            ax.get_xaxis().get_major_formatter().set_scientific(False)      # x축과 y축의 값을 있는 그대로 보여주기 위해 단위 등의 과학적 표기 비활성화
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            ax.ticklabel_format(useOffset=False)        # x축 간격을 값과 상관없이 일정하게 설정 (이는 x축 간격을 값에 맞게 설정한 것과는 다르다.)
                                                        # 이렇게 하는 이유는 일봉 차트의 경우 x축을 날짜로 지정하면 토요일 및 일요일과 같이 휴장하는 날에는 해당 차트가 비게 되기 때문이다. - 휴장 부분 표현 X


    # Figure 저장
    def save(self, path):
        plt.savefig(path)
