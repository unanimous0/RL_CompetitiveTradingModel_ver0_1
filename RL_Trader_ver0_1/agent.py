# -*- coding: utf-8 -*-

# Author: EunHwan Koh

# 에이전트 모듈(agent.py)은 투자 행동을 수행하고 투자금과 보유 주식을 관리하기 위한 에이전트 클래스를 가진다.
"""
<Attributes>
initial_balance: 초기 투자금
balance: 현금 잔고
num_stocks: 보유 주식 수
portfolio_value: 포트폴리오 가치(= 투자금 잔고 + 주식 현재가 * 보유 주식 수)

<Methods>
reset(): 에이전트의 상태를 초기화
set_balance(): 초기 자본금을 설정
get_states(): 에이전트 상태를 획득
decide_action(): 탐험 또는 정책 신경망에 의한 행동 결정
validate_action(): 행동의 유효성 판단
decide_trading_unit(): 매수 또는 매도할 주식 수 결정
act(): 행동 수행
"""

import numpy as np


class Agent:
    # 에이전트의 상태를 구성하는 값의 갯수
    # RLTrader에서의 에이전트 상태는 2개의 값을 가지므로 2차원이다.
    STATE_DIM = 2           # 에이전트 상태의 차원: 주식 보유 비율 & 포트폴리오 가치 비율

    # 매매 수수료 및 세금 미고려
    TRADING_CHARGE = 0      # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0         # 거래세 미고려 (실제 0.3%)

    # # <<보완>> 매매 수수료 및 세금 고려
    # TRADING_CHARGE = 0.00015        # 매수 또는 매도 수수료 0.015%
    # TRADING_TAX = 0.003             # 거래세 0.3%

    # 행동
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL]     # 인공 신경망에서 확률을 구할 행동들 (정책 신경망이 확률을 계산할 행동들)
                                            # 본 프로젝트에서는 매수와 매도에 대한 확률만 계산하고, 매수와 매도 중에서 결정한 행동을 할 수 없을 때만 관망 행동을 한다.
    NUM_ACTIONS = len(ACTIONS)              # 인공 신경망에서 고려할 출력값의 갯수


    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=0.05):
        # Environment 객체
        self.environment = environment      # 현재 주식 가격을 가져오기 위해 환경을 참조

        # 최소 매매 단위, 최대 매매 단위, 지연 보상 임계치
        # max_trading_unit을 크게 잡으면 결정한 행동에 대한 확신이 높을 때 더 많이 매수 또는 매도할 수 있다.
        self.min_trading_unit = min_trading_unit       # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit       # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold    # 지연 보상 임계치 (손익률이 이 값을 넘으면 지연 보상이 발생한다.)

        # Agent 클래스의 속성
        self.initial_balance = 0        # 초기 자본금
        self.balance = 0                # 현재 현금 잔고
        self.num_stocks = 0             # 보유 주식 수
        self.portfolio_value = 0        # 포트폴리오 가치 (= balance + num_stocks * {현재 주식의 가격})
        self.base_portfolio_value = 0   # 직전 학습 시점의 포트폴리오 가치 --> 이 기준 포트폴리오 가치는 목표 수익률 또는 기준 손실률을 달성하기 전의 과거 포트폴리오 가치로,
                                        # 현재 포트폴리오 가치가 증가했는지, 감소했는지를 비교할 기준이 된다. (현재 수익이 발생했는지 혹은 손실이 발생했는지를 판단할 수 있는 값이 된다.)

        self.num_buy = 0                # 매수 횟수
        self.num_sell = 0               # 매도 횟수
        self.num_hold = 0               # 관망 횟수
        self.immediate_reward = 0       # 즉시 보상 (에이전트가 가장 최근에 행한 행동에 대한 즉시 보상 값)
                                        # 이 즉시 보상은 행동을 수행한 시점에서 수익이 발생한 상태면 1을, 아니면 -1을 준다.

        # Agent 클래스의 상태
        self.ratio_hold = 0             # 주식 보유 비율 (최대로 보유할 수 있는 주식 수 대비 현재 보유하고 있는 주식 수의 비율)
        self.ratio_portfolio_value = 0  # "포트폴리오 가치 비율" (직전 지연 보상이 발생했을 때의 포트폴리오 가치 대비 현재의 포트폴리오 가치의 비율)


    # 에포크 마다 당연히 에이전트의 상태는 초기화되어야 하므로(학습된 신경망이 초기화되는 것이 아님) 학습 단계에서 한 에포크마다 에이전트의 상태를 초기화한다.
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0


    # 에이전트의 초기 자본금 설정
    def set_balance(self, balance):
        self.initial_balance = balance


    # 에이전트의 상태를 반환
    def get_states(self):
        # 주식 보유 비율은 현재 상태에서 가장 많이 가질 수 있는 주식 수(=포트폴리오가치/현재주가) 대비 현재 보유한 주식의 비율이다.
        # 주식 보유 비율 = 보유 주식 수  / (포트폴리오 가치 / 현재 주가)
        # 주식 수가 너무 적으면 매수의 관점에서 투자에 임하고, 주식 수가 너무 많으면 매도의 관점에서 투자에 임하게 된다.
        # 즉 보유 주식 수 혹은 비율이 투자 행동 결정에 영향을 줄 수 있기 때문에 이 값을 에이전트의 상태로 정하고 정책 신경망의 입력에 포함한다.
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())

        # "포트폴리오 가치 비율" = 포트폴리오 가치 / 기준 포트폴리오 가치
        # "포트폴리오 가치 비율" : 기준 포트폴리오 가치 대비 현재 포트폴리오 가치의 비율 (이때 기준 포트폴리오 가치는 직전에 목표 수익 또는 손익률을 달설했을 때의 포트폴리오 가치)
        # 이 포트폴리오 가치 비율이 0에 가까우면 손실이 큰 것이고, 1보다 크면 수익이 발생했다는 뜻이다.
        # 즉 수익률이 투자 행동 결정에 영향을 줄 수 있기 때문에 이 값을 에이전트의 상태로 정하고 정책 신경망의 입력값에 포함한다.
        self.ratio_portfolio_value = self.portfolio_value / self.initial_balance
        #TODO 분모가 (self.initial_balance가 아닌) 이게 맞는 것 같으므로 확인해볼 것 (둘이 우선 0으로 같긴하다.) (책 설명으로는 아래 내가 쓴게 맞음)
        # self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value

        return (self.ratio_hold, self.ratio_portfolio_value)


    # 행동 결정
    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.0

        # 탐험 결정 by E-Greedy
        if np.random.rand() < epsilon:      # 엡실론의 확률로 무작위 행동을 한다.
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)    # 액션의 수는 2개이다(관망 제외 매수 및 매도) --> 그러나 randint(2)는 0~2 중 랜덤 값이므로 탐험 결과로 매수,매도,관망 중 하나를 선택한다.
        else:                               # (1-엡실론)의 확률로 정책 신경망을 통해 (최적화 된, argmax로) 행동을 결정한다.        (이 부분 관련 설명이 policy_network의 predict 메서드에 있다.)
            exploration = False
            probs = policy_network.predict(sample)      # 각 행동에 대한 확률 - 정책 신경망 클래스의 predict() 함수를 사용하여, 현재 상태에서의 매수와 매도의 확률을 받아온다.
            action = np.argmax(probs)                   # 이렇게 받아온 확률 중에서 가장 큰 값을 선택하여 행동으로 결정한다. (action은 가장 큰 확률 값을 갖는 행동의 위치를 나타낸다.)
            confidence = probs[action]                  # 가장 큰 확률 값을 가져 선택된 행동의 그 확률 값 (단순히 바로 위의 action에 해당하는 (prob의) 확률 값)
        """
        # # <<보완>> 
        # # 위의 경우 정책 신경망 출력 값이, (즉 결과로 나온 확률 중 MAX값을 선택해도 그 확률 자체가 낮을 수도 있으므로), 
        # # 매우 낮은 상황에서 (즉 선택한 행동이 최선인지 불확실한 경우에) 매수 또는 매도를 수행할 수 있다.
        # # 따라서 정책 신경망의 출력 값이 어느 정도의 확률값(임계값, threshold) 이상일 경우에만 결정된 행동을 결정하도록 한다.
        # else:
        #     exploration = False
        #     probs = policy_network.predict(sample)      # 각 행동에 대한 확률
        #     action = np.argmax(probs) if np.max(probs) >= 0.1 else Agent.ACTION_HOLD      # 이때의 threshold는 0.1이다.
        #     confidence = probs[action]
        
        # 유의사항:
        #    만일 이런식으로 threshold를 줘버리면, visualizer에서 구현한 plot 함수에서 가지 값(매수 확률, 매도 확률) 중 하나를 argmax로 결정하고 있어,
        #    실제로는 관망을 선택했음에도 visulaizer에서는 매수 또는 매도 중 하나를 선택해버리는 상황이 생길 수 있다.     
        """

        return action, confidence, exploration      # action: 결정한 행동, confidence: 결정에 대한 확신도, exploration: 무작위 투자 유무


    # 결정한 행동이 특정 상황에서는 수행할 수 없을 수도 있다. 예를 들어 매수를 결정했는데 잔금이 1주를 매수하기에도 부족한 경우나,
    # 매도를 결정했는데 보유하고 있는 주식이 하나도 없는 경우에 결정한 행동을 수행할 수 없다.
    # 따라서 결정한 행동을 수행할 수 있는지를 validate_action()을 통해 확인한다.
    def validate_action(self, action):
        # 디폴트 값이 True이며 아래 조건에 걸리면 False가 된다.
        validity = True
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:     # 우리가 최소 거래 단위를 1로 해놨으니 self.min_trading_unit을
                validity = False                                                                                    # 안 곱해도 되지만, 최소 거래 단위가 1보다 크면 꼭 곱해줘야한다.
        elif action == Agent.ACTION_SELL:
            # 매도할 주식 잔고가 있는지 확인
            if self.num_stocks <= 0 :
                validity = False
        return validity


    # 매수/매도 단위 결정
    # 정책 신경망이 결정한 행동의 확률이 높을 수록 매수 또는 매도하는 단위를 크게 정해준다.
    # 높은 확률로 매수를 결정했으면 더 많은 주식을 매수하고, 높은 확률로 매도를 결정했으면 더 많은 보유 주식을 매도한다.
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading


    # 투자 행동 수행 함수 (행동 준비)
    # 인자로 들어오는 action은 탐험 또는 정책 신경망을 통해 결정한 행동으로 매수와 매도를 의미하는 0 또는 1의 값을 갖는다.
    # confidence는 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률 값이다.
    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻어오기 (매수 금액, 매도 금액, 포트폴리오 가치를 계산할 때 사용된다.)
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화 (에이전트가 행동할 때마다 (state가 바뀔 때 마다) 결정되기 때문에 초기화해야한다. (즉시 보상은 누적되는 값이 아니므로 행동마다 초기화하는 것이 맞다.)
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)

            # 매수 후의 잔금을 확인 (실제 매수 전에 미리 가격을 계산해보고 매수 후 잔금이 어떻게 될지 확인하는 것이다.)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수 (매수 후에 잔금이 0보다 적으면 안되기 때문에 확인해준다.)
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                     self.min_trading_unit
                )       # (최소가 1, 최대가 2라면) 최소 1단위에서 최대 2단위까지 최대한 매수하며, 1.5단위의 경우에는 max보다 작고 min보다 크기 때문에
                        # 그 단위만큼 사는 것이다. (이 1.5단위가 보유 현금으로 최대 구매 가능한 매수 단위인 것이다. 물론 int가 있어서 1.5는 1로 대체될 것이다.)
                        # 즉 "self.balance / (curr_price * (1+self.TRADING_CHARGE))" 가 최대구매가능단위를 뜻하므로 1.5가 나오면
                        # 앞의 min과 max에 따라 1.5만큼 최대 구매 하는 것 (다른 값이 나오면 최대 최소 비교해 본 상황에 따라 다를 것)
                        # 또한, 적어도 1주를 살 수 있는지를 이 메서드 제일 첫 부분인 self.validate_action에서 확인하고 들어오므로 현 메서드 내에서의 잔고로 최소 1주는 살 수 있다.
                        ### (max와 min을 통해, 결정한 매수 단위가 최대 단일 거래 단위를 넘어가면 최대 단일 거래 단위로 제한하고, 최소한 최소 거래 단위만큼을 살 수 있다.)

            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount       # 보유 현금을 갱신
            self.num_stocks += trading_unit     # 보유 주식 수를 갱신
            self.num_buy += 1       # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)

            # TODO 매도 후의 잔금 확인은 없는가? --> 잔금 확인이 아니라, 즉 balance<0이 아니라 self.num_stocks<0을 확인해야할 것 같음 --> 체크 필수!
            #  --> A: 그래서 바로 아래 줄에서 min으로 최소 매도 수량을 정한다. 그렇다면 Q-1.최소 매도 수량은 1일텐데 아예 없으면 문제가 있지 않은가 라는 질문에 대한 답은
            #  --> A-1: 이 메서드의 제일 첫 부분에 self.validate_action으로 매수/매도가 최소 단위로라도 가능한지 확인부터 한다.

            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도 (결정한 매도 단위가 현재 보유한 주식 수보다 많으면 안되므로, 현재 보유 주식 수를 최대 매도 단위로 제한한다.)
            trading_unit = min(trading_unit, self.num_stocks)

            # 수수료를 적용하여 총 매도 금액 산정
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.balance += invest_amount       # 보유 현금을 갱신
            self.num_stocks -= trading_unit     # 보유 주식 수를 갱신
            self.num_sell += 1      # 매도 횟수 증가

        # 관망
        # 관망은 아무 것도 하지 않으므로 보유 주식 수나 잔고에 영향을 미치지 않는다. 대신 가격 변동은 있을 수 있으므로 포트폴리오 가치가 변결 될 수 있다.
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1      # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        # profitloss: 기준 포트폴리오 가치에서 현재 포트폴리오 가치의 등락률을 계산한다.
        # base_portfolio_value: 기준 포트폴리오 가치는 과거에 학습을 수행한 시점의 포트폴리오 가치를 의미한다.
        self.portfolio_value = self.balance + curr_price * self.num_stocks      # 위의 과정을 통해 세 가지 변수 모두 값이 변경되었으므로 포트폴리오 가치를 갱신해준다.
        profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value     # 비율 (소수 / %)

        # 즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        # delayed_reward_threshold는 지연 보상 임계치로, 손익률이 이 값을 넘으면 지연 보상이 발생한다.
        if profitloss > self.delayed_reward_threshold:      # 포트폴리오 가치의 등락률인 profitloss가 지연 보상 임계치를 수익으로 초과하는 경우
            delayed_reward = 1
            # 목표 수익률을 달성하였으므로(달성하고) 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value

        elif profitloss < -self.delayed_reward_threshold:   # 포트폴리오 가치의 등락률인 profitloss가 지연 보상 임계치를 손실로 초과하는 경우
            delayed_reward = -1
            # 손실 기준치를 초과하였으므로(초과하고) 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value

        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
        # RLTrader는 지연 보상(delayed_reward)이 0이 아닌 경우 학습을 수행한다.
        # 즉 지연 보상 임계치를 초과하는 수익이 났으면 이전에 했던 행등들을 잘했다고 판단하여 긍정적으로(positive) 학습하고,
        # 지연 보상 임계치를 초과하는 손실이 났으면 이전 행동들에 문제가 있다고 판단하여 부정적으로(negative) 학습한다.