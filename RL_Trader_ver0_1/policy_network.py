# Author: EunHwan Koh

# 정책 신경망 모듈(policy_network.py)은 투자 행동을 결정하기 위해 신경망을 관리하는 정책 신경망 클래스를 가진다.
"""
<Attributes>
model: LSTM 신경망 모델
prob: 가장 최근에 계산한 각 투자 행동별 확률    # Agent 모듈을 봐도 prob은 각 행동들에 대한 확률을 모아놓은 것이며, 이 확률은 predict 메서드를 통해 구핼 수 있다.

<Methods>
reset(): prob 변수를 초기화
predict(): 신경망을 통해 투자 행동별 확률을 계산
train_on_batch(): 입력으로 들어온 배치 데이터로 학습을 진행한다.
save_model(): 학습한 신경망을 파일로 저장
load_model(): 파일로 저장한 신경망을 로드

<LSTM 신경망의 구조>
Input Layer(17차원) - Hidden Layer1(256차원) - Hidden Layer2(256차원) - Hidden Layer3(256차원) - Output Layer(2차원)
입력층은 17차원: 학습 데이터 차원인 15차원과 에이전트 상태 차원인 2차원을 합함.
출력층은 2차원: 투자 행동인 매수와 매도로 2차원을 가짐.
"""


import numpy as np
from keras.optimizers import sgd    # Learning Algorithm: Stochastic Gradient Descent (SGD)
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, l_rate=0.01):
        self.input_dim = input_dim
        self.l_rate = l_rate

        # LSTM 신경망
        self.model = Sequential()       # keras에서 Sequential 클래스: 전체 신경망을 구성하는 클래스

        self.model.add(LSTM(256, input_shape=(1, input_dim), return_sequences=True, stateful=False, dropout=0.5))    # Hidden Layer 1
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))    # Hidden Layer 2
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))   # Hidden Layer 3
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))       # Output Layer
        self.model.add(Activation('sigmoid'))   # Activation Function

        self.model.compile(optimizer=sgd(lr=l_rate), loss='mse')
        self.prob = None


    def reset(self):
        self.prob = None

    # TODO 바로 아래 줄에 쓴 주석 맞는지 체크 필수!
    # 신경망을 통해서 학습 데이터와 에이전트 상태를 합한 17 차원의 입력을 받아서(즉 state를 입력 받아서), 매수와 매도가 수익을 높일 것으로 판단되는 확률(즉 Q-Learning의 Qpred 같은)을 구한다.
    # Sequential 클래스의 predict() 함수는 여러 샘플을 한번에 받아서 신경망의 출력을 반환한다. 하나의 샘플에 대한 결과만을 받고 싶어도 입력값을 샘플의 배열로 구성해야 하기 때문에 2차원 배열로 재구성한다.
    # 정책 신경망의 출력은 매수, 매도를 했을 때의 포트폴리오 가치를 높일 확률을 의미한다.

    # 이 predict 매서드는 Agent 클래스의 decide_action 매서드가 사용한다.
    # decide_action 메서드는 탐험 시에는 매수,매도,관망 중 한 가지를 랜덤으로 선택하게 되며, 탐험이 아닐 시에는 이 predict 함수가 17차원의 sample을 입력으로 받아 매수,매도 2가지를 output으로 내놓게 된다.
    # "관망은 어디 있느냐"라는 질문에 대한 답으로는, 탐함이 아니면 우선 실제 행동을 실행하는 Agent 클래스의 act 메서드가 첫 부분에서 validate_action을 통해 매수, 매도를 할 수 없는 상황인 경우 관망을 선택한다.
    # 매수, 매도를 할 수 있는 것이 validate되면 act 메서드는 decide_action 메서드에서 결정된 action을 토대로 실제 매수와 매도 중에서 행동을 실행하게 된다.
    def predict(self, sample):
        # TODO sample이 state일 것이고, 이를  reshape를 하면 2차원의 [1,None]일 것이다. 또한 2차원이므로 [0]으로 필요한 1차원 배열을 가져와 predict함수에 넣는다 --> 체크 필수!
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob


    # 입력으로 들어온 학습 데이터 집합 x와 레이블 y로 정책 신경망을 학습시킨다.
    # keras의 train_on_batch()는 입력으로 들어온 학습 데이터 집합(즉 1개의 배치)으로 신경망을 한 번 학습한다.
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)


    # 정책 신경망을 학습하는데 많은 시간과 컴퓨팅 자원이 소모되므로, 한번 학습한 정책 신경망을 저장해 놓고 필요할 때 불러와서 사용하도록 한다.
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)     # (weight를 network로 간주)


    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)