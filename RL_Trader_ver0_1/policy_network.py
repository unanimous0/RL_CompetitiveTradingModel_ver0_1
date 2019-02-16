# Author: EunHwan Koh

# 정책 신경망 모듈(policy_network.py)은 투자 행동을 결정하기 위해 신경망을 관리하는 정책 신경망 클래스를 가진다.

    # 정책 경사(Policy Gradient)는 분류(Classfication) 문제를 푸는 강화학습으로 볼 수 있다. 즉 (Q-러닝 처럼 기대 손익(수익률)을 예측하는 것이 아니라)
    # 현재 상태에서 어떤 행동을 하는 것이 가장 좋을지를 확률적으로 판단하는 것이다. (어떠한 상태에서의 주어진 행동에 대한 확률 값들을 인공 신경망으로 모델링한다.)
    # 본 프로젝트에서 행동은 정책 신경망(Policy Network)에서 행동별 확률을 구함으로써 결정하고, 정책 경사는 결정된 행동을 바탕으로 투자를 진행하면서 발생하는 보상과 학습 데이터로 학습한다.
    # 인공신경망(LSTM)으로 각 행동에 대한 확률을 구해서 높은 확률을 갖는 행동을 선택하고, 그 선택에 따른 결과들을 즉시 보상과 지연 보상으로 긍적적인 학습과 부정적인 학습을 하는 것이다.
    # 그리고 에포크가 늘어날 수록, 즉 경험이 늘어날 수록 긍정적인 보상을 얻을 수 있는 방향으로 (=LSTM에서 얻는 확률이 더 확실하게 나오는 방향으로) 행동을 하게 될 것이다.
    # 확률을 구하는 것 자체는 인공신경망(LSTM)의 역할이며, 여기서 선택된 행동을 바탕으로 즉시/지연 보상을 얻으며 긍정적 행동을 해나가는 방향으로 진행시키는 것이 정책 경사의 역할이다.

    # TODO (강화학습 기법 중 하나인 정책 경사는 현 상태에서 어느 행동이 좋을지를 확률적으로 판단하는 것이니 인공 신경망과의 결합이 필수인 것인가?)
    #   "정책 (경사) 기반 강화학습(Policy(Gradient)-based Reinforcement Learning): 가치 함수를 토대로 행동을 선택하지 않고, 상태(State)에 따라 행동을 선택 <-- 인공신경망이 정책(Policy)를 근사(Gradient)한다.
    #    정책 기반 강화학습에서는 인공신경망이 정책을 근사한다." --> 인공신경망의 입력은 상태(의 특징 벡터)가 되고 출력은 각 행동을 할 확률이 된다. 따라서, 정책신경망에서는 출력 층의 활성 함수로 Softmax 함수를 사용한다.
    #    (※ 정책 신경망(Policy Network): 정책 기반 강화학습에서 정책을 근사하는 인공신경망) & (※ 본 프로젝트에서 에이전트의 상태(State)는 '행동'과 '보유 주식 수'를 의미한다.)
    #    또한 이 정책 경사는 목표함수 J(𝛉)를 최적화하는 방법은 '경사 상승법(Gradient Ascent)'을 사용한다. 이는 '경사 하강법(Gradient Descent)'과 다르게 오류함수를 최소화하는 것이 아니라 목표함수를 최대화하는 것이다."
    #    출처: [밑바닥부터 시작하는 딥러닝, 사이토 고키 저, 한빛출판사]


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
출력층은 2차원: 투자 행동인 매수와 매도에 대한 확률로 2차원을 가짐.
"""


import numpy as np
from keras.optimizers import sgd    # Learning Algorithm: Stochastic Gradient Descent (SGD)
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, l_rate=0.01):     # 여기서 input_dim은 15+2로 17차원이며, output_dim은 매수/매도에 대한 확률로 2차원이다.
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
        # TODO 책 [밑바닥부터 시작하는 딥러닝]에 다음과 같은 문장이 있으므로, self.model.compile() 함수를 다시 따져봐야 한다. (책 읽은 후 다시 확인 필수!)
        #   또한 마지막 activation 함수의 인자가 반드시 'softmax'일 필요는 없는가?
        """
          Softmax
            정책신경망에는 model.complie이 없습니다.
            딥살사에서는 인공신경망의 학습을 MSE 오차를 줄이는 방식으로 진행했습니다.
            정책신경망에서는 MSE 오차 함수를 사용하지 않기 때문에 따로 오류 함수를 정의해야 합니다.
            (※정책신경망 : 정책 기반 강화학습에서 정책을 근사하는 인공신경망)
        """

        self.prob = None


    def reset(self):
        self.prob = None

    # 신경망을 통해서 학습 데이터와 에이전트 상태를 합한 17차원의 입력을 받아서(즉 state를 입력 받아서), 매수와 매도가 수익을 높일 것으로 판단되는 확률(즉 Q-Learning의 Qpred 같은)을 구한다.
    # sample을 2차원으로 reshape 하는 이유: Sequential 클래스의 predict() 함수는 여러 샘플을 한번에 받아서 신경망의 출력을 반환한다. 하나의 샘플에 대한 결과만을 받고 싶어도 입력값을 샘플의 배열로 구성해야하기 때문에 2차원 배열로 재구성한다.
    # TODO 아래 TODO에도 써놨지만 2차원으로 재구성하는 이유는 알겠지만, 왜 3차원으로 재구성했다가 2차원으로 돌아오는지를 모르겠다.

    # 이 predict 매서드는 Agent 클래스의 decide_action 매서드가 사용한다. decide_action 메서드는 탐험 시에는 매수,매도,관망 중 한 가지를 랜덤으로 선택하게 되며,
    # 탐험이 아닐 시에는 이 predict 함수가 17차원의 sample을 입력으로 받아 매수,매도 2가지를 output으로 내놓게 된다. 정책 신경망의 출력은 매수, 매도를 했을 때의 포트폴리오 가치를 높일 확률들(매수에 대한 확률 하나와 매도에 대한 확률 하나로 총 2차원)이다.
    # "관망은 어디 있느냐"라는 질문에 대한 답으로는, 탐험이 아니면 우선 실제 행동을 실행하는 Agent 클래스의 act 메서드가 시작 부분에 있는 validate_action 함수 호출을 통해 매수, 매도를 할 수 없는 상황인 경우 관망을 선택한다.
    # 매수, 매도를 할 수 있는 것이 validate되면(True값이 나오면) act 메서드는 decide_action 메서드에서 결정된 action (매수와 매도 중 확률값이 큰 행동)을 토대로 실제 행동을 실행하게 된다.
    def predict(self, sample):
        # TODO 만일 sample의 shape가 (-1,17)이라면, "self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]"은
        #   self.prob = self.model.predict(np.array(sample)) 과 같을텐데 저런 식으로 굳이 reshape를 해준 이유가 무엇일까? (뭐하러 2차원을 3차원으로 만들었다가 뒤에 [0]을 붙여서 다시 2차원으로 빼는 것인가? 그리고 왜 괄호가 중복으로 쌓여있을까?
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob


    # 입력으로 들어온 학습 데이터 집합 x와 레이블 y로 정책 신경망을 학습시킨다.
    # keras의 train_on_batch()는 입력으로 들어온 학습 데이터 집합, 즉 "1개"의 배치로 신경망을 "한 번" 학습한다.
    # train_on_batch: "Runs a single gradient update on a single batch of data."
    #   Returns: "Scalar training loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).
    #   The attribute model.metrics_names will give you the display labels for the scalar outputs."
    #   출처: [https://keras.io/models/model/]
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)


    # 정책 신경망을 학습하는데 많은 시간과 컴퓨팅 자원이 소모되므로, 한번 학습한 정책 신경망을 (HDF5 파일 형식으로) 저장해 놓고 필요할 때 불러와서 사용하도록 한다.
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)     # (weight를 network로 간주)


    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)