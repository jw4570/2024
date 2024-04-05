# 슬롯머신

#### 슬롯머신

* ##### 강의 자료 https://drive.google.com/drive/folders/1v3MJeNfn1lvlo_WtB7Fy9yR3e0sFFOoi
* ##### 코랩 https://colab.research.google.com/drive/1ar9zmEt79Ge7zeX-0_4WfNrB-oXDIA-G#scrollTo=04OSWNnYoS3F

### 상태가 없는 강화학습 | 보상 추정 | 데이터 절약 | 

> ### 상태가 없는 강화학습
> 상태가 없는 강화학습 ex) 슬롯머신

> ### 보상 추정
> 평균값 사용

> ### 보상 업데이트 데이터 절약
> 평균값에서 값이 추가될 때 업데이트 방법

# 슬롯머신(밴딧) 구현하기

## 1. gymnasium 설치
표준정규분포 불러오기
```python
!pip install gymnasium[classic-control]
```

평균 0, 분산 1의 표준정규분포에서 랜덤한 값을 불러온다
```python
import numpy as np
np.random.normal()
```

## 2. 슬롯머신 환경 만들기
```python
import numpy as np
import gymnasium as gym

class BanditEnv(gym.Env):
    def __init__(self, num_bandits):
        self.num_bandits = num_bandits # 슬롯머신의 갯수
        self.action_space = list(range(num_bandits)) # 0~n-1번째 슬롯머신 선택 가능
        self.observation_space = [0]

    def reset(self):
        # self.mean = np.random.normal(size=self.num_bandits) * 10
        # 보상을 정해둠
        self.mean = [8, 9, 7.5, 7, 8.5, 7, 6, 7.5, 8, 8.5]
        return 0

    def step(self, action):
        state = 0 # 슬롯머신의 상태가 없음 > 항상 같음
        mean = self.mean[action]
        reward = mean + np.random.normal()
        done = False
        return state, reward, done, {}
self.mean = [8, 9, 7.5, 7, 8.5, 7, 6, 7.5, 8, 8.5]
```
표준정규분포에서 불러온 값을 슬롯머신의 보상으로 사용할 수 있지만, 제대로된 결과를 실험하기 위해 임의로 값을 설정한다

```python
self.mean = [8, 9, 7.5, 7, 8.5, 7, 6, 7.5, 8, 8.5]
```

슬롯머신에서는 상태가 존재하지 않는다
**(슬롯머신을 선택한 후에 어떤 보상을 받더라도 영향을 미치지 않음)**

보상은 슬롯머신의 가치 + 랜덤값(정규분포)을 사용한다
```python
def step(self, action):
        state = 0 # 슬롯머신의 상태가 없음 > 항상 같음
        mean = self.mean[action]
        reward = mean + np.random.normal()
        done = False
        return state, reward, done, {}
```

## 3. 정책 만들고 실행하기

```python
import numpy as np

class MyPolicy():
    def __init__(self, num_bandits):
        self.num_bandits = num_bandits # 술롯머신 개수 저장

        # 처음 가치 추정치
        initial_q = 100

        self.q = [initial_q]*num_bandits # 평균값
        self.n = [0]*num_bandits # 선택 횟수
        self.epsilon = 0.01 # 랜덤 선택할 확률

    def __call__(self, state):
        action = [np.argmax(self.q), np.random.randint(self.num_bandits)][np.random.random() < self.epsilon] # epsilon보다 작을 경우 랜덤 선택
        # action = np.argmax(self.q)

        # argmax >
        # max(list(range(self.num_bandits)), key = lambda x: self.q[x])
        return action
```
argmax 함수는 배열에서 가장 큰 값의 인덱스를 반환하는 함수이며 다음과 같이 나타낼 수 있다
1. **numpy 라이브러리 사용**
```python
np.argmax(self.q)
```
2. **max 함수로 구현**
```python
max(list(range(self.num_bandits)), key = lambda x: self.q[x])
```

```python
num_bandits = 10 # 슬롯머신 개수
choices = 10000 # 선택 횟수

env = BanditEnv(num_bandits)
state = env.reset()
total_reward = 0
agent = MyPolicy(num_bandits)
reward = 0

cnt = [0]*env.num_bandits


for _ in range(choices):
    action = agent(state)
    state, reward, done, _ = env.step(action)
    # print("Action:", action, "Reward:", reward)

    agent.n[action] += 1
    agent.q[action] += (reward - agent.q[action]) / agent.n[action] # 평균값을 데이터 효율적으로 업데이트 및 저장
    total_reward += reward

print("Total reward:", total_reward)
# print("보상 합:     ", agent.s)
print("선택 횟수:   ", agent.n)
print("기댓값:      ", agent.q)
# print(positive(agent.q))

# import matplotlib.pyplot as plt
# for machine in range(env.num_bandits):
#     plt.scatter([machine]*(len(history[machine])-1), history[machine][1:])
```

**데이터를 효율적으로 업데이트 하는 방법**
```python
agent.q[action] += (reward - agent.q[action]) / agent.n[action]
```


1. Q의 모든 수를 저장
2. Q의 합과 선택 횟수를 저장 (평균값을 데이터 효율적으로 저장)
3. 슬롯 머신 랜덤 선택 도입

쌤.com/2024

*ppt에 있는 것과 추가로 생각한 강화학습 코드 짜기*
