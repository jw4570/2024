# 슬롯머신

#### 슬롯머신

* ##### 강의 자료 https://drive.google.com/drive/folders/1v3MJeNfn1lvlo_WtB7Fy9yR3e0sFFOoi
* ##### 코랩 https://colab.research.google.com/drive/1ar9zmEt79Ge7zeX-0_4WfNrB-oXDIA-G#scrollTo=04OSWNnYoS3F

### 상태가 없는 강화학습 | 보상 추정 | 데이터 절약 | 

> ### 상태가 없는 강화학습
> 상태가 없는 강화학습 ex) 슬롯머신

> ### 보상 추정
> 평균값 사용

> #### argmax 함수
> 가장 큰 값의 인덱스 반환

> ### 보상 업데이트 데이터 절약
> 평균값에서 값이 추가될 때 업데이트 방법

# 슬롯머신(밴딧) 구현하기
### 1. gymnasium 설치
```python
!pip install gymnasium[classic-control]
```
표준정규분포 불러오기
```python
import numpy as np
np.random.normal()
```
평균 0, 분산 1의 표준정규분포에서 랜덤한 값을 불러옵니다

### 2. 슬롯머신 환경 만들기
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

1. Q의 모든 수를 저장
2. Q의 합과 선택 횟수를 저장 (평균값을 데이터 효율적으로 저장)
3. 슬롯 머신 랜덤 선택 도입

쌤.com/2024

ppt에 있는 것과 추가로 생각한 강화학습 코드 짜기 (숙제)
