#  CS231n: Deep Learning for Computer Vision

스탠포드 대학의 CS231n 강의를 수강하며 진행한 과제와 학습 내용을 정리하는 레포지토리입니다.

## 프로젝트 개요
- **수강 기간**: 2025년 12월 24일 ~ (진행 중)
- **목표 일정**: 5주 완성 (수강 4주 + 프로젝트 1주)
- **현재 상태**: 3주차 진행 중

## 주차별 진행 상황 (Roadmap)

### 1주차 (완료)
- [x] 강의 1강 ~ 4강 수강
- [x] Assignment 1 시작 및 완료
- **핵심 내용**: Image Classification, kNN, Linear Classifiers, Neural Networks

### 2주차 (완료)
- [x] 강의 4강 ~ 9강 수강
- [x] Assignment 2 시작
- **핵심 내용**: Backpropagation, CNN Architectures, Training Neural Networks

### 3주차 (진행 중)
- [x] 강의 10강 (Video Understanding) 수강
- [ ] 강의 11강 ~ 13강 수강
- [x] Assignment 2 종료
- [ ] Assignment 3 시작

### 4주차 (예정)
- [ ] 강의 14강 ~ 18강 수강
- [ ] Assignment 3 종료
- [ ] Project 주제 선정 (Kaggle 활용 나만의 모델 구현)

---

## Daily Log & TIL

### 1월 9일 이전
- 1강 ~ 9강 수강 완료
- Assignment 1, 2 진행

### 2026년 1월 10일
- **Assignment2: Image Captioning with Vanilla RNNs** 완료
- **학습 내용***
  - 강의를 들었을 때는 확실하지 않던 Vanilla RNNs에 대한 개념 이해
  - Embedding에 대한 이해와 PyTorch 활용에 관한 이해

### 2026년 1월 11일
- **Lecture 10: Video Understanding** 수강 완료
- **학습 내용**:
  - Video Classification에서 3D CNN을 사용할 때 연산량 문제와 해결책(T, H, W Trade-off)
  - Training(Short clips)과 Testing(Average over clips)의 차이점 이해

---

## Lecture and Assignment reviews
*(과제나 개념 공부를 하면서 얻었던 깨달음)*

### Lecture 1: Introduction
#### 배운 점
1. **이 강의를 통해 배우고자 하는 것**
   - Computer Vision이라는 분야에서 사용되는 Deep Learning 기술 (예로부터 Vision의 진화는 Intelligence의 진화를 이끌었다)
2. **전반적인 Computer Vision의 발전과 AI의 발전 과정 및 간단한 Overview**

### Lecture 2: Image Classification with Linear Classifiers

> **Main Keywords:** Image Classification, K-Nearest Neighbors, Hyperparameters, Linear Classifiers, Softmax, SVM

#### 배운 점

1. **이미지 분류의 문제점**
   - 이미지를 분류하는 과정에서 여러 문제점 존재 (배경, 다양한 종류, 그림자, 다양한 형태 등)
2. **L1 Distance에 대한 이해**
   - 좌표축을 따라 이동하면서 거리를 측정하기에 축이 회전한다면 거리가 달라짐
3. **Hyperparameters 설정 방식**
   - 값을 설정할 때 여러 방식이 존재 (Validation Set을 사용, Cross-Validation 방식을 사용 등)
4. **Softmax Loss의 역할**
   - 우리가 Linear Classifiers에서 $Wx + b$ 로 얻은 scores를 확률로 바꿔주는 장치
5. **Softmax Initialization (Sanity Check)**
   - Softmax의 Initialization 상황에서 모든 scores가 같다면, loss는 $-\log(1/C) = \log(C)$ 이어야 함 (추후 Sanity Check에서 사용)
6. **SVM Loss의 원리**
   - Classes의 scores에 대해서 정답 class의 scores과 얼마나 차이가 나는지를 확인 ($\max(0, s_j - s_{y_i} + 1)$ 을 사용)
7. **SVM Initialization**
   - SVM에서 $W$가 작아서 $s$가 0에 가깝다면, loss는 $C - 1$

---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. K-NN의 공간 문제
**Q.** 수업 중, "K-NN의 방식에선 빈 공간이 많으면 의미가 없다"라는 말에 대한 의문
> **A.** 우리가 판단하려는 공간을 충분히 채울 만큼의 다양한 sample을 확보해야 함.

##### 2. KL Divergence와 Cross Entropy
**Q.** 수업 중 나온 Kullback-Leibler divergence와 Cross Entropy의 관계에 대한 의문
> **A.** Cross Entropy는 KL divergence를 사용한 단순한 objective function이며, 주로 One-hot encoding에 사용됨.

##### 3. Softmax Loss vs SVM Loss (정확도 측면)
**Q.** SVM은 단순히 정답 scores가 다른 scores보다 margin 이상의 차이를 보여주면 loss가 늘어나지 않음. 반면 Softmax는 loss가 아무리 정확하더라도 0이 될 수 없기 때문에, "Softmax가 더욱 정확한 결과를 낸다고 할 수 있는지"에 관한 의문.
> **A.**
> - **SVM:** 특정 조건만 만족한다면 더 이상 그 부분의 성능 향상을 요구하지 않음.
> - **Softmax:** 아무리 classify가 성공했더라도 계속해서 loss를 발생시키기에, 더욱 확실하게 분류하도록(좋은 성능을 내게) 유도함.

### Lecture 3: Regularization and Optimization

> **Main Keywords:** Regularization, Optimization, Gradient Descent, SGD, Momentum, RMSProp, Adam

#### 배운 점

1. **Regularization에 대해**
   - 기존엔 단순히 Overfitting을 막는 추상적인 존재라고 생각하였지만, 이 Regularization이 단순한 수식으로 표현될 수 있다는 점을 깨달음
   - 또한 Regularization strength가 Hyperparameter라는 점도 새롭게 배움
   - Overfitting을 막기도 하지만, curvature를 더함으로 optimization을 더욱 잘되게 한다는 점 또한 배움
2. **Gradient의 종류**
   - Numerical gradient와 Analytic gradient가 존재.
   - Analytic gradient를 사용하지만 혹시나 발생할 에러를 위해 이를 Numerical gradient와 비교하는 gradient check 필요
3. **SGD에 대해**
   - N개의 데이터에 대해 전부 연산하는 것은 비효율적이므로 batch size만큼의 mini batch를 가져와서 이 데이터를 이용하여 gradient를 구함
   - 하지만 단순히 특정 데이터들에서만 gradient를 얻기에 중간에 Noise로 인한 문제와 local minima나 saddle point에서 갇히는 문제 등이 발생 -> 이를 해결하기 위해 SGD + Momentum이 사용
   - x -= learning_rate * dx
4. **SGD + Momentum에 대해***
   - velocity라는 새로운 변수를 도입해서 이전의 정보들을 현재 gradient와 합침 -> 이는 상대적으로 Noise에 간섭 받을 확률을 낮춤
   - 또한 velocity를 도입함으로써, 기존 방식이던 SGD에서 minimal value에 근접할수록 학습 속도가 느려지는 문제를 해결
   - vx = rho * vx + dx, x -= learning_rate * vx
   - 이와 비슷하지만 기울기를 이동할 곳에서 구하는 Nesterov 방식도 존재
5. **RMSPROP에 대해**
   - SGD에서의 문제점인 특정 데이터 축으로 과도하게 gradient가 descent되는 현상을 막기 위해 도입
   - grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx, x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
   - 가파른 방향은 조금씩 진행하고, 평평한 방향은 빠르게 진행
6. **Adam에 대해**
   - first_moment와 second_moment를 도입해서, RMSProp + Momentum을 구현한 방법
   - Bias correction 필요 -> beta1, beta2가 0.9, 0.999 같은 굉장히 큰 값으로 설정되기에 first_moment와 second_moment의 초기값은 0에 가까워지는 문제가 발생
7. **Learning rate에 대해**
   - 이 값은 Hyperparameter
   - 좋은 Hyperparameter를 찾기 위해선 여러 방법이 존재 (Learning rate를 학습 중 변경, Cosine의 형태에 맞게 변경, Linear, Inverse sqrt, Linear Warmup 이후 이전 방식 사용)
8. **Order Optimization에 대해**
   - 기존의 방식은 First-Order Optimization
   - Second-Order Optimization의 경우 현재의 gradient에 맞는 이차 곡선을 확인 후, 이 곡선의 최솟값이 있는 곳에 빠르게 도달
   - 하지만 이 방식은 O(N^3)이 필요한 만큼 오래 걸림 -> 그럼에도 속도가 빠름 
---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. AdamW에서 Weight Decay 구현 문제
**Q.** AdamW에서 Weight Decay를 따로 구현해준다는데 compute_gradient에서 이미 구해준 값을 굳이 다시 넣는 이유에 대한 의문.
> **A.** 우리가 의도한 페널티가 first_unbias / (np.sqrt(second_unbias) + 1e-7)에서 scaling의 영향을 받기 때문에 명시적으로 learning_rate * lambda * w를 적어줘야함.

##### 2. Second-Order Optimization 실제 사용 여부
**Q.** 교수님의 말씀에 의하면, 만약 파라미터의 개수가 적은 모델을 학습시키려고 한다면 Second-Order Optimization을 활용할 수 있다고 하셨는데 연산의 복잡성에도 불구 이를 사용할 이유에 대한 의문.
> **A.** 훨씬 적은 횟수로 정답을 찾고, Learning Rate 튜닝 스트레스가 없음, 하지만 파라미터가 많으면 계산이 불가능할 정도로 느리기에 계산이 가능한 범위에서는 사용하는 것을 추천.

### Lecture 4: Neural Networks and Backpropagation

> **Main Keywords:** 

#### 배운 점

1. 
   -

---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

### Lecture 5: Image Classification with CNNs

> **Main Keywords:** 

#### 배운 점

1. 
   -

---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

### Lecture 6: CNN Architectures

> **Main Keywords:** 

#### 배운 점

1. 
   -

---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

### Lecture 7: Recurrent Neural Networks

> **Main Keywords:** 

#### 배운 점

1. 
   -

---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

### Lecture 8: Attention and Transformers

> **Main Keywords:** 

#### 배운 점

1. 
   -

---

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

  






