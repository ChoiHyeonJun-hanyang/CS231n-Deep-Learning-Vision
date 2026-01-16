# CS231n - Deep Learning for Computer Vision

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

### 3주차 (완료)
- [x] 강의 10강 ~ 13강 수강
- [x] Assignment 2 종료
- [x] Assignment 3 시작

### 4주차 (예정)
- [ ] 강의 14강 ~ 18강 수강
- [ ] Assignment 3 종료
- [ ] Project 주제 선정 (Kaggle 활용 나만의 모델 구현)

---

## Daily Log & TIL

### 2025년 12월 24일 ~ 2026년 1월 8일
1. Lecture 1~9 수강
2. Assignment1 완료 및 Assignment2 - Batch Normalization ~ Convolution Neural Networks 완료

### 2026년 1월 9일 
1. Assignment2 - PyTorch on CIFAR-10 종료

### 2026년 1월 10일
1. Assignment2 - Image Captioning with Vanilla RNNs
2. GitHub 레포지토리 생성 및 과제 옮기기

### 2026년 1월 11일
1. 지금까지의 수강한 Lecture 1 ~ 9 Review
2. Lecture 10: Video Understanding 수강

### 2026년 1월 12일
1. Lecture 11: Large Scale Distributed Training 수강

### 2026년 1월 13일
1. Lecture 12: Self-Supervised Learning 수강
2. GitHub에 강의 5, 6강 정리

### 2026년 1월 14일
1. Assignment3 - Image Captioning with Transformers 시작
2. GitHub에 강의 7, 8강 정리

### 2026년 1월 15일
1. Assignment3 - Image Captioning with Transformers 진행

### 2026년 1월 16일
1. Assignment3 - Image Captioning with Transformers 종료
2. Lecture 13: Generative Models 1 수강

### 2026년

### Lecture 1: Introduction
#### 배운 점
1. **이 강의를 통해 배우고자 하는 것**
   - Computer Vision이라는 분야에서 사용되는 Deep Learning 기술 (예로부터 Vision의 진화는 Intelligence의 진화를 이끌었다)
2. **전반적인 Computer Vision의 발전과 AI의 발전 과정 및 간단한 Overview**

---

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
   - 우리가 Linear Classifiers에서 Wx + b 로 얻은 scores를 확률로 바꿔주는 장치
5. **Softmax Initialization (Sanity Check)**
   - Softmax의 Initialization 상황에서 모든 scores가 같다면, loss는 -log(1/C) = log(C) 이어야 함 (추후 Sanity Check에서 사용)
6. **SVM Loss의 원리**
   - Classes의 scores에 대해서 정답 class의 scores과 얼마나 차이가 나는지를 확인 (max(0, s_j - s_{y_i} + 1) 을 사용)
7. **SVM Initialization**
   - SVM에서 W가 작아서 s가 0에 가깝다면, loss는 C - 1

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

---

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

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. AdamW에서 Weight Decay 구현 문제
**Q.** AdamW에서 Weight Decay를 따로 구현해준다는데 compute_gradient에서 이미 구해준 값을 굳이 다시 넣는 이유에 대한 의문.
> **A.** 우리가 의도한 페널티가 first_unbias / (np.sqrt(second_unbias) + 1e-7)에서 scaling의 영향을 받기 때문에 명시적으로 learning_rate * lambda * w를 적어줘야함.

##### 2. Second-Order Optimization 실제 사용 여부
**Q.** 교수님의 말씀에 의하면, 만약 파라미터의 개수가 적은 모델을 학습시키려고 한다면 Second-Order Optimization을 활용할 수 있다고 하셨는데 연산의 복잡성에도 불구 이를 사용할 이유에 대한 의문.
> **A.** 훨씬 적은 횟수로 정답을 찾고, Learning Rate 튜닝 스트레스가 없음, 하지만 파라미터가 많으면 계산이 불가능할 정도로 느리기에 계산이 가능한 범위에서는 사용하는 것을 추천.

---

### Lecture 4: Neural Networks and Backpropagation

> **Main Keywords:** Neural Networks, Activation Function, Backpropagation, Chain Rule, Computation Graph, Fully-Connected Layer

#### 배운 점

1. **Activation Function에 대해**
   - Activation function을 사용하는 이유는 linear * linear = linear이고 이는 class를 분류하는 것에 있어서 한계가 발생하기에 non-linearity를 만들기 위해서 사용
   - 주로 ReLU (max(0, x))를 사용
   - Sigmoid의 경우 0과 그 근처를 제외한 나머지 부분에선 gradient가 0에 근접하는 문제가 발생하므로 middle layer에서 사용하지 않음 (너무 좁은 범위로 값을 옮기려고 하기에)
2. **BackPropagation에 대해**
   - Input layer와 가까울수록, gradient를 도출해내는 것은 매우 힘듦
   - 따라서 Chain Rule을 사용하여 Upstream gradient * Local gradient 의 방식으로 gradient를 구함
   - add gate, mul gate, copy gate, max gate로 gradient flow를 표현 가능하며 각각 gradient 유지, 상대 input 곱해주기, gradient 더해주기, 승자독식의 형태를 가짐
   - 
#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Neural Network의 크기
**Q.** ppt에선 Neural Network의 크기를 regularier로 사용하지 말고, strong regularization을 사용하라고 하던데 이 이유에 대한 의문.
> **A.** Neural Network의 크기를 키워서 더욱 복잡한 분류도 가능하게 하는 것은 중요, Overfitting을 방지하기 위해서 model의 성능을 저하시키기보단 Regularization을 강하게 해 Underfitting을 유도하는 것이 바람직.

---

### Lecture 5: Image Classification with CNNs

> **Main Keywords:** Image features, Convolution Layer, Padding, Stride, Pooling Layer, Translation Equivariance

#### 배운 점

1. **Image features에 대해**
   - Image를 raw data로 넣는 것이 기존의 방법이라면, 다양한 방법을 통해 Image의 features를 추출하여 이를 model training에 이용
   - Ex) Color Histogram, Histogram of Oriented Gradients(HoG), CNN을 통한 features 추출 
2. **CNN의 개념 등장 과정**
   - 기존의 Neural Networks는 N x C x H x W로 이루어진 여러개의 Image data를 N x (C * H * W)의 형태로 변형하여 계산 -> 이는 Spatial structure of images를 붕괴
   - 이를 극복하기 위해서 filter라는 개념을 도입한 Convolution Neural Networks라는 개념이 등장
3. **기존의 Fully Connected Layer에 대해**
   - 32 x 32 x 3 image -> stretch to 3072 x 1로 이미지를 변경한 후 Wx + b의 연산을 진행
   - 이를 추상적으로 이해한다면, W가 (D, C)의 형태일때 우리는 Image를 C개의 template와 비교 후 activation 과정을 걸쳐서 scores를 도출함
   - 여기서 비교란 내적으로 진행, 내적의 경우 template와 image가 비슷하다면 높은 값이 나오고 다르다면 0이 나옴
4. **Convolution Layer에 대해**
   - C_in x H x W 형태의 image가 있다면 이를 F개의 filter를 사용, 각각의 filter size는 C_in x HH x WW
   - 이 결과로 F개의 H_out x W_out 크기의 activation map을 얻을 수 있음, activation map의 각각의 element는 Filter * 특정 위치의 pixels + bias로 계산
   - H_out = (H - HH + 2P) / S + 1, W_out = (W - WW + 2P) / S + 1 (Padding, Stride 도입 이후)
   - 이러한 Conv 연산 이후 Activation function을 사용 -> non-linearity를 도입하기 위해
   - 상대적으로 out layer에 가까운 Conv의 filter일수록 더욱 디테일한 형태를 인식할 수 있음
5. **Padding에 대해**
   - 위의 방식대로 진행한다면 Conv 연산 후, 나온 activation map의 크기가 기존의 image보다 작아지는 문제가 발생
   - 이 문제를 해결하기 위해 기존의 Image에서 가장자리를 확대한 후, 해당 pixel의 값을 0으로 채우는 Padding이 도입
6. **Stride에 대해**
   - Filter를 3 x 3의 크기로 사용한다면 activation map의 element가 확인할 수 있는 부분은 1 + 2 * L x 1 + 2 * L
   - 우리가 한 element가 전체 사진을 전부 인식하게끔 만드는데 너무 많은 Layer가 필요함
   - 이를 극복하기 위해 Stride라는 개념을 도입
   - 이는 Dowmsample의 예시로 이를 도입 시, Output = (W - K + 2P) / S + 1이 되고 output이 S로 나눠지기에 층이 깊어질수록 한 칸이 담고 있는 정보가 지수적으로 증가
7. **Pooling Layer에 대해**
   - Stride와 마찬가지로 Downsample을 하는 하나의 방법임
   - Hyperparmeters: Kernel Size (=Filter Size), Stride, Pooling function
   - Max Pool 방식의 경우 ReLU와 유사한 점이 존재하고, non-linearity를 도입하기에 꼭 ReLU를 사용해야 하는것은 아님
   - 하지만 Avg Pool 방식은 linear + linear이기에 명시적인 Activation function이 필요요
 8. **Translation Equivariance와 CNN**
   - Conv와 Translate는 순서가 바뀌어도 같은 결과를 도출
   - 이는 Features of images가 위치에 관계없다는 것을 의미

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. filter 및 가중치의 학습 방식
**Q.** filter나 weight가 backpropagation에 의해서 학습되고 이는 우리가 따로 주제를 선정하지 않더라도 알아서 학습을 한다는 점에 의문
> **A.** 처음에 weight initialization에서 random하게 설정, 각각의 가중치 부분마다 filter마다 약간의 차이가 존재 이 차이가 filter간의 균형을 깨고, 이 점이 filter로 하여금 각각 다른 부분을 학습시킴

##### 2. Pooling Layer에서의 Padding
**Q.** Convolution Layer의 경우 크기를 맞춰주기 위해서 Padding을 해주는데 Pooling Layer에서도 Padding을 하는지에 대한 의문
> **A.**
> - Convolution Layer의 경우 목적이 Feature을 추출하는 것이고 Padding의 역할은 크기 유지 + 가장자리 정보에 대한 인식 저하를 막는 것
> - Pooling Layer는 정보를 버리는 것이 목적이기에 가장자리에 0을 추가하여 크기를 보존하려고 할 필요가 없음 (Downsample의 목적에서 벗어남)

##### 3. Translation Equivariance와 CNN의 관계 
**Q.** 교수님의 설명을 해석하건대 Translation Equivariance는 CNN에게 굉장히 중요한 개념이라던데 그 이유에 대한 의문 
> **A.**
> - 기존의 방식인 MLP의 경우 형태를 분류할 수는 있지만, 형태가 image에서 등장하는 위치가 달라진다면 새로운 패턴으로 인식
> - 반면 CNN의 경우엔 image에 대해서 weight sharing filter를 통해서 각각 다른 부분을 관찰함, 여기서 이 Equivariance에 의해 다른 위치에서 같은 패턴이 나타난다고 하여도 CNN은 weight sharing filter를 통해 이해 가능

---

### Lecture 6: CNN Architectures

> **Main Keywords:** Normalization, Dropout, Activation Functions, CNN Architectures, Weight Initialization, Data Preprocessing, Data augmentation, Transfer Learning, Hyperparameter Selection

#### 배운 점

1. **Normalization에 대해**
   - 데이터를 우리가 원하는 예쁜 모양으로 강제로 맞춰주는 과정
   - 모델이 학습하면서 레이어를 거칠 때마다 특정 데이터가 너무 크거나 작은 문제 발생 -> Normalization을 통해서 학습 속도가 증가함
   - Layer Norm은 같은 image 내의 데이터를 정규화, Batch Norm은 같은 데이터 차원 내에서의 데이터를 정규화
   - 이외에도 Group Norm, Instance Norm 존재
2. **Dropout에 대해**
   - forward pass에서 random하게 몇몇 뉴런을 특정 확률에 따라 0으로 둠
   - 여기서 Probability of dropping은 Hyperparameter임
   - 이는 학습 과정에서 model로 하여금 데이터를 학습할 때, 일부 특징을 무시하고 학습하도록 강제하기에 Regularization의 기능을 함
   - test time에선 모든 특징을 전부 반영해서 classify해야하기에 모든 뉴런을 활성화시킨 상태에서 최종 output에 p(확률)을 곱해줘야함
3. **Activation Functions에 대해**
   **Sigmoid**
      - 값을 0과 1 사이의 확률값으로 변경해주는 Sigmoid의 경우, 모든 정의역에서 미분가능하지만, 문제가 존재
      - 많은 층에서 Sigmoid를 연속적으로 사용하는 경우, 계속해서 gradient 값이 작아짐 -> 이는 Backpropagation 과정에서 문제가 발생
      - 특히 절댓값이 큰 경우, gradient가 0에 가까워지는 문제가 커짐
   **ReLU**
      - Non-linearity 추가 가능
      - 하지만 양수와 음수가 많이 섞인 데이터의 경우, ReLU는 0보다 작은 값을 0으로 처리하기에 해당 가중치 부분이 죽는 경우가 발생
      - 이를 해결하기 위해 GELU, Leaky ReLU 등 다양한 응용형태가 존재
4. **CNN Architectures에 대해** 
   **VGGNet**
      - 3x3 Conv (stride 1, pad 1), 2x2 MAX POOL (stride 2)를 사용
      - 3x3 Conv를 3번 연달아서 사용하는 것은 7x7 Conv와 같은 효과를 냄 (effective receptive field의 측면에서), 하지만 # of parameters 27C << 49C 로 3x3 Conv가 훨씬 효율적
      - 또한 Activation Funtion은 주로 Convolution 연산 이후 실행하는데 activation function의 실행 횟수가 늘어나기에 더욱 복잡한 형태를 분류 가능
   **ResNet**
      - 기존의 Conv, ReLU와 Pool을 순서대로 쌓는 방식의 model은 깊이가 깊어질수록 모델의 성능이 떨어지는 문제가 존재
      - 이는 모델의 성능이 매우 좋아 과적합 (Overfitting) 의 상태가 된 것이 아니라, 모델의 학습이 제대로 이루어지지 않아서 발생한 문제
      - 이 문제를 해결하기 위해 Identity mapping을 이용한 ResNet이 등장
      - H(x) = F(x) + x의 형태와 2개의 3x3 Conv Layer가 있는 residual block로 구성
      - 2x2 Max Pool (stride = 2) 을 이용, 이를 이용시 H와 W의 길이가 절반으로 줄고 전체 데이터의 크기는 1/4배 되기에 데이터 손실을 줄이기 위해서 Filter의 개수를 2배로 늘림림
5. **Weight Initialization에 대해**
   - Weight Initialization 과정에서 이를 랜덤하게 생성할 때 곱해지는 값에 따라 Layer가 깊어지면서 Weight의 mean과 std가 폭발적으로 증가하거나 감소할 수 있음
   - 우리가 원하는 Initialization은 Layer의 깊이에 관계없이 항상 일정한 mean과 std를 가지는 것
   - 이를 위해 Weight matrix (D_in x D_out) 의 크기 중 하나인 D_in을 활용한 sqrt(2/D_in) 을 곱해주는 Kaiming Initialization을 활용 
6. **Data Preprocessing에 대해**
   - 구해놓은 per-channel mean을 data에서 빼고 per-channel std로 나눠줌
   - 다양한 Class의 다양한 사진을 모아둔 ImageNet의 평균과 표준편차를 이용하는 것도 괜찮음
7. **Data augmentation에 대해**
   - 원본 데이터를 일부 변경하는 것
   - Horizontal Flips: 상하좌우 변환
   - Random crops and scales: Training에선 랜덤한 위치에 랜덤한 크기의 box를 sample로 사용, Test에선 다양한 크기의 박스들을 우리가 고정한 위치에서 여러번 테스트한 후, 이를 평균내서 classify
   - Color Jitter: 밝기 조절, Cutout: Image에서 일부를 특정 색으로 처리
   - 이 방식들은 전부 모델들로 하여금 해당 형태의 본질적인 특성을 인식하게 유도, 이는 일종의 Regularization의 효과를 가짐
8. **Transfer Learning에 대해**
   - 기존에 학습된 CNN model의 마지막 FC Layer를 제외한 나머지를 그 모델이 class를 분류할 때 사용하는 feature를 얻는 것에 사용하는 것
   - 우리는 우리가 분류할 데이터와 해당 모델이 학습한 데이터의 차이 정도와 데이터 개수를 확인 후, 이후 수정사항을 결정
9. **Hyperparameter Selection에 대해**
    - 처음 손실값을 확인, 이후 적은 양의 데이터를 Overfitting 시킴
    - 적절한 Learning Rate를 찾을 때까지 학습을 반복
    - 이 과정에서 사용되는 Learning Rate는 크게 Random Search와 Grid Search로 나눌 수 있음
    - Random Search는 Grid Search에선 알기 힘든 Hyperparameter의 중요도를 확인할 수 있기에 더욱 효율적인 경우가 많음

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 3x3 Conv & 7x7 Conv
**Q.** 3x3 Conv 3개와 7x7 Conv를 비교할 때, 단순히 가중치 저장을 위한 메모리만을 고려하였는데, Layer가 추가로 생기면서 발생하는 시간적, 메모리적 측면에 대한 의문
> **A.** 3x3 Conv가 곱셈 횟수가 적으니 시간적으로는 forward, backward 에서 전부 7x7 Conv보다 빠르지만, backpropagation을 위해 저장해야하는 중간값의 메모리는 증가함. 

##### 2. ResNet의 장점
**Q.** 교수님께선 ResNet이 다른 Networks에 비해 기존의 모델을 이용할 수 있어서 효율적이라고 하는데, 이 말에 대한 의문
> **A.** 우리가 더 작은 모델을 가져왔다고 가정, 이 모델의 정확도를 증가시키기 위해 뒤에 추가적인 Layer를 도입했다고 한다면, 기존의 방식의 경우 Random하게 가중치를 초기화하기에 작은 모델의 성능을 그대로 구현하기 힘듦. 하지만 ResNet은 추가된 Layer의 가중치를 0으로 설정 후, 이전 block의 데이터를 이후 block의 데이터로 보내는 방식을 사용해서 더 작은 모델의 성능에서부터 모델 디자인을 시작할 수 있음.

---

### Lecture 7: Recurrent Neural Networks

> **Main Keywords:** Recurrent Neural Networks, Truncated Backpropagation, Image Captioning, Long Short Term Memory (LSTM)

#### 배운 점

1. **Vanilla RNNs에 대해**
   - h_t = f_W(h_{t-1}, x_t), y_t = f_Why(h_t)의 과정을 거쳐서 output y_t가 도출
   - 여기서 사용되는 W, Why는 모든 time step에서 동일하게 사용
   - 보통 activation function으로 tanh가 사용 (값의 폭발을 막기 위해서)
   - One to Many에선 첫번째 input을 제외한 input을 이전 RNNs Block의 output으로 대체
   - 학습은 Backpropagation을 사용해서 학습
2. **Truncated Backpropagation에 대해**
   - Time의 값이 커질수록 backpropagation이 힘들어지기에, Backpropagation을 some smaller number of steps에서만 진행
   - Forward pass는 처음부터 끝까지 전부 이용
   - 또한 Backpropagaion 과정에서 필요한 Loss도 특정 small number of steps에서만 추출, 이렇게 구한 gradient를 모든 block에 적용 가능 -> 가중치를 공유하기 때문
3. **RNN tradoffs**
   **RNN의 장점**
      - input의 길이가 상관 없음
      - 모든 가중치를 공유하기에 상대적으로 모델의 크기에 비해 메모리 소모량이 적음
   **RNN의 단점**
      - Recurrent computation이 매우 느림. 이는 Current hidden state를 구하기 위해선 Prev hidden state가 필요하기 때문
      - hidden state의 값이 시간이 지나면서 다른 input과 많이 섞이기 때문에 이전의 hidden state의 값에 도달하기가 힘듦
4. **Image Captioning with RNNs**
   - 분류하고자 하는 Image를 CNN에 넣어서 feature을 추출해내고, 이 특징들을 RNN의 처음 hidden state에 넣음
   - 그리고 <Start> 토큰을 input에 넣고 다음의 input은 이전 block의 output을 넣어서 One to Many 형식의 RNNs을 구현
   - Visual Question Answering의 경우 CNN에서 feature을 뽑아내고 이 특징을 활용한 RNN을 통해 특정 질문을 학습한 후 그 질문에 대한 확률을 얻어냄
5. **Multilayer RNNs에 대해** 
   - RNNs에서 hidden layer를 서로 연결해서 구현
   - 각각의 층마다 다른 가중치를 사용함
   - 구하려는 hidden state의 time 값이 크고, depth가 깊을수록 hidden state를 구할 때 필요한 연산량이 많아짐
6. **LSTM에 대해서**
   - Vanilla RNN Gradient Flow는 기본적으로 gradient가 계속해서 같은 값이 곱해지는 형식이기에 그 값에 따라 매우 커지거나 작아짐
   - 총 4개의 gate가 추가로 도입되었으며, 가장 중요한 아이디어는 forget gate로 이전의 정보를 ResNet의 아이디어처럼 Highway 형식으로 다음 cell state에 일정부분 그대로 넘겨줌
   - 이 아이디어는 ResNet의 아이디어와 비슷하지만 약간은 다른 점이 존재함
   - ResNet은 원본 input 그대로를 보내지만, LSTM의 경우 forget gate의 값을 곱해가며 보내기에 길이가 길어진다면 만약 f가 1과 매우 가까운 값이 아닌 경우에 gradient vanishing이 발생할 수 있음
   - 그럼에도 불구하고, forget gate를 1로 만든다면 ResNet과 비슷하게 동작하기에 이를 극복 가능

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Vanilla RNNs에서 사용하는 activation function
**Q.** PPT에선 Vanilla RNNs에서 hidden state를 구할 때, tanh를 사용하던데 이 활성화함수는 sigmoid와 마찬가지로 특정 범위에서 gradient vanishing이 발생해서 backpropagation이 발생하는 문제에 대한 의문
> **A.** tanh는 모든 값들을 (-1, 1)의 범위에 가두는 성질을 가지고 있음. RNNs는 기존의 h_t를 W_hh에 계속해서 곱하면서 다음 구조로 넘기는 형식이기에 만약 값을 특정 범위로 제한하지 않는다면 값이 급증하거나 급락하는 문제가 발생. 이를 막기 위해서 gradient vanishing의 문제가 있더라도 tanh를 사용하였고, 이는 초기 RNNs의 성능을 제한하게 되면서 이후 LSTM의 등장 배경이 됨.

##### 2. Truncated Backpropagation에서의 문제점
**Q.** Truncated Backpropagation에서 마지막 부분 일부에서만 loss를 구한다고 하면 그 부분보다 앞에서 큰 오류가 발생했을 때, 이를 확인하고 고칠 수가 없는 것에 대한 문제점
> **A.** loss를 확인한 부분보다 앞에서 오류가 발생했다면, 이 방법에선 이를 고칠 수 없음. 그럼에도 불구하고 이렇게 하지 않는다면 메모리 사용에서의 문제와 앞선 부분의 오류를 확인한다고 해도 backpropagation에서 gradient vanishing의 문제로 우리가 원하는 부분에 대한 학습이 잘 안될 가능성이 높음.

##### 3. LSTM에서 Forget gate 문제
**Q.** ResNet과 비슷한 아이디어를 사용한 LSTM에선 왜 굳이 gradient vanishing이 발생할 확률을 forget gate를 도입해서 남겨두었는지에 대한 의문
> **A.** ResNet과 달리 LSTM은 tanh를 사용, h_t는 c_t의 값을 tanh를 사용해서 값을 -1부터 1 사이로 고정시키기에 c를 계속해서 더해도 괜찮아보이지만 tanh의 미분값은 1 - tanh^2이고 만약 tanh가 c가 계속 더해져서 큰 값을 가지거나 작은 값을 가지게 된다면 이 미분값이 거의 0에 수렴하게 되고, 이 점은 gradient vanishing이 아닌 학습 자체를 불가능하게 만들 수 있음. 

---

### Lecture 8: Attention and Transformers

> **Main Keywords:** Attention, Self-Attention Layer, Transformer

#### 배운 점

1. **Attention의 등장 배경**
   - RNNs을 사용해서 특정 언어를 다른 언어로 변환하려고 한다면, 우리는 Encoder과 Decoder라는 2개의 RNNs를 정의해야 함
   - Encoder에서 모든 input에 대한 정보를 다 담은 context vector를 만든 후, 이 vector를 Decoder의 input이나 hidden state에 사용하는 방식
   - 하지만 이 방식의 문제점은 input이 길어진다면 고정된 C 벡터에 너무 많은 정보를 넣다보니 상대적으로 정보의 손실이 발생
   - 이를 극복하기 위해 decoder state를 encoder에서 사용 -> decoder state * hidden state를 통해 alignment scores을 구하고 이를 softmax를 통해 attention weights를 구함 이를 각각의 hidden state와 곱하여 특정 decoder state에서의 context vector를 구함
   - 이 과정에서 decoder state를 사용해서 모든 input이 아닌 특정 input에 어느정도를 집중한다는 아이디어가 Attention으로 발전
2. **Attention Layer에 대해**
   - Query와 Input vectors인 X로 만든 Key, Value를 이용하여 Key와 Query를 내적을 통해 Similarities를 얻고 이를 Softmax를 통해 Attention Weights로 만들고, Value와 곱해서 Output vector를 도출
   - Self-Attention Layer는 Query 또한 Key, Value와 마찬가지로 X를 통해서 만듦
   - Self-Attention Layer는 모든 Q, K, V가 같은 X를 통해 만들어지기에 X의 순서가 달라진다고 하더라도, 단순히 Y의 순서가 달라질 뿐 값이 달라지지는 않음 -> Permutation equivariant
   - 하지만 Self-Attention Layer은 Time이라는 정보가 없기에 Sequence의 순서를 모르고 이를 해결하기 위해 Positional encoding을 추가
   - Masked Self-Attention Layer는 우리가 참고하면 안되는 정보에 대해서 -infinity로 설정해서 softmax를 0에 근접하게 만들고 이는 주로 Language model에서 이후의 단어들을 참고하면 안되는 경우에 사용 (예측)
   - Multiheaded Self-Attention Layer는 H개의 전문가 그룹을 세워놓고, 각각의 전문가들이 알아낸 정보를 W_o를 통해 Concatenate하는 방식
   - CNNs의 장점인 parallel한 연산 가능과 RNNs의 장점인 sequence의 길이에 상관 없다는 점을 합친 Self-Attention이 연산비용이 비싸고, 메모리 효율이 안 좋더라도 현재 거의 모든 부분에서 사용
3. **Transformer에 대해**
   - Transformer는 Transformer block이 연결되어 있는 형태
   - Transformer block은 input data X에 대해 Self-Attention + residual connection -> Layer Normalziation -> MLP + residual connection -> Layer Normalization의 과정을 수행하는 block
   - 이런 Transformer를 사용해서 LLM을 구현할 수 있는데 처음에 Embedding Matrix [V x D]를 사용해서 언어를 컴퓨터가 이해할 수 있는 값으로 바꾸고, 마지막에 Projection Matrix와 softmax를 이용해서 모델이 구한 값을 다시 단어들의 확률로 바꿔서 도출
   - Pre-Norm Transformer: Layer Normalization is outside the residual connections -> identity function을 정확히 넘겨줄 수 없음, 따라서 기존의 순서를 Layer Normalization -> Self-Attention + residual connection -> Layer Normalization -> MLP + Self-Attention 의 순서로 변경
   - RMSNorm: Layer Normalization보다 Root-Mean-Square Normalization이 상대적으로 학습을 더욱 안정적으로 만듦
   - SwiGLU MLP: 기존의 MLP 대신에 Y = (𝜎 (𝑋𝑊1) ⊙ 𝑋𝑊2) 𝑊3 를 사용 (H = 8D/3을 사용)
   - MoE: 기존의 방식인 1개의 MLP 대신 E개의 전문가 MLP를 만들어서 parameter를 늘리더라도 각각의 데이터에 대해 특정 전문가 집단만을 활성화시켜 더욱 정확도를 늘리는 방식
   - ViT: image를 D개의 patches로 나눈 후, 이를 D차원의 input X로 transformer에 넣은 후, 마지막에 Pooling을 사용하여 이를 우리가 분류하려는 C개의 차원으로 바꿈
   - ViT의 과정에서 masking을 사용하지 않고 (이미지는 모든 부분을 고려해야 하기 때문에), Positional Encoding을 사용 (이미지는 공간 정보가 중요하기에)

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Attention의 발전 과정
**Q.** 지금까지 배운 모든 Neural Networks는 전부 Learnable parameters를 학습시킬 때, Backpropagation을 사용하는 데 어떻게 Transformer 또는 Attention과 같은 성능 좋은 모델을 설계할 수 있는 지에 대한 의문 
> **A.** 기본적으로 혁신적인 구조가 등장하는 과정은 기존의 모델의 Bottleneck을 해결하는 과정에서 새로운 구조의 모델이 등장.
> - ResNet의 경우, 기존의 CNNs Architecture가 가진 문제인 Layer가 깊어질수록 학습이 gradient vanishing 때문에 잘 안되는 점을 고치기 위해서 H(x) = F(x) + x 와 같은 방식을 도입하여 설계.
> - LSTM의 경우, 기존의 RNNs Architecture에서 time의 값이 커질수록 초기의 값들이 너무 많이 희석돼서 전달이 잘 안되는 문제를 해결하기 위해서 다양한 gate를 도입하여 문제 해결.
> - Attention의 경우, RNNs을 이용한 Translator에서 발생하는 Context vector 문제를 해결하기 위한 과정에서 Attention이라는 개념 등장. 

##### 2. Transformer의 범용성
**Q.** 상대적으로 간단해 보이는 Transformer의 작동 방식에도 불구, 어떻게 이렇게 많은 분야에서 사용될 수 있는지에 대한 의문
> **A.**
> - 기존 모델인 CNN과 RNN은 근처 픽셀끼리 뭉쳐야 효율적이다, 순서대로 봐야 효율적이다 라는 인간의 편견을 가짐, Transformer는 그러한 편견을 가지지 않고 설계되었기에 더욱 효율적.
> - CNN, RNN은 상황에 상관없이 항상 같은 가중치가 연산에서 사용, 하지만 Transformer는 input data X에 따라 가중치가 변화하는 특성을 가지기에 데이터에 유연하게 대처 가능.
> - CNN, RNN은 기존의 데이터가 이후의 구조까지 연결되려면 많은 Layer를 거쳐야 함. 하지만 Transformer는 Self-Attention 한번을 통해 멀리 떨어진 Pixel 끼리의 정보 교환이 가능. 

---

### Lecture 9: Object Detection, Image Segmentation, Visualizing and Understanding

> **Main Keywords:** 

#### 배운 점

1. 
   -

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

---

### Lecture 10: Video Understanding

> **Main Keywords:** 

#### 배운 점

1. 
   -

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

---

### Lecture 11: Large Scale Distributed Training

> **Main Keywords:** 

#### 배운 점

1. 
   -

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

---

### Lecture 12: Self-supervised Learning

> **Main Keywords:** 

#### 배운 점

1. 
   -

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

---

### Lecture 13: Generative Models 1

> **Main Keywords:** 

#### 배운 점

1. 
   -

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

---




