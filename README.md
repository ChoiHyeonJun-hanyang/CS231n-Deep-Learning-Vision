# CS231n - Deep Learning for Computer Vision

스탠포드 대학의 CS231n 강의를 수강하며 진행한 과제와 학습 내용을 정리하는 레포지토리입니다.

## 프로젝트 개요
- **수강 기간**: 2025년 12월 24일 ~ (진행 중)
- **목표 일정**: 6주 완성 (수강 5주 + 프로젝트 1주)
- **현재 상태**: 5주차 진행 중

## 주차별 진행 상황 (Roadmap)

### 1주차 (완료)
- [x] 강의 1강 ~ 4강 수강
- [x] Assignment 1 시작 및 완료

### 2주차 (완료)
- [x] 강의 4강 ~ 9강 수강
- [x] Assignment 2 시작

### 3주차 (완료)
- [x] 강의 10강 ~ 13강 수강
- [x] Assignment 2 종료
- [x] Assignment 3 시작

### 4주차 (완료)
- [x] 강의 14 ~ 15강 수강
- [x] Assignment 3 - Self-Supervised Learning for Image Classification까지

### 5주차 (진행중)
- [x] 강의 16 ~ 17강 수강
- [ ] 강의 18강 수강
- [x] Assignment 3 종료
- [ ] Project 주제 선정 (Kaggle 활용 나만의 모델 구현)

---

## Daily Log

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

### 2026년 1월 17일
1. GitHub에 강의 9강 정리

### 2026년 1월 18일
1. GitHub에 강의 10강 정리
2. Lecture 14: Generative Models 2 수강 시작 (GANs와 Diffusion Models: Intuition까지 진행)

### 2026년 1월 19일
1. Lecture 14: Generative Models 2 수강
2. Assignment3 - Self-Supervised Learning for Image Classification 시작

### 2026년 1월 20일
1. Assignment3 - Self-Supervised Learning for Image Classification 종료
2. GitHub에 강의 11강 정리
3. Lecture 15: 3D Vision 수강 시작 (Constructive Solid Geometry까지 진행)

### 2026년 1월 21일
1. Lecture 15: 3D Vision 수강
2. GitHub에 강의 12강 정리

### 2026년 1월 22일
1. Assignment3 - Denoising Diffusion Probabilistic Models
2. GitHub에 강의 13강 정리

### 2026년 1월 23일
1. Lecture 16: Vision and Language 수강 시작 (Flamingo까지 진행)

### 2026년 1월 24일
1. Lecture 16: Vision and Language 수강
2. Assignment3 - CLIP and Dino

### 2026년 1월 25일
1. Lecture 17: Robot Learning 수강

### 2026년 1월 26일
1. GitHub에 강의 14강 정리
2. Lecture 18: Human-Centered AI 수강 시작

### 2026년 1월 27일
1. Lecture 18: Human-Centered AI 수강

---

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

##### 4. SVM Loss 개념에 대한 문제
**Q.** SVM을 Large Margin Classifier로 알고 있었는데 SVM Loss로만 배우는 것에 대한 의문
> **A.**
> - 기존의 Large Margin Classifier의 SVM은 수학적으로 깔끔하게 떨어지는 Global Optimum을 구하는 것이 목표, 이를 해결하기 위해 Lagrange Multiplier, Kernel Method를 사용 -> 이 방식의 경우 데이터가 적더라도 완벽한 해를 찾을 수 있음.
> - 하지만 Deep Learning에서의 주 목적은 완벽한 해를 구하는 것이 아닌 데이터가 많은 경우에 더욱 빨리 근접한 해를 찾는 것이 목표, 이를 해결하기 위해 SGD와 같은 Optimizer를 사용 -> 더욱 빨리 계산하기 위해 제약조건 + 미분과 같은 복잡한 연산보다 Loss 함수를 간단한 부분으로 세워서 하는 것이 효율적. 이 Loss function이 max(0, s_j - s_yi + margin)의 형태    

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
   - 하지만 Avg Pool 방식은 linear + linear이기에 명시적인 Activation function이 필요
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
   - **Sigmoid**
      - 값을 0과 1 사이의 확률값으로 변경해주는 Sigmoid의 경우, 모든 정의역에서 미분가능하지만, 문제가 존재
      - 많은 층에서 Sigmoid를 연속적으로 사용하는 경우, 계속해서 gradient 값이 작아짐 -> 이는 Backpropagation 과정에서 문제가 발생
      - 특히 절댓값이 큰 경우, gradient가 0에 가까워지는 문제가 커짐
   - **ReLU**
      - Non-linearity 추가 가능
      - 하지만 양수와 음수가 많이 섞인 데이터의 경우, ReLU는 0보다 작은 값을 0으로 처리하기에 해당 가중치 부분이 죽는 경우가 발생
      - 이를 해결하기 위해 GELU, Leaky ReLU 등 다양한 응용형태가 존재
4. **CNN Architectures에 대해** 
   - **VGGNet**
      - 3x3 Conv (stride 1, pad 1), 2x2 MAX POOL (stride 2)를 사용
      - 3x3 Conv를 3번 연달아서 사용하는 것은 7x7 Conv와 같은 효과를 냄 (effective receptive field의 측면에서), 하지만 # of parameters 27C << 49C 로 3x3 Conv가 훨씬 효율적
      - 또한 Activation Funtion은 주로 Convolution 연산 이후 실행하는데 activation function의 실행 횟수가 늘어나기에 더욱 복잡한 형태를 분류 가능
   - **ResNet**
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
   - **RNN의 장점**
      - input의 길이가 상관 없음
      - 모든 가중치를 공유하기에 상대적으로 모델의 크기에 비해 메모리 소모량이 적음
   - **RNN의 단점**
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

> **Main Keywords:** Vision Transformers, Semantic Segmentation, Upsampling, Object Detection, Fast R-CNN, RPN, YOLO, DETR, Instance Segmentation, Mask R-CNN, CAM, Grad-CAM, RoI Pool

#### 배운 점

1. **ViT에 대해**
   - Image를 patches로 나눈 후, 이를 D차원으로 만든 후, Positional Embedding을 추가
   - 이 input을 Transformer에 넣은 후, input으로 Learnable CLS Token을 넣고 이 값을 통해서 나온 output을 C차원으로 만들어서 scores를 확인
   - 또는 D차원의 값을 C차원으로 맞추기 위해, Average Pooling을 사용함
3. **MoE에 대해**
   - Transformer 내에 있는 MLP에 대해 기존의 가중치 값을 W1: [D x 4D], W2: [4D x D] -> W1: [E x D x 4D], W2: [E x 4D x D]로 변경
   - Router가 E보다 작은 A개의 특정 분야에서의 전문가를 고른 후 이 그룹을 이용해서 결과 도출
   - E의 값은 Hyperparameter
4. **Semantic Segmentation에 대해**
   - Semantic Segmentation의 목표는 각 픽셀별로 우리가 원하는 label를 기준으로 classify하는 것
   - 즉 Semantic Segmentation은 어떤 객체가 어디에 있는지를 알고 싶은 것이 아닌, 특정 pixel이 어느 label에 속하는지 알고 싶어함
   - **Sliding Window**
      - 각 Pixel별로 근처의 맥락을 파악하기 위해서 pixel size보다 더 큰 patch를 CNN에 넣어서 classify
      - pixel 개수 * CNN 만큼의 연산이 필요해서 매우 비효율적이고, pixel 별로 다른 CNN을 실행하기에 features을 공유하지 못해서 발생하는 비효율성의 문제도 존재
   - **Convolution**
      - pixel size보다 더 큰 patch를 CNN에 넣는 것이 아닌, image 자체를 CNN에 넣는 방법
      - 하지만 CNN architectures는 보통 layer가 깊어질 수록, pool이나 stride를 사용해서 data의 크기를 줄이는 방식을 사용
      - 이 점은 모든 pixel에 class를 할당하는 Semantic Segmentation의 경우엔 기존 이미지의 크기를 output에서 유지해야 하기에 문제가 발생
   - **Fully Convolutional**
      - (Without dowmsampling) input: [3 x H x W] -> Convolutions: [D x H x W] -> Scores: [C x H x W] -> Predictions: [H x W]의 형태로 진행
      - 이 방식의 경우 전체 이미지를 downsampling 없이 CNN에 넣기에 CNN의 연산값이 너무 비싸짐
      - (With downsampling and upsampling) input: [3 x H x W] -> High-res: [D1 x H/2 x W/2] -> Med-res: [D2 x H/4 x W/4] -> Low-res: [D3 x H/4 x W/4] -> Med-res -> High-res -> [C x H x W] -> Predictions: [H x W]
      - Loss function의 경우, 모든 pixel에 대해 Softmax를 사용하고 이를 전부 더함 -> 이 loss를 이용해서 backpropagation을 할 수 있음
   - **U-Net**
      - Fully Convolution with upsampling과 비슷하지만, Downsampling을 하기 전 feature 정보를 이후 upsampling을 할 때, 전달해서 사용
5. **Upsampling에 대해**
   - Nearest Neighbor: 크기를 키운 후, 해당 크기를 전부 같은 값으로 채움
   - Bed of Nails: 크기를 키운 후, 왼쪽 위에 해당 값을 채우고 나머지 값을 0으로 채움
   - Max Unpooling: Max Polling을 할 때, 어느 pixel에서 값이 max였는 지에 대한 position 정보를 저장한 후, 이를 Unpooling할 때 크기를 키우고 해당 position에 그 정보를 대입, 나머지 값은 0으로 채움
   - **Learnable Upsampling**
      - 초기의 경우엔 Encoding 과정에서 사용된 filter의 가중치를 Convolution Matrix로 바꾼 뒤, 이 값을 Decoder 과정에서 Convolution Matrix의 Transpose를 사용해서 원본의 값을 복구
      - 하지만 이 경우 Convolution Matrix에는 0이 많기에 원본을 복구하는 것에 문제가 발생할 수 있고, 이 점을 해결하기 위해서 이 Weight를 학습시키는 방식으로 발전
6. **Object Detection에 대해**
   - 기존의 Classification에 해당 class에 해당하는 객체의 Localization 정보도 필요
   - **Single Object**
      - CNN을 통해 나온 feature 정보를 사용하여 Class Scores와 Box Coordinates에 관한 정보들을 추출하고 각각 Softmax, L2 Loss를 사용하여 손실값을 합쳐서 최종 Loss를 구하는 방식
   - **Multiple Objects**
      - image를 여러개의 crops로 나눈 후, CNN을 실행해서 해당 crop이 배경인지, 아니면 다른 object인지 판단하는 방식
      - 하지만 무작정 여러개의 crops로 나누는 것은 너무나 많은 CNN 연산을 필요로 하고 이는 computationally expensive
      - Selective Search: object가 있을것 같은 위치를 찾고 그 위치에서만 CNN을 실행
      - **Slow R-CNN**
         - object가 있을만한 Regions을 224 x 224의 크기로 조정 후, 이 값을 각각의 CNN에 넣어서 Bbox reg, SVMs을 통해 object의 좌표와 class label 값을 얻음
         - 하지만 이 방식의 경우 마찬가지로 너무나 많은 CNN 연산을 필요로 함
      - **Fast R-CNN**
         - 전체 imgae를 CNN에 넣어서 features를 추출 -> 해당 features에서 Object가 있을만한 부분을 CNN을 실행하여 Object category와 Box offset을 구함
         - RoI 부분을 CNN으로 돌린다라는 개념은 Slow R-CNN과 비슷하지만 CNN의 크기와 연산량을 고려하건대 훨씬 좋은 성능을 보여줌
         - Faster R-CNN으로 발전하는 경우, CNN 대신 RPN을 사용, 또한 이렇게 구한 위치정보를 RoI pooling을 통해 Crop하고 이후 Per Region Network를 실행
      - **Region Proposal Network**
         - Input image를 Image features로 변환 후 이를 20 x 15 크기의 boxes로 나눠서 object일지 아닐지를 판단 후 boxes의 크기를 정함
         - 이는 Image -> Features -> Objectness, Bbox reg 확인이라는 2개의 단계로 구분 -> 이는 비효율적
   - **Single-Stage Object Detectors**
      - **YOLO**
         - B개의 bounding boxes에서 P(object): box에 object가 있을 확률, P(class): box내의 object가 어떤 class에 속하는지 를 구함
         - You Only Look Once의 줄임말로 real-time object detection임
         - S x S grid로 input을 나누고, 이를 각각 Bouding boxes + confidence와 Class probability map로 나눠서 object detect와 classification을 따로 수행한 후 이를 합쳐서 Final detections를 제작
      - **DETR**
         - 간단한 object detection pipeline, image를 CNN을 사용해서 features를 추출한 후, 이를 transformer encoder-decoder에 넣어서 예측하는 방식
         - 랜덤하게 boxes를 지정해서 그 부분을 확인하는 것이 아닌, transformer가 학습을 통해서 boxes를 알아서 정하는 방식을 채택
         - Transformer는 spatial 정보를 확인할 수 없기에, positional encoding을 이용
         - Encoder에서 얻는 정보들을 decoder의 hidden에 넣고 object queries를 이용하여 나온 output을 FFN을 통해 최종 결과 도출
7. **Instance Segmentation에 대해**
   - Object Detection을 pixel 단위로 하는 방식
   - **Mask R-CNN**
      - Fast R-CNN에서 RoI pooling한 후, Mask Prediction을 추가로 하는 방식
      - 기존에는 c차원의 Classification Scores와 4C차원의 Box coordinates (per class), 어떤 class가 발견될지 모르기에 모든 class의 mask를 C x 28 x 28에 저장
8. **Visual Viewpoint에 대해**
   - Linear Classifier는 가중치에 있는 template와 image의 유사성을 보는 것과 비슷
   - CNN architectures에서 첫번째 layer의 filter를 visualize한다면 상대적으로 간단한 정보들만을 나타내고 있음
   - Saliency maps: pixel의 중요도를 시각화한 maps, 우리가 원하는 것은 특정 pixels의 중요도이기에 pixel이 변화하면서 발생하는 scores의 gradient를 확인
   - CAM: 마지막 CNN의 features (H x W x K)를 Global Average Pooling을 통해 각 채널의 평균값을 낸 후, 가중치를 각 채널마다 넣어서 각 채널의 중요도를 확인, 가중치를 역추적해서 어느 채넣의 정보가 중요한지 확인하고 해당 값을 히트맵에서 강조
   - 하지만 CAM은 마지막 Conv의 features만을 반영함
   - 이를 해결하기 위해서 Grad-CAM을 도입
   - layer의 값이 변경되었을 때, gradient의 값이 크다면 중요한 layer라는 의미
   - Feature map에서 먼저 미분을 하고, 각 pixel 혹은 layer의 중요도인 가짜 w인 a를 만들고 이를 w 대신 사용
   - CNN의 경우, filter가 정보를 계속해서 압축하고 섞어버림 -> 미분을 통한 역추적 필요
   - 하지만 ViT와 같이 Transformer의 경우 거의 생략 없이 행렬 연산으로 진행 -> 역추적이 쉽고 단순히 가중치에 접근 가능
   - Transformer에서 CLS Token이 있다면 Attention map을 확인, 만약 없다면 각 층의 Attention Matrix는 Similarities를 곱하면 됨
   - guided backprop: Backward pass에서도 입력이 양수인 부분만 그대로 전달, 이를 사용 시 이미지 시각화가 잘 됨
9. **RoI Pool에 대해**
   - Object가 있을만한 부분을 특정 크기의 2x2 block으로 Max Pool을 사용해서 만듦
   - 하지만 이런 경우엔 Region features가 손실이 발생할 수 있음
   - 이를 해결하기 위해 RoI Align을 사용
   - RoI Align는 Object가 있을만한 block을 특정 크기의 Grid로 나눈 후, 각 Grid 안에 4개의 가상 점 (꼭 pixel에 걸치지 않아도 됨)을 균등하게 찍음
   - 이후 소수점인 값에 대해 근처 Pixel 값을 따져가며 적절한 값을 찾은 후, 이를 활용하여 Max, Average Pool을 진행 
    
#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. MoE에서의 학습
**Q.** 초기에 가중치를 랜덤하게 설정했을 때, 이 약간의 차이가 특정 그룹을 특정 분야에서의 전문가로 만든다는데, 이 가능성에 대한 의문 
> **A.** 초기에 가중치가 랜덤하게 설정되는 과정에서 특정 그룹이 우리가 해결하고자 하는 문제를 다른 그룹에 비해 약간이라도 더 좋은 성능을 낸다면, 그 그룹으로 하여금 그 문제를 더욱 잘 풀게끔 만들면서 각각의 그룹이 특정 분야에 더 좋은 성능을 내는 방향으로 학습이 진행

##### 2. Slow R-CNN과 Fast R-CNN에서의 차이
**Q.** Slow R-CNN과 Fast R-CNN 모두 이미지가 있을만한 부분을 CNN을 돌린다는 점에서 같은데 이 부분에서의 속도 차이에 대한 의문
> **A.**
> - Slow R-CNN: 이미지 원본의 일부를 CNN으로 넣어서 처리하기에 특징 추출 + 분류 + 위치 확인과 같은 어려운 작업을 전부 시행 -> 따라서 연산량이 많음
> - Fast R-CNN: 이미지에서 특징을 추출해서 이를 CNN에 넣기에 분류 + 위치 확인 정도의 상대적으로 쉬운 작업만 시행 -> 연산량이 적음

##### 3. DETR에서 Decoder의 object query
**Q.** DETR에서 transformer decoder에 들어가는 input인 object queries의 역할에 대한 의문
> **A.** object queries가 직접적으로 해당 patch를 의미하는 것은 아니지만, Transformer의 특성에 따라서 output이 증가하는 특성을 가지고 이 output은 FFN을 통해 block을 의미하기에 object queries의 개수를 늘리면 확인할 boxes의 개수를 늘리는 것과 같은 효과를 얻음. 

---

### Lecture 10: Video Understanding

> **Main Keywords:** Video Classification, Single-Frame CNN, Late Fusion, Early Fusion, 3D Conv, Measuring Motion, Two-Stream Networks, Long-term temporal structure, I3D, ViTs for Video, 

#### 배운 점

1. **Video Classification에 대해**
   - Video = 2D + Time, 우리가 하고자 하는 것은 Video를 보고 class label을 결정 (주로 객체나 행동)
   - 하지만 이를 Image Classification과 같은 방식을 사용해서 처리하기엔 2D image의 Image Classification을 Time의 횟수만큼 반복해야 하고 이는 너무 많은 연산량을 요구
   - 영상이 1920 x 1080의 images를 초당 30프레임 반복하는 형식이라면, Classification의 효율성을 위해 프레임들 중 일부만 이미지의 크기를 조절하여 분류에 사용
   - Training에서는 일부 프레임만을 학습에 사용한다면, Test에선 다양한 clips를 만든 후 각각에서 Classification을 한 후 이 값을 평균내서 최종 결과 도출
   - Test에선 모든 프레임을 전부 확인해야 하기에 이러한 방식을 채택 (Dropout의 방식과 비슷)
   - Clip의 중요도가 각각의 Clip마다 다르기에 Clip Classifier을 통해서 실제로 해당 Clip이 주제와 맞는지를 비교한 후, 중요도를 넘김
2. **Single-Frame CNN에 대해**
   - sampling 과정을 통해 선택된 video frames를 개별적으로 2D CNN에서 분류
   - video의 내용이 크게 변하지 않는 경우에 유리 (Time이 흐르더라도 상대적으로 고정된 이미지에 가깝기 때문에)
3. **Fusion에 대해**
   - **Late Fusion**
      - (with FC layers) 각각의 image frame을 CNN을 통해서 분류한다는 점에서 Single-Frame CNN과 비슷하지만, 이로 모인 features를 Flatten 시킨 후 MLP를 통해 최종 Class scores를 도출
      - 하지만 이 경우엔, T가 커질수록 feature의 개수가 많아지고 마지막 FC가 복잡해짐에 따라 parameter의 개수가 많아지면서 비효율성이 증가
      - 이 문제를 해결하기 위해 Average Pooling을 사용, (T x D x H' x W')의 형태로 Flatten 시킨 Features를 시공간을 기준으로 Average Pool을 사용하여 D차원으로 만든 후, 간단한 Linear 연산을 통해 결과 도출
      - 하지만 이 경우에도 CNN을 통해 특징을 추출한 후, Average Pool을 사용했기에 각각의 image에 있는 low-level motion을 비교하기가 어려움 (Pool의 정보 손실에 의해)
   - **Early Fusion**
      - Features을 추출하고 Pool을 할 경우 정보의 손실이 발생하니, 이를 해결하기 위해서 정보를 합친 후, 2D CNN을 사용
      - 기존의 Image frames는 (T x 3 x H x W)이기에 2D CNN에 넣기 위해서 (3T x H x W)로 크기 변경
      - 하지만 다양한 정보를 가진 Image frames를 한번에 합쳐서 넣는 것은 충분하지 않음  (시간의 정보가 사라지기에)
      - 이 문제를 해결하기 위해 공간과 시간의 정보들을 천천히 합치는 3D CNN와 3D Pooling을 사용
4. **3D Conv에 대해**
   - 기존의 2D Conv와 같이 filter를 사용하여 input을 특정 구역으로 나눠서 해석한 후, 이를 통해 특징을 추출하는 방식
   - 하지만 2D Conv와는 달리, input의 차원이 C x T x H x W로 바뀌고 Time 또한 H, W와 같이 filter에 곱해지는 방식으로 진행
   - 이 방식을 사용할 시의 장점은 Time이 달라지더라도 같은 가중치 filter로 해결 가능하다는 것, 만약 특정 패턴이 Time이 다른 곳에 나타난다면 2D Conv에선 다른 시간대이기에 이를 확인하기 위해서 다른 가중치가 필요
5. **Measuring Motion에 대해**
   - Video에서 actions을 분류할 때, 움직임을 분석하는 것은 도움이 됨
   - Motion을 측정하는 방법은 frames 사이에 특정 pixel이 얼마나 움직였는 지를 확인
6. **Two-Stream Networks에 대해**
   - Input Video (T x 3 x H x W) -> Single frame (3 x H x W) (Spatial stream ConvNet) + Multi-frame optical flow (2(T-1) x H x W) (Temporal stream ConvNet) 으로 나눈 후 시간과 공간 해석을 달리하여 연산의 효율성을 챙김
   - Temporal stream ConvNet에선 Early fusion을 사용하는데, 이 이유는 이미 Optical flow가 시간 순서의 정보를 가지고 있음
7. **Long-term temporal structure에 대해**
   - CNN을 여러개 사용하고, 여기서 나온 정보를 시간의 흐름에 따라 고려하는 방식이 필요 -> 이는 RNN의 개념과 비슷
   - Multi-layer RNN의 개념과 비슷한 Recurrent Convolutional Network를 사용, RNN의 개념인 이전의 hidden state를 현재의 input과 합쳐서 결론을 내는 것을 가져와서 각각의 연산을 단순한 linear 연산이 아닌 Conv 연산으로 대체한 방식
   - Features for layer L, timestep t = tanh(2D Conv(Features from layer L, timestep t-1: W_h) + 2D Conv(Features from layer L-1, timestep t: W_x))
   - 하지만 Recurrent CNN을 사용한다면 RNN의 단점인 현재의 hidden state를 구할 때, 이전의 hidden state를 전부 구해야 한다는 문제가 있고 이는 굉장히 비효율적
   - Spatio-Temporal Self-Attention (Nonlocal Block): CNN을 통해 얻어낸 Features를 Residual Connection을 포함한 Transformers에 넣은 후 Output을 얻어내는 방식
   - 3D CNN -> Nonlocal Block -> 3D CNN -> Nonlocal Block -> ... 의 순서로 진행
   - 3D CNN의 경우, depth를 깊게 하며 시야를 넓힘. 하지만 이 경우엔 parameter의 개수가 너무 많아지기에 중간에 Nonlocal Block을 사용해서 현재는 CNN이 동시에 인식하지 못하는 부분을 한번의 연산으로 연결
8. **Inflating 2D Networks to 3D (I3D)에 대해**
   - 2D (K_h x K_w) Conv / Pool을 3D (K_t x K_h x K_w) Conv / Pool로 변경, 2D에서 사용하던 가중치를 K_t만큼 나눠서 사용
   - 기존의 image (3 x H x W)를 (3 x K_t x H x W)로 바꾼 후, 3D CNN에서 학습한다면 2D Model의 성능부터 학습 시작 가능
9. **ViTs for Video에 대해**
   - Factorized attention: self attention의 경우 시공간의 정보가 섞여 연산량이 증가, 연산량을 줄이기 위해 시간과 공간을 따로 분리
   - Pooling module: 초반 layer에선 많은 토큰을 받는 대신, 채넣의 개수를 적게 (디테일한 정보 위주), 후반 layer에선 적은 토큰을 받는 대신, 채널을 늘림 (함축된 의미)
   - Video masked autoencoders: 자기주도학습과 비슷, Video의 일정 부분을 가린 후, 남은 정보로 유추하여 가려진 부분을 복원하는 방식으로 학습
10. **Temporal Action Localization에 대해**
   - long untrimmed video sequence에선 다른 동작들을 연달하서 구분할 수 있어야 함.
   - 이를 해결하기 위해 Faster R-CNN과 비슷한 방식을 사용 -> temporal proposals을 만들고 이를 통해 classify 진행
11. **Audio Information에 대해**
   - 행동을 판단할 때, Motion 정보와 Spatial 정보 외에도 영상의 Audio 정보 또한 중요
   - Spectrogram: 음향 정보를 이미지로 변환하는 방법

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Video Classification에서의 이미지 손실
**Q.** 영상에서의 변화는 갑작스럽지 않기에, frames 중 일부만 sampling을 통해서 Video Classification에 사용되는 점은 납득했으나, 이미지를 112 x 112와 같은 크기로 조절할 시 정보 손실 우려 의문
> **A.** 이미지의 크기를 줄이면서 공간이 잃은 정보를 여러 개의 frames가 채워줌, 또한 이미지를 그대로 넘길 시 연산량의 문제도 존재

##### 2. Single Frame에서의 최종 선택
**Q.** Single Frame CNN 방식에선 각각의 Frame을 다른 CNN에 넣고 나온 결과를 어떤 식으로 결론내는 지에 대한 의문
> **A.** 주로 각각의 이미지가 생각하는 확신과 그 class label을 반영하는 Average Predictions을 활용

##### 3. Training에서의 Data
**Q.** Training을 할 때, Sports-1M과 같은 경우, Class label이 500개로 굉장히 많고 이러한 경우엔 우리가 이 dataset의 일부분만 학습하고 싶다고 할 때, 상대적으로 뽑히지 않은 class data에 대한 Prediction 문제
> **A.** 그런 경우 문제가 발생할 수 있음. 이를 방지하기 위해서 class 별 data의 개수를 맞춰서 가져오는 방식을 채택하며, 만약 데이터 개수가 부족할 시 Data Augmentation을 사용하여 데이터 개수를 증가  

##### 4. Accuracy on UCF-101에서의 의문
**Q.** Accuracy on UCF-101을 본다면 3D CNN의 정확도가 너무 낮고, Spatial only도 낮은 것에 반해 Temporal only의 정확도가 높은 것에 대한 의문
> **A.**
> - 3D CNN: 데이터의 부족으로 인해 Overfitting이 발생하여 정확도가 낮음
> - Spatial only: 마찬가지로 이미지를 학습시키는 것과 비슷하기에 Overfitting으로 인해 정확도가 상대적으로 낮음
> - Temporal only: Motion이라는 정보가 Video을 분류하는 것에 있어서 Key Information이기에 Overfitting이 발생할 가능성이 낮고, 발생하더라도 문제가 적게 발생

---

### Lecture 11: Large Scale Distributed Training

> **Main Keywords:** GPU hardware, Large Scale Distributed Training, Data Parallelism, FSDP, HSDP, Activation Checkpointing, MFU, Context Parallelism, Pipeline Parallelism, Tensor Parallelism

#### 배운 점

1. **GPU Hardware에 대해**
   - Graphics Processing Unit의 줄임말로, 처음엔 graphic의 연산을 위해서 고안된 장치. 최근엔 다양한 parallel processing에서 사용
   - parallel processing이란 큰 작업을 아주 작은 단위로 쪼개서 수천개의 코어가 연산을 동시에 처리하는 방식으로 시간을 획기적으로 단축
   - (NVIDIA H100 기준) 사용 가능한 132개의 SMs가 존재, 이는 각각 independent parallel core임
   - 각각의 SM에는 128개의 FP32 Cores가 있고 각각 a*x + b의 연산 (multiply + add)의 연산을 실행하기에 128 * 2 = 256 FLOP/cycle의 성능을 가짐
   - 또한 4 Tensor Cores는 AX + B (A, X, B: Tensor)을 clock cycle마다 실행하기에, [16 x 4][4 x 8] + [16 x 8] = 16 * 4 * 8 * 2 = 1024 -> 1024 * 4 = 4096 FLOP/cycle의 성능을 가짐
   - 기술이 발전하면서, FP32의 처리 속도 증가보다, Tensor Cores의 처리 속도가 기하급수적으로 빨라짐
2. **Using Multiple GPUs에 대해**
   - (H100 기준) GPU 내의 통신은 3352 GB/sec
   - (Llama3 Cluster 기준) Server: 8개의 GPU를 뭉쳐놓은 것, 900 GB/sec between GPUs -> Rack: 2개의 Server를 뭉쳐놓은 것, 총 GPU 16개 -> Pod: 192 Racks, 3072 GPUs, 50GB / sec between GPUs -> Cluster: 8 Pods, 24,576 GPUs, < 50GB / sec between GPUs
   - 우리의 목표는 이 큰 Cluster를 하나의 큰 Computer로 생각하고 이를 통해서 규모가 큰 Neural Network를 학습시키는 것
3. **Data Parallelism에 대해**
   - 기존의 방식은 Loss를 하나의 GPU가 N개의 samples에 대해서 계산한 후, 평균 값을 사용하는 방식으로 진행
   - DP의 아이디어는 MN개의 samples에 대해서 M개의 GPU가 각각 MN개의 samples 중에서 N개의 samples에 대한 loss의 평균을 구한 후, 이 M개의 loss를 다시 또 평균을 내는 방식
   - Gradient가 Linear하기에, 즉 loss의 gradient를 서로 분리해서 구할 수 있기에 가능한 방식
   - 우리의 목표는 여러 개의 GPUs를 이용해서 하나의 큰 GPU가 동작하는 것처럼 구현하는 것이고, 이를 위해서 forward -> backward 이후 각 GPU마다의 gradient를 평균을 내서 update해야 함
   - 이 과정에서 backward pass 한번을 수행하기 위해 모든 GPU 사이의 정보 통신이 필요하고, 이는 다른 연산과정에 비해 비싼 통신을 필요로 함으로 bottleneck이 됨
   - 이를 해결하기 위해 backward를 하면서 동시에 통신하여 통신과정에서 사용되는 시간의 손실을 줄임
   - 하지만 이 방식의 경우 모든 GPU가 weight, grad, Adam b1, b2와 같은 정보들을 전부 가지고 있어야 함 -> 이는 굉장한 메모리 손실
   - **Fully Sharded Data Parallelism (FSDP)**
      - GPU마다 자신만의 고유한 가중치를 지니고 이를 다른 GPU에게 공유하는 방식 (ex. W1 ~ W3까진 GPU1, W4 ~ W6까진 GPU2와 같은 방식)
      - forward pass나 backward pass에서 자신에겐 없는 가중치가 필요할 때, 그 가중치를 가진 GPU에게 받는 방식 (forward에서의 마지막 가중치는 backward에서의 처음 가중치이기에 삭제 x, 나머지 가중치는 삭제)
      - 마찬가지로 통신에 발생하는 시간의 효율성을 높이기 위해 가중치를 가져오는 것, backward pass, aggregate gradient를 동시에 진행
      - 예를 들어 W3에 대한 gradients를 보내고 update, backward with W2, Fetch W1을 전부 동시에 진행
      - 이 아이디어의 경우, DP에 비해 메모리 효율성에서 좋지만 통신 과정이 많아지면서 속도가 느려짐
   - **Hybrid Sharded Data Parallel (HSDP)**
      - GPUs를 Group으로 나눠서 그 Group 내에서 FSDP를 진행
      - 이 방식으로 할 경우, 상대적으로 Cluster에 비해 통신 속도가 빠른 Rack이나 Pod 내에서 Forward pass의 Weight 공유, Backward pass의 Weight 공유, Gradients 공유 총 3번의 통신이 필요. 상대적으로 통신 속도가 느린 Cluster 내에서는 Gradients 공유 총 1번의 통신이 필요 -> 통신이 빠른 곳에서의 통신을 많이, 통신이 느린 곳에서의 통신은 적게
      - FSDP나 HSDP를 통해서 기존 DP에서 800GB의 메모리가 필요했다면 80개의 GPU를 사용하였을 때, GPU당 10GB의 메모리만 사용
      - 하지만 backward pass를 위해서 기존에 구해놓은 activations를 저장하는 과정에서 메모리 문제가 발생
4. **Activation Checkpointing에 대해**
   - 기존의 방식대로 모든 값을 다 저장하는 경우, Forward + Backward에는 O(N)의 연산과 메모리가 필요
   - Full Recomputation: n + (n-1) + (n-2) + ... + 1 = O(N^2)의 연산이 필요, 메모리는 O(1) -> O(N^2)의 연산은 너무나도 많음
   - C개의 checkpoints를 도입, 이 경우 O(N^2/C)의 연산과 O(N/C)의 메모리가 필요 -> C로 sqrt(N)을 주로 사용: (Computation: O(N*sqrt(N)), Memory: O(sqrt(N)))
5. **많은 양의 GPUs를 학습시키는 방법**
   - 효율성을 위해 개별 GPU의 메모리를 최대한 사용 
   - DP 사용 (GPUs < 128, parameters < 1B), FSDP + activation checkpointing (parameters > 1B), HSDP + activation checkpointing (GPUs > 256)
   - 만약 GPUs > 1K, params > 50B라면, CP, PP, TP와 같은 다른 방식을 사용
   - global batch size, local batch size, HSDP Dimension과 같은 다양한 Parameter를 세팅해야 함 -> Maximize Model Flops Utilization (MFU)
6. **HFU와 MFU**
   - **HFU**
      - 현재 사용하는 GPU가 어느 상황에서 최적의 성능을 보이는 지를 확인 후, 이 값을 이용
      - 하지만 HFU는 학습하는 Model에 따라 달리질 수 있는 Data augmentation, Optimizer, Preprocessing의 부분을 담지 못함 -> MFU를 Maximize
   - **MFU**
      - 이론상 Model에서 사용되는 FLOP_theoretical을 구한 후, 사용하는 Hardware의 이론상 최고 성능으로 나눠 이론상 걸려야하는 시간을 계산
      - 구한 값을 t_theoretical이라 할 때, 이를 실제로 걸린 시간 (t_actual)로 나눠서 MFU를 구함
      - MFU가 30% 이상이면 좋은 값이고, 40% 이상이면 굉장히 좋은 값임
      - Recent Devices로 할 경우, FLOPs 연산의 속도 증가보다 통신 속도의 증가가 적기에 MFU의 값이 낮아질 수 있음
7. **Context Parallelism에 대해**
   - Transformers는 L-length의 sequence을 받기에, 여러개의 GPUs가 single long sequence를 나눠서 처리하게끔 함
   - Normalization, residual connections에선 가중치를 공유할 필요도 없고, parallelizable하기에 매우 쉬움
   - MLP: 병렬화하기 쉽지만, 가중치를 가지기에 가중치를 복사하고 gradients를 공유하는 과정이 추가로 필요
   - **Attention**
      - QKV Projection: MLP와 같이 gradients를 공유할 필요가 있음
      - Attention operator: 병렬화시키기 어렵기에 Ring Attention, Ulysses를 사용
      - Ring Attention: Ring의 형태로 GPUs를 배치한 후, K, V를 옆으로 계속 공유하는 방식으로 Attention Scores를 구함 (총 GPU의 개수만큼 실행) -> 메모리를 효율적으로 사용하고, 문장 길이 대처능력도 좋지만 구현이 어려움
      - Ulysses: 기존의 방식은 Data: (Sequences, Heads)로 나누어서 전문가 그룹인 Heads를 통해 Sequences를 분석했다면, Sequences를 모든 GPU끼리 공유시키고 Heads를 나눠서 Multihead Attetion을 실행 -> GPU마다 특정 Heads에 집중해서 관찰하기에 max parallelism = # of heads
8. **Pipeline Parallelism에 대해**
   - Model이 여러개의 Layer로 구성되어 있다면, Layer 별로 담당하는 GPU를 달리 하는 방식
   - 하지만 이 방식의 경우, GPU1의 작업이 끝나야 GPU2의 작업이 들어갈 수 있는 Sequential dependencies가 존재 -> Max MFU with N-way PP = 1/N 이고 이는 N이 커질 수록 안 좋아짐
   - 따라서 microbatches를 많이 늘린 후, 각각의 batch를 순서대로 실행하여 bubble의 크기를 최소화하는 방식으로 진행
   - 하지만 microbatch의 개수를 늘릴 수록, backward를 하지 못하는 경우 메모리 부족 문제가 발생할 수 있기에 Activation checkpointing을 도입
9. **Tensor Parallelism에 대해**
   - Tensor multiply XW = Y에서 W를 열에 따라서 W_i로 나눈 후, 이 값들이 열에 따라서 나뉜 Y_i를 만드는 점을 이용하여 각각의 GPU가 W_i를 통해 Y_i를 구함
   - 하지만 이 방식은 만약 다음 연산이나 action이 Y를 열기준이 아닌 행 기준으로 고려해야 하는 연산이라면 모든 GPU가 연산할 때까지 기다려야 함.
   - 2 consecutive TP layers를 사용, X: [NxD], W: [DxD], Y: [NxD], U[DxD], Z[NxD] 일 때, (GPU1을 기준, 총 GPU 4개) W1: [D x D/4]를 통해 Y1: [N x D/4]를 만들고 U1: [D/4 x D]와 연산하며 GPU1에서의 연산을 진행
   - 하지만 TP의 경우 Row Parallelism을 한 후 (이 상황에선 U1 이후), All-reduce를 필요로 하기 때문에 통신 속도가 빠른 GPUs끼리 사
   - TP, CP, PP, and DP를 전부 동시에 사용하여 ND Parallelism을 실행 할 수 있음

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Cluster에서의 tradeoff
**Q.** Cluster과 같이 GPU의 개수를 많게 할수록, GPU간의 연산 속도는 감소하는데 저런 큰 규모는 속도를 조금 포기한 대신 총 연산량을 더 늘리려는 목적인지에 대한 의문
> **A.** 통신의 부분은 인공지능 모델의 부분 중 큰 부분을 차지하지 않는 경우가 많고, 대부분은 연산에 소모되는 시간. 따라서 통신 속도를 포기하면서 총 연산량을 얻는 것은 현명한 선택임. 또한 큰 규모의 Cluster는 상대적으로 GPUs 간의 통신 속도는 느려지지만, 통신 과정에서 연산을 동시에 진행하거나, 혹은 batch size를 최대화시켜서 통신 대비 연산의 비율을 높게 가져가는 방식들을 통해서 이를 해결.

##### 2. Cluster에서의 학습
**Q.** 이렇게 큰 Cluster를 사용해서 학습시키는 경우, 길게보면 몇달간 학습시킨다고 교수님이 말하셨는데, 그렇다면 learning_rate 같은 Hyperparameter를 어떻게 최적의 값을 찾고 진행하는 지에 대한 의문
> **A.**
> - Scaling Laws 활용: 상대적으로 작은 규모의 모델을 사용하여 최적의 Hyperparameter을 찾고, 모델의 크기가 커질 때, 최적의 Hyperparameter가 어떻게 변하는지 보여주는 수식을 사용하여 해결.
> - 검증된 스케줄러 활용: 최적의 값을 하나만 정하는 것이 아닌, 학습 진행 상황에 따라 변하는 스케줄러를 사용. (ex. Linear Warmup + Cosine Delay (Lecture 3에서 배움))
> - 학습 중 수술: 학습 도중에 Loss가 커지거나 줄어들지 않는다면, 학습을 중단한 후 예전에 저장해뒀던 시점으로 돌아가 Hyperparameter의 값을 수정 후 다시 실행.

##### 3. CP에서의 Normalization
**Q.** CP는 하나의 Sequence를 여러 개로 나눠서 각각을 GPU가 맡는 방식으로 한다고 하는데, 그렇다면 Layer Normalization에서의 정보 공유가 왜 필요 없는 지에 대한 의문
> **A.** LayerNorm은 하나의 sample에 대해서 그 내부의 값들을 Normalize 하는 방식이고, 여기에서의 Layer Normalization은 Sequence를 Normalization하는 것이 아닌 Sequence token이 지닌 값을 Embedding을 통해서 (N, D) -> (N, D, E)로 만들고 E에 대해서 Normalization을 하는 것이기에 D를 기준으로 나누는 CP의 경우 문제가 발생하지 않음

---

### Lecture 12: Self-supervised Learning

> **Main Keywords:** Self-Supervised Learning, Pretext Tasks, Masked Auto Encoders, Contrastive representational learning, Momentum Contrastive Learning, Contrastive Predictive Coding, DINO

#### 배운 점

1. **Self-Supervised Learning에 대해**
   - 기존의 large-scale training 방식의 경우 Label이 붙어있는 데이터를 통해서 학습하기에 많은 양의 labeled data가 필요
   - 이를 해결하기 위해서 data (no labels) 자체를 이용해서 Pretext Task를 진행하여 Encoder를 학습시키고 학습시킨 Encoder를 data (with labels) 를 이용하여 Classify하는 방식의 Self-Supervised Learning을 생각
   - Downstream Task에선 Encoder가 이미 학습되어 있기에, 상대적으로 dataset (no labels)에 비해 적은 양의 dataset (with label)으로 해결 가능
   - Pretext Task에선 Encoder를 통해 Learned Representation을 추출하고 이를 활용하여 자신들만의 Tasks를 Decoder (or Classifier, Regressor)를 사용해서 해결하는 방식으로 학습
   - Downstream Task에선 Encoder를 가져와 Learned Representation을 추출하고 이를 shallow network를 활용하여 Classification하는 방식
   - Self-supervised learning을 확인하는 방법은 Encoder로 추출한 특징을 통해서 Downstream tasks나 혹은 classification을 얼마나 잘 수행하는 지를 확인
2. **Self-supervised pretext tasks에 대해**
   - Pretext tasks를 통해 학습한 모델은 훌륭한 features를 추출할 수 있음, 또한 Pretext tasks에선 우리가 원하는 대로 data에 label을 생성하는 것이 가능 (Visual common sense를 발견하는 것에 초점)
   - Predict rotations: model이 주어진 data가 얼마나 회전했는지를 알려면, 해당 object가 어떤 시각적 특징을 가져야 하는 지를 알아야만 알 수 있다는 가정을 통해서 학습 -> 예를 들어서 0, 90, 180, 270과 같은 4개의 경우만을 고려한다고 한다면, 한 개의 raw data를 4개의 rotated data로 변경한 뒤, 이를 각각 ConvNet을 사용하여 각각의 image가 얼마나 회전된 이미지인지를 도출하는 방식 (이 경우 4-way classification)
   - Predict relative patch locations: 특정 이미지 부분을 9개의 조각으로 나눈 후, 중앙의 조각만을 주고 특정 조각이 이 조각 기준 어디에 존재해야 할지를 맞춰야 하는 방식 (이 경우 8-way classification)
   - Solving jigsaw puzzles: 앞선 방식과 마찬가지로 특정 이미지 부분을 9개 조각으로 나누지만, 이 경우엔 이 조각들로 하여금 각각의 이미지가 어디 조각에 들어가야 하는지를 맞추는 방식
      - 만약 이 경우를 classification으로 해결하려고 한다면 총 경우의 수가 9!이 나오고 이를 실행 시, 제대로 된 학습이 어렵기 때문에 가능한 9!의 경우의 수 중에서 가장 다른 모양새로 생기게끔 배치된 64개의 경우의 수 중 하나를 맞추는 방식으로 바꿈 (이는 학습에서의 효율성과 메모리에서의 효율성을 증가, 이 경우 64-way classification)
   - Predict missing pixels: Encoder를 통해 image의 features를 얻은 후, 이 Features를 Channel-wise Fully Connected를 통해서 Decoder Features로 바꾼 후 이를 Decoder에 넣어서 복원한 이미지를 원본 이미지랑 비교하는 방식
      - 위의 Tasks에서의 Loss는 다음과 같이 정의, Loss = L_reconstruction + L_adversarial_learning, L_recon = ||M * (x - Encoder((1 - M) * x))||^2 => Masked 안된 부분을 Encoder를 통해 특징을 추출하고 복원한 값과 원본에서 삭제된 부분을 비교, L_adv는 GANs에서 설명
   - Image coloring: Grayscale image (L channel)을 통해서 Color information (ab channels)를 예측하고 Grayscale과 Color inform을 섞어서 (L, ab) channels의 이미지로 복원 후 raw data와 비교
      - Split-brain Autoencoder: L, ab의 예시처럼 channels끼리 나눠서 자기가 가지지 못한 정보를 예측하고 예측한 각각의 정보를 합쳐서 원본 이미지와 비교하는 방식 (cross-channel predictions)
   - Video coloring: Image coloring과 달리, Video의 경우엔 처음에 Color 정보가 있는 사진을 주고, 약간의 시간이 지난 뒤의 사진에서 물체의 움직임을 추적해서 색을 예측하는 방식 (Tracking regions or objects without labels)
      - 이 방식의 목적은 물체를 분류하는 것이 아닌 특정 물체 (or pixels)의 위치가 얼마나 이동했는지를 확인하는 것, Reference Frame과 Input Frame을 비교하여 물체가 얼마나 이동했는지를 확인하고 이동한 위치의 색 정보를 Reference Colors에서 가져와서 칠하는 방식
      - 여기서 이전 image와 현재의 image를 비교하고 이전 image의 color information을 가져오는 것은 Attention과 비슷 (Reference Frame을 Key, Reference Colors를 Value로 놓고, Target Frame을 Query로 놓는다면 Cross-Attention과같은 개념)
   - 하지만 이런 방식으로 훈련된 Encoder의 경우, Pretext tasks를 해결하는 Representation만을 찾게끔 되기에 정작 Downstream tasks에서 좋은 정확도를 못 보여주는 경우 존재 -> Contrastive representation의 배경
3. **Masked Auto Encoders (MAE)에 대해**
   - ViT와 비슷하게, Image를 같은 크기의 patch로 나눈 후 50, 75와 같은 확률로 Masked한 뒤 이를 복원하는 것
   - 높은 Masking 비율을 사용하는 것은 모델로 하여금 복원하는 것을 더욱 어렵게 만들고, 학습이 어렵다는 것은 모델이 더욱 의미있는 결과를 낼 가능성이 크다는 의미
   - **MAE Encoder**
      - 오직 ummasked patches만 사용, 이 patches에 대해서 Linear projection을 통해 Embedding한 후, Positional embeddings을 추가 (Transformer blocks를 사용하고, 이 block은 시공간의 정보를 고려하지 않기에)
      - input patches가 input에 비해 작은 부분만 사용하기에 Encoder를 Decoder에 비해 아주 깊게 만들어서 작은 정보 내에서도 핵심 정보들을 추출
   - **MAE Decoder**
      - MAE Encoder를 통해 나온 Features tokens와 Masked tokens (positional encodings 추가)를 합친 후, 이를 input으로 사용 (원래 이미지 크기만큼의 입력으로 만듦)
      - Encoder에 비해 Input의 크기가 크기 때문에 Encoder의 크기에 비해 Decoder의 크기가 작게 설정 (Asymmetrical design)
      - 또한 Decoder는 단순히 복원할 때 사용하기에 post-training에서는 사용하지 않음, 또한 U-Net과 달리 Skip Connection이 존재하지 않기에 Encoder와 무관하게 설정할 수 있음
   - MSE loss가 사용되고, 오직 masked된 (복원한) patches에서만 사용 -> unmasked된 부분이 약간 달라지는 것은 상관없기에
4. **Linear Probing vs Full Fine-tuning**
   - Linear Probing: Encoder의 가중치는 고정해두고, 고정된 Learned Representation에 대해서 Downstream tasks를 위한 Linear layer만을 학습시키는 방법
      - 이 방법을 사용할 시에는 제한된 조건 하에서 pre-training의 representation 퀄리티를 확인할 수 있음
   - fine-tuning: 상대적으로 많은 양의 데이터 (unlabeled)를 사용해서 학습한 Encoder를 통해서 Downstream tasks를 할 때, 약간의 데이터 (labeled)를 사용해서 Encoder와 Linear (or FC)를 동시에 학습
      - 이 방법을 사용해서 새로운 tasks에서의 최대 성능을 확인할 수 있음
5. **Contrastive representation learning에 대해**
   - Reference sample을 통해서 만들어진 sample을 positive sample이라고 하고 다른 reference sample에 의해 생성된 sample을 negative sample이라고 함
   - 기존의 문제인 Downstream tasks의 정확도가 낮은 걸 해결하기 위해서, positive sample과의 비교값은 높게, negative sample과의 비교값은 낮게 하여 해당 이미지의 객체의 특징을 보다 자세하게 파악하게끔 유도
   - InfoNCE loss: L = -Ex[log(exp(s(f(x)), s(f(x+))) / (exp(s(f(x)), s(f(x+))) + 나머지 N-1개의 negative sample과의 exp(scores)))] -> log 내부의 값은 positive pair에 대한 scores를 나머지 scores와 softmax 돌린 것과 같은 결과, 만약 log 내부의 값이 높아진다면 positive와의 연관성이 높고, negative와의 연관성이 낮은 것이기에 우리가 원하는 목표를 달성할 수 있음 (-Ex[]는 손실함수이기에, Cross entropy loss와 비슷)
   - MI[f(x), f(x+)] - log(N) >= -L -> MI[f(x), f(x+)] >= log(N) - L 로 바꿀 수 있고 이 값을 잘 생각해보면, L은 우리가 최소화하려는 값이고 만약 N이 커진다면 log(N) - L의 값 또한 커지기 때문에 MI도 증가함 (MI는 상호의존성을 의미)
   - **SimCLR**
      - raw data x에 대해 data augmentation을 통해 positive pair인 x_i, x_j -> 이를 f(encoder)를 통해 Learned Representation인 h_i, h_j -> 이를 g(projection head)를 통해 z_i, z_j (scores, 주로 Cosine similarity)로 만들고 이를 Contrastive loss에 사용
      - data augmentation의 경우 random cropping, random color distortion, and random blur를 사용
      - Affinity matrix를 사용하여 2N개의 samples에 대해서 서로의 similarities를 기록 -> 이후 positive pair를 찾을 때 사용
      - 학습 이후 우리는 각각의 sample에서 특징을 추출할 수 있는 f를 사용하여 downstream tasks를 수행
      - 하지만 InfoNCE loss의 특징인 N이 커져야 positive pair의 연관성이 높아진다는 점에 의해 N을 높이다가 메모리 부족 문제가 발생할 수 있음 -> 이를 위해 MoCo를 사용할 수도 있음
   - **Momentum Contrastive Learning (MoCo)**
      - query와 key data를 나눠서 구분, query에는 positive pair만 저장, key data는 **queue**에 저장 -> query의 정보는 gradient를 구한 후, backward pass를 통해 update, 하지만 key의 정보는 gradient를 구하지 않음
      - 하지만 이런 방식으로 할 시, key의 정보가 변하지 않기에 key = key * momentum + query * (1 - momentum)의 방식으로 slowly progressing
      - query data의 경우는 backpropagation을 해야하기에 중간의 activations를 저장해야 하지만, key data는 이 정보가 필요 없기에 상대적으로 query에 비해 양이 많은 key의 activations를 저장할 필요가 없어서 메모리 부족 문제 완화 (key data는 최종적인 feature vector)
      - MoCo V2: SimCLR와 MoCo의 아이디어를 섞은 버전, non-linear projection head, strong data augmentation을 SimCLR으로부터 가져오고, momentum-updated queues를 MoCo에서 가져오는 방식을 채택 (서로의 장점을 섞은 버전)
         - Non-linear projection head와 strong data augmentation을 사용해서 꼼수를 막고, 데이터의 좋은 특징들을 추출하도록 함
   - **Contrastive Predictive Coding (CPC)**
      - 앞에서 나온 SimCLR, MoCo가 특정 data sample에 대해서 contrastive learning을 하였다면, CPC는 Sequence-level에서 contrastive learning을 하는 방법
      - 순서가 존재하는 이미지를 input으로 넣은 후, g_enc를 통해 features를 추출 -> 이 특징들을 RNN과 같은 모델을 사용하여 특정 구간 t까지의 특징들에 대한 요약본을 형성 -> 이를 통해서 이 다음 input들의 features를 예측하는 방법
      - scores(z_(t+k), c_t) = z_(t+k)T(W_k * c_t), 여기서의 W_k는 c_t와 곱해져서 우리가 예상하는 features를 생성하고 이를 원본 features와 비교 (W_k는 trainable matrix)
      - Visual context의 경우, 특히 image는 시간적 정보가 존재하지 않기에, image를 약간씩 겹치는 patches로 나눈 후, 이를 CPC에 넣어서 이후의 patches의 값을 예측하는 방식으로 사용
   - **DINO: Self-Distillation with No Labels**
      - 기본적으로 x를 통해서 만든 x1, x2를 각각 student와 teacher에게 보냄
      - Distillation의 개념처럼, student는 teacher를 보고 학습하는 형태 -> student와 teacher는 랜덤한 가중치로 초기화 -> teacher는 student보다 조금 더 차분하고 안정적인 값을 제공, 이를 통해 student가 학습 -> 이 과정에서 backpropagation을 통해 student가 변한다면 그 변한 값을 teacher는 momentum update를 통해 조금씩 변화시키며, student의 평균적인 값을 표현
      - loss: -p2logp1, p1은 학생의 것, p2는 선생의 것 

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Self-supervised model의 강력함
**Q.** Self-supervised model이 supervised model에 비해 상대적으로 더욱 강력한 feature representation이 가능하다고 하는데 이 말에 대한 의문
> **A.**
> - Self-supervised model의 경우, Pretext tasks를 정확하게 수행하려면 해당 object의 구조와 맥락을 이해해야 하고 이는 Image에서 어떤 부분이 중요한 부분인지를 알아야 할 필요가 Model에게 존재, 이러한 강점은 단순히 classification의 정확도를 높일 뿐만 아니라 Detection, Segmetation과 같은 분야에서도 좋은 성능을 보여줄 가능성이 높음
> - 하지만 Supervised model의 경우, 고양이를 분류하고자 할 때, 고양이의 특징들을 확인하지 않고 단순히 털만 보고 고양이라고 추측하여 만약 이 추측이 label과 일치할 경우 더 이상 그 부분에 대해서 학습하지 않는 문제가 발생 (Shorcut Learning) 

##### 2. Channel-wise Fully Connected의 사용 이유
**Q.** Predict missing pixels에서 Encoder에서 추출한 특징들을 바로 Decoder로 사용하는 것이 아닌 Channel-wise Fully Connected를 사용해서 넘기는 이유에 대한 의문
> **A.**
> - Encoder를 CNN으로 구성한다고 가정한다면, 원본 데이터에서 사라진 범위가 크기에 저 범위보다 크게 인식하기 위해선 많은 양의 layer가 필요, 하지만 이는 굉장히 비효율적이기에 우리는 추출된 특징들의 Receptive Field가 제한적이어서 전체 문맥을 파악하기 힘들고 이로 인해 Pixel간의 정보를 공유해야만 이미지를 이해할 수 있음
> - 하지만 모든 특징들에 대해서 비교를 하려고 한다면 Features의 채널이 128개일 때, 총 128 * 128번의 채널간의 연결이 발생하고 이는 굉장한 연산의 크기 문제를 발생 -> 이를 해결하기 위해 Channel-wise Fully Connected를 도입하 기존엔 O(C^2)이었던 연산의 수가 O(C)로 줄어드는 효과를 제공, 또한 각 채널의 이미지 특징들을 서로 비교하였기에 우리가 원하던 정보의 공유 또한 달성

##### 3. SimCLR에서의 g의 의미
**Q.** SimCLR에서 g가 projection head라고 하는데 Decoder가 아닌 이유와 굳이 f와 g의 과정을 나눠서 해결하는 이유
> **A.** Downstream tasks에서 원하는 Encoder는 이미지에 대해서 좋은 특징들을 추출할 수 있는 Encoder인데, 만약 우리가 f와 g를 구분하지 않고 하나의 모델로 구현한다면, sample x_i, x_j가 흑백사진과 같이 색 정보가 사라진 images일 때, contrastive loss를 구하려는 과정에서 색 정보와 같은 Downstream tasks에서는 중요할 수 있는 정보를 제거하여 불변성만을 학습하려고 할 수 있음. 이 한계를 극복하기 위해서 최대한 좋은 특징을 뽑는 f를 사용한 후, contrastive loss를 구하는 데 필요한 정보만 남기고 나머지를 버리는 g라는 함수를 training 과정에서만 사용하여 학습 후에 post-training에서 사용하지 않음

##### 4. MoCo에서의 Queue의 의미
**Q.** MoCo에서 key data를 저장할 때, Queue를 사용하는데 만약 정보를 저장하는 것이면 Stack을 사용해도 되는 것인지에 대한 의문
> **A.** Key data는 Backpropagation을 실행하지 않기에 변화가 오직 momentum에 의해서 결정됨. 이 방식은 slowly progressing으로 query가 변하는 방향으로 따라가긴 하지만, 결국 시간이 지나면 둘 사이의 값의 차이가 커지기에, 상대적으로 값이 차이나는 key (오랫동안 momentum을 이용한 key)를 뽑아낼 수 있는 구조인 Queue를 사용 (선입선출)

---

### Lecture 13: Generative Models 1

> **Main Keywords:** Generative Models, Autoregressive Models, Variational Autoencoders, Autoencoder, ELBO

#### 배운 점

1. **Supervised vs Unsupervised Learning**
   - Supervised Learning은 Data (x, y)를 통한 학습으로 function: x -> y를 학습시키는 것이 목표 (Classification, regression, object detection, semantic segmentation, image captioning, etc.)
   - Unsupervised Learning은 Data x를 통해 data의 숨겨진 구조들이나 특징들을 학습하는 것이 목표 (Clusting, dimensionality reduction, density estimation, etc.)
   - Unsupervised Learning은 Label 없이 Data를 사용해야 하므로, 특정 데이터를 그룹으로 나눌 수 있는 정보들에 집중하게 되고, 이는 Clusting이나 혹은 데이터의 차원을 중요한 정보만 남기고 줄이는 방식의 dimensionality reduction에서 좋은 성능을 보이게 함
2. **Generative vs Discriminative Models**
   - **Discriminative Model**
      - 이 Model에서 찾고 싶어하는 distribution은 p(y|x) -> 즉 x가 어떤 분포로 존재하건, 아무런 상관 없이 특정 데이터 x에 대한 class label y의 분포를 알고 싶어함
      - 이 경우엔 만약 data x의 값이 우리가 정의한 label에 존재하지 않는다면, 우리는 우리가 정의한 y에서의 확률만을 알 수 있기에 unreasonable inputs에 대해서 대처 불가능
      - Feature를 labels를 이용하여 학습한 후, 이를 classification에 활용
   - **Generative Model**
      - 이 Model이 알고 싶어하는 distribution은 p(x) -> 즉 모든 x에 대한 분포를 알고 싶어함 -> 가능한 모든 x가 분포에 고려되기 때문에, 우리가 가진 class label의 데이터에 대해선 높은 확률을 보이고, 우리가 가지지 않은 class label의 데이터에 대해선 낮은 값을 보임 -> 낮은 값이 나온다면 우리의 데이터에는 없는 정보이기에 unreasonable inputs에 대해서 대처 가능
      - Outliers를 감지하고, labels 없이 Feature를 학습하며, 새로운 data를 Sample할 수 있음 -> 하지만 이 Sample된 image가 어떤 class인지 모르기에 상대적으로 실용성이 떨어짐
      - Conditional Generative Model: p(x|y)를 알고 싶어함 -> 주어진 class label y에 대한 x의 distribution
         - Bayes' Rule: p(x|y) = p(y|x) * p(x) / p(y) 를 통해서 Discriminative, Generative, Conditional Generative를 연결할 수 있음
         - 만약 우리가 가진 모든 p(x|y)에서 확률 값이 낮게 나온다면, 이는 Outliear
         - 새로운 data를 주어진 labels 내에서 생성 가능하고 이는 상대적으로 일반적인 Generative Model에 비해 실용적
3. **Taxonomy of Generative Models에 대해**
   - Generative Models는 크게 2가지로 분류 가능 -> Model이 직접적으로 P(x)를 구하는 방식 (Explicit density), P(x)에 대해 직접적으로 구하진 못하지만, P(x)로부터 Sample할 수 있는 방식 (Implicit density)
   - **Explicit density**
      - Tractable density: Autoregressive, 직접적으로 P(x)를 구하는 방식
      - Approximate density: Variational Autoencoder (VAE), P(x)를 직접적으로 구하기 힘든 경우, 이를 추론하는 방식
   - **Implicit density**
      - Direct: Generative Adversarial Network (GAN), 직접적으로 P(x)를 통해서 sample하는 방식
      - Indirect: Diffusion Models, P(x)로부터 sample을 낸 결과와 비슷하게끔 내기 위한 여러번의 동작을 수행하는 방식
4. **Autoregressive Models에 대해**
   - Maximum Likelihood Estimation: 우리는 원본 데이터의 분포를 알고 싶은 것이기에 이를 찾기 위해서 Estimation이 필요 -> dataset에 존재하는 data를 우리가 찾은 p(x)라는 distribution에 넣은 후, 이 값들을 전부 곱하여 값이 최대한 크게 만들어진다면, p(x)는 우리의 원본 데이터를 잘 표현하는 것과 같음
   - Goal: p(x) = f(x, W)와 같은 explicit function을 구현하는 것, W는 p(x)의 곱들을 늘리는 MLE를 활용하여 학습
   - 만약 x가 sequence라면, p(x) = p(x1, x2, ... , xT)로 표현가능 하고, Chain Rule 사용 시, p(x) = p(x1) * p(x2 | x1) *  ... * p(xT | x1, x2, ... , xT-1)로 표현할 수 있음 -> Conditional PDF의 경우 이전 내용을 고려하여 결과를 낸다는 점에서 RNN의 아이디어와 비슷하기에 이를 통해서 구현 가능, 또한 Language Prediction에서의 Transformer 처럼 Masked Transformer를 사용하여 RNN와 같은 방식으로 사용 가능
   - x가 Image일 땐, 각 pixel은 0 ~ 255 사이의 값을 3개 가지고 있기에 이 값을 예측하는 방식으로 진행, 3 x H x W의 image를 각각의 pixels value로 나눈 후, 이 값을 RNN이나 Transformer를 넣음
   - 하지만 이 경우엔, 1024 x 1024 image에서 subpixels가 3M개가 나오고 이는 굉장히 비효율적이기에, subpixels로 나누는 것이 아닌 tiles로 나눈 후 이를 RNN이나 Transformer에 넣는 방식으로 사용
   - Autoregressive Models은 학습에서 정답 데이터를 통해서 학습하기에 RNN이나 Transformer의 결과에 상관없이 학습이 빠르게 진행되지만, 학습 후의 Sampling 과정에서 앞선 모든 것이 계산되어야 이후의 값을 계산할 수 있다는 문제에 의해 속도가 오래 걸릴 수 있음
5. **Autoencoders에 대해**
   - Idea: inputs x를 Unsupervised method를 통해 학습하여 features z를 추출하고자 함 -> 여기서의 features는 object에 대한 핵심 특성을 담고 있어야 함 (downstream tasks에서도 사용할 수 있게끔)
   - labels 없이 학습시키는 방법으로 Encoder를 통해 추출한 features z를 Decoder의 inputs으로 넣은 후 이를 통해 Reconstructed input data를 만드는 방식을 사용 (Encoder와 Decoder는 보통 MLP, CNN, Transformer를 주로 사용)
   - 이렇게 추출한 데이터를 기존의 원본 데이터와 L2 Loss를 사용하여 비교하고 학습하는 것에 사용 -> 이 때, x에 비해 z를 매우 작게 설정함으로써 복원과정을 어렵게 하여 학습의 효율을 올림 (z의 공간이 작기에 방대한 양의 정보를 담고 있는 x에서 중요한 데이터들만을 z에 저장해야함)
   - 이후 test과정에선, 학습시킨 Encoder를 사용하여 Features를 추출한 후, 이를 통해 label을 추측하고 실제 label과 비교 (softmax)
   - 만약 우리가 새로운 z를 만들어낼 수 있다면, 이를 Decoder의 inputs으로 사용 시 새로운 이미지를 만들어낼 수 있음 -> 만약 z를 (50, 100)에서 저장하였을 때, 우리가 랜덤한 z를 0에서 생성하였다면, Decoder는 제대로된 z를 받지 못하기 때문에 Noise가 나올 수 있음
5. **Variational Autoencoders (VAEs)에 대해**
   - p의 분포에 대해 직접적으로 수식으로 표현할 순 없지만, p보다 무조건 작은 값을 수식으로 표현할 수 있고, 이 값을 최대화시킨다면 p를 최대화시키는 것과 비슷하기에 lower bound를 optimize하는 방식  
   - Autoencoder에서 새로운 z를 생성하는 것이 어렵다는 단점이 있었기에, 이를 해결하기 위해서 z를 우리가 아는 분포가 되게끔 강제하는 방식
   - 기본적으로 모델은 우리가 아는 prior p(z)를 통해서 sample된 z가 특정 과정을 거쳐서 우리의 데이터 x를 만들어내는 방식의 구조
   - **Training**
      - 기본 아이디어는 Maximum likelihood를 사용, 이를 사용하기 위해선 p(x)를 찾아야하지만 이는 marginalize해도 불가능
      - Bayes' Rule: p(x) = p(x|z) * p(z) / p(z|x) -> p(z | x)를 구할 수 있는 방법이 없음 -> 학습을 가능케하기 위해서 p(z|x)와 비슷한 값을 유도할 수 있는 q(z| x) distribution을 도입
      - z를 통해 p(x|z)를 구하던 block을 Decoder Network라고 하고, x -> q(z|x)를 하기 위한 block을 Encoder block이라고 함 -> Encoder block의 경우, latent codes z의 distribution을 추측하는 것이 목적
      - p(x|z) = N(mu, std^2), q(z|x) = N(mu(z|x), var(z|x))라고 가정하여, Encoder의 output을 q(z|x)의 mean, var를 나오도록 설계, 또한 Decoder도 p(x|z)의 mean을 output으로 나오도록 설계
      - log p(x|z) = -(1/var) * ||x - mu|| ^ 2 + C 이기에 이를 최대화하는 것은 x와 mu(x|z)의 차이를 줄이는 것이고, 이는 z를 통해 생성한 x의 분포값이 실제 데이터와 비슷해야 한다는 것을 의미
   - **ELBO**
      - MLE를 하기 위해 우리가 필요한 값은 p(x) or log p(x)
      - log p(x) = log (p(x|z) * p(z) / p(z|x)) = log ((p(x|z) * p(z) * q(z|x)) / (p(z|x) * q(z|x))) = log p(x|z) - log (q(z|x) / p(z)) + log (q(z|x) / p(z|x))
      - log p(x)는 z를 기준으로 상수이기에, Ez[log p(x)] = log p(x) -> log p(x) = Ex[log p(x|z)] ... (1) - Ez[log (q(z|x) / p(z))] ... (2) + Ez[log (q(z|x) / p(z|x))] ... (3) 로 변경 가능함
      - (1): x를 통해서 얻어질 수 있는 z에 대한 log p(x|z)에 대한 기댓값, (2): KL-Divergence: q(z|x)와 p(z)가 얼마나 다른지를 의미 -> x를 통해서 추측한 분포가 실제 z와 얼마나 잘 맞는지를 나타냄 -> 특히 KL Divergence에는 수식 내에서 q(z|x)가 우리가 원하는 범위 내에서 있게끔 강제하는 값이 있기에 z를 고정시키는 효과도 존재
      - (3): 우리가 구할 수는 없지만 log p(x)를 최대화시키는 문제에서 이를 빼고 (1) + (2)만을 Optimization하더라도 log p(x)를 Optimization하는 효과와 비슷함
   - **Overview**
      - **Training**
         - 1. input을 encoder에 넣은 후, z의 distribution에 대한 정보를 얻어냄
         - 2. Prior loss: 여기서 KL Divergence가 z의 분포가 unit Gaussian이 되게끔 강제함 -> (2)
         - 3. Encoder의 output인 q(z|x)를 통해 z를 sample -> 여기서 z = Normal(mu, var)의 형식으로 선언 시, Backpropagation이 안될 가능성이 있기에, z = mu + std * eps, eps = Normal(0, 1)의 방식을 사용 (Reparameterization trick)
         - 4. sampling을 통해 얻은 z를 decoder에 넣어서 예상한 원본 데이터 분포의 평균을 얻음
         - 5. Reconstruction loss: 예측한 분포의 평균과 x를 L2 loss를 통해 비교 -> (1)
      - **Sampling**
         - 1. z를 prior distribution을 통해 sample (주로 unit Gaussian)
         - 2. Decoder에 z를 input으로 넣은 후 얻은 분포를 통해 새로운 이미지를 생성
   - Disentangling: feature을 diagonal에 저장 (diagonal gaussian을 사용) -> 서로에게 영향을 주지 않음 -> disentangling되는 경우가 존재

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. Generative Model의 정확성
**Q.** 교수님이 말하시기에, Self-Supervised Model이나 Contrastive Learning Model이 Unconditional Generative Model 보다 좋은 성능을 낼 확률이 높은데, 만약 데이터의 Distribution을 안다고 가정했을 때, 어떻게 Generative Model의 정확도가 낮을 수 있는지에 대한 의문
> **A.**
> - 물론 데이터에 대한 완벽한 Distribution을 알 수 있다면, 고차원의 데이터에선 이를 완벽하게 추정하는 것은 불가능에 가까움. 또한 Generative Model은 생성한 data가 특정 분포에 잘 맞게끔 생성해야 하기에 상대적으로 Image Generating에서 중요하지 않은 정보인 배경 복원에 많은 노력을 쏟음.
> - 반면 Contrastive Learning은 이미지를 생성하는 것이 아니라, Positive Pair는 가깝게, Negative Pair는 멀게 임베딩하는 것이 목표임. 덕분에 불필요한 픽셀 정보는 무시하고, 물체의 형태나 의미같은 핵심 특징(Key Features) 학습에만 집중할 수 있어 Downstream Task 성능이 더 우수함.

##### 2. Implicit density의 모호성
**Q.** Implicit density의 설명에는 P(x|y)를 구할 수 없지만, 이를 통해서 sample하는 방식인데, 어떻게 P(x|y)를 모르는 상태에서 이를 통한 sample이 가능한 지에 대한 의문 
> **A.** 원본 data의 distribution를 수식으로 정의하는 것은 불가능함. 따라서 이를 직접 구하는 대신 우리가 잘 아는 Distribution (ex. Gaussian Distribution)를 입력 받고, 원본 데이터와 유사한 형태로 변환해주는 함수 (Neural Network)를 학습시키는 방식으로 진행, 즉 우리는 원본 데이터 분포를 모름에도 이를 비슷하게 구현해주는 Generator를 생성하여 이를 통한 sampling을 가능하게 함.

---

### Lecture 14: Generative Models 2

> **Main Keywords:** Generative Adversarial Networks, Diffusion Models, Rectified Flow, Latent Diffusion Models, Diffusion Transformer, Diffusion Distillation, Generalized Diffusion

#### 배운 점

1. **Generative Adversarial Networks에 대해**
   - **추상적인 이해**
      - p_data라는 보편적으로 통용되는 굉장히 복잡한 Distribution을 찾고, 이를 통해서 sample하는 것이 목적
      - 우리가 가진 data x_i는 p_data의 분포에 존재, 하지만 p_data라는 분포는 굉장히 복잡한 분포기에 이를 우리가 수학적으로 표현하는 것은 불가능에 가까움
      - GANs에선 p_data를 찾는 대신, p_data와 최대한 비슷하게 생긴 p_model을 찾고 여기서 sampling을 수행
   - **구체적인 이해**
      - latent variable z를 간단한 p(z) (주로 Unit Gaussian) 를 통해 얻음, 여기서 z는 x를 설명하는 압축된 정보
      - p(z)를 통해 얻어낸 z를 Generator Network G에 넘겨서 우리가 새롭게 만든 sample x를 만듦
      - 여기서 만든 x는 p_G (Generator Distribution)을 통해 만들어진 sample이고 우리는 p_G = p_data가 되기를 원함
   - **학습 방법**
      - 이를 학습시키는 방법은 위와 같은 방식을 통해서 만들어낸 이미지를 실제 데이터의 이미지와 같이 Discriminator Network에 input으로 넣음 (Generator는 Discriminator를 자신이 생성한 이미지로 속이는 것이 목적)
      - 이후 Discriminator가 어떤 사진이 진짜이고 어떤 사진이 가짜인지를 판단하는 방식으로 학습을 진행
      - 이 학습이 진행되는 과정은, 처음엔 Generator의 성능이 매우 낮아서 비교적 적은 학습에도 Discriminator의 정확도가 증가하게 되고, 이를 속이는 것이 목적인 Generator가 이미지를 더욱 사실적으로 만드는 방향으로 학습을 진행 (길항작용과 비슷)
      - Model 내부에서의 학습은 Backpropagation을 통해 진행되며, Discriminator의 판단이 Generator이 만든 이미지를 통해 Generator로 이동하며 학습을 진행
   - **Objective function**
      - min_G max_D(E_x,p_data[logD(x)] + E_z,p(z)[log(1 - D(G(z)))]), D(x) = P(x is real)
      - **max_D**
         - x ~ p_data, D(x)에서 D를 증가시키는 것은 Discriminator가 진짜 데이터인 x를 진짜라고 분류, z ~ p(z), 1 - D(G(z))에서 이 값을 증가시키려면 D(G(z))의 값이 0이어야 하고, 이 의미는 Discriminator가 z로부터 만들어진 가짜 이미지 G(z)를 가짜라고 분류
         - 즉 max_D는 Discriminator가 생성 이미지와 원본 이미지를 제대로 분류하기 위한 Objective function임
      - **min_G**
         - z ~ p(z), 1 - D(G(z))에서 이 값을 감소시키려면, D(G(z))의 값이 1이어야하고, 이 말은 Generator가 만든 image를 Discriminator가 제대로 구분하지 못하는 상황을 의미
         - 즉 min_G는 Discriminator가 구분하지 못할 정도로 실제 이미지와 비슷한 생성형 이미지를 Generator가 만들도록 시키는 Objective function임
      - E_x + E_z = V라고 한 후, 이를 통해 gradient descent를 진행 -> D의 경우는 maximize가 목표기에 gradient를 더해주고, G의 경우는 minimize가 목표기에 gradient를 빼줌
      - **문제점**
         - Generator와 Discriminator은 독자적인 2개의 Neural Networks이고 대부분의 경우에서 두 Networks의 parameter 개수가 다른데도, 한 Objective function 내에서 하나는 maximize, 다른 하나는 minimize 해야한다는 점에서 어려움이 존재
         - V는 loss function이 아니기에 단순히 V만 보고는 이 model의 성능 파악이 힘듦, 만약 V의 값이 크다면 Discriminative model의 성능이 매우 좋은 것을 의미
         - 또한 log(1 - D(G(z))의 경우, 초반에는 D(G(z))의 값이 0일 가능성이 높음 (Generator의 학습은 느린 반면에 Discriminator의 학습은 상대적으로 빠르기에)
         - 위의 상황과 같은 경우엔, Discriminator의 학습 속도와 달리 D(G(z))를 통한 미분에 학습을 의지하는 Generator는 더욱 더 학습이 느려지거나 학습이 거의 불가능해지는 Gradient vanishing 문제에 직면
         - 기존의 목적인 D(G(z))를 최소화 시키는 조건을 만족하면서 학습 초기의 Gradient vanishing을 해결하기 위해서 log(1 - D(G(z))를 -log(D(G(z))로 변경
      - Inner objective: D_G(x) = p_data(x) / (p_data(x) + p_G(x)) -> p_G의 수준이 높을수록 값이 낮아짐, 현재 p_G가 만들어낸 수준을 확률로 계산
      - Outer objective: p_G(x) = p_data(x), 즉 우리는 Discriminator가 생성된 이미지 x를 받았을 때와 진짜 이미지를 받았을 때, 이를 구별하는 것이 거의 불가능할 정도로 최적화되는 것을 원함
         - 하지만 이 방식의 경우, 우리가 사용하는 D와 G의 neural networks가 이를 표현할 수 있을지에 대한 확신이 불가
         - 또한 유한한 데이터를 사용해서 p_G를 p_data와 동일하게 만드려는 것이 불가능할 수 있음 -> Global minimum에 도달 불가
   - **GAN Architectures**
      - DC-GAN: Generator와 Discriminator를 단순히 5-layer CNNs의 구조로 구현 -> ViT가 떠오르기 전에 GANs가 연구의 대상이었기 때문에
      - StyleGAN: 기존의 latent variable인 z를 Mapping network를 통해서 w로 변환한 후, 이를 AdalN이라는 정규화 과정에서 사용, 또한 데이터의 품질을 올리기 위해서 Noise를 중간에 섞어가면서 sampling을 시도
         - AdalN: 기존의 데이터가 가지고 있던 평균값과 표준편차값을 우리가 원하는 b와 w로 변경하는 Normalization -> z를 통해 얻어낸 w에 대한 스타일을 강제로 주입
   - Latent Space Interpolation: Latent space는 전부 unit gaussian distribution이기에 zt = t*z0 + (1-t)*z1과 같은 방식으로 각각의 특징을 섞은 새로운 latent variable을 만들 수 있음
   - 장점: Objective function을 포함한 구조가 간단함, 굉장히 좋은 Image generation을 보여줌
   - 단점: 학습의 정도를 파악할 수 있는 직관적인 loss function이 존재 x, 학습이 불안정하고, 많은 양의 데이터와 큰 크기의 모델을 학습하는 것에 있어서 적합하지 않음
2. **Diffusion Models에 대해**
   - **추상적인 이해**
      - GAN와 비슷하게 z를 noise distribution p_noise에서 sample (마찬가지로 p_noise는 보통 unit Gaussian, z는 항상 이미지와 같은 크기여야 함)
      - 다양한 noise levels인 t에 대해서 x_t라는 노이즈에 의해서 약간 변형된 이미지를 생성
      - 이렇게 생성된 이미지를 t와 같이 neural network에 넣어서 약간의 노이즈를 제거하는 방식으로 학습을 진행
      - 이후 학습된 Neural network를 이용하여, p_noise에서부터 완전한 노이즈값 x1을 가져와서 neural network를 많이 돌린 후, 완전한 노이즈로부터 노이즈를 제거하면서 새로운 이미지를 생성하는 방식
   - **Rectified Flow**
      - z ~ p_noise, x ~ p_data, t ~ Uniform[0, 1] -> x_t = (1 - t)*x + t*z, v = z - x
      - x_t는 원본데이터를 노이즈로 약간씩 변경한 데이터이고, 우리의 목표는 L = |f(x_t, t) - v|^2를 최소화시키는 것 -> v는 단순히 x에서부터 z를 연결한 선이고, Neural Network f를 통해서 노이즈를 제거한 값이 우리가 원하는 이미지의 형태와 근접해지기를 원함 (즉 f라는 함수가 z - x를 수행하는 모델이 되기를 원함)
      - Training에선 f라는 neural network를 여러 개의 data와 여러 개의 t값들을 통해서 다양한 이미지에서 어느 정도의 노이즈를 복원하는지에 대한 방식을 L2 Loss를 통해서 학습
      - Sampling에선 x ~ p_noise의 데이터를 우리가 원하는 # of steps만큼 f를 통해 연산하며 나온 값을 통해 이동하면서 최종 x0 (생성된 최종 이미지)에 도달하는 것이 목적
      - **Conditional Rectified Flow**
         - 기존의 Unconditional Rectified Flow는 완전한 noise인 z부터 복원을 시작하였을 때, 도착지가 어디일지 모름 (p_data 내부에 존재하는 이미지 중 하나)
         - 이 문제를 해결하기 위해, f의 input에 우리가 원하는 이미지 정보인 y를 넣어줌으로써, y에 해당하는 구역으로 가게끔 v값을 조정
         - sampling의 경우에도, training에서 특정 label y에 대해 기억해둔 구역으로 이동하는 방식으로 새로운 이미지를 생성
         - **Classifier-Free Guidance (CFG)**
            - Hyperparameter인 특정 확률값 (주로 0.5)을 세팅한 후, 학습의 50%는 Unconditional Rectified Flow의 방식대로 f를 학습시키고, 나머지 50%는 Conditional Rectified Flow의 방식대로 f를 학습시킴
            - 우리가 y의 값에 null을 넣는다면, p(x)로 향하는 velocity값이 나올 것이고, y의 값에 실제 class label을 넣는다면, p(x|y)로 향하는 velocity값이 나올 것임
            - 이후 Sampling 과정에서, 우리는 가장 y같은 이미지를 생성하는 것을 원하기에 Uncoditional velocity와 Conditional velocity를 구한 후, v_cfg = v_cond + w * (v_cond - v_uncond)의 방식으로 더욱 conditional한 velocity 값을 이용하며 진행
            - 이 과정을 통해서 y의 그림을 그리면서 동시에 보편적인 특성을 제거하면서, y만의 특성을 주로 고려하여 이미지를 생성 -> 이는 생성된 이미지의 quality를 더욱 높여줌
            - 하지만 이 방식의 경우, 기존의 Rectified Flow에 비해 velocity를 구하는 과정에서 Neural Network의 연산이 2배로 필요하기 때문에 연산 비용 측면에서 문제가 발생
      - **Optimal Prediction**
         - 만약 t = 1이라면, x_t = 0*x + 1*z이기에 완전한 noise이고, 이 경우에 Optimal Prediction은 많은 데이터에 대해서 손실을 최소화시켜야 하기에 optimal v의 방향이 p_data의 평균을 향하는 방향의 반대 방향이 됨
         - 만약 t = 0이라면, x_t = 1*x + 0 *z이기에 원본 데이터이고, 이 경우에 Optimal Prediction은 마찬가지로 손실을 최소화시켜야 하기에 optimal v의 방향이 p_noise의 평균을 향하는 방향이 됨
         - 하지만 t가 0.5와 비슷한 중간값인 경우, 주어진 t에 대해서 x_t를 만족하는 수많은 (x, z) pairs가 있기 때문에, 어디로 가야할지 모르는 model은 수많은 (x, z)를 이용하여 optimal v를 평균값으로 설정 -> 이는 모호한 이미지를 만드는 결과
         - t ~ Uniform[0, 1]이었기에, Optimal v를 찾기 쉬운 1과 0 근처가 Optimal v를 찾기 어려운 0.5 근처와 비슷하게 학습이 되기에, 0과 1 근처에선 잘 대처하던 model이 0.5 근처에서 애매한 결과를 내는 문제가 발생
         - **Noise Schedules**
            - Optimal v를 구하기 어려운 부분에 더 자주 noise를 주는 Noise Schedules를 사용 -> 주로 logit-normal sampling (상대적으로 학습이 쉬운 0과 1 근처를 적게 학습하고, 학습할 때 어려운 구간인 0.5 근처를 t가 더욱 자주 나오게해서 학습을 더 많이 시킴)
            - 만약 data의 resolution이 높다면, noise를 강하게 해야 model의 학습이 어려워지기에 0.75 근처에서 t의 빈도를 높이는 Noise Schedules를 사용
   - **Latent Diffusion Models (LDMs)**
      - 이미지로부터 특징들을 추출해내는 Encoder와 추출된 특징을 통해서 원본이미지를 복원하는 Decoder를 학습시킴 (Encoder와 Decoder는 CNNs with attention을 사용하고 이는 이전에 Video Understanding에서 배운 Non-local block과 비슷)
      - 학습된 Encoder를 고정한 상태에서 추출해낸 Features (or Latent variables)에 Noise를 추가하여 Noisy Latent를 만들고 이를 Diffusion model을 통해서 Latent의 noise를 줄여가면서 Denoised Latent를 얻음
      - 이후 Sampling 과정에서 Latent의 노이즈를 제거하면서 학습한 Diffusion model을 사용해서 랜덤하게 sample된 latent를 Denoised latent로 만들고 이를 기존에 학습한 Decoder를 통해서 최종 이미지로 변환
      - Encoder와 Decoder를 학습시키는 방법은 VAE를 이용하는 것, 하지만 VAE의 경우 Blurry한, 즉 좋지 않은 이미지를 주기 때문에, KL prior weight를 작게 세팅함 -> KL Divergence은 z를 unit gaussian으로 강제하는 objective function으로, 이 값을 작게 하여 기존의 VAE의 문제를 해결하려고 함
      - 또한 GAN에서 아이디어를 얻어, Discriminator를 Decoder로 통해 복원한 이미지에 사용하여 더욱 사실적인 이미지를 만들도록 강제함 (Modern LDM의 pipelines는 VAE + GAN + diffusion)
   - **Diffusion Transformer (DiT)**
      - Transformer block을 이용한 Diffusion, Noised Latent를 ViT와 같이 patches로 나눈 후, label y, timestep t와 같은 조건들을 Embedding하여 DiT Block에 넣는 방식
      - 기본적인 형태는 Transformer block과 같지만, Block 내부의 4개의 행동 이후 Scale, Shift를 실행 -> Scale과 Shift는 Embedding된 조건들을 MLP를 통해서 alpha, beta, gamma와 같은 값들로 바꾸고 이를 각각의 Scale과 Scale, Shift에 input으로 넣음 (파라미터 (평균, 분산)을 동적으로 바꿔줌, 이는 StyleGAN에서 AdaLN과 같)
      - Cross-Attention: 기존의 Transformer block에서 Self-attention과 FFN 사이에 LayerNorm + Multi-Head Cross-Attention (block's input, Conditioning)을 실행
      - Joint-Attention: LayerNorm과 Self-Attention을 하기전에 Input tokens과 Conditioning을 Concatenate해서 하나의 input으로 고려하고 진행
      - Cross-Attention과 Joint-Attention은 text to image 같이 조건의 정보량이 많은 경우에 주로 사용
   - **Diffusion Distillation**
      - Distillation (지식 증류)는 이미 학습시켜놓은 Neural Networks의 지식을 활용해서 새로운 Neural Network를 학습시키는 방법
      - Sampling 과정에서 Diffusion Model을 사용하려면, 우리가 정해놓은 Timesteps만큼 무거운 Diffusion Model의 연산을 해야 한다는 문제 존재 -> 이를 줄이기 위해서 Distillation 사용
      - 기존의 diffusion model의 경우, timestep을 하나씩 밟으며 진행 -> 이를 학습하여 학생 diffusion model을 생성, 학생은 선생이 2번에 가는 곳을 한번에 가는 방식으로 학습 -> 이를 반복한다면 z -> x를 1 ~ 4번까지 줄일 수 있음
      - 또한 CFG의 Sampling 과정에서 우리는 Uncoditional, Conditional velocity를 따로 구하고, 이 값을 통해 v_cfg를 구하는 방식으로 진행되었기에, timestep마다 2번의 model 연산이 필요함
      - 하지만 Distillation을 이용한다면, 학생은 기존의 모델이 2번의 연산을 통해 구한 v_cfg를 학습시켜서 한 번의 연산으로 해결할 수 있음
   - **Generalized Diffusion**
      - x ~ p_data, z ~ p_noise, t ~ p_t -> x_t = a(t)*x + b(t)*z, y_gt = c(t)*x + d(t)*z, y_pred = f(x_t, t), L = |y_gt - y_pred|^2
      - Rectified Flow: a(t) = 1-t, b(t) = t, c(t) = -1, d(t) = 1의 형태로 표현 가능
      - Variance Preserving (VP): a(t) = sqrt(sigma(t)), b(t) = sqrt(1 - sigma(t)) -> a^2 + b^2 = 1이므로 x와 z가 unit gaussian이라면 x_t도 unit gaussian을 유지
      - Variance Exploding (VE): a(t) = 1, b(t) = sigma(t) -> 데이터 x를 x_t에 그대로 넘겨주면서, z의 값을 timestep이 지날수록 키워서 x의 분포를 유지하면서 Noise로 가득채우는 방법
      - x_prediction: y_gt = x (c(t) = 1, d(t) = 0) -> 정답을 바로 맞추게끔 하기
      - eps-prediction: y_gt = z (c(t) = 0, d(t) = 1) -> x_t를 주고 추가된 Noise z가 뭔지 맞추게 하기
      - v-prediction: y_gt: b(t)*z - a(t)*x -> 노이즈에서 원본으로 가는 방향이 어디인지와 속도가 어느정도인지 맞추게끔 하기
   - Diffusion의 forward process는 Gaussian Noise를 더하는 것이고 이는 Diffusion process, U-net이 Noise를 원래의 이미지로 복원하는 과정을 거침 -> Foward pass에서 더해지는 Gaussian Noise의 분포를 알고 있기 때문에 (q(x_t|x_(t-1))), backward process를 VAE와 마찬가지로 lower bound를 Optimization하는 방식으로도 생각할 수 있음 (p(x_(t-1)|x_t))
   - scores function s(x) = gradient of log p(x)라고 하였을 때, 이 scores function은 확률이 가장 높은 곳으로 향하는 방향을 나타내고, 우리는 앞서서 말한 것처럼 Gaussian을 어떤 방향으로 더했는지 알기때문에 Noise가 더해진 방향의 반대로 진행한다면, scores를 올리는 결과를 도출 -> 확률값을 올리는 것 (Optimization)
   - 지금까지는 Timestep에 따른 Discrete한 상황을 생각했다면, Continuous noising process에서는 SDE를 푸는 방식으로 neural network를 학습시킬 수 있음
3. **Autoregressive Models with VAE**
   - 기존의 Autoregressive의 경우, sequential의 문제나 너무나 많은 메모리를 요구한다는 점에서 비효율적이었지만, VAE의 Encoder를 사용해서 압축한 Latent를 넣는 방식으로 비효율성을 해결
   - Encoder에서 이미지를 언어와 같이 정수로 바꿔서 다룸, 이 값들을 Autoregressive model이 LLM이 하는 방식처럼 다음 단어를 맞추는 방식대로 진행하고 이로 인해 나온 정수값들을 Decoder가 이미지로 복원하는 방식

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. VAE와 GAN에서의 z의 차이에 의한 효율 문제
**Q.** VAE에선 학습 과정에서 x를 통해서 추출한 features을 unit gaussian 분포에 맞게끔 고정시키고 이를 랜덤으로 뽑아서 sampling하였다면, GAN에선 처음부터 특정 분포에서 랜덤하게 값을 뽑아서 sampling을 하는데 이 방식에서의 GAN이 더 좋은 효율을 보여주는 것이 아닌지에 대한 의문
> **A.**
> - VAE는 Explicit density 모델이기에, p(x)의 형태를 구하는 것이 목적이고, GAN은 Implicit density 모델이기에 p(x)의 형태를 알 필요 없이 단순히 이와 비슷한 sample을 만드는 것이 목적
> - 이 이유로 sampling 과정에서는 GAN이 더욱 효율적일 수 있지만, VAE는 z와 x의 연결관계를 가지고 있는 반면에 GAN은 z와 x의 연결관계가 미약하기 때문에, x를 통해서 z를 구하는데 복잡하고 많은 양의 연산이 필요 (VAE에선 Decoder가 z를 통해서 x를 만들어내기에 x를 통한 z의 학습이 가능하지만, GAN에서는 x를 통한 z의 학습이 불가능)

##### 2. Classifier-Free Guidance에서의 f
**Q.** CFG에서 f에서 Unconditional에서의 학습과 Conditional에서의 학습을 동시에 진행할 때, 이 학습의 결과를 하나의 Neural Network에서 저장하기에 상대적으로 저하될 수 있는 f의 생성능력에 대한 의문
> **A.**
> - 모델의 구조 상, 우리가 input을 null로 넣거나, class label로 넣는 식으로 구분해서 넣어주고, 모델에서 사용하는 Attention 매커니즘 덕분에 모델 자체의 성능이 크게 저하되지 않음
> - 또한 성능이 저하된다고 할지라도, Conditional - Unconditional의 과정을 통해 조건부 특징을 극단적으로 증폭시키기에 문제가 없음

##### 3. Latent Diffusion Models의 Encoder와 Decoder
**Q.** LDMs에서 Encoder와 Decoder를 VAE의 것으로 가져오면서, KL의 가중치를 낮게 설정한다던데, 물론 VAE의 단점인 Blurry images를 해결할 수 있지만 z를 Model이 원하는 대로 저장한다고 하면 Sampling 과정에서 랜덤한 노이즈를 Unit Gaussian에서 sample할 때, 이 z가 특징값을 가진다고 장담할 수 없는 문제에 대한 의문  
> **A.**
> - GAN의 Discriminator가 사실적인 그림을 만들도록 압박하면서, z의 공간이 멋대로 흩어지더라도, 그 사이의 빈 공간들을 자연스러운 이미지로 메우는 효과가 존재
> - KL을 무시하고 학습을 종료한 뒤, 정규화되지 않는 z의 분포를 측정해서 이 분포에 맞게끔 Noise를 sampling

##### 4. Variance Exploding (VE)
**Q.** VE 방식을 보면 x를 그대로 넘기면서 z를 굉장히 큰 값이랑 곱해서 원본 데이터의 분포를 유지하면서도 Noise로 덮일 수 있도록 한다고 하였는데, b(t)를 기존의 방식과 맞게 1 / (1-t)와 같은 수식으로 이용하는지에 대한 의문
> **A.** pixel의 정보는 0 ~ 255라는 숫자로 기록되어 있고, 우리의 목표는 x를 그대로 넘기면서 t가 1에 가까워질수록 Noise로 이미지가 뒤덮이게끔 해야 하기에, t값이 커질 때, 300이나 500과 같은 큰 값을 집어넣음. 

---

### Lecture 15: 3D Vision

> **Main Keywords:** 

#### 배운 점

1. 
   -

#### 내가 가진 의문 & 답변 (AI 활용)

##### 1. 
**Q.** 
> **A.** 

---




