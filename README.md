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

# Lecture 1: Introduction
## 배운 점
1. **이 강의를 통해 배우고자 하는 것**
   - Computer Vision이라는 분야에서 사용되는 Deep Learning 기술 (예로부터 Vision의 진화는 Intelligence의 진화를 이끌었다)
2. **전반적인 Computer Vision의 발전과 AI의 발전 과정 및 간단한 Overview**

# Lecture 2: Image Classification with Linear Classifiers

> **Main Keywords:** Image Classification, K-Nearest Neighbors, Hyperparameters, Linear Classifiers, Softmax, SVM

## 배운 점

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

## 내가 가진 의문 & 답변 (AI 활용)

### 1. K-NN의 공간 문제
**Q.** 수업 중, "K-NN의 방식에선 빈 공간이 많으면 의미가 없다"라는 말에 대한 의문
> **A.** 우리가 판단하려는 공간을 충분히 채울 만큼의 다양한 sample을 확보해야 함.

### 2. KL Divergence와 Cross Entropy
**Q.** 수업 중 나온 Kullback-Leibler divergence와 Cross Entropy의 관계에 대한 의문
> **A.** Cross Entropy는 KL divergence를 사용한 단순한 objective function이며, 주로 One-hot encoding에 사용됨.

### 3. Softmax Loss vs SVM Loss (정확도 측면)
**Q.** SVM은 단순히 정답 scores가 다른 scores보다 margin 이상의 차이를 보여주면 loss가 늘어나지 않음. 반면 Softmax는 loss가 아무리 정확하더라도 0이 될 수 없기 때문에, "Softmax가 더욱 정확한 결과를 낸다고 할 수 있는지"에 관한 의문.
> **A.**
> - **SVM:** 특정 조건만 만족한다면 더 이상 그 부분의 성능 향상을 요구하지 않음.
> - **Softmax:** 아무리 classify가 성공했더라도 계속해서 loss를 발생시키기에, 더욱 확실하게 분류하도록(좋은 성능을 내게) 유도함.
  






