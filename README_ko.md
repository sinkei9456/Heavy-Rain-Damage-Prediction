# 호우 피해 예측: R 코드를 옮긴 파이썬 버전

[English](README.md)

이 저장소에는 호우 피해 예측 기술 매뉴얼의 R 코드를 옮긴 파이썬 버전([python/](python/))이 있습니다. R 코드(`Modeling.R`, `Prediction.R`, `Sub code #1/`)는 현재 파일에서는 지웠고 커밋 [4582c73](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657) 기록에 남아 있으며, 이 문서의 R 행 번호는 그 파일로 연결됩니다. 호우 사건이 지나간 시군구마다 강우, 시군구 통계, 과거 피해를 바탕으로 피해액을 예측합니다. 권역마다 모델 18개를 학습하고, 마지막 연도로 가장 좋은 모델을 고르며, 18개 예측값의 범위를 예측 구간으로 함께 냅니다.

원래 데이터는 더 이상 구할 수 없습니다. 그래서 파일 이름, 열 이름, 인코딩이 같은 **가짜(합성) 데이터** 생성기를 함께 넣었습니다([python/make_synthetic_data.py](python/make_synthetic_data.py)). 이 문서의 모든 결과는 합성 데이터에서 나온 것이며, 실제 피해에 대해서는 아무것도 말해 주지 않습니다.

## 목차

1. [폴더 구성](#1-폴더-구성)
2. [코드가 하는 일](#2-코드가-하는-일)
3. [코드가 읽는 데이터](#3-코드가-읽는-데이터)
4. [R 코드 설명](#4-r-코드-설명)
5. [R 코드에서 찾은 문제점과 특이 동작](#5-r-코드에서-찾은-문제점과-특이-동작)
6. [파이썬 버전](#6-파이썬-버전)
7. [파이썬 버전 실행 방법](#7-파이썬-버전-실행-방법)
8. [R 과 파이썬 결과 비교](#8-r-과-파이썬-결과-비교)
9. [참고 문헌](#9-참고-문헌)

## 1. 폴더 구성

```
Technical-Manual-for-Heavy-rain-damage-prediction-technique/
├── python/
│   ├── data/                      make_synthetic_data.py 가 만든 합성 데이터
│   │   ├── Dat_extra/             00_area.csv ... 13_nDRZ.csv (외부 표 14개)
│   │   ├── Dat_input/idat.csv     예측할 사건 (2019년, 피해액 열 없음)
│   │   └── Dat_train/tdat.csv     피해액이 있는 사건 (2008-2018년)
│   ├── heavyrain/                 공통 코드
│   │   ├── __init__.py
│   │   ├── clreg.py               두 그룹 회귀 (clreg.R)
│   │   ├── develop.py             권역별 모델 18개, 예측, 검증 (dvfun.R)
│   │   ├── features.py            데이터 읽기, 외부 표 붙이기 (xxfun.R)
│   │   ├── learners.py            lm, glmnet, knn, rf, gbm 과 caret 식 설정값 고르기
│   │   ├── linear.py              최소제곱, 로지스틱·Firth 로지스틱 회귀, step()
│   │   ├── linesearch.py          Firth 적합의 단계 크기 탐색 (mbest 에서 옮김)
│   │   ├── plots.py               검증 그림
│   │   └── preprocess.py          caret preProcess (주성분)
│   ├── make_synthetic_data.py     data/ 를 만듦
│   ├── predict.py                 예측 (Prediction.R)
│   ├── requirements.txt
│   └── train.py                   학습과 검증 (Modeling.R)
├── README.md                      영문 문서
└── README_ko.md                   이 문서(한국어)
```

## 2. 코드가 하는 일

```
Dat_train/tdat.csv  (시군구와 사건마다 한 행, 피해액 y)
  -> 외부 표 14개를 붙임 (1년 전 값), 빈 값은 0                       makedat()
  -> 피해액이 하위 36 % 미만인 사건을 뺌                               Modeling.R
  -> 권역 1-4 마다:
       입력 묶음 4개의 주성분 (분산 90 %, 최소 2개)
       log(y) 를 예측하는 모델 18개: 방법 6가지 x 입력 조합 3가지       dev()
  -> 검증: 마지막 연도를 뺀 해로 학습해 마지막 연도를 예측,
     권역별 최선 모델 = RMSE 가 가장 작은 모델                         hval_out()
  -> 모든 모델을 전체 연도로 다시 학습해 저장
Dat_input/idat.csv  (새 사건)
  -> 같은 방식으로 입력을 만들어 사건마다 예측값 18개                  pred_out()
  -> 18개의 최솟값과 최댓값(예측 구간) + 최선 모델의 값                Prediction.R
```

**입력.** 대부분의 열이 무엇인지는 코드에 적혀 있지 않습니다. "추정"이라고 쓴 설명은 이름을 보고 짐작한 것입니다.

| 묶음 | 열 | 뜻 | 쓰는 방식 |
|:--|:--|:--|:--|
| 0 | `ndate`, `tot`, `area` (권역 1 은 `sido_code` 도) | 사건 일수(코드에서 계산), `tot` 은 총강우로 추정, 시군구 면적 | 그대로 |
| 1 | `d7` ... `d1` | 사건 7일 전부터 1일 전까지의 일강우로 추정 | 주성분 |
| 2 | `x1` ... `x24` | 1시간부터 24시간까지의 최대 강우로 추정 | 주성분 |
| 3 | `grdp`, `money`, `property_tax`, `building`, `city_rate`, `people`, `vulnerable` | 시군구 경제·인구 통계, 1년 전 값 | 주성분 |
| 4 | `recovery_cost`, `tlength`, `owgr`, `cnt1-4`, `sum1-4`, `avg1-4`, `ccnt`, `ctpc`, `cDRZ`, `nDRZ` | 과거 피해와 복구비, 재해위험지구(`DRZ`)와 기타 시설로 추정, 1년 전 값 | 주성분 |

**모델 18개.** 방법마다 log(y) 를 세 번 학습합니다. 1번은 묶음 0 과 묶음 1–2 의 주성분, 2번은 여기에 묶음 3, 3번은 묶음 4 까지 더합니다. 모델 이름은 `lm1` 부터 `clreg3` 까지입니다.

| 방법 | R | 설정값 고르기 (10겹 교차검증, log y 의 RMSE 가 가장 작은 값) |
|:--|:--|:--|
| lm | caret `lm` | 없음 |
| glmnet | caret `glmnet`, 입력을 평균 0·표준편차 1 로 맞춤 | alpha 5개(0.1–1) x lambda 5개(caret 기본 목록) |
| knn | caret `knn` (knnreg) | k = 5, 7, 9, 11, 13 |
| rf | caret `rf`, 나무 1,000 그루 | mtry: 2부터 입력 수까지 5개 |
| gbm | caret `gbm` | 나무 500 또는 1,000 그루 x 나무당 분기 10, 15, 20 번 x 학습률 0.005, 0.01, 0.02, 끝 노드당 최소 10행 |
| clreg | 자체 코드 `clreg.R` | 그룹 기준값은 마지막 연도 RMSE 로, 판별 확률 기준값은 교차검증 ROC 로 |

**clreg** 는 피해액 기준값으로 사건을 작은 피해 그룹과 큰 피해 그룹으로 나눕니다. 사건이 어느 그룹인지는 Firth 로지스틱 회귀로 판별하고, 그룹마다 따로 로그 선형회귀로 예측합니다. 기준값은 y 의 10 %, 15 %, ..., 90 % 분위수 가운데에서 고릅니다.

**데이터 역할.**

| 데이터 | 쓰는 곳 | 역할 |
|:--|:--|:--|
| 마지막 연도를 뺀 해 | 검증 단계에서 모델 18개 학습 | 학습 |
| 마지막 연도 | 모델 18개 비교, 권역별 최선 모델, 예측 구간이 실제값을 덮는지 | 검증 |
| 전체 연도 | 예측에 쓸 최종 모델 | 최종 학습 |
| 학습 행을 10묶음으로 나눈 폴드 | glmnet, knn, rf, gbm 의 설정값, clreg 의 판별 확률 기준값 | 교차검증 |
| clreg 에 넘긴 행 가운데 마지막 연도 | clreg 의 피해액 기준값 | clreg 내부 검증 |

검증 단계에서 clreg 는 마지막 연도 직전 해까지를 받으므로, clreg 내부 검증 연도는 마지막에서 두 번째 해입니다. 최종 단계에서는 마지막 연도입니다.

## 3. 코드가 읽는 데이터

모든 파일은 EUC-KR 인코딩의 CSV 입니다. 이름 열(`sgg` 또는 `sido`)이 있는 표는 붙이기 전에 그 열을 뺍니다.

| 파일 | 붙이는 기준 | 쓰는 열 |
|:--|:--|:--|
| `Dat_train/tdat.csv` | 없음 | `sgg_code`, `sido_code`, `region` (1-4), `beg_date`, `end_date`, `y`, `tot`, `d1`-`d7`, `x1`-`x24` |
| `Dat_input/idat.csv` | 없음 | 위와 같고 `y` 만 없음 |
| `Dat_extra/00_area.csv` | `sgg_code` | `area` |
| `Dat_extra/01_grdp.csv` ... `08_recovery_cost.csv` | `sgg_code`, `year` + 1 | `grdp`, `money`, `property_tax`, `building`, `city_rate`, `people`, `vulnerable`, `recovery_cost` |
| `Dat_extra/09_odat.csv` | `sido_code`, `year` + 1 | 코드에 열 이름이 없음. 합성 데이터: `tlength`, `owgr` |
| `Dat_extra/10_ddat.csv` | `sgg_code`, `year` + 1 | 코드에 열 이름이 없음. 합성 데이터: `cnt1-4`, `sum1-4`, `avg1-4` |
| `Dat_extra/11_pdat.csv` | `sgg_code`, `year` + 1 | 코드에 열 이름이 없음. 합성 데이터: `ccnt`, `ctpc` |
| `Dat_extra/12_cDRZ.csv`, `13_nDRZ.csv` | `sgg_code`, `year` + 1 | `cDRZ`, `nDRZ` |

"`year` + 1" 은 t년 표를 t+1년 사건에 붙인다는 뜻입니다.

**합성 데이터** ([python/make_synthetic_data.py](python/make_synthetic_data.py), 난수 시드 2019):

- 권역 4개 x 시도 3개 x 시군구 8개 = 시군구 96개. 2008–2018년 피해 사건 1,924건, 2019년 예측용 사건 202건.
- 호우마다 시간별 강우 계열을 만들고, `x1`–`x24` 는 그 계열의 1–24시간 최대 누적 강우, `tot` 은 총합으로 했습니다.
- 피해액은 강우, 면적, 취약 인구 비율, 도시화율, 사건 전 1주 강우, 권역, 시군구별 숨은 취약도로 정해지는 로그 선형식에 잡음을 더해 만들었습니다. 피해액 중앙값은 295,674 입니다(합성 데이터에서는 천 원 단위로 둠).
- 외부 표 3개에서 행의 2 %를 빼 두었고(코드가 0으로 채움), `nDRZ` 는 거의 항상 0 입니다(주성분 전처리에서 빠짐).

## 4. R 코드 설명

R 파일은 현재 파일에는 없습니다. 아래 행 번호는 커밋 [4582c73](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657) 의 파일로 연결됩니다.

### 4.1 Modeling.R

| 행 | 하는 일 |
|:--|:--|
| [16-32](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L16-L32) | 패키지를 불러오고 `Rfun/` 의 함수를 `source()` |
| [36-49](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L36-L49) | `tdat.csv` 를 읽고 `ndate`(일수), `year` 를 더한 뒤 `makedat()` 로 외부 표를 붙임 |
| [53-61](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L53-L61) | y 가 36 % 분위수와 100 % 분위수 사이인 사건만 남김("Outlier elimination") |
| [67-85](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L67-L85) | 검증: 마지막 연도를 뺀 해로 권역마다 `dev()`, 마지막 연도에 `pred_out()`, `hval_out()`. `sido = T` 는 권역 1 만 |
| [88-117](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L88-L117) | 권역별 최선 모델(`optm.rds`), `validation.txt`, `validation.pdf` |
| [120-180](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L120-L180) | 모델 18개의 최소–최대 예측 구간(`pint_mm()`), `coverage.pdf`, `validation.csv` |
| [185-193](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L185-L193) | 전체 연도로 권역마다 `dev()` 를 다시 해 `obj1.rds` ... `obj4.rds` 로 저장 |

### 4.2 Prediction.R

| 행 | 하는 일 |
|:--|:--|
| [31-44](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L31-L44) | `idat.csv` 를 읽고 같은 입력을 만듦 |
| [49-53](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L49-L53) | 권역 모델 4개와 `optm.rds` 를 불러옴 |
| [57-72](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L57-L72) | 권역마다 `tryCatch()` 안에서 `pred_out()` |
| [73-89](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L73-L89) | 행마다 `pred_lwr` = 18개의 최솟값, `pred_upr` = 최댓값, `pred_fit` = 최선 모델 값 |
| [93-100](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L93-L100) | `prediction.csv` 저장 |

### 4.3 Sub code #1/xxfun.R: `makedat()`

[5–214행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L5-L214). 표 14개를 `merge(..., all.x = T)` 로 붙입니다. `00_area` 를 뺀 표는 먼저 `year + 1` 을 합니다([예: 29행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L29)). [206행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L206)에서 모든 빈 값을 0으로 바꾸고, [209–210행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L209-L210)에서 시군구와 시작일 순으로 정렬합니다.

### 4.4 Sub code #1/pcfun.R

`calc_pc1()`, `calc_pc2()` 를 정의하지만, 다른 파일 어디에서도 부르지 않습니다.

### 4.5 Sub code #1/cvfun.R

`cv.folds()`([5–7행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/cvfun.R#L5-L7))는 행을 섞어 k개 폴드에 차례로 나눕니다. `cv_cm()`([11–42행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/cvfun.R#L11-L42))은 혼동 행렬을 계산하는데, clreg 는 여기서 넘겨준 기준값만 다시 받아 씁니다.

### 4.6 Sub code #1/dvfun.R

- **`dev()`** ([4–582행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L4-L582)): 입력 묶음([11–41](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L11-L41)), 묶음마다 `preProcess(method = c("center", "scale", "zv", "nzv", "pca"), thresh = 0.9)`([43–72](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L43-L72)), 입력 조합 3가지와 log(y) 회귀식([74–134](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L74-L134)), 10겹 교차검증([137–142](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L137-L142)), lm·glmnet·knn·rf·gbm([150–391](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L150-L391)), clreg 3개([394–537](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L394-L537)).
- **`pred_out()`** ([587–797](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L587-L797)): 저장된 주성분 변환과 모델 18개를 적용하고, 로그 예측값마다 `exp()` 를 취해 권역·시군구·연도·날짜 순으로 정렬합니다.
- **`hval_out()`** ([802–879](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L802-L879)): 예측값에 실제 y 를 붙여 모델마다 RMSE, NRMSE(RMSE ÷ y 의 범위), MAPE 를 계산하고, RMSE 가 가장 작은 모델을 최선으로 고르며 막대그림을 그립니다.
- **`pint_mm()`** ([884–1006](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L884-L1006)): 검증 사건마다 예측값 18개의 최솟값과 최댓값, 실제값이 그 안에 드는 사건 수, 그림 2장(원래 단위, 로그 단위).

### 4.7 Sub code #1/clreg.R

- 피해액 기준값 후보([60–76](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L60-L76)): 학습과 검증 y 를 합친 분위수 0.10–0.90 (0.05 간격). 학습 데이터에서 한쪽 그룹이 비는 후보는 뺍니다.
- 후보마다 판별 확률 기준값([81–299](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L81-L299)): 학습 행으로 10겹 교차검증(`set.seed(i)`). 폴드마다 학습 행으로 주성분을 만들고, Firth 로지스틱 회귀(`glm(..., method = "firthglm.fit")`)와 `step()` 으로 `area:tot` 항을 넣을지 정합니다. 모은 ROC 곡선에서 민감도 + 특이도가 가장 큰 점의 기준값을 씁니다.
- 피해액 기준값 고르기([304–749](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L304-L749)): 후보마다 학습 행으로 그룹별 로그 선형회귀(각자 주성분과 `step()`)와 로지스틱 회귀를 학습하고, 검증 행을 그룹에 나눠 예측해 RMSE 가 가장 작은 후보를 고릅니다.
- 다시 학습([754–1056](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L754-L1056)): 고른 기준값으로 학습과 검증 행 전체를 다시 학습합니다.
- `predict.clreg()` ([1090–1238](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L1090-L1238)): 로지스틱 회귀로 그룹을 정하고, 그 그룹의 회귀로 예측한 뒤 `exp()` 를 취합니다.

## 5. R 코드에서 찾은 문제점과 특이 동작

**문제점.** 파이썬 버전에서 고친 것은 코드 실행을 막는 1번과 2번뿐입니다. 나머지는 R 과 같은 결과를 내도록 그대로 두었고, 고치고 싶은 사람을 위해 여기에 적습니다.

| 번호 | 문제 | 위치 | 파이썬 버전 |
|:--|:--|:--|:--|
| 1 | 스크립트는 `Rfun/*.R` 을 불러오지만 파일은 `Sub code #1/` 에 있고, 데이터 폴더(`Dat_train/`, `Dat_extra/`, `Dat_input/`)와 출력 폴더(`Robj/`, `Output/`)도 저장소에 없어서 그대로는 실행되지 않음 | [Modeling.R 28-32](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L28-L32) | 폴더를 실행 옵션으로 받음 |
| 2 | 예측 입력에 행이 없는 권역이 하나라도 있으면 `Prediction.R` 이 멈춤. `tryCatch()` 가 `NULL` 을 돌려주고, 이어서 `apply(ptmp, 1, min)` 이 "dim(X)는 반드시 양의 값을 가지는 길이를 가져야 합니다" 오류를 냄. 출력 파일도 만들지 않음(권역 4 를 빼고 돌려서 확인) | [Prediction.R 57-89](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L57-L89) | 그 권역을 건너뛰고 알림 |
| 3 | "이상치 제거"가 실제로는 피해액 하위 36 %를 버리고 위쪽은 하나도 버리지 않음. 모델이 작은 사건을 보지 못하므로 작은 사건은 크게 예측할 가능성이 큼 | [Modeling.R 53-61](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L53-L61) | 그대로 |
| 4 | 권역 1 모델에서 `sido_code`(시도 코드)를 범주가 아니라 크기가 있는 숫자로 넣음 | [dvfun.R 11-15](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L11-L15) | 그대로 |
| 5 | clreg 피해액 기준값은 학습과 검증 y 를 합친 분위수인데, 두 그룹이 모두 있는지 확인할 때는 학습 데이터 분위수만 씀 | [clreg.R 63-76](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L63-L76) | 그대로 |
| 6 | 외부 표의 빈 값을 0으로 채워서 실제 0과 구별할 수 없음 | [xxfun.R 206](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L206) | 그대로 |
| 7 | knn 이 표준화 없이 거리를 재서, 값이 큰 열(면적, 강우)이 이웃을 정함. 8장의 R 실행에서 knn1, knn2, knn3 의 예측이 모든 권역에서 똑같았음 | [dvfun.R 229-270](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L229-L270) | 그대로 |
| 8 | log(y) 로 학습하고 `exp()` 로 되돌릴 때 보정이 없어서, 예측값이 평균 피해보다 작게(중앙값에 가깝게) 나옴 | [dvfun.R 647-652](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L647-L652) | 그대로 |
| 9 | `pcfun.R` 은 쓰이지 않음 | [pcfun.R](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/pcfun.R) | 옮기지 않음 |
| 10 | 사소한 것: 모든 행이 그룹 1 일 때 `predict.clreg()` 가 그룹을 0으로 적음([1212행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L1212)). `hval_out()` 이 R² 를 계산하고 버림([837행](https://github.com/sinkei9456/Technical-Manual-for-Heavy-rain-damage-prediction-technique/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L837)). lm2, lm3 앞에서는 `set.seed(1)` 을 다시 하지 않아 다른 모델과 교차검증 폴드가 다름(쓰이지 않는 lm 교차검증 점수에만 영향) | | 그대로 (결과에 영향 없음) |

**일부러 그대로 옮긴 특이 동작.** 오류는 아니지만 R 패키지의 동작 방식이 숫자를 바꾸므로, 파이썬 버전도 똑같이 했습니다.

- **Firth 로지스틱 회귀의 `step()`.** R 의 `add1.glm()`, `drop1.glm()` 은 후보 모델을 Firth 방식이 아니라 보통 최우추정(`glm.fit`)으로 다시 맞추고, 현재 모델은 Firth 이탈도를 그대로 씁니다. 그래서 `area:tot` 항은 최우추정 이탈도 + 2 x 순위가 작아질 때 들어가고, Firth 로 다시 맞춘 뒤 다시 빠질 수도 있습니다([linear.py `step_firth()`](python/heavyrain/linear.py)).
- **Firth 적합.** mbest 는 (점수' I⁻¹ 점수)² ≤ 2e-8 에서 멈추는데, 이 기준이 느슨해서 결과가 반복 경로에 따라 달라집니다. 그래서 mbest 의 뉴턴 단계와 단계 크기 탐색을 그대로 옮겼습니다.
- **ROC 기준값.** PRROC 의 ROC 곡선 표에서는 기준값 칸이 비율 칸과 한 줄 어긋나 있습니다. 고른 줄에 적힌 기준값을 그대로 씁니다.
- **caret 의 glmnet.** caret 은 glmnet 의 기본 경로(lambda 100개)를 계산한 뒤, 고른 lambda 에서는 가장 가까운 두 경로 값 사이를 선형 보간해 예측합니다. 또 glmnet 은 y 도 내부에서 표준화하므로 같은 lambda 라도 뜻이 달라집니다.
- **knn 동점.** caret 의 C 코드는 거리 제곱이 k번째와 정확히 같은 이웃을 더 넣습니다.
- **주성분.** caret 은 주성분 하나로 분산 90 %에 닿아도 최소 2개를 남깁니다.

## 6. 파이썬 버전

### 6.1 R 과 파이썬 대응

| R | 파이썬 |
|:--|:--|
| `Modeling.R` | [train.py](python/train.py) |
| `Prediction.R` | [predict.py](python/predict.py) |
| `makedat()`, 분위수로 행 빼기 | `features.makedat()`, `features.drop_small_damage()` |
| caret `preProcess(..., "pca", thresh)` | `preprocess.CaretPCA` |
| lm, glmnet, knn, rf, gbm 의 caret `train()` | `learners.caret_train()` 과 `LM`, `Glmnet`, `KNN`, `RF`, `GBM` |
| `glm(..., method = "firthglm.fit")`, `step()` | `linear.FirthLogistic`, `linear.step_firth()`, `linear.step_lm()` |
| `PRROC::roc.curve()` + `which.max()` | `clreg.roc_youden()` |
| `clreg()`, `predict.clreg()` | `clreg.Clreg.fit()`, `.predict()` |
| `dev()`, `pred_out()`, `hval_out()`, `pint_mm()` | `develop.dev()`, `pred_out()`, `hval_out()`, `pint_mm()` |
| ggplot 그림 | `plots.validation_bars()`, `plots.coverage()` |
| `saveRDS()` / `readRDS()` | joblib (`obj1.joblib` ... `obj4.joblib`) 와 `optm.json` |

### 6.2 R 코드와 다른 점

| 항목 | R 코드 | 파이썬 버전 |
|:--|:--|:--|
| 폴더 | 고정(`Rfun/`, `Dat_train/` 등) | 옵션 `--data-dir`, `--out-dir`, `--model-dir` |
| 입력 행이 없는 권역 | 멈춤 | 건너뜀 |
| 난수 | R | numpy. 그래서 교차검증 폴드, rf, gbm 이 R 실행 결과와 다름(8장) |
| rf | randomForest | randomForest 와 같은 규칙으로 키운 scikit-learn 나무 숲(부트스트랩 행을 실제로 복사, 노드 크기 5, mtry) |
| gbm | gbm 패키지 | 같은 설정의 scikit-learn `GradientBoostingRegressor`(분기 수 + 1 개 끝 노드를 좋은 분기부터 키움, 나무마다 행 절반, 끝 노드당 최소 10행). 구현은 다름 |
| glmnet | Fortran, 수렴 허용치 1e-7 | scikit-learn 좌표 하강법, 허용치 1e-12 |
| lm 교차검증 점수 | 계산하지만 어디에도 안 씀 | 계산하지 않음 |
| 모델 저장 | `.rds` | `.joblib` 과 `.json`. CSV 출력은 열 구성과 인코딩이 같음 |
| 그림 | ggplot | matplotlib, 같은 내용, 그림 안 글자는 영문 |

## 7. 파이썬 버전 실행 방법

확인한 환경: Python 3.12.3, numpy 2.5.2, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.5.2, matplotlib([requirements.txt](python/requirements.txt) 참고).

```bash
pip install -r python/requirements.txt
cd python
python make_synthetic_data.py                 # data/ 를 만듦 (이미 들어 있음)
python train.py --data-dir data --out-dir output --model-dir model --mc 8
python predict.py --data-dir data --model-dir model --out output/prediction.csv
```

| 스크립트 | 옵션 | 기본값 | 뜻 |
|:--|:--|:--|:--|
| train.py | `--data-dir` | data | `Dat_train/`, `Dat_extra/` 가 있는 폴더 |
| | `--out-dir` | output | 검증 결과 폴더 |
| | `--model-dir` | model | 모델 저장 폴더 |
| | `--mc` | 2 | 병렬 작업 수 (R `mc`) |
| | `--seed` | 1 | 난수 시드 (R `set.seed(1)`) |
| | `--encoding` | euc-kr | 모든 CSV 의 인코딩 |
| predict.py | `--data-dir` | data | `Dat_extra/`(와 `Dat_input/`)가 있는 폴더 |
| | `--input` | data/Dat_input/idat.csv | 예측할 사건 |
| | `--model-dir` | model | train.py 가 저장한 모델 |
| | `--out` | output/prediction.csv | 결과 파일 |
| | `--encoding` | euc-kr | 모든 CSV 의 인코딩 |

**train.py 결과물**

- `output/validation.txt`: 검증 연도에서 권역별 모델 18개의 RMSE, NRMSE, MAPE 와 권역별 최선 모델.
- `output/validation.csv`: 검증 사건마다 예측값 18개, `pred_lwr`·`pred_upr`(18개의 최솟값·최댓값), `pred_fit`(그 권역 최선 모델의 값), `y`, `cover`(y 가 구간 안이면 1).
- `output/validation.pdf`: 권역마다 한 쪽. 가로축은 방법(R 그림처럼 알파벳 순), 막대는 입력 조합 1, 2, 3 (색 3가지), 세로축은 검증 연도 RMSE, 제목에 최선 모델 이름.
- `output/coverage.pdf`: 1쪽은 피해액 단위, 2쪽은 로그 단위. 가로축은 모든 권역의 검증 사건 순서, 검은 세로선은 예측값 18개의 최솟값부터 최댓값까지, 색이 채워진 원은 실제 피해액(색은 사건을 구별할 뿐 뜻은 없음), 빈 원은 예측값 18개의 평균, 제목은 실제값이 세로선 안에 든 사건 수.
- `model/obj1.joblib` ... `obj4.joblib`, `model/optm.json`.

**predict.py 결과물**: `prediction.csv`. 키 열(`region`, `sgg_code`, `year`, `beg_date`, `end_date`), 예측값 18개, `pred_lwr`, `pred_upr`, `pred_fit`.

참고: 모델 파일은 pickle 입니다. 믿을 수 있는 파일만, 학습 때와 같은 scikit-learn 버전으로 불러오세요. 합성 데이터에서 `train.py --mc 16` 은 20코어 서버에서 3.6분 걸렸습니다(R `Modeling.R` 은 코드에 적힌 `mc = 2` 로 18분. 병렬 설정이 달라 공정한 속도 비교는 아닙니다).

## 8. R 과 파이썬 결과 비교

두 버전을 같은 합성 데이터로 돌렸습니다. R 환경은 4.5.3 버전, caret 7.0.1, glmnet 5.0, gbm 2.3.1, randomForest 4.7-1.2, mbest 0.6.1, PRROC 1.4 입니다. R 스크립트는 고치지 않았고, 코드가 기대하는 폴더(`Rfun/`, `Dat_*`, `Robj/`, `Output/`)만 맞춰 주었습니다. 비교는 세 단계로 했습니다.

**8.1 같은 입력으로 부분별 비교.** 난수가 들어가지 않는 부분이라 반올림 오차 수준까지 같아야 합니다.

표 1. R 과 파이썬의 가장 큰 차이 (상대 차이, 단위 없음)

| 부분 | 가장 큰 차이 | 비고 |
|:--|--:|:--|
| 입력 묶음 4개의 주성분 (권역 1) | 5e-14 | 빠지는 열과 주성분 개수(7, 2, 2, 10)도 같음 |
| lm 과 `step()` 의 선택 | 8e-13 | `area:tot` 를 넣는 경우와 안 넣는 경우 4가지, 선택도 모두 같음 |
| Firth 로지스틱 회귀와 `step()` 의 선택 | 2e-14 | 7가지, 선택도 모두 같음 |
| glmnet lambda 목록 | 1e-10 | |
| glmnet 예측, R 을 허용치 1e-14 로 돌린 값과 비교 | 2e-6 | R 기본 허용치(1e-7)로는 R 자체가 최대 6e-3 어긋남 |
| clreg, 권역 1, R 의 폴드 사용: 기준값 후보 표 | 7e-12 | 후보마다 ROC 면적, 판별 확률 기준값, RMSE |
| clreg, 권역 1, R 의 폴드 사용: 최종 예측 | 7e-12 | |

**8.2 R 의 난수 폴드를 넣은 전체 흐름 비교.** caret 7 은 난수를 하나 뽑아 그 값을 폴드의 시드로 씁니다. R 에서 이 순서를 그대로 다시 하면 R 모델 파일에 저장된 폴드와 똑같았습니다(8번 확인 중 8번). 이 폴드와 clreg 의 폴드를 파이썬에 넣어 전체를 돌렸습니다.

표 2. 개별 예측값의 가장 큰 상대 차이, 입력 조합 3가지 x 권역 4개 (단위 없음)

| 방법 | 검증 (2008–2017년 모델로 2018년 예측, 115건) | 예측 (전체 연도 모델로 2019년 예측, 202건) |
|:--|--:|--:|
| lm | 3e-12 | 1e-13 |
| glmnet | 3e-3 | 3e-3 |
| knn | 8e-15 | 7e-15 |
| clreg | 6e-11 | 1e-10 |
| rf | 0.25 | 0.28 |
| gbm | 0.42 | 0.21 |

폴드가 같으면 파이썬은 모든 곳에서 R 과 같은 설정값을 고릅니다(모든 권역의 glmnet1, knn1, clreg1–3 은 R 이 저장한 모델과 직접 대조). glmnet 의 차이는 R 자체의 수렴 허용치 때문이고, rf 와 gbm 은 모델 안에서 쓰는 난수(부트스트랩 행, 후보 변수, 부분 표본)가 서로 달라서 다릅니다.

**8.3 평소 쓰는 방식: 언어마다 자체 난수.** 실제로 쓸 때 두 버전이 보이는 차이입니다.

표 3. 권역별 최선 모델과 검증 연도 RMSE (단위: 천 원)

| 권역 | 최선 모델, R | 최선 모델, 파이썬 | RMSE, R | RMSE, 파이썬 | 검증 사건 수 |
|:--|:--|:--|--:|--:|--:|
| 1 | gbm2 | gbm2 | 676,357 | 685,701 | 20 |
| 2 | lm3 | lm3 | 1,546,954 | 1,546,954 | 22 |
| 3 | rf2 | rf2 | 426,548 | 451,201 | 9 |
| 4 | clreg3 | clreg3 | 2,630,022 | 2,674,436 | 64 |

표 4. 파이썬 검증 RMSE ÷ R 검증 RMSE, 입력 조합 3가지 x 권역 4개 가운데 최소와 최대 (단위 없음)

| 방법 | 최소 | 최대 |
|:--|--:|--:|
| lm | 1.00 | 1.00 |
| glmnet | 0.96 | 1.09 |
| knn | 1.00 | 1.06 |
| rf | 0.98 | 1.06 |
| gbm | 0.74 | 1.14 |
| clreg | 0.76 | 1.04 |

- 최소–최대 예측 구간이 실제 피해액을 덮은 검증 사건은 115건 가운데 R 39건, 파이썬 40건입니다.
- 2019년 예측(202건)에서 파이썬 최종 예측 `pred_fit` ÷ R 최종 예측의 중앙값은 1.000 이었습니다. 99 %가 10 % 안, 전부가 20 % 안에 들었습니다(최소 0.90, 최대 1.10).
- 출력 파일의 행과 열 구성은 R 과 같습니다.

정리하면, 난수가 없는 부분은 R 과 같은 숫자를 내고, 난수가 있는 부분은 개별 숫자는 다르지만 같은 선택과 비슷한 오차를 냅니다.

## 9. 참고 문헌

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22.
- Kuhn, M. (2008). Building predictive models in R using the caret package. *Journal of Statistical Software*, 28(5), 1–26.
- Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease. *ACM Transactions on Mathematical Software*, 20(3), 286–307.
- R 코드가 쓰는 R 패키지: caret, glmnet, gbm, randomForest, mbest(Firth 회귀), PRROC(ROC 곡선), dplyr, ggplot2, doParallel.
