---
title: "XGBoostExplainer����������Ă��邩���ׂ�"
author: Satoshi Kato (@katokohaku)
output: 
  html_document:
    keep_md: yes
    toc: yes
  md_document:
    variant: markdown_github
---




# �ړI

XGBoost�̗\���𕪉�����c�[��`XGBoostExplainer`�́A����C���X�^���X�ɂ��ē���ꂽXGBoost�ɂ��\�����ʂ��A�ǂ̂悤�ɍ\������Ă��邩�������Ă����B

<img src="img/sample.png" width="400">

�R���Z�v�g�Ƃ��ẮArandomforest�ɂ�����forestfloor�Ɠ������Afeature contribution�̍l�������A�X�̃C���X�^���X��xgboost�ɂ��\�����ʂ̉����̂��߂ɓK�p���Ă���B

�T���I�ȃf�[�^���͂Ɏg�������Ȃ炨���ނ˖��Ȃ����A�_���Ȃǂ̏ڍׂȎ������������Ȃ��������߁A��̓I�ɉ�������Ă��邩�������悤�Ƃ��č������B�����ŁA`XGBoostExplainer`�̎�����ǂ������āA��������Ă��邩���ׂ��B

## �֘A�V���[�Y

1. �Ƃ肠�����g���Ă݂�i���̋L���j
2. �\�����ʂ̉����v���Z�X��step-by-step�Ŏ��s����
3. �w�K����xgboost�̃��[�����o��step-by-step�Ŏ��s����
4. �\�����ʂ�breakdown��step-by-step�Ŏ��s����

## �Q�l

�J�����̏Љ�L��

> [NEW R package that makes XGBoost   interpretable](https://medium.com/applied-data-science/new-r-package-the-xgboost-explainer-51dd7d1aa211)


# �Ƃ肠�����g���Ă݂�

���łɓ��{��̏Љ�L��������B

> [xgboost �̒���`���Ă݂�](https://qiita.com/vascoosx/items/efb3177ecf2ead5d8ce0)

`xgboostExplainer`�̃}�j���A���ɂ���example����R�s�y�𒭂߂�B

## �C���X�g�[��

�{�Ƃ̋L���ɏ]����github����C���X�g�[��


```r
install.packages("devtools") 
library(devtools) 
install_github("AppliedDataSciencePartners/xgboostExplainer")
```
## XGB���f���̊w�K�Ɨ\��

�����`xgboost`�p�b�P�[�W�t���̃T���v���f�[�^�ŁA�����̐H����L�m�R�ƓŃL�m�R��2�l���ށB�ׂ����`���[�j���O�́A�K�v�ɉ�����autoxgb������Ń`���[�j���O����Ƃ悢���A����͏ȗ��B


```r
library(xgboost)
#> Warning: package 'xgboost' was built under R version 3.5.3
require(tidyverse)
library(xgboost)
library(xgboostExplainer)

set.seed(123)

data(agaricus.train, package='xgboost')

X = as.matrix(agaricus.train$data)
y = agaricus.train$label
table(y)
train_idx = 1:5000

train.data = X[train_idx,]
test.data = X[-train_idx,]

xgb.train.data <- xgb.DMatrix(train.data, label = y[train_idx])
xgb.test.data <- xgb.DMatrix(test.data)

param <- list(objective = "binary:logistic")
xgb.model <- xgboost(param =param,  data = xgb.train.data, nrounds=3)
# 
# col_names = colnames(X)
# 
# pred.train = predict(xgb.model,X)
# nodes.train = predict(xgb.model,X,predleaf =TRUE)
# trees = xgb.model.dt.tree(col_names, model = xgb.model)
```

## �ʂ̗\�����ʂ̉���

`xgboostExplainer`�̃}�j���A���ɂ���example�̃R�s�y�i�Â��j�B���x��wrap����Ă��邽�߂킸��3�s��step-by-step����������B

### STEP.1. �w�K�ς�XGB���f�����烋�[���Z�b�g�ileaf�܂ł̃p�X�j��񋓂��ăe�[�u����

`base_score`�I�v�V������xgboost�̃I�v�V�������̂܂܂ŁA�^�[�Q�b�g�W�c�̃N���X�s�ύt��\�����O�m���B���Ȃ킿����F���၁300:700������ł���Ώۂł���΁A`base_score = 0.3`�ƂȂ�(�f�t�H���g��1:1��\��0.5)�B


```r
library(xgboostExplainer)

explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.5, trees = NULL)
#> 
#> Creating the trees of the xgboost model...
#> Getting the leaf nodes for the training set observations...
#> Building the Explainer...
#> STEP 1 of 2
#> 
#> Recalculating the cover for each non-leaf... 
#>   |                                                                         |                                                                 |   0%  |                                                                         |===                                                              |   4%  |                                                                         |=====                                                            |   8%  |                                                                         |========                                                         |  12%  |                                                                         |==========                                                       |  16%  |                                                                         |=============                                                    |  20%  |                                                                         |================                                                 |  24%  |                                                                         |==================                                               |  28%  |                                                                         |=====================                                            |  32%  |                                                                         |=======================                                          |  36%  |                                                                         |==========================                                       |  40%  |                                                                         |=============================                                    |  44%  |                                                                         |===============================                                  |  48%  |                                                                         |==================================                               |  52%  |                                                                         |====================================                             |  56%  |                                                                         |=======================================                          |  60%  |                                                                         |==========================================                       |  64%  |                                                                         |============================================                     |  68%  |                                                                         |===============================================                  |  72%  |                                                                         |=================================================                |  76%  |                                                                         |====================================================             |  80%  |                                                                         |=======================================================          |  84%  |                                                                         |=========================================================        |  88%  |                                                                         |============================================================     |  92%  |                                                                         |==============================================================   |  96%  |                                                                         |=================================================================| 100%
#> 
#> Finding the stats for the xgboost trees...
#>   |                                                                         |                                                                 |   0%  |                                                                         |======================                                           |  33%  |                                                                         |===========================================                      |  67%  |                                                                         |=================================================================| 100%
#> 
#> STEP 2 of 2
#> 
#> Getting breakdown for each leaf of each tree...
#>   |                                                                         |                                                                 |   0%  |                                                                         |======================                                           |  33%  |                                                                         |===========================================                      |  67%  |                                                                         |=================================================================| 100%
#> 
#> DONE!
```
### STEP.2. Get multiple prediction breakdowns from a trained xgboost model

�}�j���A���ɂ� step2�Ƃ���̂����A���̓p�b�P�[�W���g�������Ȃ�X�L�b�v�ł��Ă��܂��B

### STEP.3. �\���Ώ�(�C���X�^���X)�ɓK�p�����etree�̃p�X���W�v���ĉ���

2�l����(`binary:logistic`)�ł́A�Б��̃N���X�ɑ�����m��p�i�����̐��l�j�̃��W�b�g�i�ΐ��I�b�Y�G�_�O���t���̐��l�j���������킳��Ă���l�q��\������B


```r
showWaterfall(xgb.model, explainer, xgb.test.data, test.data,  2, type = "binary")
#> 
#> 
#> Extracting the breakdown of each prediction...
#>   |                                                                         |                                                                 |   0%  |                                                                         |======================                                           |  33%  |                                                                         |===========================================                      |  67%  |                                                                         |=================================================================| 100%
#> 
#> DONE!
#> 
#> Prediction:  0.811208
#> Weight:  1.457879
#> Breakdown
#>        intercept  gill-size=broad        odor=foul        odor=none 
#>      -0.27084657       1.61423045      -0.67129347       0.45408751 
#>       odor=anise      odor=almond cap-color=yellow 
#>       0.13628094       0.13073006       0.06468987
```

![](1_just_use_wrapper_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

(�Q�l) `binary:logistic`�̏ꍇ�A`base_score`�Őݒ肵�����O�m���� �x�[�X���C���Ƃ���intercept�ɔ��f�����B���L�̗�ł�intercept�������������Ă��邱�Ƃɒ��ڂ��ꂽ���B


```r
explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.2, trees = NULL)
```

```r
showWaterfall(xgb.model, explainer, xgb.test.data, test.data,  8, type = "binary")
#> 
#> 
#> Extracting the breakdown of each prediction...
#>   |                                                                         |                                                                 |   0%  |                                                                         |======================                                           |  33%  |                                                                         |===========================================                      |  67%  |                                                                         |=================================================================| 100%
#> 
#> DONE!
#> 
#> Prediction:  0.05456225
#> Weight:  -2.852306
#> Breakdown
#>               intercept               odor=foul         gill-size=broad 
#>             -1.65714093             -0.67129347             -0.46943285 
#> spore-print-color=green 
#>             -0.05443858
```

![](1_just_use_wrapper_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

����́A`xgboostExplainer`�ɂ��Axgboost�̃��f���Ɨ\�����ʂ���**�������o����A�ǂ��J����Ă��邩����**���ڍׂɌ��Ă����B
