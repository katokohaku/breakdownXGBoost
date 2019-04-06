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

����́A`xgboostExplainer`�ɂ���āAxgboost�̃��f���Ɨ\�����ʂ���**�������o����A�ǂ��J����Ă��邩**�Ƀt�H�[�J�X���Ēǂ�������B

# �֘A�V���[�Y

1. [�Ƃ肠�����g���Ă݂�](http://kato-kohaku-0.hatenablog.com/entry/2018/12/14/002253)
2. �\�����ʂ̉����v���Z�X��step-by-step�Ŏ��s����i���̋L���j
3. �w�K����xgboost�̃��[�����o��step-by-step�Ŏ��s����
4. �\�����ʂ�breakdown��step-by-step�Ŏ��s����


# �\�����ʂ̉����v���Z�X��step-by-step�Œ��߂�

`showWaterfall(..., type = "binary")`�̒��g�𔲂��������Ȃ���A�s�x�A�������o����Ă��邩���Ă����B

�Ȃ��A���̋L���ȍ~�ł́A`objective = "binary:logistic"`�������Đi�߂�B`objective = "reg:linear"`�͂��V���v���Ȏ葱���ł���A�O�҂��킩��Ό�҂͎��R�ɗ����ł���B

## �����FXGB���f���̊w�K�Ɨ\��

`xgboostExplainer`�̃}�j���A���ɂ���example����R�s�y�B


```r
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
```

## �����̎葱��(overview)

����C���X�^���X�̗\�����ʂ���������菇�́F

1. �w�K����xgboost�̃��f������\�����[���𒊏o����
2. `showWaterfall()`�̂Ȃ���
    1. �w�肵���C���X�^���X�̗\�����ʂ�\�����[�����番���č\������
    2. �����č\�������\�����ʂ�watarfall chart�ŉ�������

�Ȃ��A

* `buildExplainer()`��**�ǂ������**���f������\�����[���𒊏o���Ă��邩
* `explainPredictions()`��**�ǂ������**�����č\�����邩

�ɂ��Ă͍���͐G��Ȃ��B

### �w�K����xgboost�̃��f������\�����[���𒊏o����

`buildExplainer()`�����o�����\�����[��(�m�[�h)�𒭂߂�B


```r
library(xgboostExplainer)

explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.5, trees = NULL)
```

����̓��[���̐��������̂ŁA��x���o�ꂵ�Ȃ����̂����O���Ă݂�F


```r
nonzero.cols <- abs(explainer) %>% 
  colSums() %>% 
  is_greater_than(0.0) %>% 
  which()

explainer %>% 
  select(nonzero.cols %>% rev()) %>% 
  set_colnames(
    colnames(.) %>% 
      str_trunc(width=10, side="left"))
#>     tree leaf   intercept  ...r=green ...g=silky ...e=broad  odor=none
#>  1:    0    2 -0.11061173  0.00000000  0.0000000  0.0000000  0.0000000
#>  2:    0    7 -0.11061173 -0.02193993  0.0000000 -0.1919848  0.0000000
#>  3:    0    8 -0.11061173  1.13508088  0.0000000 -0.1919848  0.0000000
#>  4:    0   10 -0.11061173  0.00000000  0.0000000  0.6632849  0.1868594
#>  5:    0   12 -0.11061173  0.00000000  0.8995779  0.6632849 -0.6576614
#>  6:    0   14 -0.11061173  0.00000000  0.0000000  0.6632849  0.1868594
#>  7:    0   15 -0.11061173  0.00000000 -0.1528694  0.6632849 -0.6576614
#>  8:    0   16 -0.11061173  0.00000000 -0.1528694  0.6632849 -0.6576614
#>  9:    0   17 -0.11061173  0.00000000  0.0000000  0.6632849  0.1868594
#> 10:    0   18 -0.11061173  0.00000000  0.0000000  0.6632849  0.1868594
#> 11:    1    2 -0.08586974  0.00000000  0.0000000  0.0000000  0.0000000
#> 12:    1    7 -0.08586974 -0.01726586  0.0000000 -0.1483337  0.0000000
#> 13:    1    8 -0.08586974  0.88295840  0.0000000 -0.1483337  0.0000000
#> 14:    1   10 -0.08586974  0.00000000  0.0000000  0.5101908  0.1433645
#> 15:    1   12 -0.08586974  0.00000000  0.7008943  0.5101908 -0.5018480
#> 16:    1   13 -0.08586974  0.00000000  0.0000000  0.5101908  0.1433645
#> 17:    1   14 -0.08586974  0.00000000  0.0000000  0.5101908  0.1433645
#> 18:    1   15 -0.08586974  0.00000000 -0.1208858  0.5101908 -0.5018480
#> 19:    1   16 -0.08586974  0.00000000 -0.1208858  0.5101908 -0.5018480
#> 20:    2    2 -0.07436510  0.00000000  0.0000000  0.0000000  0.0000000
#> 21:    2    7 -0.07436510 -0.01523278  0.0000000 -0.1291144  0.0000000
#> 22:    2    8 -0.07436510  0.76272457  0.0000000 -0.1291144  0.0000000
#> 23:    2   10 -0.07436510  0.00000000  0.0000000  0.4407547  0.1238637
#> 24:    2   12 -0.07436510  0.00000000  0.5980809  0.4407547 -0.4287505
#> 25:    2   13 -0.07436510  0.00000000  0.0000000  0.4407547  0.1238637
#> 26:    2   14 -0.07436510  0.00000000  0.0000000  0.4407547  0.1238637
#> 27:    2   15 -0.07436510  0.00000000 -0.1059506  0.4407547 -0.4287505
#> 28:    2   16 -0.07436510  0.00000000 -0.1059506  0.4407547 -0.4287505
#>     tree leaf   intercept  ...r=green ...g=silky ...e=broad  odor=none
#>      odor=foul  odor=anise  ...=almond  ...bruises  ...=yellow
#>  1:  0.7088975  0.00000000  0.00000000  0.00000000  0.00000000
#>  2: -0.2745896  0.00000000  0.00000000  0.00000000  0.00000000
#>  3: -0.2745896  0.00000000  0.00000000  0.00000000  0.00000000
#>  4: -0.2745896  0.00000000  0.00000000  0.00000000 -1.00780012
#>  5: -0.2745896  0.00000000  0.00000000  0.00000000  0.00000000
#>  6: -0.2745896  0.00000000 -1.02528500  0.00000000  0.06468987
#>  7: -0.2745896  0.00000000  0.00000000 -0.05144537  0.00000000
#>  8: -0.2745896  0.00000000  0.00000000  0.93244731  0.00000000
#>  9: -0.2745896  0.03208427  0.03399723  0.00000000  0.06468987
#> 10: -0.2745896 -1.04934438  0.03399723  0.00000000  0.06468987
#> 11:  0.5493453  0.00000000  0.00000000  0.00000000  0.00000000
#> 12: -0.2125787  0.00000000  0.00000000  0.00000000  0.00000000
#> 13: -0.2125787  0.00000000  0.00000000  0.00000000  0.00000000
#> 14: -0.2125787  0.00000000 -0.78526446  0.00000000  0.00000000
#> 15: -0.2125787  0.00000000  0.00000000  0.00000000  0.00000000
#> 16: -0.2125787  0.05522851  0.05139002  0.00000000  0.00000000
#> 17: -0.2125787 -0.83485874  0.05139002  0.00000000  0.00000000
#> 18: -0.2125787  0.00000000  0.00000000 -0.04271214  0.00000000
#> 19: -0.2125787  0.00000000  0.00000000  0.74060173  0.00000000
#> 20:  0.4766099  0.00000000  0.00000000  0.00000000  0.00000000
#> 21: -0.1841252  0.00000000  0.00000000  0.00000000  0.00000000
#> 22: -0.1841252  0.00000000  0.00000000  0.00000000  0.00000000
#> 23: -0.1841252  0.00000000 -0.67413428  0.00000000  0.00000000
#> 24: -0.1841252  0.00000000  0.00000000  0.00000000  0.00000000
#> 25: -0.1841252  0.04896816  0.04534281  0.00000000  0.00000000
#> 26: -0.1841252 -0.71778146  0.04534281  0.00000000  0.00000000
#> 27: -0.1841252  0.00000000  0.00000000 -0.03984976  0.00000000
#> 28: -0.1841252  0.00000000  0.00000000  0.63572177  0.00000000
#>      odor=foul  odor=anise  ...=almond  ...bruises  ...=yellow
```

`nrounds = 3`�ɑΉ�����`tree`���R�A���ꂼ��̖؂�leaf�ɂ��ǂ蒅�����߂̃��[���̑g�ݍ��킹�ƁA�X�V�����\���̗ʂ��킩��B


### �w�肵���C���X�^���X�̗\�����ʂ�\�����[�����番���č\������


�w�肵���C���X�^���X�̗\�����ʂ��A`buildExplainer()`���č\����������(rules breakdown)�𒭂߂�B


```r
# showWaterfall(xgb.model, explainer, xgb.test.data, test.data,  2, type = "binary")

DMatrix = xgb.test.data
data.matrix = test.data
idx = 2
type = "binary"
threshold = 1e-04
limits = c(NA, NA)

breakdown = explainPredictions(xgb.model, explainer, slice(DMatrix, as.integer(idx)))
```

���[���������̂Ŕ�[���̂��̂������߂�B

```r
breakdown_summary = as.matrix(breakdown)[1, ]
breakdown_summary[which(abs(breakdown_summary) > 0.0)]
#> cap-color=yellow      odor=almond       odor=anise        odor=foul 
#>       0.06468987       0.13073006       0.13628094      -0.67129347 
#>        odor=none  gill-size=broad        intercept 
#>       0.45408751       1.61423045      -0.27084657
```


����́A`objective = "binary:logistic"`�Ȃ̂ŁA�\���l(�ΐ��I�b�Y)�͋t���W�b�g�ϊ��ɂ��\������(�N���X�����m��)�ɕϊ������

```r
(weight = rowSums(breakdown))
#> [1] 1.457879

(pred = 1/(1 + exp(-weight)))
#> [1] 0.811208
```

### �����č\�������\�����ʂ���������

�v���b�g�̕��j�͈ȉ��̒ʂ�B

1. �x�[�X���C��(Intercept)��擪�ɕ\���A�c����C���p�N�g(�v����)���傫�����ɕ\��
2. �C���p�N�g������(`< threshold`)�̃��[���́A���̑�(`other_impact`)�ɂ܂Ƃ߂�


```r
data_for_label = data.matrix[idx, ]
i = order(abs(breakdown_summary), decreasing = TRUE)
breakdown_summary = breakdown_summary[i]
data_for_label = data_for_label[i]

intercept = breakdown_summary[names(breakdown_summary) == "intercept"]
data_for_label = data_for_label[names(breakdown_summary) != "intercept"]
breakdown_summary = breakdown_summary[names(breakdown_summary) != "intercept"]
```


`threshold`�̒l�̓I�v�V�����w��ŕύX�ł��A�C���p�N�g�̑傫�ȃ��[�������ɍi�荞��ŕ\���ł���B�₽��ׂ������[���������ς�������������Ƃ��ł��Ȃ�����((tree depth�̃`���[�j���O�Ɏ��s����overfit���Ă����ł͂��邵�A��̃C���X�^���X�̐����p�r�Ƃ��Ă͎��s���Ă���Ƃ�������B))`threshold`�̓f�t�H���g�̂܂܂ŗǂ������B


```r
i_other = which(abs(breakdown_summary) < threshold)
other_impact = 0
if (length(i_other > 0)) {
  other_impact = sum(breakdown_summary[i_other])
  names(other_impact) = "other"
  breakdown_summary = breakdown_summary[-i_other]
  data_for_label = data_for_label[-i_other]
}
if (abs(other_impact) > 0) {
  breakdown_summary = c(intercept, breakdown_summary, other_impact)
  data_for_label = c("", data_for_label, "")
  labels = paste0(names(breakdown_summary), " = ", data_for_label)
  labels[1] = "intercept"
  labels[length(labels)] = "other"
} else {
  breakdown_summary = c(intercept, breakdown_summary)
  data_for_label = c("", data_for_label)
  labels = paste0(names(breakdown_summary), " = ", data_for_label)
  labels[1] = "intercept"
}

breakdown_summary
```

�撣���ĕ��ёւ����B

### watarfall chart�ɂ��\�����ʂ̕`��

�Z�o�����l�̏o�́B`getinfo`�̉ӏ��̓��x���̎�������������낤�Ƃ��Ă�H�@�L�m�R�̃f�[�^���ƃX�L�b�v�����B


```r

if (!is.null(getinfo(DMatrix, "label"))) {
  cat("\nActual: ", getinfo(slice(DMatrix, as.integer(idx)), "label"))
}
cat("\nPrediction: ", pred)
#> 
#> Prediction:  0.811208
cat("\nWeight: ", weight)
#> 
#> Weight:  1.457879
cat("\nBreakdown")
#> 
#> Breakdown
cat("\n")
print(breakdown_summary)
#>        intercept  gill-size=broad        odor=foul        odor=none 
#>      -0.27084657       1.61423045      -0.67129347       0.45408751 
#>       odor=anise      odor=almond cap-color=yellow 
#>       0.13628094       0.13073006       0.06468987
```


�\������(�N���X�����m��)�ŕ\�����邽�߂̍H�v������B`inverse_logit_labels`��`logit`�́Ay����ΐ��I�b�Y����N���X�����m���ɓǂݑւ��邽�߂̊֐��B

((inverse_logit_trans`�̓R�����g�A�E�g���Ă����삷��B���K�V�[���낤���H))


```r
inverse_logit_trans <- scales::trans_new("inverse logit", transform = plogis, inverse = qlogis)

inverse_logit_labels = function(x) {
  return(1/(1 + exp(-x)))
}

logit = function(x) {
  return(log(x/(1 - x)))
}

ybreaks <- logit(seq(2, 98, 2)/100)

waterfalls::waterfall(
  values = breakdown_summary,
  rect_text_labels = round(breakdown_summary, 2),
  labels = labels, 
  total_rect_text = round(weight, 2),
  calc_total = TRUE,
  total_axis_text = "Prediction") + 
  scale_y_continuous(labels = inverse_logit_labels,
                     breaks = ybreaks, 
                     limits = limits) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

![](2_show_waterfall_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

`waterfalls::waterfall()`�́A�O���̃p�b�P�[�W�����̂܂ܗ��p����B[�J����](https://jameshoward.us/software/waterfall/)���Q�Ƃ��ꂽ���B

����́A`buildExplainer()`���A�ǂ̂悤�Ȏ葱���Ŏw�肵���C���X�^���X�̗\�����ʂ𕪉��č\�����Ă���̂��ڍׂɌ��Ă����B
