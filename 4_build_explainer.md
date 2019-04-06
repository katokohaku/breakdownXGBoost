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

����́A`xgboostExplainer`�ɂ���āAxgboost�̊w�K�ς݃��f������**���[�����ǂ�����Ē��o����Ă��邩**�Ƀt�H�[�J�X���A�K�Xxgboost�̎��������Ȃ���ǂ�������B

Introduction to Boosted Trees (PDF)
https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

XGBoost: A Scalable Tree Boosting System
https://arxiv.org/abs/1603.02754

# �֘A�V���[�Y

1. �Ƃ肠�����g���Ă݂�
2. �\�����ʂ̉����v���Z�X��step-by-step�Ŏ��s����
3. �w�K����xgboost�̃��[�����o��step-by-step�Ŏ��s����i���̋L���j
4. �\�����ʂ�breakdown��step-by-step�Ŏ��s����


# �����FXGB���f���̊w�K�Ɨ\��

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


# �w�K����xgboost�̃��[�����o


`buildExplainer()`�̒��g�𔲂��������Ȃ���Astep-by-step�Œ��߂�



```r
# explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.5, trees = NULL)
# function (xgb.model, trainingData, type = "binary", base_score = 0.5, trees_idx = NULL) 
# {
trainingData = xgb.train.data
type = "binary" 
base_score = 0.5
trees_idx = NULL
```

## xgb.model.dt.tree()�ɂ��p�X�𒊏o

[`xgboost::xgb.model.dt.tree()`](https://www.imsbio.co.jp/RGM/R_rdfile?f=xgboost/man/xgb.model.dt.tree.Rd&d=R_CC)���g���B


```r
col_names = attr(trainingData, ".Dimnames")[[2]]
col_names %>% head()
#> [1] "cap-shape=bell"    "cap-shape=conical" "cap-shape=convex" 
#> [4] "cap-shape=flat"    "cap-shape=knobbed" "cap-shape=sunken"

cat("\nCreating the trees of the xgboost model...")
#> 
#> Creating the trees of the xgboost model...
trees = xgb.model.dt.tree(col_names, model = xgb.model, trees = trees_idx)

trees %>% 
  mutate(Feature = str_trunc(Feature, width=12, side="left")) %>% 
  select(-Missing)
#>    Tree Node   ID      Feature Split  Yes   No      Quality       Cover
#> 1     0    0  0-0    odor=foul   0.5  0-1  0-2 2711.3557100 1250.000000
#> 2     0    1  0-1 ...ize=broad   0.5  0-3  0-4 1263.3979500  901.000000
#> 3     0    2  0-2         Leaf    NA <NA> <NA>    0.5982857  349.000000
#> 4     0    3  0-3    odor=none   0.5  0-5  0-6  264.4693910  202.250000
#> 5     0    4  0-4 ...lor=green   0.5  0-7  0-8  203.9635310  698.750000
#> 6     0    5  0-5 ...or=yellow   0.5  0-9 0-10  121.6305770  157.500000
#> 7     0    6  0-6 ...ing=silky   0.5 0-11 0-12   74.7532349   44.750000
#> 8     0    7  0-7         Leaf    NA <NA> <NA>   -0.5991260  685.500000
#> 9     0    8  0-8         Leaf    NA <NA> <NA>    0.5578948   13.250000
#> 10    0    9  0-9  odor=almond   0.5 0-13 0-14   65.5069656  148.000000
#> 11    0   10 0-10         Leaf    NA <NA> <NA>   -0.5428572    9.500000
#> 12    0   11 0-11 ...?=bruises   0.5 0-15 0-16   26.8929482   38.250000
#> 13    0   12 0-12         Leaf    NA <NA> <NA>    0.5200000    6.500000
#> 14    0   13 0-13   odor=anise   0.5 0-17 0-18   62.2878838  143.250000
#> 15    0   14 0-14         Leaf    NA <NA> <NA>   -0.4956522    4.750000
#> 16    0   15 0-15         Leaf    NA <NA> <NA>   -0.5838926   36.250000
#> 17    0   16 0-16         Leaf    NA <NA> <NA>    0.4000000    2.000000
#> 18    0   17 0-17         Leaf    NA <NA> <NA>    0.5957143  139.000000
#> 19    0   18 0-18         Leaf    NA <NA> <NA>   -0.4857143    4.250000
#> 20    1    0  1-0    odor=foul   0.5  1-1  1-2 1489.0952100 1145.302000
#> 21    1    1  1-1 ...ize=broad   0.5  1-3  1-4  691.4842530  825.759888
#> 22    1    2  1-2         Leaf    NA <NA> <NA>    0.4634756  319.542114
#> 23    1    3  1-3    odor=none   0.5  1-5  1-6  142.5185550  186.003708
#> 24    1    4  1-4 ...lor=green   0.5  1-7  1-8  114.8126300  639.756165
#> 25    1    5  1-5  odor=almond   0.5  1-9 1-10   69.8917542  144.674194
#> 26    1    6  1-6 ...ing=silky   0.5 1-11 1-12   42.7613907   41.329517
#> 27    1    7  1-7         Leaf    NA <NA> <NA>   -0.4640480  627.485962
#> 28    1    8  1-8         Leaf    NA <NA> <NA>    0.4361763   12.270212
#> 29    1    9  1-9   odor=anise   0.5 1-13 1-14   75.4965286  135.787827
#> 30    1   10 1-10         Leaf    NA <NA> <NA>   -0.4301576    8.886355
#> 31    1   11 1-11 ...?=bruises   0.5 1-15 1-16   16.6019783   35.249847
#> 32    1   12 1-12         Leaf    NA <NA> <NA>    0.4107886    6.079669
#> 33    1   13 1-13         Leaf    NA <NA> <NA>    0.4617254  127.362411
#> 34    1   14 1-14         Leaf    NA <NA> <NA>   -0.4283619    8.425421
#> 35    1   15 1-15         Leaf    NA <NA> <NA>   -0.4537036   33.327763
#> 36    1   16 1-16         Leaf    NA <NA> <NA>    0.3296103    1.922086
#> 37    2    0  2-0    odor=foul   0.5  2-1  2-2  934.9316410  956.514771
#> 38    2    1  2-1 ...ize=broad   0.5  2-3  2-4  434.1681820  689.965515
#> 39    2    2  2-2         Leaf    NA <NA> <NA>    0.4022448  266.549286
#> 40    2    3  2-3    odor=none   0.5  2-5  2-6   87.7709885  156.324432
#> 41    2    4  2-4 ...lor=green   0.5  2-7  2-8   73.6840591  533.641052
#> 42    2    5  2-5  odor=almond   0.5  2-9 2-10   44.8767204  121.285667
#> 43    2    6  2-6 ...ing=silky   0.5 2-11 2-12   27.4686470   35.038773
#> 44    2    7  2-7         Leaf    NA <NA> <NA>   -0.4028375  523.192078
#> 45    2    8  2-8         Leaf    NA <NA> <NA>    0.3751199   10.448951
#> 46    2    9  2-9   odor=anise   0.5 2-13 2-14   48.7504120  113.642014
#> 47    2   10 2-10         Leaf    NA <NA> <NA>   -0.3680061    7.643652
#> 48    2   11 2-11 ...?=bruises   0.5 2-15 2-16   11.4850969   29.765743
#> 49    2   12 2-12         Leaf    NA <NA> <NA>    0.3515948    5.273030
#> 50    2   13 2-13         Leaf    NA <NA> <NA>    0.4004391  106.384308
#> 51    2   14 2-14         Leaf    NA <NA> <NA>   -0.3663105    7.257702
#> 52    2   15 2-15         Leaf    NA <NA> <NA>   -0.3922864   28.009958
#> 53    2   16 2-16         Leaf    NA <NA> <NA>    0.2832851    1.755784
# %>%
#   mutate_at(.vars = vars("Quality","Cover"), .funs = round)
```

�uLeaf�m�[�h��Quality��0����Ȃ��́H�v�Ǝv����������Ȃ����A�}�j���A����

> Quality: either the split gain (change in loss) **or the leaf value**

�Ə����Ă���ALeaf�m�[�h�ł�Quality�̃Z����**�\�����ʂ̊i�[�ꏊ�Ƃ��ė��p����Ă���**���Ƃɒ��ӁB�i���Ƃ��炱��𗘗p����j((�_����ǂ�ł���R�[�h��ǂݐi�߂Ă��āA�����ł͂܂����B))

[`predict(..., predleaf = TRUE)`](https://www.rdocumentation.org/packages/xgboost/versions/0.71.2/topics/predict.xgb.Booster)���w�肷�邱�ƂŁA�\���l�̑���ɌP���f�[�^�̃C���X�^���X���etree�ŏ�������Leaf�̃m�[�h�ԍ����擾�ł���B
�����`NROW(trainingData)=5000, nrounds = 3`�Ȃ̂ŁA5000�s3��̏���Leaf�̍s�񂪓�����B


```r
cat("\nGetting the leaf nodes for the training set observations...")
#> 
#> Getting the leaf nodes for the training set observations...
nodes.train = predict(xgb.model, trainingData, predleaf = TRUE)

nodes.train %>% dim()
#> [1] 5000    3
nodes.train <- NULL
```

�������A�擾���ꂽ�̂��A�w�K�Ɏg��ꂽ�C���X�^���X�̗\�����ʂ̏�񂪁A`buildExplainer()`�̂ǂ����Ŏg���Ă���`�Ղ͌�������Ȃ�����((���̎����A�S�ʓI�Ƀ��K�V�[���c���Ă���X���������āA�ǂ������Ă�������r���ōs���~�܂肾�����A�݂����Ȃ��Ƃ��A���������ł���))�B

## �\���l�̍ĕ��z

`xgboostExplainer`�́ALeaf�̗\���l��e�m�[�h�ɍĕ��z���邱�ƂŁA�\�����ʂ𕪉��E��������B

�ȉ��ł́A`buildExplainer()`�̃G���W�������ł���`xgboostExplainer:::getStatsForTrees()`�̈�A�̃X�e�b�v���g���[�X����B


```r

# tree_list = xgboostExplainer:::getStatsForTrees(trees, nodes.train, type = type, base_score = base_score)
# function (trees, nodes.train, type = "binary", base_score = 0.5) 
# {

```

�Ȃ��A�֐�������`data.table::copy()'�ɂ��A`- attr(*, ".internal.selfref")=<externalptr>`�ŁA**�f�[�^�̍X�V���Q�Ɠn���Ƃ��čs���Ă���**�̂ɒ��ӁB

## Cover (H)�̍Čv�Z

`xgb.model.dt.tree()`�̏�񂩂�A�e�m�[�h�܂ł̓񎟂̌��z�̘a�����o���B�����Cover��̏�񂻂̂��̂̂͂��Ȃ̂����A`xgb.model.dt.tree()`�̏o�͂���Cover�͐��x�ɖ�肪����((�r���̊ۂߕ��ɖ�肪����̂���))�Ƃ̂��ƂŁA������ƂƂ������x�Ōv�Z���Ȃ����B

Leaf�m�[�h��Cover����A�t�����ɂ��ǂ�Ȃ��瑫�����킹�Ă����΂悢


```r
type = "binary"
base_score = 0.5

tree_datatable = data.table::copy(trees)

tree_datatable[, leaf := (Feature == "Leaf")]
non.leaves = which(tree_datatable[, leaf] == F)

tree_datatable[, H := Cover]

cat("\n\nRecalculating the cover for each non-leaf... \n")
#> 
#> 
#> Recalculating the cover for each non-leaf...
print.counter = 1
for (i in rev(non.leaves)) {
  left = tree_datatable[i, Yes]
  right = tree_datatable[i, No]
  
  tree_datatable[i, H := (tree_datatable[ID == left, H] + 
                            tree_datatable[ID == right, H])]
  
  if(print.counter < 5){
    print(i - 1)
    
    bind_rows(tree_datatable[i, ],
              tree_datatable[ID == left, ], 
              tree_datatable[ID == right,]) %>% 
      select(-Tree,-Node,-Feature,-Split,-Missing) %>% 
      print()
    print.counter <- print.counter + 1
  }
}
#> [1] 47
#>      ID  Yes   No    Quality     Cover  leaf         H
#> 1: 2-11 2-15 2-16 11.4850969 29.765743 FALSE 29.765742
#> 2: 2-15 <NA> <NA> -0.3922864 28.009958  TRUE 28.009958
#> 3: 2-16 <NA> <NA>  0.2832851  1.755784  TRUE  1.755784
#> [1] 45
#>      ID  Yes   No    Quality      Cover  leaf          H
#> 1:  2-9 2-13 2-14 48.7504120 113.642014 FALSE 113.642010
#> 2: 2-13 <NA> <NA>  0.4004391 106.384308  TRUE 106.384308
#> 3: 2-14 <NA> <NA> -0.3663105   7.257702  TRUE   7.257702
#> [1] 42
#>      ID  Yes   No    Quality    Cover  leaf        H
#> 1:  2-6 2-11 2-12 27.4686470 35.03877 FALSE 35.03877
#> 2: 2-11 2-15 2-16 11.4850969 29.76574 FALSE 29.76574
#> 3: 2-12 <NA> <NA>  0.3515948  5.27303  TRUE  5.27303
#> [1] 41
#>      ID  Yes   No    Quality      Cover  leaf          H
#> 1:  2-5  2-9 2-10 44.8767204 121.285667 FALSE 121.285663
#> 2:  2-9 2-13 2-14 48.7504120 113.642014 FALSE 113.642010
#> 3: 2-10 <NA> <NA> -0.3680061   7.643652  TRUE   7.643652
```

Cover��H���ׂĂ݂�ƁA�������ɑ����Z�̌��ʂ�����Ă���..�B

## ���z(G)��weight�̍Čv�Z

`xgboost::xgb.model.dt.tree()`�̏o�͂̂����ALeaf�m�[�h�ł�Quality�̃Z����**�\�����ʂ̊i�[�ꏊ�Ƃ��ė��p����Ă���**�̂ŁA������N�_�ɂ��āA�e�e�m�[�h��weight���ĕ��z����B

��`��A$w^*_j=-\frac{G_j}{H_j+\lambda}$ �Ȃ̂����A����ȍ~$\lambda$�������Ă��Ȃ��̂ŁA$G_j = \sum_{i \in I_j}^{} g_i$ �Ƃ͌����ȈӖ��ł͈Ⴄ�̂�������Ȃ����A�����͓ǂݎ��Ȃ������B


```r
base_weight = log(base_score/(1 - base_score))

tree_datatable[, previous_weight := base_weight]
tree_datatable[1, previous_weight:=0]

tree_datatable[leaf==T, weight := base_weight + Quality]

tree_datatable[leaf==T, G := -weight * H]
```


�ȍ~�́Around(Tree)�P�ʂɕ����ď�����i�߂�B


```r
tree_list = split(tree_datatable,as.factor(tree_datatable$Tree))
tree_list %>% str(1)
#> List of 3
#>  $ 0:Classes 'data.table' and 'data.frame':	19 obs. of  15 variables:
#>   ..- attr(*, ".internal.selfref")=<externalptr> 
#>  $ 1:Classes 'data.table' and 'data.frame':	17 obs. of  15 variables:
#>   ..- attr(*, ".internal.selfref")=<externalptr> 
#>  $ 2:Classes 'data.table' and 'data.frame':	17 obs. of  15 variables:
#>   ..- attr(*, ".internal.selfref")=<externalptr>
```

�eround��Leaf�ȊO�̐e�m�[�h��$G$���v�Z����B��`����A$I_j = I_{left, j}+I_{right, j}$ �Ȃ̂ŁA$G_j = \sum_{i \in I_{left,j}}^{} g_i + \sum_{i \in I_{right,j}}^{} g_i = G_{L,j} + G_{R,j}$ �̑����Z���v�Z����΂悢((�_�����ɂ́AGR �� G �| GL$�Ə����Ă���iAlgrithm.1�j))�B


```r
num_tree_list = length(tree_list)
treenums =  as.character(0:(num_tree_list-1))
t = 0
cat('\n\nFinding the stats for the xgboost trees...\n')
#> 
#> 
#> Finding the stats for the xgboost trees...
# pb <- txtProgressBar(style=3)
for (tree in tree_list){
  t=t+1
  num_nodes = nrow(tree)
  non_leaf_rows = rev(which(tree[,leaf]==F))
  for (r in non_leaf_rows){
    left = tree[r,Yes]
    right = tree[r,No]
    leftG = tree[ID==left,G]
    rightG = tree[ID==right,G]
    
    tree[r,G:=leftG+rightG]
    w=tree[r,-G/H]
    
    tree[r,weight:=w]
    tree[ID==left,previous_weight:=w]
    tree[ID==right,previous_weight:=w]
    
    # bind_rows(tree[r,], tree[ID==left,], tree[ID==right,]) %>% print()
  }
  
  tree[,uplift_weight:=weight-previous_weight]
  # setTxtProgressBar(pb, t / num_tree_list)
}

tree_list %>% str(1)
#> List of 3
#>  $ 0:Classes 'data.table' and 'data.frame':	19 obs. of  16 variables:
#>   ..- attr(*, ".internal.selfref")=<externalptr> 
#>   ..- attr(*, "index")= int(0) 
#>   .. ..- attr(*, "__ID")= int [1:19] 1 2 11 12 13 14 15 16 17 18 ...
#>  $ 1:Classes 'data.table' and 'data.frame':	17 obs. of  16 variables:
#>   ..- attr(*, ".internal.selfref")=<externalptr> 
#>   ..- attr(*, "index")= int(0) 
#>   .. ..- attr(*, "__ID")= int [1:17] 1 2 11 12 13 14 15 16 17 3 ...
#>  $ 2:Classes 'data.table' and 'data.frame':	17 obs. of  16 variables:
#>   ..- attr(*, ".internal.selfref")=<externalptr> 
#>   ..- attr(*, "index")= int(0) 
#>   .. ..- attr(*, "__ID")= int [1:17] 1 2 11 12 13 14 15 16 17 3 ...
```

## Tree Breakdown�̍\�z

`buildExplainerFromTreeList()`�����s����ƁA

* �etree��`getTreeBreakdown()`���Ă΂�A
    * �eLeaf��` getLeafBreakdown()`���Ă΂�A
        * `findPath()`��Leaf �� root node �܂ł̃p�X���擾
        * Leaf �� root node�܂ł�uplift�Ɋ�Â��āA�e�p�X��Impact���W�v
        * root node��uplift��Intercept�Ƃ���


```r
cat("\n\nSTEP 2 of 2")
#> 
#> 
#> STEP 2 of 2
explainer = xgboostExplainer:::buildExplainerFromTreeList(tree_list, col_names)
#> 
#> 
#> Getting breakdown for each leaf of each tree...
#>   |                                                                         |                                                                 |   0%  |                                                                         |======================                                           |  33%  |                                                                         |===========================================                      |  67%  |                                                                         |=================================================================| 100%

explainer %>% str(0)
#> Classes 'data.table' and 'data.frame':	28 obs. of  129 variables:
#>  - attr(*, ".internal.selfref")=<externalptr>
```

�{���I�ȕ����ł͂Ȃ��̂ŁAstep-by-step�͏ȗ��B`explainer`�̒��g�͈ȑO�����ʂ�B


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

![](4_build_explainer_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

