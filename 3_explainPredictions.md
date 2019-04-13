---
title: "XGBoostExplainerが何をやっているか調べる"
author: Satoshi Kato (@katokohaku)
output: 
  html_document:
    keep_md: yes
    toc: yes
  md_document:
    variant: markdown_github
---



# 目的

今回は、インスタンスの予測結果が再構成されるプロセスをstep-by-stepで眺める。
`buildExplainer()`の中身を抜き書きしながら、都度、**何から何が取り出されているか**見ていく。


# 関連シリーズ

1. [とりあえず使ってみる](http://kato-kohaku-0.hatenablog.com/entry/2018/12/14/002253)
2. [予測結果の可視化プロセスをstep-by-stepで実行する](http://kato-kohaku-0.hatenablog.com/entry/2018/12/14/231803)
3. 予測結果を分解再構成するプロセスをstep-by-stepで実行する（この記事）
4. 学習したxgboostのルール抽出をstep-by-stepで実行する

# 準備

## XGBモデルの学習と予測

`xgboostExplainer`のマニュアルにあるexampleからコピペ。


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
## 予測ルールを抽出する

`buildExplainer()`で、学習したxgboostのモデルから予測ルールを抽出する。


```r
library(xgboostExplainer)

explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.5, trees = NULL)
```

# 結果を分解再構成する

## overview

指定したインスタンスの予測結果を、`buildExplainer()`が学習したモデルから再構成した結果(breakdown)を眺める。

再構成の大まかな手順は以下の通り：

1. 指定したインスタンスが各roundでどのleafに落ちるか予測結果を得る
2. 各roundで該当するleafの各特徴量の予測値への貢献量(予測ルール)を得る
3. 2を集計し、round全体での予測ルールを得る


## 対象の予測

今回は省略のため、1インスタンスだけを対象として指定するが、下記idxは複数のインスタンスを指定するベクトルでもよい。


```r
require(data.table)
# breakdown = explainPredictions(xgb.model, explainer, slice(DMatrix, as.integer(idx)))
# function (xgb.model, explainer, data) 

DMatrix = xgb.test.data
idx = 2

data = slice(DMatrix, as.integer(idx))

nodes = predict(xgb.model, data, predleaf = TRUE)
print(nodes)
#>      [,1] [,2] [,3]
#> [1,]   17   13   13
```

`xgboost:::predict.xgb.Booster()`は、`predleaf = TRUE`オプションを指定することで、あるインスタンスの予測時に、それぞれのroundのtreeでどのleafに落ちたか？を返してくれる

## 初期化

インスタンスの数（行）× Interceptを含むすべてのルール（列）からなるゼロ行列を準備する。


```r
colnames = names(explainer)[1:(ncol(explainer) - 2)]
  
preds_breakdown = data.table(matrix(0, nrow = nrow(nodes), ncol = length(colnames)))
setnames(preds_breakdown, colnames)
  
num_trees = ncol(nodes)

cat("\n\nExtracting the breakdown of each prediction...\n")
#> 
#> 
#> Extracting the breakdown of each prediction...

preds_breakdown.init <- preds_breakdown
```

## 対象のtreeの取り出し

今後のトレースを楽にするために「すべての行が0の列を削除し、残った列の列名を短縮する」関数を自作した。


```r
selectNonzeroToShort <- function(data, w=12){
  select_if(data,
            .predicate = function(x){ sum(abs(x)) > 0} , 
            .funs = function(x){ str_trunc(x, width=w, side="left") }) 
}
```

最初のtreeだけ少し丁寧にトレースする（残りは繰り返しなので省略）

```r
x=1

nodes_for_tree = nodes[, x]
nodes_for_tree
#> [1] 17
```

`buildExplainer()`でとりだしたLeafのうち、このroundで対象となる木のLeafを列挙。


```r
tree_breakdown = explainer[tree == x - 1]
tree_breakdown %>% selectNonzeroToShort()
#>     ...or=yellow ...?=bruises odor=almond  odor=anise  odor=foul
#>  1:   0.00000000   0.00000000  0.00000000  0.00000000  0.7088975
#>  2:   0.00000000   0.00000000  0.00000000  0.00000000 -0.2745896
#>  3:   0.00000000   0.00000000  0.00000000  0.00000000 -0.2745896
#>  4:  -1.00780012   0.00000000  0.00000000  0.00000000 -0.2745896
#>  5:   0.00000000   0.00000000  0.00000000  0.00000000 -0.2745896
#>  6:   0.06468987   0.00000000 -1.02528500  0.00000000 -0.2745896
#>  7:   0.00000000  -0.05144537  0.00000000  0.00000000 -0.2745896
#>  8:   0.00000000   0.93244731  0.00000000  0.00000000 -0.2745896
#>  9:   0.06468987   0.00000000  0.03399723  0.03208427 -0.2745896
#> 10:   0.06468987   0.00000000  0.03399723 -1.04934438 -0.2745896
#>      odor=none ...ize=broad ...ing=silky ...lor=green  intercept leaf
#>  1:  0.0000000    0.0000000    0.0000000   0.00000000 -0.1106117    2
#>  2:  0.0000000   -0.1919848    0.0000000  -0.02193993 -0.1106117    7
#>  3:  0.0000000   -0.1919848    0.0000000   1.13508088 -0.1106117    8
#>  4:  0.1868594    0.6632849    0.0000000   0.00000000 -0.1106117   10
#>  5: -0.6576614    0.6632849    0.8995779   0.00000000 -0.1106117   12
#>  6:  0.1868594    0.6632849    0.0000000   0.00000000 -0.1106117   14
#>  7: -0.6576614    0.6632849   -0.1528694   0.00000000 -0.1106117   15
#>  8: -0.6576614    0.6632849   -0.1528694   0.00000000 -0.1106117   16
#>  9:  0.1868594    0.6632849    0.0000000   0.00000000 -0.1106117   17
#> 10:  0.1868594    0.6632849    0.0000000   0.00000000 -0.1106117   18
```

## 対象のleafのとりだし

round=1のtreeのなかのLeaf ==17が、今回の予測ルール（予測値に各特徴量が寄与する変化量）。これを取り出して、`preds_breakdown`に足し合わせていく。


```r
preds_breakdown_for_tree = 
  tree_breakdown[match(nodes_for_tree, tree_breakdown$leaf), ]

print("preds_breakdown_for_tree")
#> [1] "preds_breakdown_for_tree"
preds_breakdown_for_tree %>%  
  selectNonzeroToShort(w=20) %>% 
  glimpse()
#> Observations: 1
#> Variables: 8
#> $ `cap-color=yellow` <dbl> 0.06468987
#> $ `odor=almond`      <dbl> 0.03399723
#> $ `odor=anise`       <dbl> 0.03208427
#> $ `odor=foul`        <dbl> -0.2745896
#> $ `odor=none`        <dbl> 0.1868594
#> $ `gill-size=broad`  <dbl> 0.6632849
#> $ intercept          <dbl> -0.1106117
#> $ leaf               <int> 17

preds_breakdown = 
  preds_breakdown + 
  preds_breakdown_for_tree[,colnames, with = FALSE]

print("preds_breakdown")
#> [1] "preds_breakdown"
preds_breakdown %>%  
  selectNonzeroToShort(w=20) %>% 
  glimpse()
#> Observations: 1
#> Variables: 7
#> $ `cap-color=yellow` <dbl> 0.06468987
#> $ `odor=almond`      <dbl> 0.03399723
#> $ `odor=anise`       <dbl> 0.03208427
#> $ `odor=foul`        <dbl> -0.2745896
#> $ `odor=none`        <dbl> 0.1868594
#> $ `gill-size=broad`  <dbl> 0.6632849
#> $ intercept          <dbl> -0.1106117
```

すべてのroudsをまとめて実行すると、以下の通り。


```r
preds_breakdown <- preds_breakdown.init
for (x in 1:num_trees) {
  print(x)
  
  nodes_for_tree = nodes[, x]
  tree_breakdown = explainer[tree == x - 1]

  preds_breakdown_for_tree = 
    tree_breakdown[match(nodes_for_tree, tree_breakdown$leaf), ]
  
  print("preds_breakdown_for_tree")
  preds_breakdown_for_tree %>%  
    selectNonzeroToShort(w=20) %>% 
    glimpse()
  
  preds_breakdown = 
    preds_breakdown + 
    preds_breakdown_for_tree[,colnames, with = FALSE]
  
  print("preds_breakdown")
  preds_breakdown %>%  
    selectNonzeroToShort(w=20) %>% 
    glimpse()
}
#> [1] 1
#> [1] "preds_breakdown_for_tree"
#> Observations: 1
#> Variables: 8
#> $ `cap-color=yellow` <dbl> 0.06468987
#> $ `odor=almond`      <dbl> 0.03399723
#> $ `odor=anise`       <dbl> 0.03208427
#> $ `odor=foul`        <dbl> -0.2745896
#> $ `odor=none`        <dbl> 0.1868594
#> $ `gill-size=broad`  <dbl> 0.6632849
#> $ intercept          <dbl> -0.1106117
#> $ leaf               <int> 17
#> [1] "preds_breakdown"
#> Observations: 1
#> Variables: 7
#> $ `cap-color=yellow` <dbl> 0.06468987
#> $ `odor=almond`      <dbl> 0.03399723
#> $ `odor=anise`       <dbl> 0.03208427
#> $ `odor=foul`        <dbl> -0.2745896
#> $ `odor=none`        <dbl> 0.1868594
#> $ `gill-size=broad`  <dbl> 0.6632849
#> $ intercept          <dbl> -0.1106117
#> [1] 2
#> [1] "preds_breakdown_for_tree"
#> Observations: 1
#> Variables: 8
#> $ `odor=almond`     <dbl> 0.05139002
#> $ `odor=anise`      <dbl> 0.05522851
#> $ `odor=foul`       <dbl> -0.2125787
#> $ `odor=none`       <dbl> 0.1433645
#> $ `gill-size=broad` <dbl> 0.5101908
#> $ intercept         <dbl> -0.08586974
#> $ leaf              <int> 13
#> $ tree              <dbl> 1
#> [1] "preds_breakdown"
#> Observations: 1
#> Variables: 7
#> $ `cap-color=yellow` <dbl> 0.06468987
#> $ `odor=almond`      <dbl> 0.08538725
#> $ `odor=anise`       <dbl> 0.08731278
#> $ `odor=foul`        <dbl> -0.4871683
#> $ `odor=none`        <dbl> 0.3302238
#> $ `gill-size=broad`  <dbl> 1.173476
#> $ intercept          <dbl> -0.1964815
#> [1] 3
#> [1] "preds_breakdown_for_tree"
#> Observations: 1
#> Variables: 8
#> $ `odor=almond`     <dbl> 0.04534281
#> $ `odor=anise`      <dbl> 0.04896816
#> $ `odor=foul`       <dbl> -0.1841252
#> $ `odor=none`       <dbl> 0.1238637
#> $ `gill-size=broad` <dbl> 0.4407547
#> $ intercept         <dbl> -0.0743651
#> $ leaf              <int> 13
#> $ tree              <dbl> 2
#> [1] "preds_breakdown"
#> Observations: 1
#> Variables: 7
#> $ `cap-color=yellow` <dbl> 0.06468987
#> $ `odor=almond`      <dbl> 0.1307301
#> $ `odor=anise`       <dbl> 0.1362809
#> $ `odor=foul`        <dbl> -0.6712935
#> $ `odor=none`        <dbl> 0.4540875
#> $ `gill-size=broad`  <dbl> 1.61423
#> $ intercept          <dbl> -0.2708466

# return(preds_breakdown)

```


## 検算

元実装の結果と合致してるか検算。分解再構成した結果の可視化。


```r
preds_breakdown %>% selectNonzeroToShort(w=20)
#>    cap-color=yellow odor=almond odor=anise  odor=foul odor=none
#> 1:       0.06468987   0.1307301  0.1362809 -0.6712935 0.4540875
#>    gill-size=broad  intercept
#> 1:         1.61423 -0.2708466

weight <- sum(preds_breakdown)
weight
#> [1] 1.457879

prediction <- exp(weight)/(1+exp(weight))
prediction
#> [1] 0.811208

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

![](3_explainPredictions_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

次回は、`buildExplainer()`が、どのような手続きで学習したxgboostのモデルから予測ルールを抽出しているのか詳細に見ていく。
