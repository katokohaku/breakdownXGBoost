---
title: "XGBoostExplainer‚ª‰½‚ð‚â‚Á‚Ä‚¢‚é‚©’²‚×‚é"
author: Satoshi Kato (@katokohaku)
output: 
  html_document:
    keep_md: yes
    toc: yes
  md_document:
    variant: markdown_github
---




```r
require(tidyverse)
library(xgboost)
library(xgboostExplainer)

set.seed(123)

data(agaricus.train, package='xgboost')

train <- agaricus.train

X = train$data
y = train$label

bst <- xgboost(data = train$data, label = train$label, max.depth = 2,
               eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#> [1]	train-error:0.046522 
#> [2]	train-error:0.022263

trees <- xgb.model.dt.tree(agaricus.train$data@Dimnames[[2]], model = bst)
trees %>% 
  mutate(Feature = str_trunc(Feature, width=12, side="left")) %>% 
  select(-Split, -Missing) 
#>    Tree Node  ID      Feature  Yes   No      Quality      Cover
#> 1     0    0 0-0    odor=none  0-1  0-2 4000.5310100 1628.25000
#> 2     0    1 0-1 ...root=club  0-3  0-4 1158.2119100  924.50000
#> 3     0    2 0-2 ...lor=green  0-5  0-6  198.1738130  703.75000
#> 4     0    3 0-3         Leaf <NA> <NA>    1.7121772  812.00000
#> 5     0    4 0-4         Leaf <NA> <NA>   -1.7004405  112.50000
#> 6     0    5 0-5         Leaf <NA> <NA>   -1.9407086  690.50000
#> 7     0    6 0-6         Leaf <NA> <NA>    1.8596492   13.25000
#> 8     1    0 1-0 ...ot=rooted  1-1  1-2  832.5450440  788.85205
#> 9     1    1 1-1    odor=none  1-3  1-4  569.7251590  768.38971
#> 10    1    2 1-2         Leaf <NA> <NA>   -6.2362447   20.46239
#> 11    1    3 1-3         Leaf <NA> <NA>    0.7847176  458.93686
#> 12    1    4 1-4         Leaf <NA> <NA>   -0.9685304  309.45282

p = rep(0.5,nrow(X))

L = which(X[,'odor=none']==0)
R = which(X[,'odor=none']==1)

pL = p[L]
pR = p[R]


yL = y[L]
yR = y[R]

GL = sum(pL-yL)
GL
#> [1] -1199
GR = sum(pR-yR)
GR
#> [1] 1315.5
G = sum(p-y)
G
#> [1] 116.5

HL = sum(pL*(1-pL))
HL
#> [1] 924.5
HR = sum(pR*(1-pR))
HR
#> [1] 703.75
H = sum(p*(1-p))
H
#> [1] 1628.25

gain = 0.5 * (GL^2/HL+GR^2/HR-G^2/H)
gain
#> [1] 2002.848
gain *2
#> [1] 4005.695


gain2 = 0.5 * (GL^2 / (HL+1) + GR^2 / (HR+1) - G^2/ (H+1))
gain2
#> [1] 2000.266
gain2*2
#> [1] 4000.531
```


