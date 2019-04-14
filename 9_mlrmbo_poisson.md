---
title: Bayesian optimization of xgboost hyperparameters for a Poisson regression in
  R
author: "Simon Coulombe"
date: '2019-01-09'
output: 
  html_document:
    keep_md: yes
    toc: yes
  md_document:
    variant: markdown_github
---



# Introduction  

Ratemaking models in insurance routinely use Poisson regression to model the frequency of auto insurance claims.  They usually are GLMs but some insurers are moving towards GBMs, such as `xgboost`.

`xgboost`has multiple hyperparameters that can be tuned to obtain a better predictive power.  There are multiple ways to tune these hyperparameters.  In order of efficiency are the grid search, the random search and the bayesian optimization search.  

In this post, we will  compare the results of  xgboost hyperparameters for a Poisson regression in R using a random search versus a bayesian search.  Two packages wills be compared for the bayesian approach:  the `mlrMBO` package and the `rBayesianOptimization` package.  

We will model the number of auto insurance claims based on characteristics of the car and driver, while offsetting for exposure.   The data comes from the `insuranceData` package.  

For most model types, `mlrMBO` can be used in combination with the `mlr` package to find the best hyperparameters directly.  As far as I know the `mlr` package doesnt [handle poisson regression](https://github.com/mlr-org/mlr/issues/515), so we will have to create our own function to maximise.


I tried the `rBayesianOptimization` package after being inspired by this post from  [Max Kuhn](http://blog.revolutionanalytics.com/2016/06/bayesian-optimization-of-machine-learning-models.html) from 2016.  I do not recommend using this package because it  [sometimes recycles hyperparameters](https://github.com/yanyachen/rBayesianOptimization/issues/4) and hasnt been updated on github since 2016.

**Keep your eyes peeled**: Max Kuhn (@topepos) [said that tidymodels might do this in the first half of 2019](https://twitter.com/topepos/status/1075151561863692290).



```r
library(xgboost)
#> Warning: package 'xgboost' was built under R version 3.5.3
library(insuranceData) # example dataset
# https://cran.r-project.org/web/packages/insuranceData/insuranceData.pdf
library(tidyverse) # for data wrangling
library(rBayesianOptimization) # to create cv folds and for bayesian optimisation
#> Warning: package 'rBayesianOptimization' was built under R version 3.5.3
library(mlrMBO)  # for bayesian optimisation
#> Warning: package 'mlr' was built under R version 3.5.3
#> Warning: package 'ParamHelpers' was built under R version 3.5.3
library(skimr) # for summarising databases
#> Warning: package 'skimr' was built under R version 3.5.3
library(purrr) # to evaluate the loglikelihood of each parameter set in the random grid search
require("DiceKriging") # mlrmbo requires this
require("rgenoud") # mlrmbo requires this
```


# Preparing the data

First, we load the dataCar data from the `insuranceData` package.  It contains 67 856 one-year vehicle insurance policies taken out in 2004 or 2005.   

The dependent variable is  `numclaims`, which represents the number of claims.    

The `exposure` variable  represents the "number of year of exposure" and is used as the offset variable.  It is bounded between 0 and 1.   

Finally, the independent variables are as follow:  

* `veh_value`, the vehicle value in tens of thousand of dollars,  
* `veh_body`, y vehicle body, coded as BUS CONVT COUPE HBACK HDTOP MCARA MIBUS PANVN RDSTR SEDAN STNWG TRUCK UTE,  
* `veh_age`, 1 (youngest), 2, 3, 4,   
* `gender`, a factor with levels F M,   
* `area` a factor with levels A B C D E F,   
* `agecat` 1 (youngest), 2, 3, 4, 5, 6  




```r
# load insurance data
data(dataCar)
mydb <- dataCar %>% select(numclaims, exposure, veh_value, veh_body,
                           veh_age, gender, area, agecat)
label_var <- "numclaims"  
offset_var <- "exposure"
feature_vars <- mydb %>% 
  select(-one_of(c(label_var, offset_var))) %>% 
  colnames()

skimr::skim(mydb ) %>% 
  skimr::kable()
#> Skim summary statistics  
#>  n obs: 67856    
#>  n variables: 8    
#> 
#> Variable type: factor
#> 
#>  variable    missing    complete      n      n_unique                     top_counts                      ordered 
#> ----------  ---------  ----------  -------  ----------  -----------------------------------------------  ---------
#>    area         0        67856      67856       6            C: 20540, A: 16312, B: 13341, D: 8173         FALSE  
#>   gender        0        67856      67856       2                  F: 38603, M: 29253, NA: 0               FALSE  
#>  veh_body       0        67856      67856       13       SED: 22233, HBA: 18915, STN: 16261, UTE: 4586     FALSE  
#> 
#> Variable type: integer
#> 
#>  variable     missing    complete      n      mean      sd     p0    p25    p50    p75    p100      hist   
#> -----------  ---------  ----------  -------  -------  ------  ----  -----  -----  -----  ------  ----------
#>   agecat         0        67856      67856    3.49     1.43    1      2      3      5      6      <U+2583><U+2586><U+2581><U+2587><U+2587><U+2581><U+2585><U+2583> 
#>  numclaims       0        67856      67856    0.073    0.28    0      0      0      0      4      <U+2587><U+2581><U+2581><U+2581><U+2581><U+2581><U+2581><U+2581> 
#>   veh_age        0        67856      67856    2.67     1.07    1      2      3      4      4      <U+2585><U+2581><U+2586><U+2581><U+2581><U+2587><U+2581><U+2587> 
#> 
#> Variable type: numeric
#> 
#>  variable     missing    complete      n      mean     sd       p0      p25     p50     p75     p100       hist   
#> -----------  ---------  ----------  -------  ------  ------  --------  ------  ------  ------  -------  ----------
#>  exposure        0        67856      67856    0.47    0.29    0.0027    0.22    0.45    0.71      1      <U+2587><U+2587><U+2587><U+2587><U+2586><U+2586><U+2586><U+2586> 
#>  veh_value       0        67856      67856    1.78    1.21      0       1.01    1.5     2.15    34.56    <U+2587><U+2581><U+2581><U+2581><U+2581><U+2581><U+2581><U+2581>
```

Insurance ratemaking often requires monotonous relationships.  In our case, we will arbitrarily force the number of claims to be non-increasing with the age of the vehicle.

The code below imports the data, one-hot encodes dummy variables,  converts the data frame to a xgb.DMatrix for xgboost consumption, sets the offset for exposure, sets the constraints and defines the 3 folds we will use for cross-validation.





```r
# one hot encoding of categorical (factor) data
myformula <- paste0( "~", paste0( feature_vars, collapse = " + ") ) %>% 
  as.formula()

dummyFier <- caret::dummyVars(myformula, data=mydb, fullRank = TRUE)
dummyVars.df <- predict(dummyFier,newdata = mydb)
mydb_dummy <- cbind(mydb %>% select(one_of(c(label_var, offset_var))), 
                    dummyVars.df)
rm(myformula, dummyFier, dummyVars.df)

# get  list the column names of the db with the dummy variables
feature_vars_dummy <-  mydb_dummy  %>% 
  select(-one_of(c(label_var, offset_var))) %>% 
  colnames()

# create xgb.matrix for xgboost consumption
mydb_xgbmatrix <- xgb.DMatrix(
  data = mydb_dummy %>% select(feature_vars_dummy) %>% as.matrix, 
  label = mydb_dummy %>% pull(label_var),
  missing = "NAN")

#base_margin: apply exposure offset 
setinfo(mydb_xgbmatrix,"base_margin", 
        mydb %>% pull(offset_var) %>% log() )
#> [1] TRUE

# a fake constraint, just to show how it is done.  
#Here we force "the older the car, the less likely are claims"
myConstraint   <- data_frame(Variable = feature_vars_dummy) %>%
  mutate(sens = ifelse(Variable == "veh_age", -1, 0))
#> Warning: `data_frame()` is deprecated, use `tibble()`.
#> This warning is displayed once per session.

# random folds for xgb.cv
cv_folds = rBayesianOptimization::KFold(mydb_dummy$numclaims, 
                                        nfolds= 3,
                                        stratified = TRUE,
                                        seed= 0)
```


# Example 1: Optimize hyperparameters using a random search   (non bayesian)

We will start with a quick example of random search.  

I don't use caret for the random search  [because it has a hard time with poisson regression](https://github.com/topepo/caret/issues/861).

First, we generate 20 random sets of hyperparameters.  I will force `gamma = 0` for half the sets.I will also hardcode an extra set of parameters named `simon_params` because I find this combination is often a good starting point.


```r
# generate hard coded parameters
simon_params <- data.frame(max_depth = 6,
                           colsample_bytree= 0.8,
                           subsample = 0.8,
                           min_child_weight = 3,
                           eta  = 0.01,
                           gamma = 0,
                           nrounds = 200) %>% 
  as_tibble()
# generate 20 random models
how_many_models <- 20
max_depth <-        data.frame(max_depth = floor(runif(how_many_models)*5 ) + 3)  # 1 ﾃ? 4
colsample_bytree <- data.frame(colsample_bytree =runif(how_many_models) * 0.8 + 0.2)  # 0.2 ﾃ? 1
subsample <-        data.frame(subsample =runif(how_many_models) * 0.8 + 0.2) # 0.2 ﾃ? 1
min_child_weight <- data.frame(min_child_weight = floor(runif(how_many_models) * 10) + 1) # 1 ﾃ? 10
eta <-              data.frame(eta = runif(how_many_models) * 0.06 + 0.002) # 0.002 ﾃ? 0.062
gamma <-            data.frame(gamma =c(rep(0,how_many_models/2), runif(how_many_models/2)*10)) # 0 ﾃ? 10
nrounds <-          data.frame(nrounds = rep(2e2,how_many_models)) # max 200

random_grid <-max_depth %>%
  bind_cols(colsample_bytree ) %>%
  bind_cols(subsample) %>%
  bind_cols(min_child_weight) %>%
  bind_cols(eta) %>%
  bind_cols(gamma) %>%
  bind_cols(nrounds)  %>% as_tibble()
# combine random and hardcoded parameters
df.params <- simon_params %>%  bind_rows(random_grid) %>%
  mutate(rownum = row_number(),
         rownumber = row_number())
list_of_param_sets <- df.params %>% nest(-rownum)
```

Here are the hyperparameters that will be tested:


```r
kable(df.params)
```



 max_depth    colsample_bytree    subsample    min_child_weight       eta        gamma      nrounds    rownum    rownumber 
-----------  ------------------  -----------  ------------------  -----------  ----------  ---------  --------  -----------
     6           0.8000000        0.8000000           3            0.0100000    0.000000      200        1           1     
     4           0.3697140        0.7176482           3            0.0447509    0.000000      200        2           2     
     5           0.7213390        0.8263462           5            0.0259997    0.000000      200        3           3     
     7           0.3004441        0.6424290           4            0.0215211    0.000000      200        4           4     
     4           0.4137765        0.6237757           7            0.0474252    0.000000      200        5           5     
     7           0.5088913        0.8314850           3            0.0141615    0.000000      200        6           6     
     7           0.2107123        0.2186650           5            0.0446673    0.000000      200        7           7     
     6           0.5059104        0.5817841           8            0.0093015    0.000000      200        8           8     
     6           0.8957527        0.7858510           1            0.0167293    0.000000      200        9           9     
     3           0.4722792        0.7541852           9            0.0105983    0.000000      200        10         10     
     4           0.5856641        0.5820957           4            0.0163778    0.000000      200        11         11     
     3           0.6796527        0.8889676           9            0.0055361    3.531973      200        12         12     
     6           0.5948330        0.5504777           4            0.0405373    2.702602      200        13         13     
     4           0.3489741        0.3958378           4            0.0545762    9.926841      200        14         14     
     6           0.8618987        0.2565432           5            0.0487349    6.334933      200        15         15     
     5           0.7347734        0.2795729           9            0.0498385    2.132081      200        16         16     
     6           0.8353919        0.4530174           9            0.0293165    1.293724      200        17         17     
     7           0.2863549        0.6149074           4            0.0266050    4.781180      200        18         18     
     4           0.7789688        0.7296041           8            0.0506522    9.240745      200        19         19     
     6           0.5290195        0.5254641           10           0.0382960    5.987610      200        20         20     
     7           0.8567570        0.9303007           5            0.0412834    9.761707      200        21         21     

Evaluate these 21 models with 3 folds and retourn the loglikelihood of each models:  




```r
start <- Sys.time()
random_grid_results <- list_of_param_sets %>% 
  mutate(booster = map(data, function(X){
    message(paste0("model #",       X$rownumber,
                   " eta = ",              X$eta,
                   " max.depth = ",        X$max_depth,
                   " min_child_weigth = ", X$min_child_weight,
                   " subsample = ",        X$subsample,
                   " colsample_bytree = ", X$colsample_bytree,
                   " gamma = ",            X$gamma, 
                   " nrounds = ",          X$nrounds))
    set.seed(1234)
    
    cv <- xgb.cv(
      params = list(
        booster = "gbtree",
        eta = X$eta,
        max_depth = X$max_depth,
        min_child_weight = X$min_child_weight,
        gamma = X$gamma,
        subsample = X$subsample,
        colsample_bytree = X$colsample_bytree,
        objective = 'count:poisson', 
        eval_metric = "poisson-nloglik"),
      data = mydb_xgbmatrix,
      nround = X$nrounds,
      folds=  cv_folds,
      monotone_constraints = myConstraint$sens,
      prediction = FALSE,
      showsd = TRUE,
      early_stopping_rounds = 50,
      verbose = 0)
    
    function_return <- list(Score = cv$evaluation_log[, max(test_poisson_nloglik_mean)], 
                            Pred = 0)
    
    message(paste0("Score :", function_return$Score))
    return(function_return)})) %>%
  mutate(Score =  pmap(list(booster), function(X){X$Score })%>% unlist())

getwd()
write_rds(random_grid_results, "./temp_files/random_grid_results.rds")
stop <- Sys.time()
stop- start
```


```r
random_grid_results <- read_rds( "temp_files/random_grid_results.rds")

random_grid_results %>%
  mutate( hardcoded = ifelse(rownum ==1,TRUE,FALSE)) %>%
  ggplot(aes( x = rownum, y = Score, color = hardcoded)) + 
  geom_point() +
  labs(title = "random grid search")+
  ylab("loglikelihood")
```

![](9_mlrmbo_poisson_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

# Example #2 Bayesian optimization using `mlrMBO`

This tutorial builds on the [mlrMBO vignette](https://mlrmbo.mlr-org.com/articles/mlrMBO.html)

First, we need to define the objective function that the bayesian search will try to maximise.  In this case we want to maximise the log likelihood of the out of fold predictions.


```r
# objective function: we want to maximise the log likelihood by tuning most parameters
obj.fun  <- smoof::makeSingleObjectiveFunction(
  name = "xgb_cv_bayes",
  fn =   function(x){
    set.seed(12345)
    cv <- xgb.cv(params = list(
      booster          = "gbtree",
      eta              = x["eta"],
      max_depth        = x["max_depth"],
      min_child_weight = x["min_child_weight"],
      gamma            = x["gamma"],
      subsample        = x["subsample"],
      colsample_bytree = x["colsample_bytree"],
      objective        = 'count:poisson', 
      eval_metric     = "poisson-nloglik"),
      data = mydb_xgbmatrix,
      nround = 30,
      folds=  cv_folds,
      monotone_constraints = myConstraint$sens,
      prediction = FALSE,
      showsd = TRUE,
      early_stopping_rounds = 10,
      verbose = 0)
    
    cv$evaluation_log[, max(test_poisson_nloglik_mean)]
  },
  par.set = makeParamSet(
    makeNumericParam("eta",              lower = 0.001, upper = 0.05),
    makeNumericParam("gamma",            lower = 0,     upper = 5),
    makeIntegerParam("max_depth",        lower= 1,      upper = 10),
    makeIntegerParam("min_child_weight", lower= 1,      upper = 10),
    makeNumericParam("subsample",        lower = 0.2,   upper = 1),
    makeNumericParam("colsample_bytree", lower = 0.2,   upper = 1)
  ),
  minimize = FALSE
)
```

After this, we generate the design, which are a set of hyperparameters that will be tested before starting the bayesian optimization.  Here we generate only 10 sets, but `mlrMBO`would normally generate 4 times the number of parameters.  I also force my  `simon_params` to be part of the design because I want to make sure at least one of of sets generated is good.  


```r
# generate an optimal design with only 10  points
des = generateDesign(n=10,
                     par.set = getParamSet(obj.fun), 
                     fun = lhs::randomLHS)  ## . If no design is given by the user, mlrMBO will generate a maximin Latin Hypercube Design of size 4 times the number of the black-box function窶冱 parameters.
# i still want my favorite hyperparameters to be tested
simon_params <- data.frame(max_depth = 6,
                           colsample_bytree= 0.8,
                           subsample = 0.8,
                           min_child_weight = 3,
                           eta  = 0.01,
                           gamma = 0) %>% as_tibble()
#final design  is a combination of latin hypercube optimization and my own preferred set of parameters
final_design =  simon_params  %>% bind_rows(des)
# bayes will have 10 additional iterations
control = makeMBOControl()
control = setMBOControlTermination(control, iters = 10)
```

Run the bayesian search:  



```r
# run this!
run = mbo(fun = obj.fun, 
          design = final_design,  
          control = control, 
          show.info = TRUE)
write_rds( run, "temp_files/run.rds")
```



```r
run <- read_rds( "temp_files/run.rds")
# print a summary with run
#run
# return  best model hyperparameters using run$x
# return best log likelihood using run$y
# return all results using run$opt.path$env$path
run$opt.path$env$path  %>% 
  mutate(Round = row_number()) %>%
  mutate(type = case_when(
    Round==1  ~ "1- hardcoded",
    Round<= 11 ~ "2 -design ",
    TRUE ~ "3 - mlrMBO optimization")) %>%
  ggplot(aes(x= Round, y= y, color= type)) + 
  geom_point() +
  labs(title = "mlrMBO optimization")+
  ylab("loglikelihood")
```

![](9_mlrmbo_poisson_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

# Conclusion:  
Bayesian optimization does generate better models, but it might be overkill if you aren't participating in a kaggle.  The table below shows the log likelihood of the best model found using the random grid and the mlrMBO and compares it my `simon params`.  




     type        loglikelihood    max_depth    colsample_bytree    subsample    min_child_weight       eta        gamma   
--------------  ---------------  -----------  ------------------  -----------  ------------------  -----------  ----------
 simon params      0.516181           6           0.8000000        0.8000000           3            0.0100000    0.000000 
 random_grid       0.516919           3           0.6796527        0.8889676           9            0.0055361    3.531973 
    mlrMBO         0.517671           5           0.6094444        0.4133019           2            0.0010006    1.356870 

# Code
The code that generated this document is located at

https://github.com/SimonCoulombe/snippets/blob/master/content/post/2019-1-09-bayesian.Rmd

## mlrMBO further reading  

Here are some resources I used to build this post:  

* [Xgboost using MLR package](http://rstudio-pubs-static.s3.amazonaws.com/336732_52d1b0e682634b5eae42cf86e1fc2a98.html)  
* [Vignette: https://mlrmbo.mlr-org.com/articles/supplementary/machine_learning_with_mlrmbo.html](https://mlrmbo.mlr-org.com/articles/supplementary/machine_learning_with_mlrmbo.html) , how to use lrMBO with mlr. I dont think you can do poisson regression using mlr.
* [Parameter tuning with mlrHyperopt](https://www.r-bloggers.com/parameter-tuning-with-mlrhyperopt/)  
* [Train R ML models efficiently with mlr](https://www.kaggle.com/xanderhorn/train-r-ml-models-efficiently-with-mlr)  

