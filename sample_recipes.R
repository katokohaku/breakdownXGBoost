require(tidyverse)
require(recipes)

iris.1 <- sample_frac(iris, 0.8)
iris.2 <- sample_frac(iris, 0.3)
rec <- recipe(Species ~ ., data = iris.1) %>% 
  step_log(Sepal.Length, Sepal.Width, base = 10)

rec_trained <- prep(rec, retain = TRUE, verbose = TRUE)
juice(rec_trained)
rec_trained %>% str(3)
bake(rec_trained, iris.1)
object.size(rec_trained)
rec_trained$template <- NULL

