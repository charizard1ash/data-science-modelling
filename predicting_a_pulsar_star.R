### Load libraries ###
library(lubridate)
library(data.table)
library(corrplot)
library(scales)
library(RColorBrewer)
library(ggplot2)
library(cowplot)
library(DT)
library(rpart)
library(e1071)
library(randomForest)
library(xgboost)
library(neuralnet)
library(shiny)


### Set variables ###
data_location <- "/Users/minathaniel/Documents/Data/Kaggle/"
function_location <- "/Users/minathaniel/Documents/R/R Scripts/Functions/"


### Import existing and set new functions ###
## Model performance.
source(file=paste0(function_location, "model_performance.R"))

## ROC.
source(file=paste0(function_location, "roc.R"))

## Permutation importance.
source(file=paste0(function_location, "permutation_importance.R"))


### Import data ###
## Pulsar star.
pul_str <- read.table(file=unz(description=paste0(data_location, "predicting-a-pulsar-star.zip"), filename="pulsar_stars.csv"), sep=",", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
setDT(pul_str)
colnames(pul_str) <- gsub("[^A-Za-z0-9_]", "_", colnames(pul_str))


### Split dataset into train and test ###
## Copy dataset.
pul_str_2 <- copy(pul_str)

## Convert all string variables to factor type.
x_char <- sapply(pul_str_2[, !c("target_class")], is.character)
x_char <- names(x_char[x_char==TRUE])
suppressWarnings(pul_str_2[, c(x_char):=lapply(.SD, as.factor), .SDcols=x_char])
rm(x_char)

## Training dataset indeces.
n_train <- sample(x=1:nrow(pul_str_2), size=floor(nrow(pul_str_2)*0.70), replace=FALSE)

## Train and test set.
dt_train <- pul_str_2[n_train]
dt_test <- pul_str_2[-n_train]


### Conduct model experiments ###
## Set empty model prediction results tables.
# names of each model vector.
model_names <- c()

# confusion matrix.
conf_mat <- list()

# performance results.
perf_res <- data.table(model=NA,
                       class=NA,
                       precision=NA,
                       recall=NA,
                       f1_score=NA,
                       weight=NA)[0]

# performance summary results.
perf_res_cons <- data.table(model=NA,
                            mean_precision=NA,
                            weighted_precision=NA,
                            mean_recall=NA,
                            weighted_recall=NA,
                            mean_f1_score=NA,
                            weighted_f1_score=NA)[0]

# roc results.
roc_res <- data.table(model=NA,
                      prob_threshold=NA,
                      tnr=NA,
                      fpr=NA,
                      fnr=NA,
                      tpr=NA,
                      precision=NA,
                      recall=NA,
                      f1_score=NA)[0]

# auc results.
auc_res <- c()

# permutation importance results.
perm_imp_res <- data.table(model_name=NA,
                           variable=NA,
                           iteration=NA,
                           loss_func=NA)[0]

## logistic regression 1.
model_names <- c(model_names, "glm 01")

# model train.
glm_01 <- glm(formula=target_class~., data=dt_train, family="binomial", control=glm.control(epsilon=1e-05, maxit=100))

# model test.
glm_01_prob <- predict(object=glm_01, newdata=dt_test, type="response")

# find optimal probability threshold.
roc_prob_thr <- roc(dt=dt_test, dt_act=dt_test[, target_class], model=glm_01, model_name="glm 01", pred_func="predict(object=glm_01, newdata=dt, type=\"response\")", prob_thresh=seq(from=0, to=1, by=0.01))
prob_thr_val <- roc_prob_thr$roc_dt[order(-f1_score)][, first(prob_threshold)] # prob~0.3

# consolidate roc results.
roc_res <- rbindlist(l=list(roc_res, glm_01_prob_thr$roc_dt), use.names=TRUE, fill=TRUE)

# consolidate auc results.
auc_res <- c(auc_res, roc_prob_thr$auc)

# set predictions based on max f1-score.
glm_01_pred <- ifelse(glm_01_prob>=prob_thr_val, 1, 0)

# model performnace results.
perf_res_lst <- model_performance(model="glm 01", predict=glm_01_pred, actual=dt_test[, target_class])

# consolidate confusion matrix results.
conf_mat[[1]] <- perf_res_lst$confusion_matrix

# consolidate model performance results.
perf_res <- rbindlist(l=list(perf_res, perf_res_lst$performance_results), use.names=TRUE, fill=TRUE)

# consolidate model summary performance results.
perf_res_cons <- rbindlist(l=list(perf_res_cons, perf_res_lst$results_consolidated), use.names=TRUE, fill=TRUE)