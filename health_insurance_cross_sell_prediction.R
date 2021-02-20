### load libraries ###
library(lubridate)
library(data.table)
library(scales)
library(RColorBrewer)
library(ggplot2)
library(cowplot)
library(caret)
library(gbm)
library(pROC)
library(mmpf)
library(parallel)


### set variables ###
data_location <- "..."
r_func_location <- "..."
no_cores <- 1


### set functions ###
source(file=paste0(r_func_location, "profiler.R"))


### import data ###
## health train
hl_train <- fread(file=paste0(data_location, "train.csv"), sep=",", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
setDT(hl_train)

## health test
hl_test <- fread(file=paste0(data_location, "test.csv"), sep=",", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
setDT(hl_test)

## health submission sample
hl_sub <- fread(file=paste0(data_location, "sample_submission.csv"), sep=",", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
setDT(hl_sub)


### feature engineering ###
## health train
hl_train[, Gender:=factor(x=Gender, levels=c("Male","Female"))]
hl_train[, Age_Bin:=cut(x=Age, breaks=c(18, seq(from=25, to=65, by=5), Inf), right=FALSE)]
hl_train[, Region_Code_2:=as.character(Region_Code)]
hl_train[, Region_Code_2:=ifelse(Region_Code_2 %in% c("28","8","46","41","15","30","29","50","3"), Region_Code_2, "other")]
hl_train[, Region_Code_2:=factor(x=Region_Code_2, levels=c("28","8","46","41","15","30","29","50","3","other"))]
hl_train[, Vehicle_Age:=factor(x=Vehicle_Age, levels=c("< 1 Year","1-2 Year","> 2 Years"))]
hl_train[, Vehicle_Damage:=factor(x=Vehicle_Damage, levels=c("Yes","No"))]
hl_train[, Annual_Premium_Bin:=cut(x=Annual_Premium, breaks=c(0, seq(from=20000, to=50000, by=5000), Inf), right=FALSE)]
hl_train[, Policy_Sales_Channel_2:=as.character(Policy_Sales_Channel)]
hl_train[, Policy_Sales_Channel_2:=ifelse(Policy_Sales_Channel_2 %in% c("152","26","124","160","156","122","157","154","151"), Policy_Sales_Channel_2, "other")]
hl_train[, Policy_Sales_Channel_2:=factor(x=Policy_Sales_Channel_2, levels=c("152","26","124","160","156","122","157","154","151","other"))]
hl_train[, Vintage_Bin:=cut(x=Vintage, breaks=c(seq(from=0, to=250, by=50), Inf), right=FALSE)]
hl_train[, Response:=factor(x=Response, levels=c(0,1))]

## health test
hl_test[, Gender:=factor(x=Gender, levels=c("Male","Female"))]
hl_test[, Age_Bin:=cut(x=Age, breaks=c(18, seq(from=25, to=65, by=5), Inf), right=FALSE)]
hl_test[, Region_Code_2:=as.character(Region_Code)]
hl_test[, Region_Code_2:=ifelse(Region_Code_2 %in% c("28","8","46","41","15","30","29","50","3"), Region_Code_2, "other")]
hl_test[, Region_Code_2:=factor(x=Region_Code_2, levels=c("28","8","46","41","15","30","29","50","3","other"))]
hl_test[, Vehicle_Age:=factor(x=Vehicle_Age, levels=c("< 1 Year","1-2 Year","> 2 Years"))]
hl_test[, Vehicle_Damage:=factor(x=Vehicle_Damage, levels=c("Yes","No"))]
hl_test[, Annual_Premium_Bin:=cut(x=Annual_Premium, breaks=c(0, seq(from=20000, to=50000, by=5000), Inf), right=FALSE)]
hl_test[, Policy_Sales_Channel_2:=as.character(Policy_Sales_Channel)]
hl_test[, Policy_Sales_Channel_2:=ifelse(Policy_Sales_Channel_2 %in% c("152","26","124","160","156","122","157","154","151"), Policy_Sales_Channel_2, "other")]
hl_test[, Policy_Sales_Channel_2:=factor(x=Policy_Sales_Channel_2, levels=c("152","26","124","160","156","122","157","154","151","other"))]
hl_test[, Vintage_Bin:=cut(x=Vintage, breaks=c(seq(from=0, to=250, by=50), Inf), right=FALSE)]


### model development ###
## profiler analysis
prof_var <- c("Gender","Age_Bin","Driving_License","Region_Code_2","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium_Bin","Policy_Sales_Channel_2","Vintage_Bin")
hl_prof <- profiler(dt_pop_1=hl_train[Response=="1"], dt_pop_2=hl_train[Response=="0"], variables=prof_var, no_cores=no_cores)
rm(prof_var)
hl_prof[, .(information_value=sum(information_value)), keyby=attribute][order(-information_value)]
saveRDS(object=hl_prof, file=paste0(data_location, "hl_prof.Rds"))

## train/test split on existing data
n_index <- sample(x=1:nrow(hl_train), size=floor(nrow(hl_train)*0.7), replace=FALSE)
hl_train_tmp <- hl_train[n_index]
hl_train_tmp[, c("id","Age","Region_Code","Vintage","Policy_Sales_Channel"):=NULL]
hl_train_tmp[, Response:=factor(x=ifelse(Response=="1", "yes", "no"), levels=c("no","yes"))]
hl_test_tmp <- hl_train[-n_index]
hl_test_tmp[, c("id","Age","Region_Code","Vintage","Policy_Sales_Channel"):=NULL]
hl_test_tmp[, Response:=factor(x=ifelse(Response=="1", "yes", "no"), levels=c("no","yes"))]
saveRDS(object=n_index, file=paste0(data_location, "n_index.Rds"))
saveRDS(object=hl_train_tmp, file=paste0(data_location, "hl_train_tmp.Rds"))
saveRDS(object=hl_test_tmp, file=paste0(data_location, "hl_train_tmp.Rds"))

names(getModelInfo())

## model training control
# training control
ctrl <- trainControl(method="boot",
                     number=1,
                     p=0.8,
                     summaryFunction=twoClassSummary,
                     classProbs=TRUE,
                     verboseIter=TRUE,
                     returnData=FALSE)

## glm
# parameter lookup (we all know there's no such here ;))
modelLookup(model="glm")

# model training
print(Sys.time())
start_time <- proc.time()
hl_glm <- train(x=hl_train_tmp[, !c("Response")], y=hl_train_tmp[, Response],
                method="glm",
                trControl=ctrl,
                family=binomial,
                model=FALSE,
                metric="ROC")
end_time <- proc.time()
print(paste0("seconds to run models: ", (end_time-start_time)[3]))

# initial model performance statistics
print(hl_glm)
summary(object=hl_glm)
varImp(object=hl_glm)

# remove additional parameters
rm(start_time, end_time)

saveRDS(object=hl_glm, file=paste0(data_location, "hl_glm.Rds"))

## gbm
# parameter lookup
modelLookup(model="gbm")

# hyper-parameters for tuning
xpnd_grid <- expand.grid(n.trees=c(600),
                         interaction.depth=c(4,6,8),
                         shrinkage=c(0.05,0.1,0.2),
                         n.minobsinnode=c(20,50))

# model training
print(Sys.time())
start_time <- proc.time()
hl_gbm <- train(x=hl_train_tmp[, !c("Response")], y=hl_train_tmp[, Response],
                method="gbm",
                trControl=ctrl,
                tuneGrid=xpnd_grid,
                verbose=FALSE,
                metric="ROC")
end_time <- proc.time()
print(paste0("seconds to run models: ", (end_time-start_time)[3]))

# initial model performance statistics
print(hl_gbm)
summary(object=hl_gbm, plotit=FALSE)
varImp(hl_gbm)

# remove additional parameters
rm(xpnd_grid, start_time, end_time)

saveRDS(object=hl_gbm, file=paste0(data_location, "hl_gbm.Rds"))

## xgboost
# parameter lookup
modelLookup("xgbTree")

# hyper-parameters for tuning
xpnd_grid <- expand.grid(nrounds=500,
                         max_depth=c(4,6,8),
                         eta=c(0.05,0.1,0.2),
                         gamma=c(0,1),
                         colsample_bytree=c(1),
                         min_child_weight=c(1),
                         subsample=c(1))

# model training
print(Sys.time())
start_time <- proc.time()
hl_train_tmp_2 <- as.matrix(model.matrix(~., data=hl_train_tmp[, !c("Response")])[, -1])
hl_xgb <- train(x=hl_train_tmp_2, y=hl_train_tmp[, Response],
                method="xgbTree",
                trControl=ctrl,
                tuneGrid=xpnd_grid,
                verbose=0,
                metric="ROC")
end_time <- proc.time()
print(paste0("seconds to run models: ", (end_time-start_time)[3]))

# initial model performance statistics
print(hl_xgb)
summary(object=hl_xgb)
varImp(hl_xgb)

# remove additional parameters
rm(hl_train_tmp_2, xpnd_grid, start_time, end_time)

saveRDS(object=hl_xgb, file=paste0(data_location, "hl_xgb.Rds"))

## random forest
# parameter lookup
modelLookup("rf")

# hyper-parameters for tuning
xpnd_grid <- expand.grid(mtry=c(2,3,4,5))

# model training
print(Sys.time())
start_time <- proc.time()
hl_rf <- train(x=hl_train_tmp[, !c("Response")], y=hl_train_tmp[, Response],
               method="rf",
               trControl=ctrl,
               tuneGrid=xpnd_grid,
               ntrees=200,
               importance=TRUE,
               metric="ROC")
end_time <- proc.time()
print(paste0("seconds to run models: ", (end_time-start_time)[3]))

# initial model performance statistics
print(hl_rf)
summary(object=hl_rf)
varImp(object=hl_rf)

# remove additional parameters
rm(xpnd_grid, start_time, end_time)

saveRDS(object=hl_rf, file=paste0(data_location, "hl_rf.Rds"))

## neural network
# parameter lookup
modelLookup("nnet")

# hyper-parameters for tuning
xpnd_grid <- expand.grid(size=c(3,5,8,10,15,20),
                         decay=c(0,0.5,1))

# model training
print(Sys.time())
start_time <- proc.time()
hl_nn <- train(x=hl_train_tmp[, !c("Response")], y=hl_train_tmp[, Response],
               method="nnet",
               trControl=ctrl,
               tuneGrid=xpnd_grid,
               maxit=150,
               metric="ROC")
end_time <- proc.time()
print(paste0("seconds to run models: ", (end_time-start_time)[3]))

# initial model performance statistics
print(hl_nn)
summary(object=hl_nn)
varImp(object=hl_nn)

# remove additional parameters
rm(xpnd_grid, start_time, end_time)

saveRDS(object=hl_nn, file=paste0(data_location, "hl_nn.Rds"))


### further model testing ###
## roc on second test dataset
par(pty="s")
roc(response=hl_test_tmp[, ifelse(Response=="yes", 1, 0)],
    predictor=predict(object=hl_glm, newdata=hl_test_tmp, type="prob")[, 2],
    plot=TRUE, col="green", lwd=2.5, legacy.axes=TRUE, print.auc=TRUE,
    main="ROC Analysis", xlab="FPR", ylab="TPR")
plot.roc(x=hl_test_tmp[, ifelse(Response=="yes", 1, 0)],
         predictor=predict(object=hl_gbm, newdata=hl_test_tmp, type="prob")[, 2],
         col="royalblue", lwd=2.5, add=TRUE, print.auc=TRUE, print.auc.y=0.45)
plot.roc(x=hl_test_tmp[, ifelse(Response=="yes", 1, 0)],
         predictor=predict(object=hl_xgb, newdata=as.matrix(model.matrix(~., data=hl_test_tmp[, !c("Response")])[, -1]), type="prob")[, 2],
         col="coral", lwd=2.5, add=TRUE, print.auc=TRUE, print.auc.y=0.40)
plot.roc(x=hl_test_tmp[, ifelse(Response=="yes", 1, 0)],
         predictor=predict(object=hl_rf, newdata=hl_test_tmp, type="prob")[, 2],
         col="purple", lwd=2.5, add=TRUE, print.auc=TRUE, print.auc.y=0.35)
plot.roc(x=hl_test_tmp[, ifelse(Response=="yes", 1, 0)],
         predictor=predict(object=hl_nn, newdata=hl_test_tmp, type="prob")[, 2],
         col="orange", lwd=2.5, add=TRUE, print.auc=TRUE, print.auc.y=0.30)
legend("bottomright", legend=c("logistic regression","gradient boosting","xgboost","random forest","neural network"), col=c("green","royalblue","coral","purple","orange"), cex=0.75, lwd=2.5, bty="n")
par(pty="m")

## optimal number of trees gbm
hl_gbm_tr <- lapply(1:hl_gbm$finalModel$n.trees,
                    function(x1) {
                        if (x1 ==1 || x1 %% 10 == 0) {
                            print(paste0("number of trees: ", x1))
                        }
                        
                        val <- hl_test_tmp[, ifelse(Response=="yes", 1, 0)]
                        pred <- predict(object=hl_gbm$finalModel, newdata=model.matrix(~., data=hl_test_tmp)[, -1], n.trees=x1, type="response")
                        
                        dt_1_roc <- roc(response=val, predictor=pred)
                        
                        dt_1 <- data.table(n_trees=x1,
                                           auc=auc(dt_1_roc))
                        
                        return(dt_1)
                    })
hl_gbm_tr <- rbindlist(l=hl_gbm_tr, use.names=TRUE, fill=TRUE)
