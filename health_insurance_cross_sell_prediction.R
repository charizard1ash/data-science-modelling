### load libraries ###
library(lubridate)
library(data.table)
library(scales)
library(RColorBrewer)
library(ggplot2)
library(cowplot)
library(caret)
library(pROC)


### set variables ###
data_location <- "..."
r_func_location <- "..."


### set functions ###
source(file=paste0(r_func_location, "permutation_importance.R"))


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
## train/test split on existing data
n_index <- sample(x=1:nrow(hl_train), size=floor(nrow(hl_train)*0.7), replace=FALSE)
hl_train_tmp <- hl_train[n_index]
hl_train_tmp[, c("id","Age","Region_Code","Vintage","Policy_Sales_Channel"):=NULL]
hl_test_tmp <- hl_train[-n_index]
hl_test_tmp[, c("id","Age","Region_Code","Vintage","Policy_Sales_Channel"):=NULL]



trainControl

## glm
hl_glm
