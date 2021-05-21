rm(list = ls())    #delete objects
cat("\014")
install.packages("glmnet")
library(glmnet)
library(tidyverse); library(modelr); ## packages for data manipulation and computing rmse easily.
library(randomForest)
library(ggplot2)
library(gridExtra)
library(MASS)
library(tree)
library(tictoc)
#library(glmnetUtils)

## read data
d <- read_csv("C:/Users/hucar/Desktop/9890 Project/2018-cleaned.csv")
names(d) <- tolower(names(d)) ## make variable names lowercase because it's easier.

set.seed(0)

n = 2477 # observations
p = 141 # predictors (144-1(observation variable)-1(sticker)-1(sector)=141)

X = as.matrix(d[,2:142])
y = d$`2019 price var`

n.train        =     floor(0.8*n)
n.test         =     n-n.train

M              =     100 # replication times
Rsq.test.rid    =     rep(0,M)  # rd = ridge
Rsq.train.rid   =     rep(0,M)
Rsq.test.ls    =     rep(0,M)  # ls = lasso
Rsq.train.ls   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  #en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.rf    =     rep(0,M)  #rf = random forest
Rsq.train.rf   =     rep(0,M)

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  #, intercept = FALSE
  # fit ridge and calculate and record the train and test R squares 
  a=0 # ridge
  cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net 0<a<1
  cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
 
  # random forest
  bag  =  randomForest(X.train , y.train , mtry= floor(sqrt(p)), importance=TRUE)
  y.train.hat  = predict(bag, newx = X.train, type = "response")
  y.test.hat   = predict(bag, newdata = X.test, type = "response")
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  
cat(sprintf("m=%3.f| Rsq.test.ls=%.2f,  Rsq.test.en=%.2f| Rsq.train.ls=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.ls[m], Rsq.test.en[m],  Rsq.train.ls[m], Rsq.train.en[m]))
}

#CV PLOT
# fit lasso and calculate and record the train and test R squares 
a=1 # lasso
cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
plot(cv.fit)
a=0.5 # elastic
cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
plot(cv.fit)
a=0 # ridge
cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
plot(cv.fit)

#4b
library(dplyr)
#Rsq.test.en %>%filter()
boxplot(Rsq.test.rid,main="Test R squared- Ridge",
        xlab="", ylab="R squared")
boxplot(Rsq.train.rid,main="Train R squared- Ridge",
        xlab="", ylab="R squared")
boxplot(Rsq.test.en,main="Test R squared- Elastic",
        xlab="", ylab="R squared")
boxplot(Rsq.train.en,main="Train R squared- Elastic",
        xlab="", ylab="R squared")
boxplot(Rsq.test.ls,main="Test R squared- Lasso",
        xlab="", ylab="R squared")
boxplot(Rsq.train.ls,main="Train R squared- Lasso",
        xlab="", ylab="R squared")
boxplot(Rsq.test.rf,main="Test R squared- Random Forest",
        xlab="", ylab="R squared")
boxplot(Rsq.train.rf,main="Train R squared- Random Forest",
        xlab="", ylab="R squared")
# pattern 2
boxplot(Rsq.test.rid,Rsq.train.rid,main="Test Rsq vs. Train Rsq- Ridge",
        xlab="test vs train", ylab="R squared")

boxplot(Rsq.test.rid,Rsq.train.rid,main="Test Rsq vs. Train Rsq- Ridge",
        xlab="test vs train", ylab="R squared",ylim=c(-0.2,0))
boxplot(Rsq.test.en,Rsq.train.en,main="Test Rsq vs. Train Rsq-Elastic Net",
        xlab="test vs train", ylab="R squared",ylim=c(-0.1,0.05))
boxplot(Rsq.test.ls,Rsq.train.ls,main="Test Rsq vs. Train Rsq-- Lasso",
        xlab="test vs train", ylab="R squared",ylim=c(-0.2,0.1))
boxplot(Rsq.test.rf,Rsq.train.rf,main="Test Rsq vs. Train Rsq-- Random Forest",
        xlab="test vs train", ylab="R squared",ylim=c(-0.2,0.1))

#Residuals
a=0 # ridge
cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit$lambda.min)
y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.test.rid   =     y.train - y.train.hat
Res.train.rid  =     y.test - y.test.hat

a=0.5 # elastic-net 0<a<1
cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit$lambda.min)
y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.test.en   =     y.test - y.test.hat
Res.train.en  =     y.train - y.train.hat
 
a=1 # lasso
cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit$lambda.min)
y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.test.ls   =     y.test - y.test.hat
Res.train.ls  =     y.train - y.train.hat
# random forest
bag  =  randomForest(X.train , y.train , mtry= floor(sqrt(p)), importance=TRUE)
y.train.hat  = predict(bag, newx = X.train, type = "response")
y.test.hat   = predict(bag, newdata = X.test, type = "response")
Res.test.rf   = y.test - y.test.hat
Res.train.rf  = y.train - y.train.hat

boxplot(Res.train.rid,main="Ridge Train Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.test.rid,main="Ridge Test Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.train.en,main="Elastic Net Train Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.test.en,main="Elastic Net Test Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.train.ls,main="Lasso Train Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.test.ls,main="Lasso Test Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.train.rf,main="Random Forest Train Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))
boxplot(Res.test.rf,main="Random Forest Test Residuals",
        xlab=" ", ylab="Residuals",ylim=c(-50,50))

library(purrr)

lasso_model <- function(m,value) {
  
  
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = 0) 
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  a0.hat.ls        =   fit$a0[fit$lambda==cv.fit$lambda.min]
  beta.hat.ls       =   fit$beta
  
  a = paste(Rsq.test.ls[m])
  b = paste(Rsq.train.ls[m])
  
  
  if (value == "Residuals") {
    return(data.frame(y = y.test,
                      y_hat = as.numeric(y.test.hat),
                      stringsAsFactors = F) %>% dplyr::mutate(residuals = y-y_hat))
  } else if (value == "Coefficients") {
    betas <- data.frame(c(1:141), as.vector(beta.hat.ls))
    colnames(betas) <- c( "feature", "value")
    return(betas)
  } else {
    return(data.frame(Rsq_test_ls = a,
                      Rsq.train.ls = b, stringsAsFactors = F))
  }}
lasso_model(1,"Results")

ridge_model <- function(m,value){
  
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit ridge and calculate and record the train and test R squares 
  a=0 # ridge
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  a0.hat.rid         =   fit$a0[fit$lambda==cv.fit$lambda.min]
  beta.hat.rid       =   fit$beta[ ,fit$lambda==cv.fit$lambda.min]
  
  a = paste(Rsq.test.rid[m])
  b = paste(Rsq.train.rid[m])
  
  
  if (value == "Residuals") {
    return(data.frame(y = y.test,
                      y_hat = as.numeric(y.test.hat),
                      stringsAsFactors = F) %>% dplyr::mutate(residuals = y-y_hat))
  } else if (value == "Coefficients") {
    betas <- data.frame(c(1:141), as.vector(beta.hat.rid))
    colnames(betas) <- c( "feature", "value")
    return(betas) 
  } else {
    return(data.frame(Rsq.test.rid = a,
                      Rsq.train.rid = b, stringsAsFactors = F))
  }}
ridge_model(1,"Results")

elastic_net <- function(m,value) {  
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net 0<a<1
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  a0.hat.en        =   fit$a0[fit$lambda==cv.fit$lambda.min]
  beta.hat.en      =   fit$beta[ ,fit$lambda==cv.fit$lambda.min]
  
  
  a = paste(Rsq.test.en[m])
  b = paste(Rsq.train.en[m])
  
  
  if (value == "Residuals") {
    return(data.frame(y = y.test,
                      y_hat = as.numeric(y.test.hat),
                      stringsAsFactors = F) %>% dplyr::mutate(residuals = y-y_hat))
  } else if (value == "Coefficients") {
    betas <- data.frame(c(1:141), as.vector(beta.hat.en))
    colnames(betas) <- c( "feature", "value")
    return(betas)
    
  } else {
    return(data.frame(rsq_test_en = a,
                      rsq_train_en = b, stringsAsFactors = F))
  }}
elastic_net(1,"Results")

##Random Forest
random_forest <- function(m,value) { 
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  bag  =  randomForest(X.train , y.train , mtry= floor(sqrt(p)), importance=TRUE)
  y.train.hat  = predict(bag, newx = X.train, type = "response")
  y.test.hat   = predict(bag, newdata = X.test, type = "response")
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  a0.hat.rf        =   fit$a0[fit$lambda==cv.fit$lambda.min]
  beta.hat.rf      =   fit$beta[ ,fit$lambda==cv.fit$lambda.min]
  
  
  
  if (value == "Residuals") {
    return(data.frame(y = y.test,
                      y_hat = as.numeric(y.test.hat),
                      stringsAsFactors = F) %>% dplyr::mutate(residuals = y-y_hat))
  } else if (value == "Coefficients") {
    betas <- data.frame(c(1:141), as.vector(beta.hat.rf))
    colnames(betas) <- c( "feature", "value")
    return(betas)
    
  } else {
    return(data.frame(Rsq.test.rf = a,
                      Rsq.train.rf = b, stringsAsFactors = F))
  }}



##Boxplots
boxplot(Rsq.test.rid,Rsq.train.rid,main="Test Rsq vs. Train Rsq- Ridge",
        xlab="test vs train", ylab="R squared")
boxplot(Rsq.test.en,Rsq.train.en,main="Test Rsq vs. Train Rsq-Elastic ",
        xlab="test vs train", ylab="R squared",ylim)
boxplot(Rsq.test.ls,Rsq.train.ls,main="Test Rsq vs. Train Rsq-- Lasso",
        xlab="", ylab="R squared")

##Plot of Cv fit shuffled_indexes =     sample(n)
shuffled_indexes =     sample(n)
train            =     shuffled_indexes[1:n.train]
test             =     shuffled_indexes[(1+n.train):n]
X.train          =     X[train, ]
y.train          =     y[train]
X.test           =     X[test, ]
y.test           =     y[test]
a=1 # lasso
cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
plot(cv.fit,main="Lasso 10-Fold Cross Validation Plot")
a=0.5 # elastic
cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
plot(cv.fit,main="Elastic Net 10-Fold Cross Validation Plot")
a=0 # ridge
cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
plot(cv.fit,main="Ridge 10-Fold Cross Validation Plot")



tic()
ridge_model_results = 1:100 %>% purrr::map(function(x) ridge_model(x,"results"))
toc()
tic()
lasso_model_results = 1:100 %>% purrr::map(function(x) lasso_model(x,"results"))
toc()

tic()
elastic_net_model_results = 1:100 %>% purrr::map(function(x) elastic_net(x,"results"))
toc()

### Work On
##ridge_model_results<-ridge_model_results %>% bind_rows() %>%
##mutate_at(vars(everything()),as.numeric) %>% dplyr::filter(Rsq.test.rid>=-1)
##boxplot(ridge_model_results$Rsq.test.rid,ridge_model_results$Rsq.train.rid,min="Test Rsq vs Train Rsq for Ridge",
##xlab="Test vs Train",ylab="R Squared",ylim=c(-.4,0),main="Ridge R^2 Test Vs Train") 

lasso_model_results<-lasso_model_results %>% bind_rows() %>%
  mutate_at(vars(everything()),as.numeric) %>% dplyr::filter(Rsq.test.las>=-1)
boxplot(lasso_model_results$Rsq.test.las,lasso_model_results$Rsq.train.las,min="Test Rsq vs Train Rsq for Lasso",
        xlab="Test vs Train",ylab="R Squared",main="Lasso R^2 Test Vs Train",ylim=c(-1,1)) 

elastic_net_model_results<-elastic_net_model_results %>% bind_rows() %>%
  mutate_at(vars(everything()),as.numeric) %>% dplyr::filter(Rsq.test.en>=-1)
boxplot(elastic_net_model_results$Rsq.test.en,elastic_net_model_results$Rsq.train.en,min="Test Rsq vs Train Rsq for Ridge",
        xlab="Test vs Train",ylab="R Squared",ylim=c(-.4,0),main="Ridge R^2 Test Vs Train") 




#tic()
#random_forest_model_results = 1:100 %>% purrr::map(function(x) random_forest(x,"results"))
#toc()

ridge_model_residuals <- ridge_model(1,"Residuals")
elastic_net_residuals <- elastic_net(1,"Residuals")
lasso_model_residuals <- lasso_model(1,"Residuals") %>% dplyr::filter(residuals <= 2000 | residuals >= -2000)


boxplot(x=ridge_model_residuals$residuals,data=ridge_model_residuals, main="Ridge Model Residuals",
        xlab="", ylab="Residuals",ylim=c(-100,50))
boxplot(x=elastic_net_residuals$residuals,data=elastic_net_residuals, main="Elastic Net Model Residuals",
        xlab="", ylab="Residuals",ylim=c(-100,400))
boxplot(x=lasso_model_residuals$residuals,data=lasso_model_residuals, main="Lasso Model Residuals", xlab="", ylab="Residuals",ylim=c(-100,400))




# fit lasso to the whole data
a=1 # lasso
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
lasso_model_results <- data.frame(r_squared = cv.fit$glmnet.fit$dev.ratio,stringsAsFactors = F) %>% dplyr::filter(r_squared >= -1 | r_squared <= 1)
lasso_model_CI <- c(mean(lasso_model_results$r_squared,na.rm=T)+1.645*(sd(lasso_model_results$r_squared,na.rm=T)/sqrt(nrow(lasso_model_results))),
                    mean(lasso_model_results$r_squared,na.rm=T)-1.645*(sd(lasso_model_results$r_squared,na.rm=T)/sqrt(nrow(lasso_model_results))))

tic()
fit.ls           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
toc()
# fit en to the whole data
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
elastic_model_results <- data.frame(r_squared = cv.fit$glmnet.fit$dev.ratio,stringsAsFactors = F) %>% dplyr::filter(r_squared >= -1 | r_squared <= 1)
elastic_model_CI <- c(mean(elastic_model_results$r_squared,na.rm=T)+1.645*(sd(elastic_model_results$r_squared,na.rm=T)/sqrt(nrow(elastic_model_results))),
                      mean(elastic_model_results$r_squared,na.rm=T)-1.645*(sd(elastic_model_results$r_squared,na.rm=T)/sqrt(nrow(elastic_model_results))))

tic()
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
toc()
#fit rid to the whole data
a=0
tic()
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
toc()
ridge_model_results <- data.frame(r_squared = cv.fit$glmnet.fit$dev.ratio,stringsAsFactors = F) %>% dplyr::filter(r_squared >= -1 | r_squared <= 1)
ridge_model_CI <- c(mean(ridge_model_results$r_squared,na.rm=T)+1.645*(sd(ridge_model_results$r_squared,na.rm=T)/sqrt(nrow(ridge_model_results))),
                    mean(ridge_model_results$r_squared,na.rm=T)-1.645*(sd(ridge_model_results$r_squared,na.rm=T)/sqrt(nrow(ridge_model_results))))



Rsq.test.en <- Rsq.test.en[Rsq.test.en>=-1 & Rsq.test.en<=1]
Rsq.test.ls <-Rsq.test.ls[Rsq.test.ls>=-1 & Rsq.test.ls<=1]
New.Rsq.test.ls <- na.omit(Rsq.test.ls)
Rsq.test.rid <- Rsq.test.rid[Rsq.test.rid>=-1 & Rsq.test.rid<=1]
Rsq.test.rf <- Rsq.test.rf[Rsq.test.rf>=-1 & Rsq.test.rf<=1]


##Confidence intervals
CI(Rsq.test.rid,ci=.90) 
CI(New.Rsq.test.ls,ci=.90)
CI(Rsq.test.en,ci=.90)
CI(Rsq.test.rf,ci=.90)



tic()
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
toc()
#For all data
# random forest(5.c)
tic()
bag2  =  randomForest(X , y , mtry= floor(sqrt(p)), importance=TRUE)
rf_model_results <- data.frame(r_squared = bag2$rsq,stringsAsFactors = F)
rf_model_CI <- c(mean(rf_model_results$r_squared,na.rm=T)+1.645*(sd(rf_model_results$r_squared,na.rm=T)/sqrt(nrow(rf_model_results))),
                 mean(rf_model_results$r_squared,na.rm=T)-1.645*(sd(rf_model_results$r_squared,na.rm=T)/sqrt(nrow(rf_model_results))))
y.hat  = predict(bag2, type = "response")
toc()

##Bar-Plots of Estimated Coefficients
elastic_coeff<-elastic_net(1,"Coefficients")
ridge_coeff<-ridge_model(1,"Coefficients")
lasso_coeff<-lasso_model(1,"Coefficients")


lasso_coeff$feature     =  factor(lasso_coeff$feature, levels = elastic_coeff$feature[order(lasso_coeff$value, decreasing = TRUE)])
elastic_coeff$feature   =  factor(elastic_coeff$feature, levels = elastic_coeff$feature[order(elastic_coeff$value, decreasing = TRUE)])
ridge_coeff$feature     =  factor(ridge_coeff$feature, levels = elastic_coeff$feature[order(ridge_coeff$value, decreasing = TRUE)])


rgPlot =  ggplot(ridge_coeff, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="blue", colour="black",main="Ridge Model Coefficients") + ggtitle("Ridge Model Coefficients")
theme_classic()

rgPlot

lsPlot =  ggplot(lasso_coeff, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="blue", colour="black",main="Lasso Model Coefficients") + ggtitle("Lasso Model Coefficients")
theme_classic()

lsPlot

enPlot =  ggplot(elastic_coeff, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="blue", colour="black",) + ggtitle("Elastic Net Model Coefficients")
theme_classic()

enPlot

rfPlot <-importance(bag2) %>% as.data.frame() %>% dplyr::mutate(feature = rownames(.)) %>%
  dplyr::arrange(-`%IncMSE`) %>%
  ggplot(aes(x=reorder(feature,-`%IncMSE`),y=`%IncMSE`)) +
  geom_col(fill="blue") +
  theme_classic() +
  xlab("feature") +
  ylab("Importance") +
  ggtitle("Random Forest Feature Importance")
toc()
grid.arrange(enPlot, lsPlot, rgPlot, rfPlot, nrow = 4)
