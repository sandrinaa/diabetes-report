
setwd("C:/Users/sandr/OneDrive/Documents/NUS/NUS Courses/Y1S1/DSA1101/Data")

set.seed(1101)
library(caret)
library("ggplot2")
library("dplyr")
library(class)
library("rpart")
library("rpart.plot")
library(e1071)
library(ROCR)

data = read.csv("diabetes_5050.csv")
raw = read.csv("diabetes_5050.csv")
length(data$Diabetes_binary)
head(data)
names(data)
dim(data)

table(data$Diabetes_binary) #binary
table(data$HighBP) #binary
table(data$HighChol) #binary
table(data$CholCheck) #binary
table(data$BMI) #categorical
table(data$Smoker) #binary
table(data$Stroke) #binary
table(data$HeartDiseaseorAttack) #binary
table(data$PhysActivity) #binary
table(data$Fruits) #binary
table(data$Veggies) #binary
table(data$HvyAlcoholConsump) #binary
table(data$AnyHealthcare) #binary
table(data$NoDocbcCost) #binary
table(data$GenHlth) #categorical
table(data$MentHlth) #categorical
table(data$PhysHlth) #categorical
table(data$DiffWalk) #binary
table(data$Sex) #binary
table(data$Age) #categorical
table(data$Education) #categorical
table(data$Income) #categorical

cor(data$HighBP, data$Diabetes_binary) # 0.3815155
cor(data$HighChol, data$Diabetes_binary) # 0.2892128
cor(data$CholCheck, data$Diabetes_binary) # 0.1153816
cor(data$BMI, data$Diabetes_binary) # 0.2933727
cor(data$Smoker, data$Diabetes_binary) # 0.08599896
cor(data$Stroke, data$Diabetes_binary) # 0.1254268
cor(data$HeartDiseaseorAttack, data$Diabetes_binary) # 0.2115234
cor(data$PhysActivity, data$Diabetes_binary) # -0.1586656
cor(data$Fruits, data$Diabetes_binary) # -0.05407656
cor(data$Veggies, data$Diabetes_binary) # -0.07929315
cor(data$HvyAlcoholConsump, data$Diabetes_binary) # -0.09485314
cor(data$AnyHealthcare, data$Diabetes_binary) # 0.02319075
cor(data$NoDocbcCost, data$Diabetes_binary) # 0.04097657
cor(data$GenHlth, data$Diabetes_binary) # 0.4076116
cor(data$MentHlth, data$Diabetes_binary) # 0.08702877
cor(data$PhysHlth, data$Diabetes_binary) # 0.213081
cor(data$DiffWalk, data$Diabetes_binary) # 0.272646
cor(data$Sex, data$Diabetes_binary) # 0.04441286
cor(data$Age, data$Diabetes_binary) # 0.2787381
cor(data$Education, data$Diabetes_binary) # -0.1704806
cor(data$Income, data$Diabetes_binary) # -0.2244487

# Based on the correlation, we can use these variables as the regressors to determine the result of the response:
# HighBP, HighChol, CholCheck, BMI, Stroke, HeartDiseaseorAttack, PhysActivity, GenHlth, PhysHlth, DiffWalk, Age, Education, Income 

table(data$HighBP, data$Diabetes_binary)
table(data$HighChol, data$Diabetes_binary)
table(data$CholCheck, data$Diabetes_binary) 
boxplot(data$BMI, data$Diabetes_binary, main = "BMI") 
table(data$Stroke, data$Diabetes_binary) 
table(data$HeartDiseaseorAttack, data$Diabetes_binary) 
table(data$PhysActivity, data$Diabetes_binary) 
boxplot(data$GenHlth, data$Diabetes_binary, main = "GenHlth") 
boxplot(data$PhysHlth, data$Diabetes_binary, main = "PhysHlth")
table(data$DiffWalk, data$Diabetes_binary)
boxplot(data$Age, data$Diabetes_binary, main = "Age") 
boxplot(data$Education, data$Diabetes_binary, main = "Education") 
boxplot(data$Income, data$Diabetes_binary, main = "Income") 

data = data[,-c(6,10,11,12,13,14,16,19)]
data$BMI = c(scale(data$BMI))
data$PhysHlth = c(scale(data$PhysHlth))
data$Age = c(scale(data$Age))

response_var = data$Diabetes_binary
index = createDataPartition(response_var, p = 0.8, list = FALSE)

train.data = data[index, ]
test.data = data[-index, ]
dim(train.data) #56554 Observations
dim(test.data) #14138 Observations

train.x = train.data[,-1]
train.y = train.data[,c("Diabetes_binary")]
test.x = test.data[,-1]
test.y = test.data[,c("Diabetes_binary")]

length(which(test.y == 0))/length(which(test.y == 1))
length(which(train.y == 0))/length(which(train.y == 1))
#both train and test have the same ratio of responses

attach(data)
attach(train.data)

##### LINEAR REGRESSION
M1.model = lm(Diabetes_binary ~ ., data = train.data)
summary(M1.model)

M1 = lm(Diabetes_binary ~ HighBP+HighChol+CholCheck+BMI+Stroke+HeartDiseaseorAttack
      +GenHlth+PhysHlth+DiffWalk+Age+Education+Income, data = train.data)
summary(M1)

M1.pred = predict(M1, newdata = test.x)

##### KNN

M2.pred = knn(train.x, test.x, train.y, k = 74)

#find the best k
accuracy = numeric(100)
for (i in 1:100){
  pred = knn(train.x, test.x, train.y, k=i) 
  confusion.matrix = table(test.y, pred)
  accuracy[i] = sum(diag(confusion.matrix))/sum(confusion.matrix)
}
accuracy

max(accuracy) #0.7513793
index = which(accuracy == max(accuracy)) ; index #74

cm.M2 = table(M2.pred, test.y); cm.M2
#       test.y
#M2.pred    0    1
#      0 4988 1427
#      1 2081 5642

k_values = seq(1, 100)
plot(k_values, accuracy, pch = 20)

##### DECISION TREE

M3 = rpart(train.data$Diabetes_binary ~ .,
             method="class",
             data= train.data,
             control=rpart.control(minsplit = 1), 
             parms=list(split='information')
)
rpart.plot(M3, type=4, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)

M3.pred = predict(M3, test.x, type = "class")
cm.M3 = table(M3.pred, test.y); cm.M3

#       test.y
#M3.pred   0    1
#      0 4623 1422
#      1 2446 5647

##### NAIVE BAYES

M4 = naiveBayes(train.data$Diabetes_binary ~ ., data = train.data)
M4.pred = predict(M4, test.x, "class") 
cm.M4 = table(M4.pred, test.y); cm.M4
#       test.y
#M4.pred    0    1
#      0 5275 2258
#      1 1794 4811

##### LOGISTIC REGRESSION
M5.model = glm(train.data$Diabetes_binary ~ ., data = train.data, family = binomial(link ="logit"))
summary(M5.model)

M5 = glm(Diabetes_binary ~ HighBP+HighChol+CholCheck+BMI+Stroke+HeartDiseaseorAttack
        +GenHlth+PhysHlth+DiffWalk+Age+Education+Income, data = train.data, family = binomial(link = "logit"))
summary(M5)

M5.pred = predict(M5, newdata = test.x, type = "response")

##### AUC
#### Linear Regression
prob1 = M1.pred
pred1 = prediction(prob1, test.y)
roc1 = performance(pred1 , "tpr", "fpr")
auc1 = performance(pred1 , measure ="auc")
auc1@y.values[[1]] #0.8257816

#### Find the optimal threshold
alpha1 = round (as.numeric(unlist(roc1@alpha.values)) ,4)
fpr1 = round(as.numeric(unlist(roc1@x.values)) ,4)
tpr1 = round(as.numeric(unlist(roc1@y.values)) ,4)
par(mar = c(5 ,5 ,2 ,5))
plot(alpha1 ,tpr1 , main = paste("TPR and FPR of Linear Regression"), xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "cadetblue")
par( new ="True")
plot(alpha1 ,fpr1 , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "pink" )
axis( side =4) # to create an axis at the 4th side
mtext(side =4, line =3, "False positive rate")
text(0.38 ,0.38 , "FPR")
text(0.58 ,0.58 , "TPR")

#### KNN
prob2 = as.numeric(paste(M2.pred))
pred2 = prediction(prob2, test.y)
roc2 = performance(pred2, "tpr", "fpr")
auc2 = performance(pred2 , measure ="auc")
auc2@y.values[[1]] #0.7518744

#### Decision Tree
prob3 = as.numeric(paste(M3.pred))
pred3 = prediction(prob3, test.y)
roc3 = performance(pred3, "tpr", "fpr")
auc3 = performance(pred3 , measure ="auc")
auc3@y.values[[1]] #0.7264111

#### Naive Bayes
prob4 = as.numeric(paste(M4.pred))
pred4 = prediction(prob4, test.y)
roc4 = performance(pred4, "tpr", "fpr")
auc4 = performance(pred4 , measure ="auc")
auc4@y.values[[1]]  #0.7133965

#### Logistic Regresison
prob5 = M5.pred
pred5 = prediction(prob5, test.y)
roc5 = performance(pred5 , "tpr", "fpr")
auc5 = performance(pred5 , measure ="auc")
auc5@y.values[[1]] #0.82636

# Find the optimal threshold
alpha5 = round (as.numeric(unlist(roc5@alpha.values)) ,4)
fpr5 = round(as.numeric(unlist(roc5@x.values)) ,4)
tpr5 = round(as.numeric(unlist(roc5@y.values)) ,4)
par(mar = c(5 ,5 ,2 ,5))
plot(alpha5 ,tpr5 , main = paste("TPR and FPR of Logistic Regression"), xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "cadetblue")
par(new ="True")
plot(alpha5 ,fpr5 , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "pink" )
axis( side =4) 
mtext(side =4, line =3, "False positive rate")
text(0.4 ,0.4 , "FPR")
text(0.6 ,0.6 , "TPR")
# The optimal threshold can be taken as 0.5

# Classified data by using optimal threshold of linear and logistic regression

M1.pred.class = ifelse(M1.pred >= 0.5, 1, 0)
cm.M1 = table(M1.pred.class, test.y); cm.M1
#       test.y
#M1.pred    0    1
#      0 5167 1651
#      1 1902 5418

M5.pred.class = ifelse(M5.pred >= 0.5, 1, 0)
cm.M5 = table(M5.pred.class, test.y); cm.M5
#       test.y
#M5.pred    0    1
#      0 5221 1702
#      1 1848 5367

prob6 = M1.pred.class
pred6 = prediction(prob6, test.y)
roc6 = performance(pred6 , "tpr", "fpr")
auc6 = performance(pred6 , measure ="auc")
auc6@y.values[[1]] #0.7486915

prob7 = M5.pred.class
pred7 = prediction(prob7, test.y)
roc7 = performance(pred7 , "tpr", "fpr")
auc7 = performance(pred7 , measure ="auc")
auc7@y.values[[1]] #0.7489037

# Plotting the ROC
par(mar = c(4 ,4 ,4 ,4))
plot(roc1 , col = "purple", main = paste("ROC Curve and AUC of the Classifiers"), xlab = "False Positive Rate", ylab = "True Positive Rate")
plot(roc2, col = "blue", add = TRUE)
plot(roc3, col = "green", add = TRUE)
plot(roc4 , col = "orange",add = TRUE)
plot(roc5 , col = "red", add = TRUE)
plot(roc6 , col = "brown", add = TRUE)
plot(roc7, col = "black", add = TRUE)

legend("bottomright", c("Linear Regression 0.8258", "KNN 0.7519", "Decision Tree 0.7264", "Naive Bayes 0.7134", "Logistic Regression 0.8264", "Classified Lin R 0.7487","Classified Log R 0.7489"),col=c("purple", "blue", "green", "orange", "red","brown", "black"), lty=1, cex = 0.8)