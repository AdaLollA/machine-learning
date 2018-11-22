library(plyr)
library(png)
library(caret)
library(corrplot)
library(doParallel)
registerDoParallel(4)


facesPaths <- dir("./faces_data", full.names=T, pattern='png')

faces <-ldply(facesPaths, function(f) { as.numeric(readPNG(f)) })
classes <-laply(facesPaths, function(f) { paste("P", substr(f, 18, 19), sep="") })  #prepend a 'P' so the classnames are valid R vairable names

faces$class <- as.factor(classes)

dim(faces)
str(faces$class)

#plot first face
plot(1:1, type='n', main=faces$class[1],xlim=c(1,50),ylim=c(1,50))
rasterImage(matrix(as.numeric(faces[1,-2501]), 50,50),0,0,50,50)

#correlation of first (top) 100 pixels
face00TopPixels = faces[,1:99]
#correlation of 100 pixels from the middle
face00CenterPixels = faces[,1200:1299]

corrplot(cor(face00TopPixels), 'circle', tl.cex = 0.3)
corrplot(cor(face00CenterPixels), 'circle', tl.cex = 0.3)

indexes_train <- createDataPartition(faces$class, p = 0.75, list = F) # indexes of training samples
indexes_test <- (1:nrow(faces))[-indexes_train] # use all samples not in train as test samples

training <- faces[indexes_train,]
testing <- faces[indexes_test,]

trControl <- trainControl(method = 'repeatedcv', 
                          number = 10, 
                          repeats = 20, 
                          returnData = F, 
                          classProbs = T, 
                          returnResamp = 'final', 
                          allowParallel = T,
                          preProcOptions = list(thresh = 0.8)
                          )

models <- list()
models$modelKnn <- train(x = training[,-ncol(training)],
                         y = training$class,
                         preProcess = NULL,
                         method = 'knn',
                         tuneGrid = expand.grid(k=1:10),
                         metric = 'Kappa',
                         trControl = trControl)

plot(models$modelKnn)


#training confusion matrix
trainPredicted <- predict(models$modelKnn, newdata = training[,-ncol(training)])
trainConfMatrix <- confusionMatrix(data = trainPredicted, reference = training$class) # create confusion matrix over training data = apparent error
trainConfMatrix
levelplot(sweep(x = trainConfMatrix$table, STATS = colSums(trainConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

#test confusion matrix
testPredicted <- predict(models$modelKnn, newdata = testing[,-ncol(testing)])
testConfMatrix <- confusionMatrix(data = testPredicted, reference = testing$class)
testConfMatrix
levelplot(sweep(x = testConfMatrix$table, STATS = colSums(testConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))


cvConfMatrix <- confusionMatrix(models$modelKnn)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))



models$modelLda <- train(x = training[,-ncol(training)],
                         y = training$class, 
                         preProcess = c('pca'), 
                         method = 'lda', 
                         tuneGrid = NULL, 
                         metric = 'Kappa', 
                         trControl = trControl)

models$modelLda$preProcess


models$modelSvmRadial<- train(x = training[,-ncol(training)],
                              y = training$class, 
                              preProcess = c('pca'), 
                              method = 'svmRadial', 
                              tuneGrid = expand.grid(sigma=3**(-3:3), C=3**(-3:3)),
                              metric = 'Kappa',
                              trControl = trControl)

results <- resamples(models)
summary(results)
bwplot(results)

