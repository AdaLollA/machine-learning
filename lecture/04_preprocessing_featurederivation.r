########################################################
# DATA NORMALIZATION
########################################################

library(caret)
library(plyr)
myData <- iris[,1:4]
boxplot(myData)
# scale features
myDataScaled <- scale(myData)
boxplot(myDataScaled)
# mean and sd are 0 and 1 for each feature accross all samples
colMeans(myDataScaled)
aaply(.data = myDataScaled, .margins = 2, .fun = sd)

# box and cox
# -- there are better, ready made boxcox functions in some packages
boxcox <- function(l, lambda) {
  if(lambda == 0) {
    log(l)
  } else {
    (l**lambda - 1) / lambda
  }
}
plot(1:100)
plot(boxcox(1:100, 1))
plot(boxcox(1:100, 2))
plot(boxcox(1:100, 0.5))
plot(boxcox(1:100, 0))
plot(boxcox(1:100, -0.5))
plot(boxcox(1:100, -2))

########################################################
# DATA PREPROCESSING: EXAMPLES
########################################################

library(corrplot)
library(caret)
library(doMC)
registerDoMC(3)
data(segmentationData)

# feature correlation as plot
corrplot(cor(iris[,1:4]), 'circle') # addgrid.col = NA
corrplot(cor(segmentationData[,-(1:3)]), tl.cex = 0.3, type = "upper") # addgrid.col = NA
# remove correlated variable using ?findCorrelation
foundCorIndexes <- findCorrelation(cor(segmentationData[,-(1:3)]), cutoff = 0.7)
foundCorIndexes
corrplot(cor(segmentationData[,-(1:3)][,-foundCorIndexes]), tl.cex = 0.3, type = "upper")

# PCA example
d <- segmentationData[,-(1:3)]
pcs <- prcomp(x = as.matrix(d), retx = T, center = T, scale. = T, tol = 0.0) # tol --> increase to leave out less important dimensions
names(pcs) # sdev: strength of PCs, rotation: translation from original space to PC space and vice versa, x: our data in PC space
plot(pcs$sdev) # strength of PCs
dim(d) # original dimensions
dim(pcs$rotation) # matrix giving us the possibility of transforming other data into our PC space - prob. less dimensions if tol > 0
str(pcs$center) # shift applied to original data to bring mean to 1
str(pcs$scale) # scale applied do original data to bring sd to 1
dim(data.frame(pcs$x)) # our data in PC space
str(data.frame(pcs$x)) # our data in PC space
# we can do the same transformation by hand
pcs_x <- scale(d, center = pcs$center, scale = pcs$scale) %*% pcs$rotation # %*% --> matrix multiplication
all(abs(pcs_x - pcs$x) < 0.00001) # some tolerance due to inaccuracy from dimensionality reduction...
# we can reverse the transformation by hand too
reversed <- scale(pcs$x %*% t(pcs$rotation), scale = 1/pcs$scale, center= -pcs$center/pcs$scale)
all(abs(d - reversed) < 0.00001) # some tolerance due to inaccuracy from dimensionality reduction...

# PCA with caret
str(mtcars[,-2]) # 10 features + 1 target variable (cyl)
model <- train(mtcars[,-2], factor(mtcars[,2]), method = 'knn', 
               preProcess = c('center', 'scale', 'pca'), # center + scale + PCA
               metric = 'Kappa', trControl = trainControl(method = 'LOOCV', preProcOptions = list(thresh = 0.9))) # set the pca threshold level to 0.9
model$preProcess # only need 4 features --> faster

# caret provides different preprocessing options
#   see: https://www.rdocumentation.org/packages/caret/versions/6.0-77/topics/preProcess
#   like: "BoxCox", "YeoJohnson", "expoTrans", "center", "scale", 
#   "range", "knnImpute", "bagImpute", "medianImpute", "pca", "ica", 
#   "spatialSign", "corr", "zv", "nzv", and "conditionalX"

########################################################
# FEATURE SELECTION
########################################################

# feature selection: feature filters and feature wrappers in caret
# http://topepo.github.io/caret/featureselection.html

# mtcars: cyl is the class label
d <- mtcars
d$cyl <- factor(d$cyl)
str(d)

# minimal example: feature filter using univariate filters
sbfRes <- sbf(x = d[,-2], y = d$cyl, sbfControl = sbfControl(functions = rfSBF, method = 'repeatedcv', repeats = 5)) # more repeats are better
sbfRes
sbfRes$optVariables # all 10 features

# minimal example: feature wrapper using recursive feature elimination
rfeRes <- rfe(x = d[,-2], y = d$cyl, rfeControl = rfeControl(functions = rfFuncs, method = 'repeatedcv', repeats = 5)) # more repeats are better
rfeRes
rfeRes$optVariables # 8/10 features

########################################################
# FEATURE IMPORTANCE FROM MODEL
########################################################

library(caret)

# mtcars: cyl is the class label
d <- mtcars
d$cyl <- factor(d$cyl)
str(d)

model <- train(x = d[,-2], y = d$cyl, method = 'rpart', metric = 'Kappa', 
               trControl = trainControl(method = 'repeatedcv', repeats = 10), 
               tuneGrid = expand.grid(cp=2**(-7:0)))
model
# after being trained, some models give indications about what they think are the features' importances
varImp(model)

