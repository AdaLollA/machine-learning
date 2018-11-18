library('lattice')

seg <- read.csv(file="C:/FH Hagenberg Master/Machine Learning/ex1/3/segmentationData.csv", header=TRUE, sep=",")

str(seg)
summary(seg)

table(seg$Class)

pairs(seg, col=seg$Class, upper.panel = NULL, pch=".")
boxplot(seg)

dim(seg)

densityplot(seg[,2], groups = seg$Class, plot.points = F) 
densityplot(seg[,3], groups = seg$Class, plot.points = F) 
densityplot(seg[,4], groups = seg$Class, plot.points = F) 
densityplot(seg[,5], groups = seg$Class, plot.points = F) 
densityplot(seg[,6], groups = seg$Class, plot.points = F) 
densityplot(seg[,7], groups = seg$Class, plot.points = F) 
densityplot(seg[,8], groups = seg$Class, plot.points = F) 
densityplot(seg[,9], groups = seg$Class, plot.points = F) 
densityplot(seg[,10], groups = seg$Class, plot.points = F) 
densityplot(seg[,11], groups = seg$Class, plot.points = F) 
densityplot(seg[,12], groups = seg$Class, plot.points = F) 
densityplot(seg[,13], groups = seg$Class, plot.points = F) 
densityplot(seg[,14], groups = seg$Class, plot.points = F) 
densityplot(seg[,15], groups = seg$Class, plot.points = F) 
