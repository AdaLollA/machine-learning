library(ggplot2)
data(diamonds)

str(diamonds)

#EX 2.5

#pairwise plotting of the features
pairs(diamonds[,7:10], col=diamonds$color, upper.panel = NULL, pch=".")

#plot shows logarithmic corellation between price and features x, y, z (bigger dimensions equal a higher price)
#plot alsow shows linear corellation between the features x and y, x and z, as well as y and z

#plotting price and each dimension
plot(diamonds[,c('price', 'x')], col=diamonds$color, pch=".")
plot(diamonds[,c('price', 'y')], col=diamonds$color, pch=".")
plot(diamonds[,c('price', 'z')], col=diamonds$color, pch=".")

#EX 2.6
boxplot(diamonds$price~diamonds$color)

#both plots shows that most of the diamonds in each color category have a price range between 0 and <= 5000 
featurePlot(x = diamonds$price, y = diamonds$color, col=diamonds$cut, scales=list(relation='free'), plot = 'density', plot.points=F) # ... with densities without points
