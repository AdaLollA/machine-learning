data(diamonds)

#2.3
summary(diamonds$price)
mean(diamonds$price)
median(diamonds$price)
stats::sd(diamonds$price)
stats::mad(diamonds$price)

Q1 <- stats::quantile(diamonds$price, 0.25)[[1]]
Q3 <- stats::quantile(diamonds$price, 0.75)[[1]]
print(Q1)
print(Q3)

iqr <- Q3-Q1
print(iqr)

#2.4
plot(diamonds$carat, diamonds$price, pch='.')
