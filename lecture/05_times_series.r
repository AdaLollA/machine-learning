########################################################
# TIME SERIES: EXAMPLES
########################################################

# derivative (simplified...)
d <- as.numeric(scale(beaver1[1:50,3]))
plot(d)
lines(d)
derivative1st <- d[-1]-d[-length(d)] # derivative = difference over values
points(derivative1st, col=2, x = 2:length(d)) # 1st derivative
lines(derivative1st, col=2, x = 2:length(d))
derivative2nd <- derivative1st[-1]-derivative1st[-length(derivative1st)]
points(derivative2nd, col=3, x = 3:length(d)) # 2nd derivative
lines(derivative2nd, col=3, x = 3:length(d))

# filtering, e.g. with a runnig median, sg-filter, etc
d <- beaver1[,3]
plot(d, type='l')
d2 <- runmed(d, k = 5)
d3 <- runmed(d, k = 15)
d4 <- runmed(d, k = 31)
matplot(data.frame(d, d2, d3, d4), type='l', lwd=2)
library(signal)
d21 <- sgolayfilt(x = d, p = 1, n = 9)
d22 <- sgolayfilt(x = d, p = 2, n = 9)
d23 <- sgolayfilt(x = d, p = 3, n = 9)
d24 <- sgolayfilt(x = d, p = 3, n = 5)
matplot(data.frame(d, d21, d22, d23, d24), type='l', lwd=2, ylab='Feature dimension', xlab='Time')
# how much filtering? use domain specific knowledge ("do I need this spike or not?")

# interpolation
d <- beaver1[1:20,3]
plot(d, type='p')
d2 <- approx(x = 1:length(d), y = d, xout = seq(1,length(d),0.2), method = 'linear') # generate more samples...
points(x = d2$x, y = d2$y, col=3, pch=4)
d2.2 <- approx(x = 1:length(d), y = d, xout = seq(1,length(d),0.2), method = 'constant')
lines(x = d2.2$x, y = d2.2$y, col=4, pch=4)
d3 <- approx(x = 1:length(d), y = d, xout = seq(1,length(d),2.8)) # ...or less samples
lines(x = d3$x, y = d3$y, col=2, pch=2)
# approx can be used to easily normalize lengths of time series to a defined amount of data points
d4 <- beaver1[1:55,3] # 55 values
d5 <- beaver1[20:40,3] # 20 values
plot(d4, type='l')
lines(d5, col=2)
stepwidth <- 0.01 # we want 100 values
d4.1 <- approx(x = seq(0,1,1/(length(d4)-1)), y = d4, xout = seq(0,1,stepwidth))$y # interpolate to 100 samples
d5.1 <- approx(x = seq(0,1,1/(length(d5)-1)), y = d5, xout = seq(0,1,stepwidth))$y # interpolate to 100 samples
matplot((data.frame(d4.1, d5.1)), type='l')

# very simple sliding window example
library(zoo)
d <- scale(beaver1[,3])
plot(d, type='l')
d2 <- rollapply(data = d, width = 3, FUN = sd) # SD 
d3 <- rollapply(data = d, width = 11, FUN = sd) # SD with shorter window
lines(x = 2:(nrow(d)-1), d2, type='l', col=2)
lines(x = 6:(nrow(d)-5), d3, type='l', col=3)
# rollapply can be applied accross multiple columns at once (careful: applying it over columns is default and might not be what you want)

# autocorrelation ACF
d <- scale(beaver1[,3])
plot(d, type='l')
myAcf <- acf(d, lag.max = length(d), plot = F)
plot(myAcf)
str(myAcf$acf) # features as list

# frequency transformation FFT
d <- as.numeric(scale(beaver1[,3]))
plot(d, type='l')
fft_mod <- Mod(fft(d)) # power
fft_arg <- Arg(fft(d)) # phase
barplot(fft_mod, ylab = 'Frequency power', xlab = 'Frequency', main = 'Frequency power spectrum')
barplot(fft_arg, ylab = 'Frequency phase', xlab = 'Frequency', main = 'Frequency phase spectrum')

# wavelets
library(wmtsa) # there are more wavelet libs
d <- scale(beaver1[,3])
plot(d, type='l')
plot(wavDaubechies('s8')) # this is the 's8' wavelet
myWav <- wavDWT(d, n.levels = 5, wavelet = 's8', keep.series = T) # DWT: discrete wavelet transformation
plot(myWav) # "multi resolution analysis" with this wavelet
str(myWav$data) # ...and these are its obtained features for different levels (=scalings of wavelet)
str(unlist(myWav$data)) # features as simple vector
# wavelets and multi resolution analysis are a very powerful concept: 
#   * they capture time and "frequency" information at once
#   * remember them if you need to go deep into time series analysis!

# DTW
library(dtw) # there are more DTW libs
plot(beaver1[,3], type='l', ylim=c(36,38.5))
lines(beaver2[,3], col=2)
# ?dtw
myDtw <- dtw(beaver1[,3], beaver2[,3], window.type = 'none')
plot(myDtw)
myDtw$distance
myDtw2 <- dtw(beaver1[,3], beaver2[,3], window.type = 'sakoechiba', window.size=15) # band restriction --> more "diagonal", higher cost
plot(myDtw2)
myDtw2$distance 
