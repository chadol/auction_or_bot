library(dplyr)
library(data.table)
library(xgboost)
library(reshape2)

setwd("D:/facebook data")

train = read.csv("train.csv", header = TRUE)
test = read.csv("test.csv", header = TRUE)
bids = read.csv("bids.csv", header = TRUE)
bids = select(bids, - bid_id, -merchandise)

# Number of unique levels per variable
apply(bids[,2:8],2,function(x){length(table(x))})

### Unique levels per variable for each bideer ###
temp = distinct(select(bids, bidder_id, ip_num))
ip_count = summarize(group_by(temp, bidder_id), ip_count = n() )
temp = distinct(select(bids, bidder_id, device))
dev_count = summarize(group_by(temp, bidder_id), dev_count = n() )
temp = distinct(select(bids ,bidder_id, auction))
auc_count = summarize(group_by(temp, bidder_id), auc_count = n() )
temp = distinct(select(bids, bidder_id, country))
cou_count = summarize(group_by(temp, bidder_id), cou_count = n() )
temp = distinct(select(bids, bidder_id, url_num))
url_count = summarize(group_by(temp, bidder_id), url_count = n() )
bid_count = summarize(group_by(bids, bidder_id), bid_count = n() )

# Merge all
feature1 = merge(merge(merge(merge(merge(merge(
  ip_count,dev_count, by="bidder_id")
  , auc_count, by="bidder_id")
  , cou_count, by="bidder_id")
  , url_count, by="bidder_id")
  , bid_count, by="bidder_id")
  , merch, by='bidder_id')

### Unique levels per variable for each bidder ###
temp = distinct(select(bids, bidder_id, ip))
ip_count = summarize(group_by(temp, bidder_id), ip_count = n())
temp = distinct(select(bids, bidder_id, device))
dev_count = summarize(group_by(temp, bidder_id), dev_count = n())
temp = distinct(select(bids ,bidder_id, auction))
auc_count = summarize(group_by(temp, bidder_id), auc_count = n())
temp = distinct(select(bids, bidder_id, country))
cou_count = summarize(group_by(temp, bidder_id), cou_count = n())
temp = distinct(select(bids, bidder_id, url))
url_count = summarize(group_by(temp, bidder_id), url_count = n())
bid_count = summarize(group_by(bids, bidder_id), bid_count = n())

# Merge all
feature1 = merge(merge(merge(merge(merge(
               ip_count,dev_count, by="bidder_id")
               , auc_count, by="bidder_id")
               , cou_count, by="bidder_id")
               , url_count, by="bidder_id")
               , bid_count, by="bidder_id")

### Bid time characteristics  ###
# Split data into 3 datasets with disjoint time spans
data1 = filter(bids, time<9.68e+15)
data2 = filter(bids, time>9.68e+15, time<9.74e+15)
data3 = filter(bids, time>9.74e+15)

time1 = summarize(group_by(data1, bidder_id), 
                  mintime = min(time),
                  maxtime = max(time) )
time1$tottime = time1$maxtime - time1$mintime
time2 = summarize(group_by(data2, bidder_id), 
                  mintime = min(time),
                  maxtime = max(time) )
time2$tottime = time2$maxtime - time2$mintime
time3 = summarize(group_by(data3, bidder_id), 
                  mintime = min(time),
                  maxtime = max(time) )
time3$tottime = time3$maxtime - time3$mintime

time = rbind(time1, time2, time3)
rm(time1, time2, time3)

temp = summarize(group_by(time, bidder_id), tottime = sum(tottime))

feature2 = merge(feature1, temp, by="bidder_id", all.x=T)
feature2$bid_density = feature2$bid_count/feature2$tottime
feature2$bid_density[is.infinite(feature2$bid_density)] = median(feature2$bid_density)

# Features computed from characteristics between 2 bids
bids = bids[order(bids$bidder_id, bids$time),]

bids$lag_id = lag(bids$bidder_id, k=1)
bids$lag_time = lag(bids$time, k=1)
bids$time_diff = bids$time - bids$lag_time
temp = filter(bids, bidder_id == lag_id)
temp2 = summarize(group_by(temp, bidder_id),                  
                  max_time_diff = max(time_diff),
                  q3_time_diff = quantile(time_diff, probs=.75),
                  q1_time_diff = quantile(time_diff, probs=.25),
                  med_time_diff = median(time_diff), #median time between bids
                  sd_time_diff = sd(time_diff))   # variance of between bid time

# number of fast bids (within 0 seconds, 4 seconds, 8 seconds)
temp3 = filter(temp, time_diff == 0 )
temp4 = summarize(group_by(temp3, bidder_id), n_zero_bid = n())
temp3 = filter(temp, time_diff<(53031366*4), time_diff!=0)
temp5 = summarize(group_by(temp3, bidder_id), n_four_bid = n())
temp3 = filter(temp, time_diff<(53031366*8), time_diff>=(53031366*4))
temp6 = summarize(group_by(temp3, bidder_id), n_eight_bid = n())
temp7 = merge(merge(temp4, temp5, by='bidder_id', all.x=T, all.y=T), 
              temp6, by='bidder_id', all.x=T, all.y=T)
temp8 = merge(temp2, temp7, by='bidder_id', all.x=T, all.y=T)

feature2 = merge(feature2, temp8, by="bidder_id", all.x=T)

feature2$n_zero_bid[is.na(feature2$n_zero_bid)] = 0
feature2$n_four_bid[is.na(feature2$n_four_bid)] = 0
feature2$n_eight_bid[is.na(feature2$n_eight_bid)] = 0
feature2$max_time_diff[is.na(feature2$max_time_diff)] = median(feature2$max_time_diff, na.rm=T)
feature2$med_time_diff[is.na(feature2$med_time_diff)] = median(feature2$med_time_diff, na.rm=T)
feature2$sd_time_diff[is.na(feature2$sd_time_diff)] = median(feature2$sd_time_diff, na.rm=T)
feature2$q1_time_diff[is.na(feature2$q1_time_diff)] = median(feature2$q1_time_diff, na.rm=T)
feature2$q3_time_diff[is.na(feature2$q3_time_diff)] = median(feature2$q3_time_diff, na.rm=T)

### Switching ###
# Switching proportion
temp = bids  
temp$lag_auc = lag(temp$auction, k=1)
temp$lag_dev = lag(temp$device, k=1)
temp$lag_country = lag(temp$country, k=1)
temp$lag_ip = lag(temp$ip_num, k=1)

temp$sw_auc[temp$auc == temp$lag_auc] = 0
temp$sw_auc[temp$auc != temp$lag_auc] = 1
temp$sw_dev[temp$dev == temp$lag_dev] = 0
temp$sw_dev[temp$dev != temp$lag_dev] = 1
temp$sw_country[temp$country == temp$lag_country] = 0
temp$sw_country[temp$country != temp$lag_country] = 1
temp$sw_ip[temp$ip == temp$lag_ip] = 0
temp$sw_ip[temp$ip != temp$lag_ip] = 1

temp2 = filter(temp, bidder_id == lag_id, time_diff < 5e+13 ) #delete observations that crosses different time datasets
temp3 = summarize(group_by(temp2, bidder_id),
                  sw_auc = mean(sw_auc),
                  sw_dev = mean(sw_dev),
                  sw_country = mean(sw_country),
                  sw_ip = mean(sw_ip)
)

feature3 = merge(feature2, temp3, by="bidder_id", all.x=T)
feature3[is.na(feature3)] = 0

# Only use switching variables for id's with at least 10 bids, set everything else to 0
feature3[feature3$bid_count<10, 14:17] = 0

# Median time between switching
temp4 = filter(temp2, temp2$auc!=temp2$lag_auc)
temp5 = summarize(group_by(temp4, bidder_id), med_sw_auc = median(time_diff))
temp4 = filter(temp2, temp2$dev!=temp2$lag_dev)
temp6 = summarize(group_by(temp4, bidder_id), med_sw_dev = median(time_diff))
temp4 = filter(temp2, temp2$country!=temp2$lag_country)
temp7 = summarize(group_by(temp4, bidder_id), med_sw_country = median(time_diff))
temp4 = filter(temp2, temp2$ip!=temp2$lag_ip)
temp8 = summarize(group_by(temp4, bidder_id), med_sw_ip = median(time_diff))
temp9 = merge(merge(merge(temp5, temp6, by="bidder_id", all=T), temp7, by="bidder_id", all=T), temp8, by="bidder_id", all=T)

feature3 = merge(feature3, temp9, by="bidder_id", all.x=T)
feature3[is.na(feature3)] = median(feature3$med_time_diff)

### Auction characteristics ###
# Number of auctions won 
temp = bids[order(bids$auction, -bids$time),] # sort for each auction, last bid will be first obs
temp$lag_auc = lag(temp$auction, k=1)
temp2 = rbind(temp[1,], # first obs of first auction
              filter(temp, auction != lag_auc))
temp3 = summarize(group_by(bids, auction),
                  max_time = max(time),
                  bid_count = n())
t_buffer = 52631366 * 3600 * 3
temp4 = filter(temp3, bid_count>500 & (max_time<(9.65E+15-t_buffer) |
                                         (max_time>9.69E+15 & max_time<9.71E+15-t_buffer) |
                                         (max_time>9.75E+15 & max_time<9.77E+15-t_buffer)))
temp5 = merge(temp2, temp4, by='auction')
temp6 = summarize(group_by(temp5, bidder_id),
                  win_count = n())
feature4 = merge(feature3, temp6, by="bidder_id", all.x=T)
feature4$win_count[is.na(feature4$win_count)] = 0
feature4 = mutate(feature4, win_density_bids = win_count/bid_count)

### Bid time after last bid in auction ###
temp = select(temp, auction, time, bidder_id)
temp = temp[order(temp$auction, temp$time),] # sort for each auction
temp$lagtime = lag(temp$time, k=1)
temp$lagauc = lag(temp$auction, k=1)
temp$time_diff = temp$time - temp$lagtime
temp = filter(temp, auction == lagauc)
temp2 = summarize(group_by(temp, bidder_id), 
                  min_time_diff_auc = min(time_diff),
                  med_time_diff_auc = median(time_diff),
                  var_time_diff_auc = var(time_diff)
)

# number of 0 second/ 1 second bids
temp3 = filter(temp, time_diff == 0 )
temp4 = summarize(group_by(temp3, bidder_id),
                  n_zero_bid_auc = n())
temp3 = filter(temp, time_diff<(53031366*4), time_diff!=0)
temp5 = summarize(group_by(temp3, bidder_id),
                  n_four_bid_auc = n())
temp3 = filter(temp, time_diff<(53031366*8), time_diff!=0)
temp6 = summarize(group_by(temp3, bidder_id),
                  n_eight_bid_auc = n())
temp7 = merge(merge(temp4, temp5, by='bidder_id', all.x=T, all.y=T), 
              temp6, by='bidder_id', all.x=T, all.y=T)
temp8 = merge(temp2, temp7, by='bidder_id', all.x=T, all.y=T)

feature5 = merge(feature4, temp8, by="bidder_id", all.x=T)
feature5$var_time_diff_auc[is.na(feature5$var_time_diff_auc)] = median(feature5$var_time_diff_auc, na.rm=T)
feature5$min_time_diff_auc[is.na(feature5$min_time_diff_auc)] = median(feature5$min_time_diff_auc, na.rm=T)
feature5$med_time_diff_auc[is.na(feature5$med_time_diff_auc)] = median(feature5$med_time_diff_auc, na.rm=T)
feature5$n_zero_bid_auc[is.na(feature5$n_zero_bid_auc)] = 0
feature5$n_four_bid_auc[is.na(feature5$n_four_bid_auc)] = 0
feature5$n_eight_bid_auc[is.na(feature5$n_eight_bid_auc)] = 0

### Country fraction dummies ###
sort(table(bids$country)/nrow(bids), decreasing=T)[1:5]
temp = bids
temp$c_dummy[!(temp$country %in% c('in', 'ng', 'id', 'tr', 'us'))] = 'other'
temp$c_dummy[is.na(temp$c_dummy)] = temp$country[is.na(temp$c_dummy)]
temp2 = summarize(group_by(temp, bidder_id, c_dummy), c_count = n())
temp3 = dcast(temp2, bidder_id ~ c_dummy, value.var="c_count")
temp4 = select(feature5, bidder_id, bid_count)
temp5 = merge(temp3, temp4, by="bidder_id")
temp5[is.na(temp5)] = 0

# country fraction
temp5$cfrac1 = temp5[["134"]]/temp5$bid_count
temp5$cfrac2 = temp5[["181"]]/temp5$bid_count
temp5$cfrac3 = temp5[["188"]]/temp5$bid_count
temp5$cfrac4 = temp5[["83"]]/temp5$bid_count
temp5$cfrac5 = temp5[["86"]]/temp5$bid_count

temp5 = select(temp5, bidder_id, cfrac1:cfrac5)
feature6 = merge(feature5, temp5, by="bidder_id", all.x=T)

# merge with training and test data
train2 = merge(train, id_unique, by="bidder_id")
train2 = select(train2, bidder_id, outcome, bidder_id)
test2 = merge(test, id_unique, by="bidder_id", all.x=T)
test2 = select(test2, bidder_id, bidder_id)
train_feat = merge(train2, feature6, by="bidder_id")
test_feat = merge(test2, feature6, by="bidder_id", all.x=T)

### test extra rows (need to add due to error in original data) ###
test_extra = test_feat$bidder_id[is.na(test_feat$ip_count)]
test_extra = data.frame(test_extra, 0)
names(test_extra) = c("bidder_id", "prediction")

test_feat_id = test_feat$bidder_id[!is.na(test_feat$ip_count)]
test_feat = filter(test_feat, !is.na(bid_count))



### Random Forest ###

# Random Forest

library(randomForest)
set.seed(1)

numtree = 5000

time=Sys.time()
rf.fit = randomForest(factor(outcome) ~ . -bidder_id-bidder_id-mintime-maxtime-tottime, 
                      data=train_feat, mtry=3, ntree=numtree)
time2=Sys.time()-time

oob.err=rf.fit$err.rate[numtree,]

time=Sys.time()
oob.err=matrix(0, nrow=numtree, ncol=10)
for (i in 1:10){ 
  rf.fit = randomForest(factor(outcome) ~ . -bidder_id-bidder_id, 
                        data=train_feat, mtry=(1+i), ntree=numtree)
  oob.err[,i]=rf.fit$err.rate[,1]  
  assign(paste('rf.fit',i,sep=''), rf.fit)
}
time2=Sys.time()-time

plot(oob.err[numtree,])
plot(oob.err[,5])
oob.err[3000,6]

rf.fitp = predict(rf.fit5, test_feat, type='prob')
pred_sub = data.frame(test_feat_id, rf.fitp[,2])
names(pred_sub) = c("bidder_id", "prediction")

pred_sub = rbind(pred_sub, test_extra)


write.csv(pred_sub, "rf_sub.csv", row.names=F)

blah = cbind(X_test,pred)
