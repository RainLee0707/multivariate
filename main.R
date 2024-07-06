library(tidyverse)
library(ggplot2)
library(corrplot)
library(factoextra)
library(FactoMineR)
library(psych)
library(cluster)
library(caret)
library(MASS)
library(plotly)

setwd("C:/Users/Rain/Desktop/多變量分析")
data <- read.csv('data_no_land1.csv')

main_col <- c('總價元', '土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', 
              '建物移轉總面積平方公尺', '建物現況格局.房', '建物現況格局.廳', 
              '建物現況格局.衛', '單價元平方公尺', '車位移轉總面積平方公尺', 
              '車位總價元', '主建物面積', '附屬建物面積', '陽台面積')

data1 <- data[main_col]



# PCA 
pca_result <- PCA(data1, graph = FALSE)
# 繪製累積方差圖
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100))

# 提取 PC1  PC2  PC3
PC1 <- pca_result$ind$coord[, 1]
PC2 <- pca_result$ind$coord[, 2]
PC3 <- pca_result$ind$coord[, 3]  
# 繪製散佈圖
plot(PC1, PC2, 
     xlab = "Principal Component 1", 
     ylab = "Principal Component 2",
     main = "PCA: PC1 vs PC2 Scatter Plot")

#因素分析
fa_result <- psych::fa(data1, nfactors = 3, rotate = "none")
print(fa_result)
# 繪製因子負荷圖
fa.diagram(fa_result)


# 階層式
hc_result <- hclust(dist(data1), method = "ward.D2")
# 繪製樹狀圖
plot(hc_result, hang = -1, cex = 0.6)
rect.hclust(hc_result, k = 2, border = 2:4)

# K-means 
kmeans_result <- kmeans(data1, centers = 2, nstart = 1)
# 將 K-means 結果繪製在 PC1, PC2 和 PC3 上
plot_ly(x = PC1, y = PC2, z = PC3, color = factor(kmeans_result$cluster)) %>%
  add_markers(marker = list(size = 5, opacity = 0.8)) %>%
  layout(scene = list(
    xaxis = list(title = "PC1"),
    yaxis = list(title = "PC2"),
    zaxis = list(title = "PC3")
  ),
  legend = list(title = list(text = "Clusters"))) %>%
  layout(title = "K-means Clustering on PC1, PC2, and PC3")



# 線性判別分析 (LDA)
data$Cluster <- factor(data$Cluster)
set.seed(42)
train_index <- createDataPartition(data$Cluster, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
lda_model <- lda(Cluster ~ ., data = train_data)
lda_pred <- predict(lda_model, newdata = test_data)
confusionMatrix(lda_pred$class, test_data$Cluster)

#QDA
set.seed(42)
train_index <- createDataPartition(data$Cluster, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
qda_model <- qda(Cluster ~ ., data = train_data)
qda_pred <- predict(qda_model, newdata = test_data)
confusionMatrix(qda_pred$class, test_data$Cluster)

