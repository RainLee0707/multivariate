import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from scipy import stats
from matplotlib.patches import Ellipse
from scipy.stats import ortho_group

matplotlib.rc('font', family='Microsoft JhengHei')

class DataProcessing:
    @staticmethod
    def chineseNumber2Int(strNum: str):
        if isinstance(strNum, float):
            return None  # 或者選擇一個適當的缺失值處理方式

        if "地下" in strNum:
            return -1
        result = 0
        temp = 1  
        count = 0  
        cnArr = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
        chArr = ['十', '百', '千', '萬', '億']
        for i in range(len(strNum)):
            b = True
            c = strNum[i]
            for j in range(len(cnArr)):
                if c == cnArr[j]:
                    if count != 0:
                        result += temp
                        count = 0
                    temp = j + 1
                    b = False
                    break
            if b:
                for j in range(len(chArr)):
                    if c == chArr[j]:
                        if j == 0:
                            temp *= 10
                        elif j == 1:
                            temp *= 100
                        elif j == 2:
                            temp *= 1000
                        elif j == 3:
                            temp *= 10000
                        elif j == 4:
                            temp *= 100000000
                    count += 1
            if i == len(strNum) - 1:
                result += temp
        return result

    @staticmethod
    def remove_last_four_digits(value):
        if pd.isna(value):
            return value  
        str_value = str(int(value))
        modified_value = str_value[:-4]
        return float(modified_value)

    @staticmethod
    def custom_replace(x):
        if pd.notnull(x):  
            return DataProcessing.chineseNumber2Int(str(x).replace("層", ""))
        else:
            return None
    def col(data):
        data['總樓層數'] = data['總樓層數'].apply(DataProcessing.custom_replace)
        data['建築完成年月日'] = data['建築完成年月日'].apply(DataProcessing.remove_last_four_digits)
        main_col = ['總價元','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積']
        data = data[main_col]
        data = data.fillna(data.median())
        
class Statistic:
    def __init__(self, data):
        self.data = data
        
    def describe_dataframe(self):
        stats = {}
        for col in self.data.columns:
            stats[col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'max': self.data[col].max(),
                'min': self.data[col].min(),
                'count': self.data[col].count()
            }
        return stats
    def count_zeros(self, columns):
        zero_counts = {}
        for column in columns:
            if column in self.data.columns:
                zero_counts[column] = (self.data[column] == 0).sum()
            else:
                zero_counts[column] = None
        return zero_counts
    
    def print_zero_counts(self, columns):
        zero_counts = self.count_zeros(columns)
        for column, count in zero_counts.items():
            if count is not None:
                print(f"{column} has {count} zeros.")
            else:
                print(f"{column} does not exist in the DataFrame.")

    def count_transaction_types(self):
        if '交易標的' in self.data.columns:
            transaction_counts = self.data['交易標的'].value_counts()
            return transaction_counts
        else:
            print("The column '交易標的' does not exist in the DataFrame.")
            return None

    def print_transaction_counts(self):
        transaction_counts = self.count_transaction_types()
        if transaction_counts is not None:
            for transaction_type, count in transaction_counts.items():
                print(f"{transaction_type}: {count} records")

class EDA:
    def __init__(self, data):
        self.data = data

    def plot_barplots(self):
        data = self.data[['總樓層數', '建築完成年月日', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛']]
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

        num_rows1 = 3
        num_cols1 = 2

        fig1, axes1 = plt.subplots(num_rows1, num_cols1, figsize=(15, 15))
        axes1 = axes1.ravel()

        for i, column in enumerate(numeric_columns):
            if i < len(axes1):
                value_counts = data[column].value_counts().sort_index()
                min_val = value_counts.index.min()
                max_val = value_counts.index.max()
                median_val = data[column].median()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes1[i])
                axes1[i].set_ylabel('Count')

                # Ensure min, median, and max are in the sorted index for plotting
                unique_values = value_counts.index.to_list()
                if median_val not in unique_values:
                    unique_values.append(median_val)
                    unique_values.sort()
                
                min_pos = unique_values.index(min_val)
                max_pos = unique_values.index(max_val)
                median_pos = unique_values.index(median_val)

                axes1[i].set_xticks([min_pos, median_pos, max_pos])  # Set x-axis ticks to positions
                axes1[i].set_xticklabels([min_val, median_val, max_val])  # Set x-axis labels to min, median, and max values

        plt.tight_layout()
        plt.show()
        
    def plot_lineplots(self):
        data = self.data[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元', '主建物面積', '附屬建物面積', '陽台面積']]
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

        num_rows = 4
        num_cols = 2

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 15))
        axes = axes.ravel()

        for i, column in enumerate(numeric_columns):
            if i < len(axes):
                sns.histplot(data[column], ax=axes[i])
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Count')

        plt.tight_layout()
        plt.show()
        # for i, column in enumerate(numeric_columns):
        #     row = i // 2
        #     col = i % 2
        #     value_counts = data[column].value_counts().sort_index()
        #     sns.lineplot(x=value_counts.index, y=value_counts.values, ax=axes[row, col])
        #     axes[row, col].set_xlabel(f'{column}')
        #     axes[row, col].set_ylabel('Count')
        #     axes[row, col].set_xticklabels([])  # Remove x-axis labels

        #     # # 只顯示四分位數x軸標籤
        #     # quartiles = [0, len(value_counts) // 4, len(value_counts) // 2, 3 * len(value_counts) // 4, len(value_counts) - 1]
        #     # labels = [str(idx) if idx in quartiles else '' for idx in range(len(value_counts))]
        #     # axes[row, col].set_xticks(value_counts.index[quartiles])
        #     # axes[row, col].set_xticklabels(value_counts.index[quartiles])

        # # 調整子圖布局
        # plt.tight_layout()
        # plt.show()

    def plot_histogram(self, column):
        """
        繪製指定欄位的直方圖
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column])
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_distribution(self, column):
        """
        繪製指定欄位的分配折線圖
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True) 
        plt.title(f'Distribution Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    def plot_boxplot(self, column):
        """
        繪製指定欄位的盒狀圖
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=self.data[column])
        plt.title(f'Boxplot of {column}')
        plt.ylabel(column)
        plt.show()

    def plot_scatter(self, column1, column2):
        """
        繪製兩個指定欄位之間的散點圖
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data[column1], y=self.data[column2])
        plt.title(f'Scatter Plot of {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.show()

    def plot_pairplot(self, columns):
        """
        繪製多個欄位之間的配對圖
        """
        plt.figure(figsize=(10, 6))
        sns.pairplot(self.data[columns])
        plt.title('Pairplot')
        plt.show()

    def plot_corr(self):
        # # Histogram
        # data= self.data
        # sns.histplot(data['土地移轉總面積平方公尺'])
        # plt.title('Histogram of 土地移轉總面積平方公尺')
        # plt.show()

        # # Box Plot
        # sns.boxplot(x='建物現況格局-房', y='總價元', data=data)
        # plt.title('Box Plot of 總價元 grouped by 建物現況格局-房')
        # plt.show()

        # # Pair Plot
        # sns.pairplot(data[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺', '車位移轉總面積平方公尺', '主建物面積']])
        # plt.title('Pair Plot')
        # plt.show()

        # Correlation Heatmap
        corr = self.data[['總價元','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.xticks(rotation=30)
        plt.title('Correlation Matrix Heatmap')
        plt.show()


        # numeric_vars = ['建築完成年月日', '土地移轉總面積平方公尺', '建物移轉總面積平方公尺', 
        #                 '車位移轉總面積平方公尺', '主建物面積', '附屬建物面積', '陽台面積']
        
        # for var in numeric_vars:
        #     plt.figure(figsize=(8, 6))
        #     sns.boxplot(x='有無管理組織', y=var, data=self.data)
        #     plt.title(f'Boxplot of {var} by 有無管理組織')
        #     plt.xlabel('有無管理組織')
        #     plt.ylabel(var)
        #     plt.xticks(rotation=0)
        #     plt.show()

    def plot_relationship(self, target_variable):
        numeric_vars = ['土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', 
                        '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', 
                        '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元', '主建物面積', '附屬建物面積', 
                        '陽台面積']
        
        for var in numeric_vars:
            if var != target_variable:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=var, y=target_variable, data=self.data)
                plt.title(f'{var} vs {target_variable}')
                plt.xlabel(var)
                plt.ylabel(target_variable)
                plt.show()

    def plot_relationship_categorical(self, target_variable):
        categorical_vars = ['交易標的', '建物型態']
        
        for var in categorical_vars:
            if var != target_variable:
                plt.figure(figsize=(8, 6))
                sns.countplot(x=var, hue=target_variable, data=self.data)
                plt.title(f'{var} vs {target_variable}')
                plt.xlabel(var)
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                plt.legend(title=target_variable)
                plt.show()

class PCAnalysis:
    def __init__(self, data):
        self.data = data
        self.pca, self.pca_df = self.perform_pca()

    def perform_pca(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        pca = PCA()  # 如果要限制主成分數量，可以在這裡指定n_components
        principal_components = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca.components_, columns=self.data.columns)
        print(pca_df.head(6).transpose())
        print("\n")
        # 找出explained variance大於1的主成分的索引
        lambda_gt_1_indices = np.where(pca.explained_variance_ > 1)[0]
        lambda_values = pca.explained_variance_[lambda_gt_1_indices]
        print("主成分的特徵值大於1的有：", lambda_gt_1_indices+1)
        print(lambda_values)
        return pca, pca_df
    
    def plot_biplot(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        principal_components = self.pca.transform(scaled_data)

        plt.figure(figsize=(12, 10))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                    edgecolor='k', s=30, alpha=0.5)
        
        scale_factor = np.max([np.abs(principal_components[:, 0]).max(), 
                            np.abs(principal_components[:, 1]).max()]) / 0.7
        
        for i, var in enumerate(self.data.columns):
            x = self.pca.components_[0, i] * scale_factor
            y = self.pca.components_[1, i] * scale_factor
            plt.arrow(0, 0, x, y, color='r', alpha=0.5, head_width=0.05, head_length=0.05)
            plt.text(x*1.2, y*1.2, var, color='g', ha='center', va='center')

        plt.xlabel(f'Principal Component 1 ({self.pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'Principal Component 2 ({self.pca.explained_variance_ratio_[1]*100:.2f}%)')
        plt.title('Biplot')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        
        plt.tight_layout()
        plt.show()

    def plot_cumulative_variance(self):
        explained_variance_ratio_cumsum = self.pca.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o', linestyle='-')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio by Number of Principal Components')
        plt.grid(True)
        plt.show()

    def plot_individual_variance(self):
        explained_variance_ratio = self.pca.explained_variance_ratio_
        for i, ratio in enumerate(explained_variance_ratio):
            print("第{}個主成分：{}%".format(i+1, ratio * 100))
        print("\n")
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, align='center')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')
        plt.xticks(range(1, len(explained_variance_ratio) + 1))
        plt.grid(True)
        plt.show()
    
    def plot_sample_projection(self):
        scaled_data = StandardScaler().fit_transform(self.data)
        projected_data = self.pca.transform(scaled_data)
        plt.figure(figsize=(10, 6))
        plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.8)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Sample Projection Plot on Principal Components 1 and 2')
        plt.grid(True)
        plt.show()

    def plot_pca_with_kmeans(self, n_clusters=3):
        scaled_data = StandardScaler().fit_transform(self.data)
        pca = PCA(n_components=2)
        projected_data = pca.fit_transform(scaled_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(projected_data)
        
        plt.figure(figsize=(10, 8))
        plt.style.use('seaborn')
        
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        for i in range(n_clusters):
            cluster_data = projected_data[labels == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i+1}', alpha=0.7)
            
            # 用共變異矩陣
            cov = np.cov(cluster_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 6 * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=np.mean(cluster_data, axis=0),
                            width=width, height=height,
                            angle=angle, color=colors[i], alpha=0.2)
            plt.gca().add_artist(ellipse)
        
        plt.title('Cluster plot', fontsize=16)
        plt.xlabel(f'Dim1 ({pca.explained_variance_ratio_[0]:.0%})', fontsize=12)
        plt.ylabel(f'Dim2 ({pca.explained_variance_ratio_[1]:.0%})', fontsize=12)
        plt.legend(title='Clusters', title_fontsize=12, fontsize=10)
        plt.tight_layout()
        plt.show()

class FactorAnalysis:
    def __init__(self, data, num_factors=3, rotation=None):
        self.data = data
        self.num_factors = num_factors
        self.rotation = rotation
        self.fa, self.factor_loadings = self.perform_factor_analysis()

    def perform_factor_analysis(self):
        fa = FactorAnalyzer(n_factors=self.num_factors, rotation=self.rotation)
        fa.fit(self.data)
        factor_loadings = fa.loadings_
        if self.rotation is not None:
            rotation_matrix = ortho_group.rvs(factor_loadings.shape[1]).reshape(-1, factor_loadings.shape[1])
            factor_loadings = factor_loadings @ rotation_matrix
        return fa, factor_loadings

    def plot_factor_loadings(self):
        y_axis_labels = ['總價元','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積']
        plt.figure(figsize=(12, 8))
        plt.imshow(self.factor_loadings, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title('Factor Loadings Heatmap')
        plt.xlabel('Factor')
        plt.xticks(range(len(self.factor_loadings.T)), ['Factor 1', 'Factor 2', 'Factor 3'],rotation=45)
        plt.yticks(range(len(y_axis_labels)), y_axis_labels)
        plt.show()

    def plot_factor_correlation_matrix(self):
        factor_corr = np.corrcoef(self.factor_loadings.T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(factor_corr, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=range(1, factor_corr.shape[0] + 1), yticklabels=range(1, factor_corr.shape[0] + 1))
        plt.title('Factor Correlation Matrix')
        plt.xlabel('Factors')
        plt.ylabel('Factors')
        plt.show()

    def plot_factor_variance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fa.get_factor_variance()[1]) + 1), self.fa.get_factor_variance()[1], marker='o', linestyle='-')
        plt.title('Explained Variance by Each Factor')
        plt.xlabel('Number of Factors')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(self.fa.get_factor_variance()[1]) + 1))
        plt.grid(True)
        plt.show()

    def print_factor_analysis_results(self):
        new_columns = ['總價元','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積']
        #df = pd.DataFrame(self.factor_loadings, columns=["Factor 1", "Factor 2", "Factor 3","Factor 4"])
        df = pd.DataFrame(self.factor_loadings, columns=["Factor 1", "Factor 2", "Factor 3"])
        new = df.transpose()
        new.columns = new_columns
        print(new.transpose())
        print("\n")
        print("Explained Variance Ratio:")
        print(self.fa.get_factor_variance())
        explained_variance_ratio = self.fa.get_factor_variance()[1]
        for i, ratio in enumerate(explained_variance_ratio):
            print("第{}個因子：{}%".format(i+1, ratio * 100))
        print("\n")

class HierarchicalClustering:
    def __init__(self, data):
        self.data = data
        self.linkage_matrix = None

    def perform_clustering(self):
        self.linkage_matrix = linkage(self.data, method='ward')

    def plot_clusters(self, num_clusters=None):
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram")
        dendrogram(self.linkage_matrix)
        plt.axhline(y=400, color='r', linestyle='--', label='分兩群的位置')
        plt.legend()
        plt.show()

        if num_clusters:
            cluster_labels = fcluster(self.linkage_matrix, num_clusters, criterion='maxclust')
            plt.figure(figsize=(10, 7))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=cluster_labels, cmap='prism')
            plt.title(f"Clusters (num_clusters={num_clusters})")
            plt.show()

    def get_cluster_stats(self, num_clusters):
        cluster_labels = fcluster(self.linkage_matrix, num_clusters, criterion='maxclust')
        cluster_stats = {}
        cluster_counts = {}  
        for cluster_id in set(cluster_labels):
            cluster_data = self.data[cluster_labels == cluster_id]
            cluster_mean = cluster_data.mean(axis=0)
            cluster_median = np.median(cluster_data, axis=0)
            cluster_std = cluster_data.std(axis=0)
            cluster_stats[cluster_id] = {
                'mean': cluster_mean,
                'median': cluster_median,
                'std': cluster_std
            }
            cluster_counts[cluster_id] = len(cluster_data)
        return cluster_stats,cluster_counts

class KMeansClustering:
    def __init__(self, data):
        self.data = np.array(data)
        self.kmeans_model = None
        self.cluster_labels = None

    def fit(self, n_clusters=2):
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.kmeans_model.fit_predict(self.data)

    def elbow_method(self, max_clusters=10):
        distortions = []
        K = range(1, max_clusters + 1)
        for k in K:
            kmeans_model = KMeans(n_clusters=k, random_state=42)
            kmeans_model.fit(self.data)
            distortions.append(kmeans_model.inertia_)

        plt.figure(figsize=(10, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def descriptive_statistics(self, main_col):
        if self.cluster_labels is None:
            raise ValueError("You need to fit the model before getting descriptive statistics.")
        
        df = pd.DataFrame(self.data, columns=main_col)
        df['Cluster'] = self.cluster_labels
        
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster].drop(columns=['Cluster'])
            stats = cluster_data.describe().T
            stats['median'] = cluster_data.median()
            stats = stats[['count', 'mean', 'std', 'median', 'min', 'max']]
            print(f"Descriptive statistics for cluster {cluster+1}:")
            print(stats)
            print("\n")

    def save_to_csv(self):
        if self.cluster_labels is None:
            raise ValueError("You need to fit the model before saving to CSV.")
        
        df = pd.DataFrame(self.data, columns=[f"Feature_{i}" for i in range(self.data.shape[1])])
        df['Cluster'] = self.cluster_labels + 1  # Add 1 to make clusters 1, 2, 3 instead of 0, 1, 2
        
        df.to_csv('kmeans.csv', index=False)
        print(f"Data saved successfully.")

class LDA:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.model = LinearDiscriminantAnalysis()
        self.scaler = StandardScaler()

    def split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'LDA 準確率: {accuracy}')
        self.plot_confusion_matrix(y_test, y_pred)

        return accuracy 

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

class QDA:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.model = QuadraticDiscriminantAnalysis()
        self.scaler = StandardScaler()

    def split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        X = X.fillna(X.mean())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'QDA 準確率: {accuracy}')
        self.plot_confusion_matrix(y_test, y_pred)
        return accuracy

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    
#data = pd.read_csv('data_新竹.csv', low_memory=False)
# data_no_land = data[data['交易標的']!= '土地']
# data_no_land.to_csv('data_no_land.csv', index=False)
data = pd.read_csv('data_no_land.csv', low_memory=False)

data['總樓層數'] = data['總樓層數'].apply(DataProcessing.custom_replace)
data['建築完成年月日'] = data['建築完成年月日'].apply(DataProcessing.remove_last_four_digits)


#selected_columns = ['總價元','交易標的','建物型態','有無管理組織','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積']
#多cluster
selected_columns = ['總價元','交易標的','建物型態','有無管理組織','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積','Cluster']
data = data[selected_columns]

#main_col = ['總價元','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積']
#多cluster
main_col = ['總價元','土地移轉總面積平方公尺', '總樓層數', '建築完成年月日', '建物移轉總面積平方公尺', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '單價元平方公尺', '車位移轉總面積平方公尺', '車位總價元','主建物面積','附屬建物面積','陽台面積','Cluster']
data1 = data[main_col]


# print(data_lda['有無管理組織'].head())
new_data = data1.fillna(data1.median())
data_log = new_data.copy()

# print(new_data.head())

# 對主要欄位進行 log transform
for col in main_col:
    data_log[col] = data_log[col].apply(lambda x: np.log(x + 1))

data_log.to_csv('data_no_land1.csv', index=False)    


# #EDA
# # eda = EDA(data)
# eda = EDA(data_log)
# eda.plot_barplots()
# eda.plot_lineplots()
# eda.plot_corr()
# eda.plot_histogram('土地移轉總面積平方公尺')
# eda.plot_histogram('建物移轉總面積平方公尺')
# eda.plot_histogram('主建物面積')
# eda.plot_histogram('總價元')
# eda.plot_distribution('車位總價元')
# eda.plot_boxplot('總價元')
# eda.plot_scatter('總價元', '建築完成年月日')
# eda.plot_relationship('有無管理組織')
# eda.plot_relationship_categorical('有無管理組織')

# # 敘述性統計
# stats_obj = Statistic(data1)
# stats = stats_obj.describe_dataframe()
# for key, value in stats.items():
#     print(f"{key}: {value}")
# stats_obj.print_zero_counts(main_col)

# #看交易標的
# stats_obj = Statistic(data)   
# stats_obj.print_transaction_counts()


# #PCA
# pca_analysis = PCAnalysis(data_log)
# # pca_analysis = PCAnalysis(new_data)
# pca_analysis.plot_cumulative_variance()
# pca_analysis.plot_individual_variance()
# pca_analysis.plot_biplot()
# pca_analysis.plot_sample_projection()
# pca_analysis.plot_pca_with_kmeans()


# # #Factor Analysis
# #不進行轉軸
# fa_analysis= FactorAnalysis(new_data, num_factors=3, rotation=None)

# # #使用varimax轉軸
# # fa_analysis = FactorAnalysis(data_log, num_factors=3, rotation='varimax')

# fa_analysis.print_factor_analysis_results()
# fa_analysis.plot_factor_correlation_matrix()
# fa_analysis.plot_factor_loadings()
# fa_analysis.plot_factor_variance()

# # 階層式分群
# hc = HierarchicalClustering(new_data)
# hc.perform_clustering()
# # hc.plot_clusters(num_clusters=2)
# cluster_stats, cluster_counts = hc.get_cluster_stats(num_clusters=2)
# for cluster_id, stats in cluster_stats.items():
#     print(f"Cluster {cluster_id}:")
#     print(f"  Mean: {stats['mean']}")
#     print(f"  Median: {stats['median']}")
#     print(f"  Std: {stats['std']}")
#     print(f"  Count: {cluster_counts[cluster_id]}")


# # Kmeans
# kmeans_clustering = KMeansClustering(new_data)
# #kmeans_clustering.elbow_method(max_clusters=10)
# kmeans_clustering.fit(n_clusters=2)
# kmeans_clustering.descriptive_statistics(main_col)
# kmeans_clustering.save_to_csv()

# # 使用LDA
# lda_model = LDA(new_data, 'Cluster')
# X_train, X_test, y_train, y_test = lda_model.split_data()
# lda_model.train(X_train, y_train)
# lda_accuracy = lda_model.evaluate(X_test, y_test)
# lda_model.plot_confusion_matrix(y_test, lda_model.model.predict(X_test))


# # 使用QDA
# qda_model = QDA(new_data, 'Cluster')
# X_train, X_test, y_train, y_test = qda_model.split_data()
# qda_model.train(X_train, y_train)
# qda_accuracy = qda_model.evaluate(X_test, y_test)
# qda_model.plot_confusion_matrix(y_test, qda_model.model.predict(X_test))
