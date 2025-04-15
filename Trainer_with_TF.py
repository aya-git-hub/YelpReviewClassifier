import h5py
import numpy as np
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ------------------------------
# 1. 加载 Vocabulary（index）文件
# ------------------------------
# 假设 Vocabulary 是你项目中定义的类，保存为 index.pck 文件
from Vocabulary import Vocabulary

print("加载词汇表（index）...")
vocab = Vocabulary.load('dataset/index')  # 默认加载文件 "index.pck"
print("词汇表加载完成。")

# ------------------------------
# 2. 从 data.h5 中读取评论数据
# ------------------------------
print("打开 HDF5 数据文件 data.h5 ...")
with h5py.File('dataset/data.h5', 'r') as h5_file:
    dset = h5_file['reviews']
    n_reviews = dset.shape[0]
    print(f"共 {n_reviews} 条评论数据。")

    # 用于保存解码后的评论文本和目标评分
    review_texts = []
    ratings = []

    # 遍历每一行，注意：矩阵前 4 列为 [stars, useful, funny, cool]，后面为 encoded text tokens
    for i in range(n_reviews):
        # 打印进度，每 1000 条显示一次
        if (i + 1) % 1000 == 0:
            print(f"正在处理第 {i + 1} 条评论...")

        row = dset[i]
        rating = row[0]  # stars 字段
        ratings.append(rating)

        # 提取评论的 encoded token，注意去掉前 4 个特征（以及填充的 0 值）
        tokens = row[4:]
        tokens = tokens[tokens != 0]  # 去除填充值

        # 利用 Vocabulary 将编码转换为单词列表
        # Vocabulary.decode 接受一个列表，返回对应的单词列表
        words = vocab.decode(tokens.tolist())
        text = " ".join(words)
        review_texts.append(text)

print("所有评论数据读取完毕。")

# ------------------------------
# 3. 利用 TF-IDF 对评论文本向量化，并聚类
# ------------------------------
print("开始对评论文本进行 TF-IDF 向量化...")
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(review_texts)
print("TF-IDF 向量化完成。")

# 使用 KMeans 将评论聚为 k 个簇（例如 k=10）
k = 10
print(f"开始使用 KMeans 对评论进行聚类 (k={k}) ...")
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_matrix)
print("聚类完成。")

# 对聚类标签进行 One-Hot 编码
cluster_labels = cluster_labels.reshape(-1, 1)
enc = OneHotEncoder(sparse_output=False)
cluster_onehot = enc.fit_transform(cluster_labels)
print("聚类标签 One-Hot 编码完成。")

# ------------------------------
# 4. 使用 VADER 提取情感特征
# ------------------------------
print("下载 VADER 词库并初始化情感分析器...")
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

print("开始提取每条评论的情感特征...")
sentiment_features = []
for i, text in enumerate(review_texts):
    if (i + 1) % 1000 == 0:
        print(f"情感分析处理 {i + 1} 条评论...")
    score = analyzer.polarity_scores(text)
    feat = [score["compound"], score["pos"], score["neu"], score["neg"]]
    sentiment_features.append(feat)
sentiment_features = np.array(sentiment_features)
print("情感特征提取完成。")

# ------------------------------
# 5. 构建回归模型输入特征
# ------------------------------
# 拼接情感特征 (4 维) 与 聚类 One-Hot 特征 (k 维)
X = np.concatenate([sentiment_features, cluster_onehot], axis=1)
y = np.array(ratings)

print(f"特征矩阵 X 的形状: {X.shape}, 目标向量 y 的长度: {len(y)}")

# 划分训练集和验证集（例如 80% / 20%）
print("划分训练集和验证集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集数量：{len(y_train)}，验证集数量：{len(y_test)}")

# ------------------------------
# 6. 使用随机森林回归预测评分
# ------------------------------
print("开始训练随机森林回归模型...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("模型训练完成。")

print("在验证集上进行预测...")
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"验证集平均绝对误差 (MAE): {mae:.3f}")

print("部分预测结果对比：")
for i in range(min(10, len(y_test))):
    print(f"实际评分: {y_test[i]}, 预测评分: {y_pred[i]:.2f}")
