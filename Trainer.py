import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# 如果没下载过 VADER 的词库需要先下载（只需运行一次）
nltk.download('vader_lexicon')

# 初始化情感分析器
analyzer = SentimentIntensityAnalyzer()

# 文件路径（请根据实际情况调整）
input_file = "dataset/yelp_useful_tok.json"

# 列表用于存储特征和目标值
features = []
targets = []

# 可选择限制读取行数（例如：10000 行）
limit = None  # 若为 None，则全部读取；否则可以设置为一个整数，例如 10000

# 读取文件，逐行提取情感特征和评分
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if limit is not None and i >= limit:
            break

        data = json.loads(line)
        # 将 token 列表还原为一句话（各单词之间空格分隔）
        tokens = data.get("text", [])
        review_text = " ".join(tokens)

        # 利用 VADER 得到情感分数
        score = analyzer.polarity_scores(review_text)
        # 构造特征向量：我们这里使用 compound, pos, neu, neg 四个分数
        feature_vector = [
            score["compound"],
            score["pos"],
            score["neu"],
            score["neg"]
        ]
        features.append(feature_vector)

        # 目标值：这里使用评论中的 stars 字段，假设已经是 0-5 的评分
        targets.append(data.get("stars", 0))

# 转换为 numpy 数组
X = np.array(features)
y = np.array(targets)

# 划分训练集与验证集（例如80%训练，20%验证）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林回归
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = rf.predict(X_test)

# 计算平均误差（这里采用平均绝对误差 MAE）
mae = mean_absolute_error(y_test, y_pred)
print(f"验证集上平均绝对误差: {mae:.3f}")

# 输出部分预测结果对比
for i in range(min(10, len(y_test))):
    print(f"实际评分: {y_test[i]}, 预测评分: {y_pred[i]:.2f}")
