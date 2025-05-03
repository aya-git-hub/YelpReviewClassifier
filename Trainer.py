import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from tqdm import tqdm
from scipy.stats import norm

class Trainer:
    def __init__(
        self,
        input_file: str,
        model_output: str = 'model/rf.joblib',
        limit: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100
    ):
        nltk.download('vader_lexicon', quiet=True)
        self.analyzer = SentimentIntensityAnalyzer()
        self.input_file = input_file
        self.model_output = model_output
        self.limit = limit
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators

    def load_data(self):
        print(f"Starting to load reviews(most: {self.limit or 'all'} ...")
        features, targets = [], []
        total = self.limit
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading data", total=total, unit="行")):
                if self.limit is not None and i >= self.limit:
                    break
                data = json.loads(line)
                review_text = " ".join(data.get("text", []))
                score = self.analyzer.polarity_scores(review_text)
                features.append([score["compound"], score["pos"], score["neu"], score["neg"]])
                targets.append(data.get("stars", 0))
        self.X = np.array(features)
        self.y = np.array(targets)
        print(f"Finish data loading, totally {len(self.y)} reviews。\n")

        print("dividing training set and test set...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Training:{len(self.y_train)} ，Test:{len(self.y_test)} .\n")

    def train(self):
        print(f"Starting Random Forset, totally {self.n_estimators} trees)…")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=1
        )
        self.model.fit(self.X_train, self.y_train)
        dump(self.model, self.model_output)
        print(f"Finish training, model has been saved to {self.model_output}\n")

    def evaluate(self):
        print("Calculating MAE…")
        self.y_pred = self.model.predict(self.X_test)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        print(f"Test set: MAE: {self.mae:.3f}\n")

    def plot_error_distribution(self, a=0.8, tol=1e-6, max_iter=100):


        errors = self.y_test - self.y_pred
        mu_err = errors.mean()
        sigma_err = errors.std()
        u = self.mae


        plt.figure()
        plt.hist(errors, bins=30, density=True, alpha=0.6, label="Error Histogram")
        x_vals = np.linspace(errors.min(), errors.max(), 200)
        pdf_vals = norm.pdf(x_vals, loc=mu_err, scale=sigma_err)
        plt.plot(x_vals, pdf_vals,
                 label=f"Normal PDF (μ={mu_err:.3f}, σ={sigma_err:.3f})")
        plt.xlabel("Error (y_test - y_pred)")
        plt.ylabel("Density")
        plt.title("Error Distribution with Fitted Normal Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. 牛顿法：解 Φ((u+x−μ)/σ) − Φ((u−x−μ)/σ) = a
        def f(x):
            return ( norm.cdf((u + x - mu_err) / sigma_err)
                   - norm.cdf((u -   x - mu_err) / sigma_err)
                   - a )
        def df(x):
            return ( norm.pdf((u + x - mu_err) / sigma_err)
                   + norm.pdf((u -   x - mu_err) / sigma_err)
                   ) / sigma_err

        x = sigma_err
        for _ in range(max_iter):
            fx  = f(x)
            dfx = df(x)
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        print(f"For a={a*100:.1f}% coverage around MAE, threshold x ≈ {x:.3f}")
        print(f"Interval: [{u - x:.3f}, {u + x:.3f}]")

        # 验证实际覆盖率
        coverage = ( norm.cdf((u + x - mu_err)/sigma_err)
                   - norm.cdf((u -   x - mu_err)/sigma_err) )
        print(f"Actual coverage: {coverage:.4f}\n")

    def plot_random_forest(self, feature_index=0, resolution=0.01):
        x_min, x_max = self.X_test[:, feature_index].min(), self.X_test[:, feature_index].max()
        x_grid = np.arange(x_min, x_max, resolution)
        mean_vector = self.X_train.mean(axis=0)
        X_grid = np.tile(mean_vector, (len(x_grid), 1))
        X_grid[:, feature_index] = x_grid
        y_grid = self.model.predict(X_grid)

        plt.figure()
        plt.scatter(self.X_test[:, feature_index], self.y_test,
                    alpha=0.6, label="Actual (test)")
        plt.plot(x_grid, y_grid, linewidth=2, label="RF Prediction slice")
        plt.xlabel(f"Feature #{feature_index}")
        plt.ylabel("Stars")
        plt.title(f"Random Forest on Feature {feature_index}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        self.load_data()
        self.train()
        self.evaluate()
        # a = 50%
        self.plot_error_distribution(a=0.5)


if __name__ == "__main__":
    trainer = Trainer(
        input_file="dataset/yelp_usefulw_tok.json",
        model_output="model/rf.joblib",
        limit=None,
        test_size=0.2,
        random_state=42,
        n_estimators=100
    )
    trainer.run()
