# ✅ Full Agentic + Multi-Agent AutoML System with Chat Integration

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# === Models ===
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# === Together API ===
together_api_key = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"

def ask_together_agent(prompt):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {together_api_key}"},
        json={
            "model": "meta-llama-3-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

# === Agent Class ===
class AutoMLAgent:
    def __init__(self, X, y):
        self.X_raw = X.copy()
        self.X = pd.get_dummies(X)
        self.y = y
        self.classification = self._detect_task_type()
        self.models = self._load_models()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = -np.inf
        self.best_info = {}
        self.results = []

    def _detect_task_type(self):
        return self.y.dtype == 'object' or len(np.unique(self.y)) <= 20

    def _load_models(self):
        return {
            "classification": {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC(),
                "Naive Bayes": GaussianNB(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            },
            "regression": {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": xgb.XGBRegressor()
            }
        }["classification" if self.classification else "regression"]

    def run(self):
        for test_size in [0.1, 0.2, 0.3]:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds) if self.classification else r2_score(y_test, preds)

                    info = {
                        "Model": name,
                        "Score": round(score, 4),
                        "Test Size": test_size,
                        "Type": "Classification" if self.classification else "Regression"
                    }

                    self.results.append(info)

                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        self.best_info = info
                except Exception:
                    continue

        return pd.DataFrame(self.results).sort_values(by="Score", ascending=False), self.best_info

    def save_best_model(self):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

# === Streamlit UI ===
st.set_page_config(page_title="Agentic AutoML AI", layout="wide")
st.title("🤖 Multi-Agent AutoML System with Chat Intelligence")

uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Variable", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        if st.button("🚀 Deploy Agent"):
            agent = AutoMLAgent(X, y)
            with st.spinner("🤖 Agent training all models..."):
                results_df, best = agent.run()
                agent.save_best_model()
                st.session_state['agent_results'] = results_df
                st.session_state['best_info'] = best
            st.success("✅ Agent training complete!")

# === Show Results from Session State ===
if 'agent_results' in st.session_state and 'best_info' in st.session_state:
    st.subheader("📈 Model Comparison")
    st.dataframe(st.session_state['agent_results'])

    st.subheader("🏆 Best Model")
    st.write(st.session_state['best_info'])

    # Reasoning Agent (LLM)
    prompt = f"""
    I am working on a {st.session_state['best_info']['Type']} problem.
    The best performing model is {st.session_state['best_info']['Model']} with a score of {st.session_state['best_info']['Score']} on test size {st.session_state['best_info']['Test Size']}.
    Suggest ways to improve it further: tuning, feature engineering, or algorithm switching.
    """
    st.subheader("🧠 Agent's Insight via Together AI")
    response = ask_together_agent(prompt)
    st.write(response)

# === Sidebar Multi-Agent Chat ===
st.sidebar.title("💬 Multi-Agent Chat")
query = st.sidebar.text_area("Ask the AI Agent something about your dataset or models")
if query:
    st.sidebar.write("🤖 Thinking...")
    answer = ask_together_agent(query)
    st.sidebar.write(answer)
