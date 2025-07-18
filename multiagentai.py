# ✅ Full Agentic + Multi-Agent AutoML System with Chat Integration

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score

# === Models ===
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb

# === Together API ===
together_api_key = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"

def ask_data_scientist_agent(prompt):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {together_api_key}"},
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": f"[DATA SCIENTIST] {prompt}"}],
        }
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

def ask_ml_engineer_agent(prompt):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {together_api_key}"},
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": f"[ML ENGINEER] {prompt}"}],
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
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC(),
                "Naive Bayes (Gaussian)": GaussianNB(),
                "Naive Bayes (Multinomial)": MultinomialNB(),
                "Naive Bayes (Complement)": ComplementNB(),
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
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": xgb.XGBRegressor(),
                "Polynomial Linear Regression": make_polynomial_model()
            }
        }["classification" if self.classification else "regression"]

    def run(self):
        for test_size in [0.1, 0.2, 0.3]:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42)
            
            if self.classification and len(np.unique(y_train)) > 2:
                sampler = SMOTE()
                X_train, y_train = sampler.fit_resample(X_train, y_train)

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

def make_polynomial_model():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

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
                st.session_state['data_preview'] = df.head(3).to_string(index=False)
            st.success("✅ Agent training complete!")

# === Show Results from Session State ===
if 'agent_results' in st.session_state and 'best_info' in st.session_state:
    st.subheader("📈 Model Comparison")
    st.dataframe(st.session_state['agent_results'])

    st.markdown("### 🧠 Agent AI Performance")
    best = st.session_state['best_info']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success(f"**Best Model:** `{best['Model']}`")
    with col2:
        st.success(f"**Score ({'Accuracy' if best['Type']=='Classification' else 'R2 Score'}):** `{best['Score']}`")
    with col3:
        st.success(f"**Test Size:** `{int(best['Test Size'] * 100)}%`")
    with col4:
        st.success(f"**Type:** `{best['Type']}`")

    st.markdown(
        f"The agent recommends using the **{best['Model']}** model with a **{int(best['Test Size'] * 100)}%** test split for your dataset, as it yielded the highest performance for a **{best['Type']}** task."
    )

    # === Reasoning Agent (LLM) ===
    prompt = f"""
    I am working on a {st.session_state['best_info']['Type']} problem.
    The best performing model is {st.session_state['best_info']['Model']} with a score of {st.session_state['best_info']['Score']} on test size {st.session_state['best_info']['Test Size']}.
    Suggest advanced improvements including hyperparameter tuning strategies, feature selection methods, data augmentation, dimensionality reduction, ensemble techniques, and possibly neural network alternatives.
    Provide suggestions in bullet points with concise reasoning.
    """
    st.subheader("🧠 Agent's Insight via Together AI")
    response = ask_data_scientist_agent(prompt)
    st.write(response)

# === Sidebar Multi-Agent Chat ===
st.sidebar.title("💬 Multi-Agent Chat")
query = st.sidebar.text_area("Ask both agents something about your dataset or models")
if query and 'best_info' in st.session_state and 'data_preview' in st.session_state:
    context = f"Dataset Sample:\n{st.session_state['data_preview']}\n\nBest Model Info: {st.session_state['best_info']}"
    combined_prompt = f"{context}\n\nUser Question: {query}"

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.markdown("**🧠 Data Scientist Agent:**")
        answer1 = ask_data_scientist_agent(combined_prompt)
        st.sidebar.write(answer1)

    with col2:
        st.sidebar.markdown("**🛠️ ML Engineer Agent:**")
        answer2 = ask_ml_engineer_agent(combined_prompt)
        st.sidebar.write(answer2)
