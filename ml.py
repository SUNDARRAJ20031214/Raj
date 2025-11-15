import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

st.title("🔮 Machine Learning Multi-Algorithm Predictor")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Select target column
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # For non-numeric data
        X = pd.get_dummies(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.subheader("⚙️ Choose ML Algorithm")
        algorithm = st.selectbox(
            "Select an algorithm",
            [
                "Logistic Regression",
                "KNN",
                "SVM",
                "Random Forest",
                "Decision Tree",
                "Naive Bayes",
            ]
        )

        # Select model based on algorithm
        if algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif algorithm == "KNN":
            model = KNeighborsClassifier()
        elif algorithm == "SVM":
            model = SVC()
        elif algorithm == "Random Forest":
            model = RandomForestClassifier()
        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algorithm == "Naive Bayes":
            model = GaussianNB()

        # Train model
        model.fit(X_train, y_train)

        # Accuracy
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        st.success(f"🎯 Model Accuracy: {acc * 100:.2f}%")

        st.subheader("🧪 Predict on New Data")

        user_input = {}
        for col in X.columns:
            val = st.text_input(f"Enter value for {col}")
            if val:
                try:
                    user_input[col] = float(val)
                except:
                    user_input[col] = val

        if st.button("Predict"):
            user_df = pd.DataFrame([user_input])
            user_df = user_df.reindex(columns=X.columns, fill_value=0)
            prediction = model.predict(user_df)[0]
            st.success(f"🔮 Prediction: {prediction}")
