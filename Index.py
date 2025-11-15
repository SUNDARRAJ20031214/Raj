import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score

st.title("🤖 Machine Learning Prediction App")
st.write("Select your dataset and algorithm to train & predict.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    all_columns = df.columns.tolist()
    target = st.selectbox("Select Target (Output) Column", all_columns)
    features = [col for col in all_columns if col != target]

    X = df[features]
    y = df[target]

    # Encode categorical variables
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Algorithm type
    task = st.radio("Select Task Type", ["Classification", "Regression"])

    model = None
    algo_name = None

    if task == "Classification":
        algo_name = st.selectbox("Select Algorithm", [
            "Logistic Regression", "Decision Tree", "Random Forest",
            "Naive Bayes", "KNN", "SVM"
        ])

        if algo_name == "Logistic Regression":
            model = LogisticRegression()
        elif algo_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algo_name == "Random Forest":
            model = RandomForestClassifier()
        elif algo_name == "Naive Bayes":
            model = GaussianNB()
        elif algo_name == "KNN":
            model = KNeighborsClassifier()
        elif algo_name == "SVM":
            model = SVC()

    else:
        algo_name = st.selectbox("Select Algorithm", [
            "Linear Regression", "Decision Tree", "Random Forest", "SVR"
        ])

        if algo_name == "Linear Regression":
            model = LinearRegression()
        elif algo_name == "Decision Tree":
            model = DecisionTreeRegressor()
        elif algo_name == "Random Forest":
            model = RandomForestRegressor()
        elif algo_name == "SVR":
            model = SVR()

    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"✅ {algo_name} model trained successfully!")

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {acc * 100:.2f}%")
        else:
            r2 = r2_score(y_test, y_pred)
            st.write(f"**R² Score:** {r2:.2f}")

        # Prediction
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
            
 
