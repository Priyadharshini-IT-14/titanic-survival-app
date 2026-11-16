import joblib
import pandas as pd
import streamlit as st

FEATURES = ["pclass", "sex", "age", "fare"]   # same order as in training


@st.cache_resource            # load the model once, not on every click
def load_model():
    return joblib.load("model.joblib")


st.title("Would this passenger have survived?")
st.write("An educational model trained on the Titanic passenger list.")

# Inputs, each with sensible limits and a default
pclass = st.selectbox("Ticket class", [1, 2, 3])
sex = st.radio("Sex", ["female", "male"], horizontal=True)
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare paid", 0, 500, 30)

if st.button("Predict"):
    # Friendly message if the model file is missing
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file is missing. Train and save it first.")
        st.stop()

    # Friendly message if something goes wrong with the input
    try:
        row = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, age, fare]],
            columns=FEATURES,
        )
        chance = model.predict_proba(row)[0][1]
        st.metric("Estimated chance of survival", f"{chance:.0%}")
    except Exception:
        st.warning("Something went wrong with these inputs. Please check them and try again.")
        st.stop()

    st.caption("A learning exercise. It describes a historical dataset and is not advice.")
