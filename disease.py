import pandas as pd
import pickle
import streamlit as st


dataset = pd.read_csv("cleaned_dataset.csv")
precaution = pd.read_csv("cleaned_precaution.csv")
description = pd.read_csv("symptom_Description.csv")
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))
nb = pickle.load(open("nb.pkl", "rb"))


def clean(string):
    return string.replace("_", " ").title()


st.title("Disease Prediction System")

r1col1, r1col2, r1col3 = st.columns(3)
with r1col1:
    s1 = st.selectbox("Select Symptom 1", ["Select"] + sorted(list(set(dataset["Symptom_1"].apply(lambda e: clean(e))))))
with r1col2:
    u_s2 = dataset[dataset["Symptom_2"] != s1.replace(" ", "_").lower()]
    s2 = st.selectbox("Select Symptom 2", ["Select"] + sorted(list(set(u_s2["Symptom_2"].apply(lambda e: clean(e))))))
with r1col3:
    u_s3 = dataset[dataset["Symptom_3"] != s2.replace(" ", "_").lower()]
    s3 = st.selectbox("Select Symptom 3", ["Select"] + sorted(list(set(u_s3["Symptom_3"].apply(lambda e: clean(e))))))

r2col1, r2col2, r2col3 = st.columns(3)
with r2col1:
    u_s4 = dataset[dataset["Symptom_4"] != s3.replace(" ", "_").lower()]
    s4 = st.selectbox("Select Symptom 4", ["Select"] + sorted(list(set(u_s4["Symptom_4"].apply(lambda e: clean(e))))))
with r2col2:
    u_s5 = dataset[dataset["Symptom_5"] != s4.replace(" ", "_").lower()]
    s5 = st.selectbox("Select Symptom 5", ["Select"] + sorted(list(set(u_s5["Symptom_5"].apply(lambda e: clean(e))))))
with r2col3:
    u_s6 = dataset[dataset["Symptom_6"] != s5.replace(" ", "_").lower()]
    s6 = st.selectbox("Select Symptom 6", ["Select"] + sorted(list(set(u_s6["Symptom_6"].apply(lambda e: clean(e))))))

if st.button("Predict Disease", use_container_width = True):
    if s1 != "Select" and s2 != "Select" and s3 != "Select" and s4 != "Select" and s5 != "Select" and s6 != "Select":
        symptoms = [s1, s2, s3, s4, s5, s6]
        symptoms_vector = cv.transform([' '.join(symptoms)])
        prediction = nb.predict(symptoms_vector)

        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        st.write(f"<h2 style='text-align: center;'><b>{'Disease Name : ' + predicted_disease.title()}</b></h2>", unsafe_allow_html = True)

        desc = description["Description"][description[description["Disease"] == predicted_disease].index[0]]
        st.write(f"<h3><b>{'Disease Description'}</b></h3>", unsafe_allow_html = True)
        st.write(desc)

        st.write(f"<h3><b>{'Precautions'}</b></h3>", unsafe_allow_html = True)
        precautions = []
        for i in precaution.columns:
            if i != "Disease":
                idx = precaution[precaution["Disease"] == predicted_disease].index[0]
                precautions.append(precaution[i][idx])

        col1, col2 = st.columns(2)
        with col1:
            for i, j in enumerate(precautions[1:3], start = 1):
                st.write("{}. {}".format(i, j.title()))
        with col2:
            for i, j in enumerate(precautions[3:], start = 3):
                st.write("{}. {}".format(i, j.title()))
    else:
        st.error("Please Select from Every Option")