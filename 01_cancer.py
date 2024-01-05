import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

## Parammetre de l'application
st.set_page_config("Franck_BATTY",layout="wide")
st.sidebar.subheader(":blue[**Franck BATTY**]")

# Charger la base de données
train=pd.read_csv("base_finale.csv")
train=train.drop('Unnamed: 0',axis=1)

# Interface utilisateur de l'application
st.title("Application de Prédiction de survie au cancer du sein")

st.write("Cette application utilise un modèle de Machine Learning pour prédire si un patient survivra ou non à une maladie mortelle comme le cancer du sein en fonction des caractéristiques fournies.")

## Affichage de la vidéo
video_file = open('sein.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

## 
st.subheader(":blue[**Veillez renseigner les données et appuyer sur le Boutton de prédiction**]")

#st.video(video_bytes)
# Fonction pour faire des prédictions
def make_prediction(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability * 100, 2)
    return prediction, probability

# Saisie des caractéristiques du client
Age  =  st.sidebar.number_input(":blue[**Âge du patient**]",train["Age"].min(),train["Age"].max(),train["Age"].min())        
Gender =  st.sidebar.selectbox(":blue[**Sexe du patient**]",train["Gender"].unique())       
Protein1 = st.sidebar.number_input(":blue[**Niveaux d’expression du Protein1**]",train["Protein1"].min(),train["Protein1"].max(),train["Protein1"].min())     
Protein2 = st.sidebar.number_input(":blue[**Niveaux d’expression du Protein2**]",train["Protein2"].min(),train["Protein2"].max(),train["Protein2"].min())      
Protein3 = st.sidebar.number_input(":blue[**Niveaux d’expression du Protein3**]",train["Protein3"].min(),train["Protein3"].max(),train["Protein3"].min())      
Protein4 = st.sidebar.number_input(":blue[**Niveaux d’expression du Protein4**]",train["Protein4"].min(),train["Protein4"].max(),train["Protein4"].min())    
Tumour_Stage = st.sidebar.selectbox(":blue[**Stade du cancer du sein de la patiente**]",train["Tumour_Stage"].unique())  
Histology = st.sidebar.selectbox(":blue[**Types de cancers du sein**]",train["Histology"].unique()) 
ER_status = st.sidebar.selectbox(":blue[**Statut du recepteur hormonal**]",train["ER status"].unique())
PR_status = st.sidebar.selectbox(":blue[**Statut du récepteur de la Progestérone**]",train["PR status"].unique())
HER2_status = st.sidebar.selectbox(":blue[**Statut du récepteur du facteur de croissance épidermique humain 2**]",train["HER2 status"].unique())
Surgery_type = st.sidebar.selectbox(":blue[**Les types de chirurgie du cancer du sein**]",train["Surgery_type"].unique())
# Créer un DataFrame à partir des caractéristiques
input_data = pd.DataFrame({"Age":[Age],
    "Gender":[Gender],
    "Protein1":[Protein1],
    "Protein2":[Protein2],
    "Protein3":[Protein3],
    "Protein4":[Protein4],
    "Tumour_Stage":[Tumour_Stage],
    "Histology":[Histology],
    "ER status":[ER_status],
    "HER2 status":[HER2_status],
    "PR status":[PR_status],
    "Surgery_type":[Surgery_type]})

# Affichages des données de l'Utilisateurs
#st.header("Affichages des données de l'Utilisateurs")
#st.write(input_data)
#st.write('---')

# Charger le modèle pré-entraîné
model = joblib.load("Forest_aleatoire.joblib")

# Définir les catégories pour le diagramme
categories = ["Mort","Vivant"]
# Prédiction
if st.sidebar.button("Prédire"):
    prediction, probability = make_prediction(input_data)
    st.subheader("Probabilités :")
    prob_df = pd.DataFrame({'Catégories': categories, 'Probabilité': probability[0]})
    fig = px.bar(prob_df, x='Catégories', y='Probabilité', text='Probabilité', labels={'Probabilité': 'Probabilité (%)'})
    st.plotly_chart(fig)

    st.subheader("Résultat de la prédiction :")
    if prediction[0] == 1:
        st.error("Le patient survivra à cette maladie mortelle du cancer du sein.")
    else:
        st.success("Le patient ne survivra pas à cette maladie mortelle du cancer du sein.")