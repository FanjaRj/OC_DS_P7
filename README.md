# OC Data Scientist - Projet 7 : Implémentez un modèle de scoring
## Contexte
Une société financière, nommée "Prêt à dépenser",  propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé.
Le développement d'un dashboard interactif est également prévu pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

## Données
https://www.kaggle.com/c/home-credit-default-risk/data

## Missions
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

## Livrables
### 1-Développement du modèle de classification
- P7_Exploration.ipynb : notebook exploration des données issues du kaggle https://www.kaggle.com/c/home-credit-default-risk/data 
- P7_Cleaning_Engineering.ipynb : notebook data cleaning (inspiré de https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/script) et feature engineering  (inspiré de https://www.kaggle.com/sz8416/6-ways-for-feature-selection)
- P7_Modeling.ipynb : notebook modélisation

### 2-Création d'un dashboard interactif : décision d'octroi de crédit
- FlaskAPI.py : code API de prédiction du score avec Flask (https://p7flaskapi.herokuapp.com/)
- Dashboard.py : code du dashboard avec Streamlit (https://share.streamlit.io/fanjarj/oc_ds_p7/main/Dashboard/dashboard.py)

### 3-Documentation du projet
- P7_note_methodologique.pdf : description de la méthodologie du modèle de scoring, de son interprétabilité (globale et locale), et de ses limites et améliorations
- P7_presentation.pdf : support de présentation
