#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:48:46 2022

@author: fanjama
"""

### Import de librairies
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import math
import pickle
import shap
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


### Import des donnees
# Features
feat = ['SK_ID_CURR','TARGET','DAYS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
        'DAYS_EMPLOYED','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']
                
# Nombre de ligne
num_rows = 150000

# Original Data
raw_train = pd.read_csv('../Dataset/application_train.csv',usecols=feat,nrows=num_rows)
raw_test = pd.read_csv('../Dataset/application_train.csv',usecols=[f for f in feat if f!='TARGET'])
raw_app = raw_train.append(raw_test).reset_index(drop=True)
del raw_train
del raw_test
#scaler = MinMaxScaler()
#scaler.fit(raw_app[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']])
#raw_app[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']] = scaler.transform(raw_app[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']])
raw_app['AGE'] = raw_app['DAYS_BIRTH'] // (-365)
raw_app['YEARS_EMPLOYED'] = raw_app['DAYS_EMPLOYED'] // (-365)
raw_app['CREDIT'] = raw_app['AMT_CREDIT'].apply(lambda x: 'No' if math.isnan(x) else 'Yes')
raw_app = raw_app.drop(['DAYS_BIRTH','DAYS_EMPLOYED'], axis=1)

# Treated Data
train = pd.read_csv('../Results/data_train.csv')
test = pd.read_csv('../Results/data_test.csv')
app = train.append(test).reset_index(drop=True)

# Modele voisin
pk_kn_in = open('../Results/knn.pkl','rb')
knn = pickle.load(pk_kn_in)

# Chargement du modèle de classification
pk_mdl_in = open('../Results/model.pkl','rb')
model = pickle.load(pk_mdl_in)

# Explainer
X_train_sm = pd.read_csv('../Results/X_train_sm.csv')
X_name = list(X_train_sm.columns)
explainer = shap.TreeExplainer(model,X_train_sm)
del X_train_sm
#explainer = shap.Explainer.load(open('../Results/explainer','rb'),model_loader="auto",masker_loader="auto")


# Features
features =['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT']

@st.cache(suppress_st_warning=True)
### Fonctions
## DATA
# Recuperation de data
def get_data(data,ID):
    if type(ID) == list:
        return data[data['SK_ID_CURR'].isin(ID)]
    else:
        return data[data['SK_ID_CURR']==ID].head(1)

# Recuperation des voisins
def get_similar_ID(ID):    
    app_id = app[app['SK_ID_CURR']==ID].drop(['SK_ID_CURR','TARGET'], axis=1)
    knn_index = knn.kneighbors(app_id,return_distance=False)
    knn_id = app['SK_ID_CURR'][app.index.isin(knn_index[0])].values.tolist()
    return knn_id

def get_stat_ID(ID):   
    app_knn = get_similar_ID(ID)
    data_knn = get_data(raw_app,app_knn).dropna()
    return len(data_knn),len(data_knn[data_knn['TARGET']==1])

## GRAPHE
# Initialisation de Graphe Radar
def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
            ax.set_yticklabels(ax.get_yticklabels(),fontsize=4)                       
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax.set_xticklabels(variables,fontsize=6) 
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)        
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        
# Graphe Radar
def radat_id_plot(ID,fig,features=features,fill=False):
    app_id = get_data(raw_app,ID)[features]
    client = app_id.iloc[0]
    ranges = [(client['AGE']-5, client['AGE']+5),
              (client['YEARS_EMPLOYED']-1, client['YEARS_EMPLOYED']+1),
              (client['AMT_INCOME_TOTAL']-500, client['AMT_INCOME_TOTAL']+500),
              (client['AMT_ANNUITY']-100, client['AMT_ANNUITY']+100),
              (client['AMT_CREDIT']-500, client['AMT_CREDIT']+500)]
    
    radar = ComplexRadar(fig,features,ranges)
    radar.plot(client,linewidth=3,color='blue')
    if fill:
        radar.fill(client, alpha=0.2)
        
def radat_knn_plot(ID,fig,features=features,fill=False):
    app_id = get_data(raw_app,ID)[features]
    data_id = app_id.iloc[0]    
    app_knn = get_similar_ID(ID)
    data_knn = get_data(raw_app,app_knn).dropna()
    data_knn['TARGET'] = data_knn['TARGET'].astype(int)
    moy_knn = data_knn.groupby('TARGET').mean()
    ranges = [(min(data_knn['AGE'])-5, max(data_knn['AGE'])+5),
              (min(data_knn['YEARS_EMPLOYED'])-1,  max(data_knn['YEARS_EMPLOYED'])+1),
              (min(data_knn['AMT_INCOME_TOTAL'])-500,  max(data_knn['AMT_INCOME_TOTAL'])+500),
              (min(data_knn['AMT_ANNUITY'])-100,  max(data_knn['AMT_ANNUITY'])+100),
              (min(data_knn['AMT_CREDIT'])-500,  max(data_knn['AMT_CREDIT'])+500)]
    
    radar = ComplexRadar(fig,features,ranges)
    radar.plot(data_id,linewidth=3,label='Client '+str(ID),color='blue')
    radar.plot(moy_knn.iloc[1][features],linewidth=3,label='Client similaire moyen avec difficultés',color='red')
    radar.plot(moy_knn.iloc[0][features],linewidth=3,label='Client similaire moyen sans difficultés',color='green')
    fig.legend(fontsize=5,loc='upper center',bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    if fill:
        radar.fill(client, alpha=0.2)
        
def shap_id(ID):
    app_id = get_data(app,ID)[X_name]
    shap_vals = explainer.shap_values(app_id)
    shap.bar_plot(shap_vals[1][0],feature_names=X_name,max_display=10)
    #shap.force_plot(explainer.expected_value[1], shap_vals[1], app_id)
    
def shap_all():
    app_all = app[X_name]
    shap_values = explainer.shap_values(app_all)
    shap.summary_plot(shap_values,feature_names=feats,max_display=10)


### DASHBOARD
## Titre & Entete
st.set_page_config(layout='wide',
                   page_title="Dashboard : décision d'octroi de crédit")
st.write("""# Dashboard : décision d'octroi de crédit""")
st.write("""_Dashboard à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client._
    """)
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

## Corps
analyse = st.sidebar.radio("Choisir votre analyse",('Client','Portefeuille'))
# Analyse Client à partir d'un ID
if analyse == 'Client':
    st.sidebar.markdown("Merci d'entrer un identifiant client :")
    ID = st.sidebar.number_input(' ', min_value=100002, max_value=456255)
    try:
        id_raw_app = get_data(raw_app,ID)
        with st.spinner('Analyse Client...'):
            st.write("## Analyse Client")
            with st.container():
                col1, col2 = st.columns([1.5,2.5])      
                with col1:
                    st.write("#### Détail du Client " + str(ID))
                    st.markdown("* **Statut : " + str(id_raw_app['NAME_FAMILY_STATUS'].values[0]) + "**")
                    st.markdown("* **Nombre d'enfant(s) : " + str(id_raw_app['CNT_CHILDREN'].values[0]) + "**")
                    st.markdown("* **Emploi : " + str(id_raw_app['NAME_INCOME_TYPE'].values[0]) + "**")
                    st.markdown("* **Crédit en cours : " + str(id_raw_app['CREDIT'].values[0]) + "**")
                with col2:
                    fig = plt.figure(figsize=(2,2))
                    st.pyplot(radat_id_plot(ID,fig))
            st.markdown("""---""")
            with st.container():
                st.write("#### Client(s) similaires ")
                try:
                    col3, col4 = st.columns([3,1])
                    with col3:
                        fig = plt.figure(figsize=(3,3))
                        st.pyplot(radat_knn_plot(ID,fig))
                    with col4:
                        N_knn, N_knn1 = get_stat_ID(ID)
                        st.markdown("* **Clients similaires : " + str(N_knn) + "**")
                        st.markdown("* **Avec difficultés de paiements : " + str(N_knn1) + "**")                
                        st.markdown("_(soit " + str(N_knn1*100/N_knn) + "% des Clients similaires en difficultés de paiements)_")
                except:
                    st.info('**_Aucun Client Similaire_**')
            st.markdown("""---""")
            with st.container():
                st.write("#### Prédiction de solvabilité ")
                pred = st.button('Calcul')
                if pred:
                    with st.spinner('Calcul...'):
                        try:
                            # http://127.0.01:5000/ is from the flask api
                            prediction = requests.get("http://127.0.0.1:5000/predict?ID=" + str(ID)).json()
                            if prediction["target"]==0:
                                st.write(':smiley:')
                                st.success('Client solvable _(Target = 0)_, prédiction de difficultés à **' + str(prediction["risk"] * 100) + '%**')
                            elif prediction["target"]==1:
                                st.write(':confused:')
                                st.error('Client non solvable _(Target = 1)_, prédiction de difficultés à **' + str(prediction["risk"] * 100) + '%**')  
                            st.write('**Interprétabilité**')
                            fig = plt.figure(figsize=(2,2))
                            st.pyplot(shap_id(ID))
                        except :
                            st.warning('Erreur programme') 
                            st.write(':dizzy_face:')                                               
                    
    except:
        st.warning('**_Client Introuvable_**')
        
# Analyse du portefeuille client        
elif analyse == 'Portefeuille':
    st.write("### Analyse Portefeuille")
    with st.spinner('Analyse Portefeuille...'):
        with st.container():            
            st.write("#### Types de profil")
            col1, col2,col3 = st.columns(3)
            with col1:
                fig = plt.figure(figsize=(4,4))
                bins = (raw_app['AGE'].max()-raw_app['AGE'].min())//5
                pt = sns.histplot(data=raw_app, x='AGE', hue='TARGET',bins=bins,palette=['green','red'],alpha=.5)
                plt.xlabel('AGE',fontsize=12)
                plt.ylabel('')
                plt.legend(['avec difficulté','sans difficulté'],loc='lower center',bbox_to_anchor=(0.5, -0.35),fancybox=True, shadow=True, ncol=5)
                st.pyplot(fig)
            with col2:
                fig = plt.figure(figsize=(3,3))                
                pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==1],raw_app['CNT_CHILDREN'][raw_app['TARGET']==1],color='red',alpha=.5,ci=None,edgecolor='black')
                pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==0],raw_app['CNT_CHILDREN'][raw_app['TARGET']==0],color='green',alpha=.5,ci=None,edgecolor='black')
                #pt = sns.barplot(x='NAME_FAMILY_STATUS', y='CNT_CHILDREN', hue='TARGET', data=raw_app,palette=['green','red'],alpha=.7)
                plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                plt.setp(pt.get_yticklabels(),fontsize=5)
                st.pyplot(fig)
            with col3:
                fig = plt.figure(figsize=(4.5,4.5))
                pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==1],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==1],color='red',alpha=.5,ci=None,edgecolor='black')
                pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==0],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==0],color='green',alpha=.5,ci=None,edgecolor='black')
                #pt = sns.barplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', hue='TARGET', data=raw_app,palette=['green','red'],alpha=.7)
                plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                plt.setp(pt.get_yticklabels(),fontsize=7)
                st.pyplot(fig)
        st.markdown("""---""")
        with st.container():
            st.write("#### Paiement de crédit")
            tg_n = np.array([len(raw_app[raw_app['TARGET']==1]),len(raw_app[raw_app['TARGET']==0]),len(raw_app[raw_app['TARGET'].isnull()])])            
            col4, col5 = st.columns(2)
            with col4:
                fig = plt.figure(figsize=(5,5))
                plt.pie(tg_n,labels=['Avec difficultés','Sans difficultés','Aucun emprunt en cours'],colors=['red','green','blue'],autopct=lambda x:str(round(x,2))+'%')
                st.pyplot(fig)
            with col5:
                df = raw_app[['TARGET','NAME_INCOME_TYPE','AMT_ANNUITY','AMT_CREDIT']]
                df['COUNT_TG'] = df['TARGET']
                tg_df = pd.concat((df.groupby(['TARGET','NAME_INCOME_TYPE']).mean()[['AMT_ANNUITY','AMT_CREDIT']],df.groupby(['TARGET','NAME_INCOME_TYPE']).count()[['COUNT_TG']]), axis = 1)
                tg_0 = tg_df.loc[0]
                tg_1 = tg_df.loc[1]
                fig = plt.figure(figsize=(2,2))                  
                pt = sns.scatterplot(tg_1['AMT_ANNUITY'],tg_1['AMT_CREDIT'],s=tg_1['COUNT_TG'].values/100,label='Avec Difficulté',color='red')
                pt = sns.scatterplot(tg_0['AMT_ANNUITY'],tg_0['AMT_CREDIT'],s=tg_0['COUNT_TG'].values/100,label='Sans Difficulté',color='green',alpha=.3)
                plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=True, ncol=5,fontsize=5)
                plt.xlabel('AMT_ANNUITY',fontsize=5)
                plt.ylabel('AMT_CREDIT',fontsize=5)
                plt.xlim([20000,40000])
                plt.ylim([400000,800000])
                plt.setp(pt.get_xticklabels(),fontsize=4)
                plt.setp(pt.get_yticklabels(),fontsize=4)                
                st.pyplot(fig)
        st.markdown("""---""")
        with st.container():
            st.write('**Interprétabilité**')
            fig = plt.figure(figsize=(3,3))
            st.pyplot(shap_all())
        
