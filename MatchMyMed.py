# Machine Learning for medical vocacional predicition
### Create by Timothy Cavazzotto
### Started at 15/07/2024
### Ended 28/10/2024


# install
!pip install shap
!pip install pydot
!pip install hyperopt
!pip install numpy
!pip install pandas
!pip install xgboost
!pip install seaborn
!pip install sklearn
!pip install matplotlib
!pip install imblearn
!pip install openpyxl
!pip install catboost
!pip install lightgbm


# import
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import pydot
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

# import from
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from hyperopt import hp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import plot_tree


#########fazer dois modelos 1 com e 1 sem os dados de interesse

dados= pd.read_excel("dados_pred.xlsx")
dados


# criação de variáveis desfecho separadas do treino
Anestesiologia = dados["Especialidade:Anestesiologia"].copy()
Cardiologia = dados["Especialidade:Cardiologia"].copy()
Cirurgia_Geral = dados["Especialidade:Cirurgia.Geral"].copy()
Clínica_médica = dados["Especialidade:Clínica.Médica"].copy()
Dermatologia = dados["Especialidade:Dermatologia"].copy()
Ginecologia_e_obstetrícia = dados["Especialidade:Ginecologia.e.Obstetrícia"].copy()
Medicina_de_família_e_comunidade = dados["Especialidade:Medicina.de.Família.e.Comunidade"].copy()
Medicina_do_trabalho = dados["Especialidade:Medicina.do.Trabalho"].copy()
Oftalmologia = dados["Especialidade:Oftalmologia"].copy()
Ortopedia_e_traumatologia = dados["Especialidade:Ortopedia.e.traumatologia"].copy()
Outras = dados["Especialidade:Outras"].copy()
Pediatria = dados["Especialidade:Pediatria"].copy()
Psiquiatria = dados["Especialidade:Psiquiatria"].copy()
Radiologia_e_diagnóstico_por_imagem = dados["Especialidade:Radiologia.e.Diagnóstico.por.Imagem"].copy()
GrupoEsp_cirurgias = dados["GrupoEsp:Cirurgias"].copy()
GrupoEsp_clinica = dados["GrupoEsp:Clínicas"].copy()
GrupoEsp_GO = dados["GrupoEsp:Ginecologia.e.Obstetrícia"].copy()
GrupoEsp_PED = dados["GrupoEsp:Pediatria"].copy()
GrupoEsp_PSIQ = dados["GrupoEsp:Psiquiatria"].copy()

dados = dados.drop(["id",
"Especialidade:Anestesiologia",	
"Especialidade:Cardiologia",
"Especialidade:Cirurgia.Geral", 
"Especialidade:Clínica.Médica", 
"Especialidade:Dermatologia", 
"Especialidade:Ginecologia.e.Obstetrícia",
"Especialidade:Medicina.de.Família.e.Comunidade",
"Especialidade:Medicina.do.Trabalho",
"Especialidade:Oftalmologia",	
"Especialidade:Ortopedia.e.traumatologia",
"Especialidade:Outras",	
"Especialidade:Pediatria",
"Especialidade:Psiquiatria",
"Especialidade:Radiologia.e.Diagnóstico.por.Imagem",
"GrupoEsp:Cirurgias",
"GrupoEsp:Clínicas",
"GrupoEsp:Ginecologia.e.Obstetrícia",
"GrupoEsp:Pediatria",
"GrupoEsp:Psiquiatria",
], axis=1)


# Divisão em treinamento, validação e teste (80/20)


# Anestesiologia
X_train_anest, X_test_anest, y_train_anest, y_test_anest = train_test_split(dados, Anestesiologia,
                                                                           test_size=0.20,
                                                                           random_state=42,
                                                                           stratify=Anestesiologia)

# Cardiologia
X_train_cardio, X_test_cardio, y_train_cardio, y_test_cardio = train_test_split(dados, Cardiologia,
                                                                               test_size=0.20,
                                                                               random_state=42,
                                                                               stratify=Cardiologia)

# Cirurgia Geral
X_train_cirgen, X_test_cirgen, y_train_cirgen, y_test_cirgen = train_test_split(dados, Cirurgia_Geral,
                                                                               test_size=0.20,
                                                                               random_state=42,
                                                                               stratify=Cirurgia_Geral)

# Clínica Médica
X_train_clinmed, X_test_clinmed, y_train_clinmed, y_test_clinmed = train_test_split(dados, Clínica_médica,
                                                                                   test_size=0.20,
                                                                                   random_state=42,
                                                                                   stratify=Clínica_médica)

# Dermatologia
X_train_derm, X_test_derm, y_train_derm, y_test_derm = train_test_split(dados, Dermatologia,
                                                                       test_size=0.20,
                                                                       random_state=42,
                                                                       stratify=Dermatologia)

# Ginecologia e Obstetrícia
X_train_ginec, X_test_ginec, y_train_ginec, y_test_ginec = train_test_split(dados, Ginecologia_e_obstetrícia,
                                                                           test_size=0.20,
                                                                           random_state=42,
                                                                           stratify=Ginecologia_e_obstetrícia)

# Medicina de Família e Comunidade
X_train_medfam, X_test_medfam, y_train_medfam, y_test_medfam = train_test_split(dados, Medicina_de_família_e_comunidade,
                                                                               test_size=0.20,
                                                                               random_state=42,
                                                                               stratify=Medicina_de_família_e_comunidade)

# Medicina do Trabalho
X_train_medtrab, X_test_medtrab, y_train_medtrab, y_test_medtrab = train_test_split(dados, Medicina_do_trabalho,
                                                                                   test_size=0.20,
                                                                                   random_state=42,
                                                                                   stratify=Medicina_do_trabalho)

# Oftalmologia
X_train_oftal, X_test_oftal, y_train_oftal, y_test_oftal = train_test_split(dados, Oftalmologia,
                                                                           test_size=0.20,
                                                                           random_state=42,
                                                                           stratify=Oftalmologia)

# Ortopedia e Traumatologia
X_train_ortop, X_test_ortop, y_train_ortop, y_test_ortop = train_test_split(dados, Ortopedia_e_traumatologia,
                                                                           test_size=0.20,
                                                                           random_state=42,
                                                                           stratify=Ortopedia_e_traumatologia)

# Outras Especialidades
X_train_outras, X_test_outras, y_train_outras, y_test_outras = train_test_split(dados, Outras,
                                                                               test_size=0.20,
                                                                               random_state=42,
                                                                               stratify=Outras)

# Pediatria
X_train_ped, X_test_ped, y_train_ped, y_test_ped = train_test_split(dados, Pediatria,
                                                                   test_size=0.20,
                                                                   random_state=42,
                                                                   stratify=Pediatria)

# Psiquiatria
X_train_psiq, X_test_psiq, y_train_psiq, y_test_psiq = train_test_split(dados, Psiquiatria,
                                                                       test_size=0.20,
                                                                       random_state=42,
                                                                       stratify=Psiquiatria)

# Radiologia e Diagnóstico por Imagem
X_train_radio, X_test_radio, y_train_radio, y_test_radio = train_test_split(dados, Radiologia_e_diagnóstico_por_imagem,
                                                                           test_size=0.20,
                                                                           random_state=42,
                                                                           stratify=Radiologia_e_diagnóstico_por_imagem)


#GRUPOS
# Cirurgias
X_train_grupo_cirurgias, X_test_grupo_cirurgias, y_train_grupo_cirurgias, y_test_grupo_cirurgias = train_test_split(
    dados, GrupoEsp_cirurgias, test_size=0.20, random_state=42, stratify=GrupoEsp_cirurgias)

# Clínicas
X_train_grupo_clinicas, X_test_grupo_clinicas, y_train_grupo_clinicas, y_test_grupo_clinicas = train_test_split(
    dados, GrupoEsp_clinica, test_size=0.20, random_state=42, stratify=GrupoEsp_clinica)

# Ginecologia e Obstetrícia
X_train_grupo_ginecologia, X_test_grupo_ginecologia, y_train_grupo_ginecologia, y_test_grupo_ginecologia = train_test_split(
    dados, GrupoEsp_GO, test_size=0.20, random_state=42, stratify=GrupoEsp_GO)

# Pediatria
X_train_grupo_pediatria, X_test_grupo_pediatria, y_train_grupo_pediatria, y_test_grupo_pediatria = train_test_split(
    dados, GrupoEsp_PED, test_size=0.20, random_state=42, stratify=GrupoEsp_PED)

# Psiquiatria
X_train_grupo_psiquiatria, X_test_grupo_psiquiatria, y_train_grupo_psiquiatria, y_test_grupo_psiquiatria = train_test_split(
    dados, GrupoEsp_PSIQ, test_size=0.20, random_state=42, stratify=GrupoEsp_PSIQ)


#  balancear as categorias para o treino
from imblearn.over_sampling import RandomOverSampler
over = RandomOverSampler(random_state=42)
X_train_anest, y_train_anest = over.fit_resample(X_train_anest, y_train_anest)
X_train_cardio, y_train_cardio = over.fit_resample(X_train_cardio, y_train_cardio)
X_train_cirgen, y_train_cirgen = over.fit_resample(X_train_cirgen, y_train_cirgen)
X_train_clinmed, y_train_clinmed = over.fit_resample(X_train_clinmed, y_train_clinmed)
X_train_derm, y_train_derm = over.fit_resample(X_train_derm, y_train_derm)
X_train_ginec, y_train_ginec = over.fit_resample(X_train_ginec, y_train_ginec)
X_train_medfam, y_train_medfam = over.fit_resample(X_train_medfam, y_train_medfam)
X_train_medtrab, y_train_medtrab = over.fit_resample(X_train_medtrab, y_train_medtrab)
X_train_oftal, y_train_oftal = over.fit_resample(X_train_oftal, y_train_oftal)
X_train_ortop, y_train_ortop = over.fit_resample(X_train_ortop, y_train_ortop)
X_train_outras, y_train_outras = over.fit_resample(X_train_outras, y_train_outras)
X_train_ped, y_train_ped = over.fit_resample(X_train_ped, y_train_ped)
X_train_psiq, y_train_psiq = over.fit_resample(X_train_psiq, y_train_psiq)
X_train_radio, y_train_radio = over.fit_resample(X_train_radio, y_train_radio)
# Grupos
X_train_grupo_cirurgias, y_train_grupo_cirurgias = over.fit_resample(X_train_grupo_cirurgias, y_train_grupo_cirurgias)
X_train_grupo_clinicas, y_train_grupo_clinicas = over.fit_resample(X_train_grupo_clinicas, y_train_grupo_clinicas)
X_train_grupo_ginecologia, y_train_grupo_ginecologia = over.fit_resample(X_train_grupo_ginecologia, y_train_grupo_ginecologia)
X_train_grupo_pediatria, y_train_grupo_pediatria = over.fit_resample(X_train_grupo_pediatria, y_train_grupo_pediatria)
X_train_grupo_psiquiatria, y_train_grupo_psiquiatria = over.fit_resample(X_train_grupo_psiquiatria, y_train_grupo_psiquiatria)


##### XGBOOST ###########################
##### Estrutura dos hiperparametros #####

xgboost_space = {
    'learning_rate': [0.01, 0.001],  # determina a contribuição de cada árvore no modelo final.
    'max_depth': [3, 5],               # número de camadas das arvores
    'n_estimators': [50, 100],       # total de arvores a serem constru?das para formar o comit? de decis?o
    'subsample': [0.80, 0.90],      # percentual de amostra utilizada para treinara cada ?rvore (amostra com reposi??o) 
    'scale_pos_weight': [1],        # Peso relativo das classes positivas
    'eval_metric': ['auc'],               # Metrica de avaliação
    'seed': [42]                          # Semente aleatória para reprodu??o dos resultados.
    }


#Predizer o grupo
# Grupo: Cirurgias
xgb_classifier_grupo_cirurgias = XGBClassifier()
grid_search_grupo_cirurgias = GridSearchCV(estimator=xgb_classifier_grupo_cirurgias,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)



# "GrupoEsp:Clínicas",
xgb_classifier_grupo_clinicas = XGBClassifier()
grid_search_grupo_clinicas = GridSearchCV(estimator=xgb_classifier_grupo_clinicas,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)



# "GrupoEsp:Ginecologia.e.Obstetrícia",
xgb_classifier_grupo_ginecologia = XGBClassifier()
grid_search_grupo_ginecologia = GridSearchCV(estimator=xgb_classifier_grupo_ginecologia,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)


# "GrupoEsp:Pediatria"
xgb_classifier_grupo_pediatria = XGBClassifier()
grid_search_grupo_pediatria = GridSearchCV(estimator=xgb_classifier_grupo_pediatria,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)



#"GrupoEsp:Psiquiatria",
xgb_classifier_grupo_psiquiatria = XGBClassifier()
grid_search_grupo_psiquiatria = GridSearchCV(estimator=xgb_classifier_grupo_psiquiatria,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)

# Executar a busca em grade
grid_search_grupo_cirurgias.fit(X_train_grupo_cirurgias, y_train_grupo_cirurgias)
grid_search_grupo_clinicas.fit(X_train_grupo_clinicas, y_train_grupo_clinicas)
grid_search_grupo_ginecologia.fit(X_train_grupo_ginecologia, y_train_grupo_ginecologia)
grid_search_grupo_pediatria.fit(X_train_grupo_pediatria, y_train_grupo_pediatria)
grid_search_grupo_psiquiatria.fit(X_train_grupo_psiquiatria, y_train_grupo_psiquiatria)



# Cirurgias
best_params_grupo_cirurgias = grid_search_grupo_cirurgias.best_params_
best_score_grupo_cirurgias = grid_search_grupo_cirurgias.best_score_
best_xgb_classifier_grupo_cirurgias = XGBClassifier(**best_params_grupo_cirurgias)
best_xgb_classifier_grupo_cirurgias.fit(X_train_grupo_cirurgias, y_train_grupo_cirurgias)

# Clínicas
best_params_grupo_clinicas = grid_search_grupo_clinicas.best_params_
best_score_grupo_clinicas = grid_search_grupo_clinicas.best_score_
best_xgb_classifier_grupo_clinicas = XGBClassifier(**best_params_grupo_clinicas)
best_xgb_classifier_grupo_clinicas.fit(X_train_grupo_clinicas, y_train_grupo_clinicas)

# Ginecologia e Obstetrícia
best_params_grupo_ginecologia = grid_search_grupo_ginecologia.best_params_
best_score_grupo_ginecologia = grid_search_grupo_ginecologia.best_score_
best_xgb_classifier_grupo_ginecologia = XGBClassifier(**best_params_grupo_ginecologia)
best_xgb_classifier_grupo_ginecologia.fit(X_train_grupo_ginecologia, y_train_grupo_ginecologia)

# Pediatria
best_params_grupo_pediatria = grid_search_grupo_pediatria.best_params_
best_score_grupo_pediatria = grid_search_grupo_pediatria.best_score_
best_xgb_classifier_grupo_pediatria = XGBClassifier(**best_params_grupo_pediatria)
best_xgb_classifier_grupo_pediatria.fit(X_train_grupo_pediatria, y_train_grupo_pediatria)

# Psiquiatria
best_params_grupo_psiquiatria = grid_search_grupo_psiquiatria.best_params_
best_score_grupo_psiquiatria = grid_search_grupo_psiquiatria.best_score_
best_xgb_classifier_grupo_psiquiatria = XGBClassifier(**best_params_grupo_psiquiatria)
best_xgb_classifier_grupo_psiquiatria.fit(X_train_grupo_psiquiatria, y_train_grupo_psiquiatria)

# Predição
y_pred_xgb_grupo_cirurgias = best_xgb_classifier_grupo_cirurgias.predict(X_test_grupo_cirurgias)
y_pred_xgb_grupo_clinicas = best_xgb_classifier_grupo_clinicas.predict(X_test_grupo_clinicas)
y_pred_xgb_grupo_ginecologia = best_xgb_classifier_grupo_ginecologia.predict(X_test_grupo_ginecologia)
y_pred_xgb_grupo_pediatria = best_xgb_classifier_grupo_pediatria.predict(X_test_grupo_pediatria)
y_pred_xgb_grupo_psiquiatria = best_xgb_classifier_grupo_psiquiatria.predict(X_test_grupo_psiquiatria)

# ROC
roc_auc_grupo_cirurgias = roc_auc_score(y_test_grupo_cirurgias, y_pred_xgb_grupo_cirurgias)
roc_auc_grupo_clinicas = roc_auc_score(y_test_grupo_clinicas, y_pred_xgb_grupo_clinicas)
roc_auc_grupo_ginecologia = roc_auc_score(y_test_grupo_ginecologia, y_pred_xgb_grupo_ginecologia)
roc_auc_grupo_pediatria = roc_auc_score(y_test_grupo_pediatria, y_pred_xgb_grupo_pediatria)
roc_auc_grupo_psiquiatria = roc_auc_score(y_test_grupo_psiquiatria, y_pred_xgb_grupo_psiquiatria)

confusion_matrix_xgb_grupo_cirurgias = confusion_matrix(y_test_grupo_cirurgias, y_pred_xgb_grupo_cirurgias)
confusion_matrix_xgb_grupo_clinicas = confusion_matrix(y_test_grupo_clinicas, y_pred_xgb_grupo_clinicas)
confusion_matrix_xgb_grupo_ginecologia = confusion_matrix(y_test_grupo_ginecologia, y_pred_xgb_grupo_ginecologia)
confusion_matrix_xgb_grupo_pediatria = confusion_matrix(y_test_grupo_pediatria, y_pred_xgb_grupo_pediatria)
confusion_matrix_xgb_grupo_psiquiatria = confusion_matrix(y_test_grupo_psiquiatria, y_pred_xgb_grupo_psiquiatria)
################################################################################
############# PREDIÇÃO DAS ESPECIALIDADES ##############



# Anestesiologia
xgb_classifier_anest = XGBClassifier()
grid_search_anest = GridSearchCV(estimator=xgb_classifier_anest,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)

# Cardiologia
xgb_classifier_cardio = XGBClassifier()
grid_search_cardio = GridSearchCV(estimator=xgb_classifier_cardio,
                                  param_grid=xgboost_space,
                                  scoring='roc_auc',
                                  error_score='raise',
                                  cv=5)

# Cirurgia Geral
xgb_classifier_cirgen = XGBClassifier()
grid_search_cirgen = GridSearchCV(estimator=xgb_classifier_cirgen,
                                  param_grid=xgboost_space,
                                  scoring='roc_auc',
                                  error_score='raise',
                                  cv=5)

# Clínica Médica
xgb_classifier_clinmed = XGBClassifier()
grid_search_clinmed = GridSearchCV(estimator=xgb_classifier_clinmed,
                                   param_grid=xgboost_space,
                                   scoring='roc_auc',
                                   error_score='raise',
                                   cv=5)

# Dermatologia
xgb_classifier_derm = XGBClassifier()
grid_search_derm = GridSearchCV(estimator=xgb_classifier_derm,
                                param_grid=xgboost_space,
                                scoring='roc_auc',
                                error_score='raise',
                                cv=5)

# Ginecologia e Obstetrícia
xgb_classifier_ginec = XGBClassifier()
grid_search_ginec = GridSearchCV(estimator=xgb_classifier_ginec,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)

# Medicina de Família e Comunidade
xgb_classifier_medfam = XGBClassifier()
grid_search_medfam = GridSearchCV(estimator=xgb_classifier_medfam,
                                  param_grid=xgboost_space,
                                  scoring='roc_auc',
                                  error_score='raise',
                                  cv=5)

# Medicina do Trabalho
xgb_classifier_medtrab = XGBClassifier()
grid_search_medtrab = GridSearchCV(estimator=xgb_classifier_medtrab,
                                   param_grid=xgboost_space,
                                   scoring='roc_auc',
                                   error_score='raise',
                                   cv=5)

# Oftalmologia
xgb_classifier_oftal = XGBClassifier()
grid_search_oftal = GridSearchCV(estimator=xgb_classifier_oftal,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)

# Ortopedia e Traumatologia
xgb_classifier_ortop = XGBClassifier()
grid_search_ortop = GridSearchCV(estimator=xgb_classifier_ortop,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)

# Outras Especialidades
xgb_classifier_outras = XGBClassifier()
grid_search_outras = GridSearchCV(estimator=xgb_classifier_outras,
                                  param_grid=xgboost_space,
                                  scoring='roc_auc',
                                  error_score='raise',
                                  cv=5)

# Pediatria
xgb_classifier_ped = XGBClassifier()
grid_search_ped = GridSearchCV(estimator=xgb_classifier_ped,
                               param_grid=xgboost_space,
                               scoring='roc_auc',
                               error_score='raise',
                               cv=5)

# Psiquiatria
xgb_classifier_psiq = XGBClassifier()
grid_search_psiq = GridSearchCV(estimator=xgb_classifier_psiq,
                                param_grid=xgboost_space,
                                scoring='roc_auc',
                                error_score='raise',
                                cv=5)

# Radiologia e Diagnóstico por Imagem
xgb_classifier_radio = XGBClassifier()
grid_search_radio = GridSearchCV(estimator=xgb_classifier_radio,
                                 param_grid=xgboost_space,
                                 scoring='roc_auc',
                                 error_score='raise',
                                 cv=5)


# Executar a busca em grade
grid_search_anest.fit(X_train_anest, y_train_anest)
grid_search_cardio.fit(X_train_cardio, y_train_cardio)
grid_search_cirgen.fit(X_train_cirgen, y_train_cirgen)
grid_search_clinmed.fit(X_train_clinmed, y_train_clinmed)
grid_search_derm.fit(X_train_derm, y_train_derm)
grid_search_ginec.fit(X_train_ginec, y_train_ginec)
grid_search_medfam.fit(X_train_medfam, y_train_medfam)
grid_search_medtrab.fit(X_train_medtrab, y_train_medtrab)
grid_search_oftal.fit(X_train_oftal, y_train_oftal)
grid_search_ortop.fit(X_train_ortop, y_train_ortop)
grid_search_outras.fit(X_train_outras, y_train_outras)
grid_search_ped.fit(X_train_ped, y_train_ped)
grid_search_psiq.fit(X_train_psiq, y_train_psiq)
grid_search_radio.fit(X_train_radio, y_train_radio)





# Obter os melhores hiperparâmetros e o melhor desempenho para Anestesiologia
best_params_anest = grid_search_anest.best_params_
best_score_anest = grid_search_anest.best_score_
best_xgb_classifier_anest = XGBClassifier(**best_params_anest)
best_xgb_classifier_anest.fit(X_train_anest, y_train_anest)

# Obter os melhores hiperparâmetros e o melhor desempenho para Cardiologia
best_params_cardio = grid_search_cardio.best_params_
best_score_cardio = grid_search_cardio.best_score_
best_xgb_classifier_cardio = XGBClassifier(**best_params_cardio)
best_xgb_classifier_cardio.fit(X_train_cardio, y_train_cardio)

# Obter os melhores hiperparâmetros e o melhor desempenho para Cirurgia Geral
best_params_cirgen = grid_search_cirgen.best_params_
best_score_cirgen = grid_search_cirgen.best_score_
best_xgb_classifier_cirgen = XGBClassifier(**best_params_cirgen)
best_xgb_classifier_cirgen.fit(X_train_cirgen, y_train_cirgen)

# Obter os melhores hiperparâmetros e o melhor desempenho para Clínica Médica
best_params_clinmed = grid_search_clinmed.best_params_
best_score_clinmed = grid_search_clinmed.best_score_
best_xgb_classifier_clinmed = XGBClassifier(**best_params_clinmed)
best_xgb_classifier_clinmed.fit(X_train_clinmed, y_train_clinmed)

# Obter os melhores hiperparâmetros e o melhor desempenho para Dermatologia
best_params_derm = grid_search_derm.best_params_
best_score_derm = grid_search_derm.best_score_
best_xgb_classifier_derm = XGBClassifier(**best_params_derm)
best_xgb_classifier_derm.fit(X_train_derm, y_train_derm)

# Obter os melhores hiperparâmetros e o melhor desempenho para Ginecologia e Obstetrícia
best_params_ginec = grid_search_ginec.best_params_
best_score_ginec = grid_search_ginec.best_score_
best_xgb_classifier_ginec = XGBClassifier(**best_params_ginec)
best_xgb_classifier_ginec.fit(X_train_ginec, y_train_ginec)

# Obter os melhores hiperparâmetros e o melhor desempenho para Medicina de Família e Comunidade
best_params_medfam = grid_search_medfam.best_params_
best_score_medfam = grid_search_medfam.best_score_
best_xgb_classifier_medfam = XGBClassifier(**best_params_medfam)
best_xgb_classifier_medfam.fit(X_train_medfam, y_train_medfam)

# Obter os melhores hiperparâmetros e o melhor desempenho para Medicina do Trabalho
best_params_medtrab = grid_search_medtrab.best_params_
best_score_medtrab = grid_search_medtrab.best_score_
best_xgb_classifier_medtrab = XGBClassifier(**best_params_medtrab)
best_xgb_classifier_medtrab.fit(X_train_medtrab, y_train_medtrab)

# Obter os melhores hiperparâmetros e o melhor desempenho para Oftalmologia
best_params_oftal = grid_search_oftal.best_params_
best_score_oftal = grid_search_oftal.best_score_
best_xgb_classifier_oftal = XGBClassifier(**best_params_oftal)
best_xgb_classifier_oftal.fit(X_train_oftal, y_train_oftal)

# Obter os melhores hiperparâmetros e o melhor desempenho para Ortopedia e Traumatologia
best_params_ortop = grid_search_ortop.best_params_
best_score_ortop = grid_search_ortop.best_score_
best_xgb_classifier_ortop = XGBClassifier(**best_params_ortop)
best_xgb_classifier_ortop.fit(X_train_ortop, y_train_ortop)

# Obter os melhores hiperparâmetros e o melhor desempenho para Outras Especialidades
best_params_outras = grid_search_outras.best_params_
best_score_outras = grid_search_outras.best_score_
best_xgb_classifier_outras = XGBClassifier(**best_params_outras)
best_xgb_classifier_outras.fit(X_train_outras, y_train_outras)

# Obter os melhores hiperparâmetros e o melhor desempenho para Pediatria
best_params_ped = grid_search_ped.best_params_
best_score_ped = grid_search_ped.best_score_
best_xgb_classifier_ped = XGBClassifier(**best_params_ped)
best_xgb_classifier_ped.fit(X_train_ped, y_train_ped)

# Obter os melhores hiperparâmetros e o melhor desempenho para Psiquiatria
best_params_psiq = grid_search_psiq.best_params_
best_score_psiq = grid_search_psiq.best_score_
best_xgb_classifier_psiq = XGBClassifier(**best_params_psiq)
best_xgb_classifier_psiq.fit(X_train_psiq, y_train_psiq)

# Obter os melhores hiperparâmetros e o melhor desempenho para Radiologia e Diagnóstico por Imagem
best_params_radio = grid_search_radio.best_params_
best_score_radio = grid_search_radio.best_score_
best_xgb_classifier_radio = XGBClassifier(**best_params_radio)
best_xgb_classifier_radio.fit(X_train_radio, y_train_radio)

# Previsões para cada especialidade
y_pred_xgb_anest = best_xgb_classifier_anest.predict(X_test_anest)
y_pred_xgb_cardio = best_xgb_classifier_cardio.predict(X_test_cardio)
y_pred_xgb_cirgen = best_xgb_classifier_cirgen.predict(X_test_cirgen)
y_pred_xgb_clinmed = best_xgb_classifier_clinmed.predict(X_test_clinmed)
y_pred_xgb_derm = best_xgb_classifier_derm.predict(X_test_derm)
y_pred_xgb_ginec = best_xgb_classifier_ginec.predict(X_test_ginec)
y_pred_xgb_medfam = best_xgb_classifier_medfam.predict(X_test_medfam)
y_pred_xgb_medtrab = best_xgb_classifier_medtrab.predict(X_test_medtrab)
y_pred_xgb_oftal = best_xgb_classifier_oftal.predict(X_test_oftal)
y_pred_xgb_ortop = best_xgb_classifier_ortop.predict(X_test_ortop)
y_pred_xgb_outras = best_xgb_classifier_outras.predict(X_test_outras)
y_pred_xgb_ped = best_xgb_classifier_ped.predict(X_test_ped)
y_pred_xgb_psiq = best_xgb_classifier_psiq.predict(X_test_psiq)
y_pred_xgb_radio = best_xgb_classifier_radio.predict(X_test_radio)

# Calcular ROC AUC para cada especialidade
roc_auc_anest = roc_auc_score(y_test_anest, y_pred_xgb_anest)
roc_auc_cardio = roc_auc_score(y_test_cardio, y_pred_xgb_cardio)
roc_auc_cirgen = roc_auc_score(y_test_cirgen, y_pred_xgb_cirgen)
roc_auc_clinmed = roc_auc_score(y_test_clinmed, y_pred_xgb_clinmed)
roc_auc_derm = roc_auc_score(y_test_derm, y_pred_xgb_derm)
roc_auc_ginec = roc_auc_score(y_test_ginec, y_pred_xgb_ginec)
roc_auc_medfam = roc_auc_score(y_test_medfam, y_pred_xgb_medfam)
roc_auc_medtrab = roc_auc_score(y_test_medtrab, y_pred_xgb_medtrab)
roc_auc_oftal = roc_auc_score(y_test_oftal, y_pred_xgb_oftal)
roc_auc_ortop = roc_auc_score(y_test_ortop, y_pred_xgb_ortop)
roc_auc_outras = roc_auc_score(y_test_outras, y_pred_xgb_outras)
roc_auc_ped = roc_auc_score(y_test_ped, y_pred_xgb_ped)
roc_auc_psiq = roc_auc_score(y_test_psiq, y_pred_xgb_psiq)
roc_auc_radio = roc_auc_score(y_test_radio, y_pred_xgb_radio)

# Criar uma tabela com os resultados de ROC AUC
roc_auc_table = pd.DataFrame({
    'Especialidade': [
        'Anestesiologia', 'Cardiologia', 'Cirurgia Geral', 'Clínica Médica',
        'Dermatologia', 'Ginecologia e Obstetrícia', 'Medicina de Família e Comunidade',
        'Medicina do Trabalho', 'Oftalmologia', 'Ortopedia e Traumatologia', 
        'Outras Especialidades', 'Pediatria', 'Psiquiatria', 'Radiologia e Diagnóstico por Imagem'
    ],
    'ROC AUC': [
        roc_auc_anest, roc_auc_cardio, roc_auc_cirgen, roc_auc_clinmed, 
        roc_auc_derm, roc_auc_ginec, roc_auc_medfam, roc_auc_medtrab, 
        roc_auc_oftal, roc_auc_ortop, roc_auc_outras, roc_auc_ped, 
        roc_auc_psiq, roc_auc_radio
    ]
})

# Exibir a tabela
print(roc_auc_table)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Matrizes de confusão para o XGBoost
confusion_matrix_xgb_anest = confusion_matrix(y_test_anest, y_pred_xgb_anest)
confusion_matrix_xgb_cardio = confusion_matrix(y_test_cardio, y_pred_xgb_cardio)
confusion_matrix_xgb_cirgen = confusion_matrix(y_test_cirgen, y_pred_xgb_cirgen)
confusion_matrix_xgb_clinmed = confusion_matrix(y_test_clinmed, y_pred_xgb_clinmed)
confusion_matrix_xgb_derm = confusion_matrix(y_test_derm, y_pred_xgb_derm)
confusion_matrix_xgb_ginec = confusion_matrix(y_test_ginec, y_pred_xgb_ginec)
confusion_matrix_xgb_medfam = confusion_matrix(y_test_medfam, y_pred_xgb_medfam)
confusion_matrix_xgb_medtrab = confusion_matrix(y_test_medtrab, y_pred_xgb_medtrab)
confusion_matrix_xgb_oftal = confusion_matrix(y_test_oftal, y_pred_xgb_oftal)
confusion_matrix_xgb_ortop = confusion_matrix(y_test_ortop, y_pred_xgb_ortop)
confusion_matrix_xgb_outras = confusion_matrix(y_test_outras, y_pred_xgb_outras)
confusion_matrix_xgb_ped = confusion_matrix(y_test_ped, y_pred_xgb_ped)
confusion_matrix_xgb_psiq = confusion_matrix(y_test_psiq, y_pred_xgb_psiq)
confusion_matrix_xgb_radio = confusion_matrix(y_test_radio, y_pred_xgb_radio)

# Criar uma tabela com as matrizes de confusão para XGBoost e CatBoost
confusion_matrices_xgb = {
    'Anestesiologia': confusion_matrix_xgb_anest,
    'Cardiologia': confusion_matrix_xgb_cardio,
    'Cirurgia Geral': confusion_matrix_xgb_cirgen,
    'Clínica Médica': confusion_matrix_xgb_clinmed,
    'Dermatologia': confusion_matrix_xgb_derm,
    'Ginecologia e Obstetrícia': confusion_matrix_xgb_ginec,
    'Medicina de Família e Comunidade': confusion_matrix_xgb_medfam,
    'Medicina do Trabalho': confusion_matrix_xgb_medtrab,
    'Oftalmologia': confusion_matrix_xgb_oftal,
    'Ortopedia e Traumatologia': confusion_matrix_xgb_ortop,
    'Outras Especialidades': confusion_matrix_xgb_outras,
    'Pediatria': confusion_matrix_xgb_ped,
    'Psiquiatria': confusion_matrix_xgb_psiq,
    'Radiologia e Diagnóstico por Imagem': confusion_matrix_xgb_radio
}

# Exibir ou salvar as matrizes de confusão conforme necessário
for especialidade, matrix in confusion_matrices_xgb.items():
    print(f"Matriz de Confusão XGBoost - {especialidade}:\n", matrix, "\n")





##### CATBOOST ################################
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
### Estrutura de hiperpar?metros ##############
cat = {
  'iterations': [100, 150],
  'learning_rate': [0.01, 0.001],  
  'max_depth': [3, 5],
  'subsample': [0.80, 0.90],      
  'scale_pos_weight': [1, 2],       
  'random_state': [42]
  }


# Inicializar o classificador CatBoost
catboostclassifier = CatBoostClassifier()


# RandomizedSearch para Anestesiologia (ANEST)
random_search_anest_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                  param_distributions=cat,
                                                  n_iter=20,  # Número de combinações a testar
                                                  scoring='roc_auc',
                                                  error_score='raise',
                                                  cv=5,
                                                  random_state=42)

# RandomizedSearch para Cardiologia (CARD)
random_search_cardio_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                   param_distributions=cat,
                                                   n_iter=20,
                                                   scoring='roc_auc',
                                                   error_score='raise',
                                                   cv=5,
                                                   random_state=42)

# RandomizedSearch para Cirurgia Geral (CIRGEN)
random_search_cirgen_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                   param_distributions=cat,
                                                   n_iter=20,
                                                   scoring='roc_auc',
                                                   error_score='raise',
                                                   cv=5,
                                                   random_state=42)

# RandomizedSearch para Clínica Médica (CLINMED)
random_search_clinmed_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                    param_distributions=cat,
                                                    n_iter=20,
                                                    scoring='roc_auc',
                                                    error_score='raise',
                                                    cv=5,
                                                    random_state=42)

# RandomizedSearch para Dermatologia (DERM)
random_search_derm_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                 param_distributions=cat,
                                                 n_iter=20,
                                                 scoring='roc_auc',
                                                 error_score='raise',
                                                 cv=5,
                                                 random_state=42)

# RandomizedSearch para Ginecologia e Obstetrícia (GINEC)
random_search_ginec_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                  param_distributions=cat,
                                                  n_iter=20,
                                                  scoring='roc_auc',
                                                  error_score='raise',
                                                  cv=5,
                                                  random_state=42)

# RandomizedSearch para Medicina de Família e Comunidade (MEDFAM)
random_search_medfam_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                   param_distributions=cat,
                                                   n_iter=20,
                                                   scoring='roc_auc',
                                                   error_score='raise',
                                                   cv=5,
                                                   random_state=42)

# RandomizedSearch para Medicina do Trabalho (MEDTRAB)
random_search_medtrab_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                    param_distributions=cat,
                                                    n_iter=20,
                                                    scoring='roc_auc',
                                                    error_score='raise',
                                                    cv=5,
                                                    random_state=42)

# RandomizedSearch para Oftalmologia (OFTAL)
random_search_oftal_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                  param_distributions=cat,
                                                  n_iter=20,
                                                  scoring='roc_auc',
                                                  error_score='raise',
                                                  cv=5,
                                                  random_state=42)

# RandomizedSearch para Ortopedia e Traumatologia (ORTOP)
random_search_ortop_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                  param_distributions=cat,
                                                  n_iter=20,
                                                  scoring='roc_auc',
                                                  error_score='raise',
                                                  cv=5,
                                                  random_state=42)

# RandomizedSearch para Outras Especialidades (OUTRAS)
random_search_outras_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                   param_distributions=cat,
                                                   n_iter=20,
                                                   scoring='roc_auc',
                                                   error_score='raise',
                                                   cv=5,
                                                   random_state=42)

# RandomizedSearch para Pediatria (PED)
random_search_ped_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                param_distributions=cat,
                                                n_iter=20,
                                                scoring='roc_auc',
                                                error_score='raise',
                                                cv=5,
                                                random_state=42)

# RandomizedSearch para Psiquiatria (PSIQ)
random_search_psiq_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                 param_distributions=cat,
                                                 n_iter=20,
                                                 scoring='roc_auc',
                                                 error_score='raise',
                                                 cv=5,
                                                 random_state=42)

# RandomizedSearch para Radiologia e Diagnóstico por Imagem (RADIO)
random_search_radio_catBoost = RandomizedSearchCV(estimator=catboostclassifier,
                                                  param_distributions=cat,
                                                  n_iter=20,
                                                  scoring='roc_auc',
                                                  error_score='raise',
                                                  cv=5,
                                                  random_state=42)
                                                  
                                                  
# Ajustar modelo CatBoost para Anestesiologia (ANEST)
random_search_anest_catBoost.fit(X_train_anest, y_train_anest)
best_params_anest = random_search_anest_catBoost.best_params_
best_score_anest = random_search_anest_catBoost.best_score_
best_catboost_classifier_anest = CatBoostClassifier(**best_params_anest)
best_catboost_classifier_anest.fit(X_train_anest, y_train_anest)

# Ajustar modelo CatBoost para Cardiologia (CARD)
random_search_cardio_catBoost.fit(X_train_cardio, y_train_cardio)
best_params_cardio = random_search_cardio_catBoost.best_params_
best_score_cardio = random_search_cardio_catBoost.best_score_
best_catboost_classifier_cardio = CatBoostClassifier(**best_params_cardio)
best_catboost_classifier_cardio.fit(X_train_cardio, y_train_cardio)

# Ajustar modelo CatBoost para Cirurgia Geral (CIRGEN)
random_search_cirgen_catBoost.fit(X_train_cirgen, y_train_cirgen)
best_params_cirgen = random_search_cirgen_catBoost.best_params_
best_score_cirgen = random_search_cirgen_catBoost.best_score_
best_catboost_classifier_cirgen = CatBoostClassifier(**best_params_cirgen)
best_catboost_classifier_cirgen.fit(X_train_cirgen, y_train_cirgen)

# Ajustar modelo CatBoost para Clínica Médica (CLINMED)
random_search_clinmed_catBoost.fit(X_train_clinmed, y_train_clinmed)
best_params_clinmed = random_search_clinmed_catBoost.best_params_
best_score_clinmed = random_search_clinmed_catBoost.best_score_
best_catboost_classifier_clinmed = CatBoostClassifier(**best_params_clinmed)
best_catboost_classifier_clinmed.fit(X_train_clinmed, y_train_clinmed)

# Ajustar modelo CatBoost para Dermatologia (DERM)
random_search_derm_catBoost.fit(X_train_derm, y_train_derm)
best_params_derm = random_search_derm_catBoost.best_params_
best_score_derm = random_search_derm_catBoost.best_score_
best_catboost_classifier_derm = CatBoostClassifier(**best_params_derm)
best_catboost_classifier_derm.fit(X_train_derm, y_train_derm)

# Ajustar modelo CatBoost para Ginecologia e Obstetrícia (GINEC)
random_search_ginec_catBoost.fit(X_train_ginec, y_train_ginec)
best_params_ginec = random_search_ginec_catBoost.best_params_
best_score_ginec = random_search_ginec_catBoost.best_score_
best_catboost_classifier_ginec = CatBoostClassifier(**best_params_ginec)
best_catboost_classifier_ginec.fit(X_train_ginec, y_train_ginec)

# Ajustar modelo CatBoost para Medicina de Família e Comunidade (MEDFAM)
random_search_medfam_catBoost.fit(X_train_medfam, y_train_medfam)
best_params_medfam = random_search_medfam_catBoost.best_params_
best_score_medfam = random_search_medfam_catBoost.best_score_
best_catboost_classifier_medfam = CatBoostClassifier(**best_params_medfam)
best_catboost_classifier_medfam.fit(X_train_medfam, y_train_medfam)

# Ajustar modelo CatBoost para Medicina do Trabalho (MEDTRAB)
random_search_medtrab_catBoost.fit(X_train_medtrab, y_train_medtrab)
best_params_medtrab = random_search_medtrab_catBoost.best_params_
best_score_medtrab = random_search_medtrab_catBoost.best_score_
best_catboost_classifier_medtrab = CatBoostClassifier(**best_params_medtrab)
best_catboost_classifier_medtrab.fit(X_train_medtrab, y_train_medtrab)

# Ajustar modelo CatBoost para Oftalmologia (OFTAL)
random_search_oftal_catBoost.fit(X_train_oftal, y_train_oftal)
best_params_oftal = random_search_oftal_catBoost.best_params_
best_score_oftal = random_search_oftal_catBoost.best_score_
best_catboost_classifier_oftal = CatBoostClassifier(**best_params_oftal)
best_catboost_classifier_oftal.fit(X_train_oftal, y_train_oftal)

# Ajustar modelo CatBoost para Ortopedia e Traumatologia (ORTOP)
random_search_ortop_catBoost.fit(X_train_ortop, y_train_ortop)
best_params_ortop = random_search_ortop_catBoost.best_params_
best_score_ortop = random_search_ortop_catBoost.best_score_
best_catboost_classifier_ortop = CatBoostClassifier(**best_params_ortop)
best_catboost_classifier_ortop.fit(X_train_ortop, y_train_ortop)

# Ajustar modelo CatBoost para Outras Especialidades (OUTRAS)
random_search_outras_catBoost.fit(X_train_outras, y_train_outras)
best_params_outras = random_search_outras_catBoost.best_params_
best_score_outras = random_search_outras_catBoost.best_score_
best_catboost_classifier_outras = CatBoostClassifier(**best_params_outras)
best_catboost_classifier_outras.fit(X_train_outras, y_train_outras)

# Ajustar modelo CatBoost para Pediatria (PED)
random_search_ped_catBoost.fit(X_train_ped, y_train_ped)
best_params_ped = random_search_ped_catBoost.best_params_
best_score_ped = random_search_ped_catBoost.best_score_
best_catboost_classifier_ped = CatBoostClassifier(**best_params_ped)
best_catboost_classifier_ped.fit(X_train_ped, y_train_ped)

# Ajustar modelo CatBoost para Psiquiatria (PSIQ)
random_search_psiq_catBoost.fit(X_train_psiq, y_train_psiq)
best_params_psiq = random_search_psiq_catBoost.best_params_
best_score_psiq = random_search_psiq_catBoost.best_score_
best_catboost_classifier_psiq = CatBoostClassifier(**best_params_psiq)
best_catboost_classifier_psiq.fit(X_train_psiq, y_train_psiq)

# Ajustar modelo CatBoost para Radiologia e Diagnóstico por Imagem (RADIO)
random_search_radio_catBoost.fit(X_train_radio, y_train_radio)
best_params_radio = random_search_radio_catBoost.best_params_
best_score_radio = random_search_radio_catBoost.best_score_
best_catboost_classifier_radio = CatBoostClassifier(**best_params_radio)
best_catboost_classifier_radio.fit(X_train_radio, y_train_radio)

# Fazer previsões para Catboost
y_pred_catBoost_anest = best_catboost_classifier_anest.predict(X_test_anest)
y_pred_catBoost_cardio = best_catboost_classifier_cardio.predict(X_test_cardio)
y_pred_catBoost_cirgen = best_catboost_classifier_cirgen.predict(X_test_cirgen)
y_pred_catBoost_clinmed = best_catboost_classifier_clinmed.predict(X_test_clinmed)
y_pred_catBoost_derm = best_catboost_classifier_derm.predict(X_test_derm)
y_pred_catBoost_ginec = best_catboost_classifier_ginec.predict(X_test_ginec)
y_pred_catBoost_medfam = best_catboost_classifier_medfam.predict(X_test_medfam)
y_pred_catBoost_medtrab = best_catboost_classifier_medtrab.predict(X_test_medtrab)
y_pred_catBoost_oftal = best_catboost_classifier_oftal.predict(X_test_oftal)
y_pred_catBoost_ortop = best_catboost_classifier_ortop.predict(X_test_ortop)
y_pred_catBoost_outras = best_catboost_classifier_outras.predict(X_test_outras)
y_pred_catBoost_ped = best_catboost_classifier_ped.predict(X_test_ped)
y_pred_catBoost_psiq = best_catboost_classifier_psiq.predict(X_test_psiq)
y_pred_catBoost_radio = best_catboost_classifier_radio.predict(X_test_radio)


# Calcular ROC AUC para cada especialidade com CatBoost
roc_auc_catBoost_anest = roc_auc_score(y_test_anest, y_pred_catBoost_anest)
roc_auc_catBoost_cardio = roc_auc_score(y_test_cardio, y_pred_catBoost_cardio)
roc_auc_catBoost_cirgen = roc_auc_score(y_test_cirgen, y_pred_catBoost_cirgen)
roc_auc_catBoost_clinmed = roc_auc_score(y_test_clinmed, y_pred_catBoost_clinmed)
roc_auc_catBoost_derm = roc_auc_score(y_test_derm, y_pred_catBoost_derm)
roc_auc_catBoost_ginec = roc_auc_score(y_test_ginec, y_pred_catBoost_ginec)
roc_auc_catBoost_medfam = roc_auc_score(y_test_medfam, y_pred_catBoost_medfam)
roc_auc_catBoost_medtrab = roc_auc_score(y_test_medtrab, y_pred_catBoost_medtrab)
roc_auc_catBoost_oftal = roc_auc_score(y_test_oftal, y_pred_catBoost_oftal)
roc_auc_catBoost_ortop = roc_auc_score(y_test_ortop, y_pred_catBoost_ortop)
roc_auc_catBoost_outras = roc_auc_score(y_test_outras, y_pred_catBoost_outras)
roc_auc_catBoost_ped = roc_auc_score(y_test_ped, y_pred_catBoost_ped)
roc_auc_catBoost_psiq = roc_auc_score(y_test_psiq, y_pred_catBoost_psiq)
roc_auc_catBoost_radio = roc_auc_score(y_test_radio, y_pred_catBoost_radio)

# Criar uma tabela com os resultados de ROC AUC para CatBoost
roc_auc_table_catBoost = pd.DataFrame({
    'Especialidade': [
        'Anestesiologia', 'Cardiologia', 'Cirurgia Geral', 'Clínica Médica',
        'Dermatologia', 'Ginecologia e Obstetrícia', 'Medicina de Família e Comunidade',
        'Medicina do Trabalho', 'Oftalmologia', 'Ortopedia e Traumatologia', 
        'Outras Especialidades', 'Pediatria', 'Psiquiatria', 'Radiologia e Diagnóstico por Imagem'
    ],
    'ROC AUC (CatBoost)': [
        roc_auc_catBoost_anest, roc_auc_catBoost_cardio, roc_auc_catBoost_cirgen, 
        roc_auc_catBoost_clinmed, roc_auc_catBoost_derm, roc_auc_catBoost_ginec, 
        roc_auc_catBoost_medfam, roc_auc_catBoost_medtrab, roc_auc_catBoost_oftal, 
        roc_auc_catBoost_ortop, roc_auc_catBoost_outras, roc_auc_catBoost_ped, 
        roc_auc_catBoost_psiq, roc_auc_catBoost_radio
    ]
})

# Exibir a tabela
print(roc_auc_table_catBoost)



# Matrizes de confusão para o CatBoost
confusion_matrix_catBoost_anest = confusion_matrix(y_test_anest, y_pred_catBoost_anest)
confusion_matrix_catBoost_cardio = confusion_matrix(y_test_cardio, y_pred_catBoost_cardio)
confusion_matrix_catBoost_cirgen = confusion_matrix(y_test_cirgen, y_pred_catBoost_cirgen)
confusion_matrix_catBoost_clinmed = confusion_matrix(y_test_clinmed, y_pred_catBoost_clinmed)
confusion_matrix_catBoost_derm = confusion_matrix(y_test_derm, y_pred_catBoost_derm)
confusion_matrix_catBoost_ginec = confusion_matrix(y_test_ginec, y_pred_catBoost_ginec)
confusion_matrix_catBoost_medfam = confusion_matrix(y_test_medfam, y_pred_catBoost_medfam)
confusion_matrix_catBoost_medtrab = confusion_matrix(y_test_medtrab, y_pred_catBoost_medtrab)
confusion_matrix_catBoost_oftal = confusion_matrix(y_test_oftal, y_pred_catBoost_oftal)
confusion_matrix_catBoost_ortop = confusion_matrix(y_test_ortop, y_pred_catBoost_ortop)
confusion_matrix_catBoost_outras = confusion_matrix(y_test_outras, y_pred_catBoost_outras)
confusion_matrix_catBoost_ped = confusion_matrix(y_test_ped, y_pred_catBoost_ped)
confusion_matrix_catBoost_psiq = confusion_matrix(y_test_psiq, y_pred_catBoost_psiq)
confusion_matrix_catBoost_radio = confusion_matrix(y_test_radio, y_pred_catBoost_radio)

confusion_matrices_catBoost = {
    'Anestesiologia': confusion_matrix_catBoost_anest,
    'Cardiologia': confusion_matrix_catBoost_cardio,
    'Cirurgia Geral': confusion_matrix_catBoost_cirgen,
    'Clínica Médica': confusion_matrix_catBoost_clinmed,
    'Dermatologia': confusion_matrix_catBoost_derm,
    'Ginecologia e Obstetrícia': confusion_matrix_catBoost_ginec,
    'Medicina de Família e Comunidade': confusion_matrix_catBoost_medfam,
    'Medicina do Trabalho': confusion_matrix_catBoost_medtrab,
    'Oftalmologia': confusion_matrix_catBoost_oftal,
    'Ortopedia e Traumatologia': confusion_matrix_catBoost_ortop,
    'Outras Especialidades': confusion_matrix_catBoost_outras,
    'Pediatria': confusion_matrix_catBoost_ped,
    'Psiquiatria': confusion_matrix_catBoost_psiq,
    'Radiologia e Diagnóstico por Imagem': confusion_matrix_catBoost_radio
}

for especialidade, matrix in confusion_matrices_catBoost.items():
    print(f"Matriz de Confusão CatBoost - {especialidade}:\n", matrix, "\n")
    


import pandas as pd
import numpy as np



# Inicializar listas para armazenar as importâncias das variáveis
importancias_xgb = []
importancias_catBoost = []
especialidades = ["Cirurgias", "Clínicas", "Ginecologia.e.Obstetrícia", "Pediatria", "Psiquiatria"]

# Loop para XGBoost
for esp in especialidades:
    # Formatar nome da especialidade para acessar o modelo e o conjunto de dados corretamente
    esp_nome = esp.lower().replace(" ", "")
    model_xgb = globals().get(f'best_xgb_classifier_{esp_nome}')
    X_train = globals().get(f'X_train_{esp_nome}')
    
    if model_xgb and X_train is not None:
        importancias = model_xgb.feature_importances_
        feature_names = X_train.columns
        for i, feature in enumerate(feature_names):
            importancias_xgb.append({
                'Especialidade': esp,
                'Algoritmo': 'XGBoost',
                'Variável': feature,
                'Importância': importancias[i]
            })
    else:
        print(f"Modelo ou dados de treino para {esp} não encontrados.")

# Loop para CatBoost
for esp in especialidades:
    esp_nome = esp.lower().replace(" ", "")
    model_catBoost = globals().get(f'best_catBoost_classifier_{esp_nome}')
    X_train = globals().get(f'X_train_{esp_nome}')
    
    if model_catBoost and X_train is not None:
        importancias = model_catBoost.get_feature_importance()
        feature_names = X_train.columns
        for i, feature in enumerate(feature_names):
            importancias_catBoost.append({
                'Especialidade': esp,
                'Algoritmo': 'CatBoost',
                'Variável': feature,
                'Importância': importancias[i]
            })
    else:
        print(f"Modelo ou dados de treino para {esp} não encontrados.")

# Criar DataFrames para cada conjunto de importâncias
df_importancias_xgb = pd.DataFrame(importancias_xgb)
df_importancias_catBoost = pd.DataFrame(importancias_catBoost)

# Concatenar ambos os DataFrames e ordenar
df_importancias_final = pd.concat([df_importancias_xgb, df_importancias_catBoost], ignore_index=True)
df_importancias_final = df_importancias_final.sort_values(by=['Especialidade', 'Algoritmo', 'Importância'], ascending=[True, True, False])

# Exibir o DataFrame final
print(df_importancias_final)

df_importancias_final.to_csv('importancias_variaveis.csv', index=False)

import shap
import matplotlib.pyplot as plt

# Gráficos SHAP para Anestesiologia - XGBoost
explainer_xgb_anest = shap.TreeExplainer(best_xgb_classifier_anest)
shap_values_xgb_anest = explainer_xgb_anest.shap_values(X_train_anest)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_anest, X_train_anest, feature_names=X_train_anest.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_anest.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Cardiologia - CatBoost
background_sample = shap.sample(X_train_cardio, 100)
explainer_catboost_cardio = shap.KernelExplainer(best_catboost_classifier_cardio.predict, background_sample)
shap_values_catboost_cardio = explainer_catboost_cardio.shap_values(X_train_cardio)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_catboost_cardio, X_train_cardio, feature_names=X_train_cardio.columns, max_display=10, show=False)
plt.savefig('shap_summary_catboost_cardio.png', dpi=1200, bbox_inches='tight')
plt.close()


# Gráficos SHAP para Radiologia e Diagnóstico por Imagem - CatBoost
background_sample = shap.sample(X_train_radio, 100)
explainer_catboost_radio = shap.KernelExplainer(best_catboost_classifier_radio.predict, background_sample)
shap_values_catboost_radio = explainer_catboost_radio.shap_values(X_train_radio)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_catboost_radio, X_train_radio, feature_names=X_train_radio.columns, max_display=10, show=False)
plt.savefig('shap_summary_catboost_radio.png', dpi=1200, bbox_inches='tight')
plt.close()

import shap
import matplotlib.pyplot as plt

# Gráficos SHAP para Anestesiologia - XGBoost
explainer_xgb_anest = shap.TreeExplainer(best_xgb_classifier_anest)
shap_values_xgb_anest = explainer_xgb_anest.shap_values(X_train_anest)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_anest, X_train_anest, feature_names=X_train_anest.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_anest.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Cardiologia - XGBoost
explainer_xgb_cardio = shap.TreeExplainer(best_xgb_classifier_cardio)
shap_values_xgb_cardio = explainer_xgb_cardio.shap_values(X_train_cardio)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_cardio, X_train_cardio, feature_names=X_train_cardio.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_cardio.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Cirurgia Geral - XGBoost
explainer_xgb_cirgen = shap.TreeExplainer(best_xgb_classifier_cirgen)
shap_values_xgb_cirgen = explainer_xgb_cirgen.shap_values(X_train_cirgen)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_cirgen, X_train_cirgen, feature_names=X_train_cirgen.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_cirgen.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Clínica Médica - XGBoost
explainer_xgb_clinmed = shap.TreeExplainer(best_xgb_classifier_clinmed)
shap_values_xgb_clinmed = explainer_xgb_clinmed.shap_values(X_train_clinmed)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_clinmed, X_train_clinmed, feature_names=X_train_clinmed.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_clinmed.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Dermatologia - XGBoost
explainer_xgb_derm = shap.TreeExplainer(best_xgb_classifier_derm)
shap_values_xgb_derm = explainer_xgb_derm.shap_values(X_train_derm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_derm, X_train_derm, feature_names=X_train_derm.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_derm.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Ginecologia e Obstetrícia - XGBoost
explainer_xgb_ginec = shap.TreeExplainer(best_xgb_classifier_ginec)
shap_values_xgb_ginec = explainer_xgb_ginec.shap_values(X_train_ginec)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_ginec, X_train_ginec, feature_names=X_train_ginec.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_ginec.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Medicina de Família e Comunidade - XGBoost
explainer_xgb_medfam = shap.TreeExplainer(best_xgb_classifier_medfam)
shap_values_xgb_medfam = explainer_xgb_medfam.shap_values(X_train_medfam)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_medfam, X_train_medfam, feature_names=X_train_medfam.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_medfam.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Medicina do Trabalho - XGBoost
explainer_xgb_medtrab = shap.TreeExplainer(best_xgb_classifier_medtrab)
shap_values_xgb_medtrab = explainer_xgb_medtrab.shap_values(X_train_medtrab)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_medtrab, X_train_medtrab, feature_names=X_train_medtrab.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_medtrab.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Oftalmologia - XGBoost
explainer_xgb_oftal = shap.TreeExplainer(best_xgb_classifier_oftal)
shap_values_xgb_oftal = explainer_xgb_oftal.shap_values(X_train_oftal)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_oftal, X_train_oftal, feature_names=X_train_oftal.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_oftal.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Ortopedia e Traumatologia - XGBoost
explainer_xgb_ortop = shap.TreeExplainer(best_xgb_classifier_ortop)
shap_values_xgb_ortop = explainer_xgb_ortop.shap_values(X_train_ortop)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_ortop, X_train_ortop, feature_names=X_train_ortop.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_ortop.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Outras Especialidades - XGBoost
explainer_xgb_outras = shap.TreeExplainer(best_xgb_classifier_outras)
shap_values_xgb_outras = explainer_xgb_outras.shap_values(X_train_outras)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_outras, X_train_outras, feature_names=X_train_outras.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_outras.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Pediatria - XGBoost
explainer_xgb_ped = shap.TreeExplainer(best_xgb_classifier_ped)
shap_values_xgb_ped = explainer_xgb_ped.shap_values(X_train_ped)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_ped, X_train_ped, feature_names=X_train_ped.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_ped.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Psiquiatria - XGBoost
explainer_xgb_psiq = shap.TreeExplainer(best_xgb_classifier_psiq)
shap_values_xgb_psiq = explainer_xgb_psiq.shap_values(X_train_psiq)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_psiq, X_train_psiq, feature_names=X_train_psiq.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_psiq.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Radiologia e Diagnóstico por Imagem - XGBoost
explainer_xgb_radio = shap.TreeExplainer(best_xgb_classifier_radio)
shap_values_xgb_radio = explainer_xgb_radio.shap_values(X_train_radio)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_radio, X_train_radio, feature_names=X_train_radio.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_radio.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Grupo de Cirurgias - XGBoost
explainer_xgb_grupo_cirurgias = shap.TreeExplainer(best_xgb_classifier_grupo_cirurgias)
shap_values_xgb_grupo_cirurgias = explainer_xgb_grupo_cirurgias.shap_values(X_train_grupo_cirurgias)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_grupo_cirurgias, X_train_grupo_cirurgias, feature_names=X_train_grupo_cirurgias.columns, max_display=10, show=False)
plt.savefig('shap_summary_xgb_grupo_cirurgias.png', dpi=1200, bbox_inches='tight')
plt.close()

# Gráficos SHAP para Grupo de Clínicas - XGBoost
explainer_xgb_grupo_clinicas = shap.TreeExplainer(best_xgb_classifier_grupo_clinicas)
shap_values_xgb_grupo_clinicas = explainer_xgb_grupo_clinicas.shap_values(X_train_grupo_clinicas)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb_gr



#############################################################################
#############################################################################
# Função para coletar as importâncias e criar um DataFrame
def get_feature_importance(classifier, feature_names):
    importance = classifier.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    return feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Criar um DataFrame para armazenar as importâncias
all_importances = []

# Adicionar as importâncias de cada especialidade ao DataFrame
all_importances.append(get_feature_importance(best_xgb_classifier_anest, X_train_anest.columns).assign(Specialty='Anestesiologia'))
all_importances.append(get_feature_importance(best_xgb_classifier_cardio, X_train_cardio.columns).assign(Specialty='Cardiologia'))
all_importances.append(get_feature_importance(best_xgb_classifier_cirgen, X_train_cirgen.columns).assign(Specialty='Cirurgia Geral'))
all_importances.append(get_feature_importance(best_xgb_classifier_clinmed, X_train_clinmed.columns).assign(Specialty='Clínica Médica'))
all_importances.append(get_feature_importance(best_xgb_classifier_derm, X_train_derm.columns).assign(Specialty='Dermatologia'))
all_importances.append(get_feature_importance(best_xgb_classifier_ginec, X_train_ginec.columns).assign(Specialty='Ginecologia e Obstetrícia'))
all_importances.append(get_feature_importance(best_xgb_classifier_medfam, X_train_medfam.columns).assign(Specialty='Medicina de Família e Comunidade'))
all_importances.append(get_feature_importance(best_xgb_classifier_medtrab, X_train_medtrab.columns).assign(Specialty='Medicina do Trabalho'))
all_importances.append(get_feature_importance(best_xgb_classifier_oftal, X_train_oftal.columns).assign(Specialty='Oftalmologia'))
all_importances.append(get_feature_importance(best_xgb_classifier_ortop, X_train_ortop.columns).assign(Specialty='Ortopedia e Traumatologia'))
all_importances.append(get_feature_importance(best_xgb_classifier_outras, X_train_outras.columns).assign(Specialty='Outras Especialidades'))
all_importances.append(get_feature_importance(best_xgb_classifier_ped, X_train_ped.columns).assign(Specialty='Pediatria'))
all_importances.append(get_feature_importance(best_xgb_classifier_psiq, X_train_psiq.columns).assign(Specialty='Psiquiatria'))
all_importances.append(get_feature_importance(best_xgb_classifier_radio, X_train_radio.columns).assign(Specialty='Radiologia e Diagnóstico por Imagem'))

# Grupos de especialidades
all_importances.append(get_feature_importance(best_xgb_classifier_grupo_cirurgias, X_train_grupo_cirurgias.columns).assign(Specialty='GrupoEsp: Cirurgias'))
all_importances.append(get_feature_importance(best_xgb_classifier_grupo_clinicas, X_train_grupo_clinicas.columns).assign(Specialty='GrupoEsp: Clínicas'))
all_importances.append(get_feature_importance(best_xgb_classifier_grupo_ginecologia, X_train_grupo_ginecologia.columns).assign(Specialty='GrupoEsp: Ginecologia e Obstetrícia'))
all_importances.append(get_feature_importance(best_xgb_classifier_grupo_pediatria, X_train_grupo_pediatria.columns).assign(Specialty='GrupoEsp: Pediatria'))
all_importances.append(get_feature_importance(best_xgb_classifier_grupo_psiquiatria, X_train_grupo_psiquiatria.columns).assign(Specialty='GrupoEsp: Psiquiatria'))

# Concatenar e exibir o DataFrame final
df_importancias_final = pd.concat(all_importances, ignore_index=True)
print(df_importancias_final)
df_importancias_final.to_csv('feature_importances.csv', index=False)


