# Diabetes Risk Prediction

Systeme d'analyse de donnees cliniques et d'estimation du risque de diabete combinant clustering non supervise (K-Means), classification supervisee, tracking MLflow, API FastAPI, interface Streamlit et orchestration Airflow.

## Description du Projet

Ce projet a ete realise dans le cadre d'un laboratoire biomedical. Il comprend deux volets principaux :
1. Un clustering non supervise avec K-Means pour decouvrir des profils homogenes de patients et identifier un groupe a haut risque.
2. Une classification supervisee entrainee a partir des groupes obtenus pour predire le niveau de risque de nouveaux patients.

Les 5 variables cliniques retenues pour l'ensemble du pipeline sont :
- Glucose
- Insulin
- BMI
- Age
- DiabetesPedigreeFunction

## Structure du Projet

```text
diabetes-risk-prediction/
├── airflow/
│   └── dags/
│       └── diabetes_training.py
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   └── dataset-diabete.csv
├── models/
│   └── pipeline.joblib
├── notebooks/
│   └── notebook.ipynb
├── src/
│   ├── classification.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── mlflow_tracking.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   └── test_predection.py
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.airflow
├── requirements.txt
└── README.md
```

## Etapes du Pipeline

### 1. Analyse Exploratoire et Pretraitement (EDA)
- Verification des dimensions, types de donnees et suppression des doublons.
- Gestion des valeurs manquantes : remplacement des faux zeros (Glucose, BloodPressure, SkinThickness, Insulin, BMI) par NaN, puis imputation avec `KNNImputer(n_neighbors=5)`.
- Traitement des valeurs aberrantes : ecretage (`clip`) selon des bornes cliniques realistes.
- Standardisation des 5 variables avec `StandardScaler`.

### 2. Clustering K-Means
- Determination du nombre optimal de clusters avec la methode du coude (inertie) et le score de silhouette pour k variant de 2 a 10.
- Entrainement de K-Means avec k = 2.
- Attribution dynamique de la categorie de risque : le cluster affichant la moyenne de Glucose la plus elevee est designe comme "high risk", l'autre comme "low risk".

### 3. Classification Supervisee
- La colonne `risk_category` sert de variable cible (y).
- Gestion du desequilibre de classes avec `RandomOverSampler` integre dans les pipelines via `imblearn`.
- Comparaison de 4 algorithmes : Logistic Regression, Random Forest, SVM, KNeighbors.
- Optimisation des hyperparametres de la regression logistique par `GridSearchCV`.
- Evaluation par matrice de confusion, precision, rappel, et F1-score pondere.
- Sauvegarde du pipeline champion dans `models/pipeline.joblib`.

### 4. Tracking et Model Registry avec MLflow
- Enregistrement des runs de clustering : parametre k, inertie, score de silhouette, artefacts scaler et modele K-Means.
- Enregistrement des runs de classification : hyperparametres, metriques (Accuracy, Precision, Recall, F1-score), matrice de confusion en image et signature du modele.
- Enregistrement du meilleur modele dans le Model Registry sous le nom `diabetes-risk-classifier` avec l'alias `champion` et le stage `Production`.

### 5. Application Web et API
- API FastAPI (`api/main.py`) : charge le modele champion depuis MLflow Registry et expose l'endpoint `/predict`.
- Interface Streamlit (`app/streamlit_app.py`) : formulaire permettant au praticien de renseigner les 5 constantes cliniques, appel de l'API et affichage du niveau de risque accompagne de recommandations cliniques.

### 6. Orchestration et Deploiement
- DAG Airflow (`airflow/dags/diabetes_training.py`) : automatise l'extraction, le pretraitement, le reentrainement et la mise a jour du modele champion.
- Docker et Docker Compose : orchestration complete des services (MLflow, FastAPI, Streamlit, PostgreSQL et Airflow).

## Installation et Utilisation Locale

### 1. Cloner le projet et installer les dependances

```bash
git clone https://github.com/Anas-EL-KHADIRI/diabetes-risk-prediction.git
cd diabetes-risk-prediction
pip install -r requirements.txt
```

### 2. Lancer le serveur MLflow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

### 3. Entrainer et enregistrer le modele

```bash
python src/train.py
```

### 4. Lancer l'API FastAPI

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 5. Lancer l'application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'interface praticien est alors accessible sur `http://localhost:8501`.

### 6. Tester l'inference

```bash
python tests/test_predection.py
```

## Lancement avec Docker Compose

Pour demarrer l'ensemble des services (MLflow, FastAPI, Streamlit, Airflow et PostgreSQL) en une seule commande :

```bash
docker compose up --build
```

Acces aux services :
- Streamlit : http://localhost:8501
- FastAPI Docs : http://localhost:8000/docs
- MLflow UI : http://localhost:5000
- Airflow UI : http://localhost:8080 (identifiants: admin / admin)

## Remarques Methodologiques pour l'Oral

1. Nature des labels de classification : La variable cible `risk_category` est derivee du clustering K-Means non supervise, et non d'un diagnostic medical preexistant. Les classifieurs apprennent donc a reproduire la frontiere geometrique etablie par K-Means.
2. Scores eleves : L'obtention de scores F1 proches de 1.00 est attendue et normale dans ce contexte, car les modeles supervises sont evalues sur des groupes nettement separes par K-Means a partir des memes variables cliniques.
3. Imputation KNN : L'utilisation de KNN avec 5 voisins preserve les correlations multivariables entre le glucose, l'insuline et l'IMC, contrairement a une simple imputation par la moyenne.
4. Ecretage des valeurs aberrantes : Le capping des valeurs extremes evite la suppression de donnees patient tout en protegeant les algorithmes bases sur les distances euclidiennes (K-Means, KNN, SVM).