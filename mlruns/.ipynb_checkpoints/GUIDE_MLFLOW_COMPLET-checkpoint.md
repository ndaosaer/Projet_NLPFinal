# 🚀 GUIDE COMPLET D'INTÉGRATION MLFLOW

## 📋 TABLE DES MATIÈRES
1. Installation et Configuration
2. Lancement du Serveur MLflow
3. Intégration dans le Notebook 04_modeling.ipynb
4. Visualisation et Comparaison des Modèles
5. Intégration MLflow avec l'API
6. Bonnes Pratiques

---

## 1️⃣ INSTALLATION ET CONFIGURATION

### Installer MLflow

```bash
pip install mlflow
```

### Vérifier l'installation

```bash
mlflow --version
```

Vous devriez voir quelque chose comme : `mlflow, version 2.14.1`

---

## 2️⃣ LANCEMENT DU SERVEUR MLFLOW

### Option A: Lancement Simple (Recommandé pour débuter)

Ouvrez un **NOUVEAU terminal** et laissez-le ouvert :

```bash
cd C:\Users\Easy Services Pro\Projet_NLPfinal
mlflow ui --port 5000
```

### Option B: Lancement avec Serveur Complet

```bash
cd C:\Users\Easy Services Pro\Projet_NLPfinal
mlflow server --host 127.0.0.1 --port 5000
```

### Vérification

Ouvrez votre navigateur et allez sur :
```
http://localhost:5000
```

Vous devriez voir l'interface MLflow ! 🎯

**⚠️ IMPORTANT:** Laissez ce terminal ouvert pendant que vous travaillez avec MLflow

---

## 3️⃣ INTÉGRATION DANS LE NOTEBOOK 04_modeling.ipynb

### Étape 1: Imports et Configuration (Première cellule)

Ajoutez ces imports au début de votre notebook :

```python
import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Configuration MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "CV_Classification_Experiments"

# Configurer MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✅ MLflow configuré:")
print(f"   URI: {MLFLOW_TRACKING_URI}")
print(f"   Expérience: {EXPERIMENT_NAME}")
print(f"   Tracking URI actif: {mlflow.get_tracking_uri()}")
```

### Étape 2: Fonction d'Entraînement avec MLflow

Remplacez votre fonction d'entraînement par celle-ci :

```python
def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, label_encoder):
    """
    Entraîner un modèle et logger dans MLflow
    
    Args:
        model: Le modèle à entraîner
        model_name: Nom du modèle (ex: "Random_Forest")
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        label_encoder: Encodeur pour décoder les labels
    
    Returns:
        model, metrics_dict
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(f"\n{'='*70}")
    print(f"🚀 ENTRAÎNEMENT: {model_name}")
    print(f"{'='*70}")
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name=model_name):
        
        # 1. ENTRAÎNER LE MODÈLE
        print(f"⏳ Entraînement en cours...")
        start_time = datetime.now()
        
        model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Modèle entraîné en {training_time:.2f} secondes")
        
        # 2. PRÉDICTIONS
        print(f"🔮 Prédictions...")
        y_pred = model.predict(X_test)
        
        # 3. CALCULER LES MÉTRIQUES
        print(f"📊 Calcul des métriques...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time_seconds': training_time
        }
        
        # Afficher les métriques
        print(f"\n📈 RÉSULTATS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # 4. LOGGER LES PARAMÈTRES
        params = {
            "model_type": model_name,
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
            "n_classes": len(label_encoder.classes_)
        }
        
        # Ajouter les hyperparamètres du modèle
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            for key, value in model_params.items():
                if not callable(value):  # Éviter de logger les fonctions
                    params[f"hyperparameter_{key}"] = str(value)
        
        mlflow.log_params(params)
        print(f"✅ Paramètres loggés")
        
        # 5. LOGGER LES MÉTRIQUES
        mlflow.log_metrics(metrics_dict)
        print(f"✅ Métriques loggées")
        
        # 6. LOGGER LE MODÈLE
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"{model_name}_CV_Classifier"
        )
        print(f"✅ Modèle loggé dans MLflow")
        
        # 7. SAUVEGARDER LOCALEMENT
        Path('../models_saved').mkdir(exist_ok=True)
        model_path = f'../models_saved/{model_name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Modèle sauvegardé localement: {model_path}")
        
        # 8. CRÉER ET LOGGER LA MATRICE DE CONFUSION
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.ylabel('Vraie Classe')
        plt.xlabel('Classe Prédite')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Sauvegarder et logger
        Path('../outputs/plots').mkdir(parents=True, exist_ok=True)
        plot_path = f'../outputs/plots/{model_name}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        print(f"✅ Matrice de confusion loggée")
        
        # 9. LOGGER LE RAPPORT DE CLASSIFICATION
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=label_encoder.classes_
        )
        
        Path('../outputs/reports').mkdir(parents=True, exist_ok=True)
        report_path = f'../outputs/reports/{model_name}_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write(f"{'='*70}\n\n")
            f.write(report)
        
        mlflow.log_artifact(report_path)
        print(f"✅ Rapport de classification loggé")
        
        # 10. LOGGER DES TAGS POUR FACILITER LA RECHERCHE
        mlflow.set_tags({
            "model_family": model_name.split('_')[0],
            "framework": "scikit-learn",
            "task": "multiclass_classification",
            "dataset": "CV_Classification"
        })
        
        print(f"{'='*70}\n")
        
        return model, metrics_dict
```

### Étape 3: Entraîner les Modèles avec MLflow

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Charger les données (adaptez selon votre code existant)
# X, y, label_encoder = ... votre code de chargement ...

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Données divisées:")
print(f"   Train: {X_train.shape[0]} exemples")
print(f"   Test: {X_test.shape[0]} exemples\n")

# Dictionnaire pour stocker les résultats
results = {}

# 1. LOGISTIC REGRESSION
print("🤖 MODÈLE 1: LOGISTIC REGRESSION")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model, lr_metrics = train_and_log_model(
    lr_model, "Logistic_Regression", 
    X_train, y_train, X_test, y_test, label_encoder
)
results['Logistic_Regression'] = lr_metrics

# 2. RANDOM FOREST
print("🤖 MODÈLE 2: RANDOM FOREST")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model, rf_metrics = train_and_log_model(
    rf_model, "Random_Forest",
    X_train, y_train, X_test, y_test, label_encoder
)
results['Random_Forest'] = rf_metrics

# 3. SVM
print("🤖 MODÈLE 3: SUPPORT VECTOR MACHINE")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model, svm_metrics = train_and_log_model(
    svm_model, "SVM",
    X_train, y_train, X_test, y_test, label_encoder
)
results['SVM'] = svm_metrics

# 4. NAIVE BAYES
print("🤖 MODÈLE 4: NAIVE BAYES")
nb_model = MultinomialNB()
nb_model, nb_metrics = train_and_log_model(
    nb_model, "Naive_Bayes",
    X_train, y_train, X_test, y_test, label_encoder
)
results['Naive_Bayes'] = nb_metrics

# 5. K-NEAREST NEIGHBORS (BONUS)
print("🤖 MODÈLE 5: K-NEAREST NEIGHBORS")
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model, knn_metrics = train_and_log_model(
    knn_model, "KNN",
    X_train, y_train, X_test, y_test, label_encoder
)
results['KNN'] = knn_metrics
```

### Étape 4: Comparer les Résultats

```python
# Créer un DataFrame de comparaison
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('f1_score', ascending=False)

print("\n" + "="*80)
print("📊 COMPARAISON DES MODÈLES")
print("="*80)
print(results_df.to_string())
print("="*80)

# Sauvegarder
results_df.to_csv('../outputs/reports/models_comparison.csv')

# Visualiser
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, (metric, ax) in enumerate(zip(metrics, axes.flat)):
    data = results_df[metric].sort_values(ascending=False)
    ax.barh(data.index, data.values, color=colors[idx], alpha=0.7)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} par Modèle')
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(data.values):
        ax.text(v, i, f' {v:.4f}', va='center')

plt.tight_layout()
comparison_path = '../outputs/plots/models_comparison.png'
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.show()

# Logger dans MLflow sous un run séparé
with mlflow.start_run(run_name="Models_Comparison"):
    mlflow.log_artifact(comparison_path)
    mlflow.log_artifact('../outputs/reports/models_comparison.csv')
    
    # Logger le meilleur modèle
    best_model = results_df['f1_score'].idxmax()
    best_score = results_df.loc[best_model, 'f1_score']
    
    mlflow.log_params({
        "best_model": best_model,
        "comparison_metric": "f1_score"
    })
    
    mlflow.log_metrics({
        "best_f1_score": best_score
    })
```

### Étape 5: Sauvegarder le Meilleur Modèle

```python
# Identifier le meilleur modèle
best_model_name = results_df['f1_score'].idxmax()
best_f1_score = results_df.loc[best_model_name, 'f1_score']

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"   F1-Score: {best_f1_score:.4f}\n")

# Charger et sauvegarder comme best_model.pkl
best_model_path = f'../models_saved/{best_model_name}_model.pkl'
with open(best_model_path, 'rb') as f:
    best_model = pickle.load(f)

with open('../models_saved/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Meilleur modèle sauvegardé: models_saved/best_model.pkl")

# Créer un fichier de métadonnées
import json

metadata = {
    'best_model': best_model_name,
    'f1_score': float(best_f1_score),
    'accuracy': float(results_df.loc[best_model_name, 'accuracy']),
    'precision': float(results_df.loc[best_model_name, 'precision']),
    'recall': float(results_df.loc[best_model_name, 'recall']),
    'training_date': datetime.now().isoformat(),
    'n_categories': len(label_encoder.classes_),
    'categories': label_encoder.classes_.tolist(),
    'mlflow_experiment': EXPERIMENT_NAME
}

with open('../models_saved/best_model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Métadonnées sauvegardées: models_saved/best_model_metadata.json")
```

---

## 4️⃣ VISUALISATION ET COMPARAISON DANS MLFLOW

### Accéder à l'Interface MLflow

1. Assurez-vous que le serveur MLflow tourne (http://localhost:5000)
2. Ouvrez votre navigateur sur cette adresse
3. Vous verrez votre expérience "CV_Classification_Experiments"

### Comparer les Modèles

1. Cliquez sur l'expérience
2. Vous verrez tous vos runs (un par modèle)
3. Sélectionnez plusieurs runs (checkbox)
4. Cliquez sur "Compare"
5. Visualisez les graphiques de comparaison

### Filtrer et Rechercher

Dans la barre de recherche, vous pouvez utiliser :

```
metrics.f1_score > 0.85
```

Ou :

```
tags.model_family = "Random"
```

---

## 5️⃣ INTÉGRATION MLFLOW AVEC L'API (OPTIONNEL)

### Charger un Modèle depuis MLflow dans l'API

Modifiez votre `api/main.py` pour charger depuis MLflow :

```python
import mlflow
import mlflow.sklearn

# Dans la classe ModelLoader, ajoutez:
def load_from_mlflow(self, run_id=None):
    """Charger le meilleur modèle depuis MLflow"""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    if run_id is None:
        # Chercher le meilleur run
        experiment = mlflow.get_experiment_by_name("CV_Classification_Experiments")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_score DESC"],
            max_results=1
        )
        run_id = runs.iloc[0].run_id
    
    # Charger le modèle
    model_uri = f"runs:/{run_id}/model"
    self.model = mlflow.sklearn.load_model(model_uri)
    
    print(f"✅ Modèle chargé depuis MLflow (run: {run_id})")
```

---

## 6️⃣ BONNES PRATIQUES

### 1. Organisation des Expériences

Créez des expériences séparées pour différents cas :

```python
# Expérience pour le développement
mlflow.set_experiment("CV_Classification_DEV")

# Expérience pour la production
mlflow.set_experiment("CV_Classification_PROD")

# Expérience pour les tests
mlflow.set_experiment("CV_Classification_TEST")
```

### 2. Tags Utiles

```python
mlflow.set_tags({
    "developer": "VotreNom",
    "version": "1.0",
    "data_version": "2024-02",
    "environment": "development"
})
```

### 3. Logger des Artifacts Supplémentaires

```python
# Logger le code source
mlflow.log_artifact("../src/preprocessing/text_cleaner.py")

# Logger les données de test
mlflow.log_artifact("../data/processed/test_set.csv")
```

### 4. Recherche de Modèles

```python
# Trouver les meilleurs modèles
best_runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.f1_score > 0.90",
    order_by=["metrics.f1_score DESC"]
)

print(best_runs[['run_id', 'metrics.f1_score', 'params.model_type']])
```

---

## 🎯 COMMANDES ESSENTIELLES

**Lancer MLflow:**
```bash
cd C:\Users\Easy Services Pro\Projet_NLPfinal
mlflow ui --port 5000
```

**Lister les expériences:**
```python
import mlflow
mlflow.list_experiments()
```

**Supprimer une expérience:**
```python
mlflow.delete_experiment(experiment_id="1")
```

**Nettoyer les runs:**
```bash
# Supprimer le dossier mlruns pour tout réinitialiser
# (Attention: perte de toutes les données!)
rmdir /s mlruns
```

---

## ✅ CHECKLIST D'INTÉGRATION

- [ ] MLflow installé (`pip install mlflow`)
- [ ] Serveur MLflow lancé (http://localhost:5000)
- [ ] Code d'intégration ajouté au notebook
- [ ] Tous les modèles entraînés et loggés
- [ ] Métriques visibles dans l'interface MLflow
- [ ] Meilleur modèle identifié et sauvegardé
- [ ] Métadonnées créées

---

## 🆘 DÉPANNAGE

**Problème: MLflow ne démarre pas**
```bash
# Vérifier si le port est utilisé
netstat -ano | findstr :5000

# Utiliser un autre port
mlflow ui --port 5001
```

**Problème: Les runs n'apparaissent pas**
```python
# Vérifier le tracking URI
print(mlflow.get_tracking_uri())

# S'assurer que c'est bien configuré
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

**Problème: Erreur lors du log du modèle**
```python
# Vérifier que le modèle est bien entraîné
assert hasattr(model, 'predict'), "Modèle non entraîné"

# Logger sans l'enregistrement
mlflow.sklearn.log_model(model, "model")  # Sans registered_model_name
```

Bonne intégration ! 🚀
