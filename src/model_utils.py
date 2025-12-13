# src/model_utils.py
"""
Fonctions utilitaires pour :
- découper le jeu de données (train / validation),
- construire le pipeline de modèle (Random Forest + préprocessing),
- faire une validation croisée AUC,
- entraîner le modèle final,
- évaluer le coût métier,
- sauvegarder le pipeline sur disque.

Les notebooks (02_model_training, 03_explainability, 04_mlflow_serving_test)
doivent importer ces fonctions plutôt que dupliquer le code.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from .data_prep import build_preprocessor
from .metrics import business_cost


# ----------------------------
# 1. Split train / validation
# ----------------------------
def split_train_valid(
    train_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Découpe le DataFrame d'apprentissage en :
    - X_train, X_valid, y_train, y_valid

    Le split est stratifié sur TARGET pour conserver le déséquilibre
    dans chaque sous-échantillon.
    """
    X = train_df.drop(columns=["TARGET"])
    y = train_df["TARGET"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_valid, y_train, y_valid


# -----------------------------------------------
# 2. Construction du pipeline Random Forest
# -----------------------------------------------
def build_rf_pipeline(
    train_df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    class_weight: str | dict | None = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Construit un pipeline sklearn :
    - étape 1 : préprocessing (imputation, encodage),
    - étape 2 : RandomForestClassifier.

    On passe train_df pour que build_preprocessor connaisse toutes les colonnes.
    """
    preprocessor = build_preprocessor(train_df)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    return pipeline


# -----------------------------------------------
# 3. Validation croisée AUC
# -----------------------------------------------
def cross_val_auc(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[float, float, np.ndarray]:
    """
    Calcule l'AUC-ROC moyenne et son écart-type par validation croisée
    stratifiée, pour un pipeline donné.

    Retourne : (auc_mean, auc_std, scores_par_fold)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )

    return float(scores.mean()), float(scores.std()), scores


# -----------------------------------------------
# 4. Entraînement + coût métier
# -----------------------------------------------
def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Entraîne le pipeline complet sur (X_train, y_train).
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_business_cost(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.10,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> tuple[float, dict]:
    """
    Calcule le coût métier moyen (FN pondéré 10x par défaut) pour un pipeline
    et un seuil donnés.

    Retourne : (coût_moyen, dictionnaire_matrice_confusion)
    """
    y_proba = pipeline.predict_proba(X)[:, 1]

    cost, confusion = business_cost(
        y_true=y,
        y_proba=y_proba,
        threshold=threshold,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        normalize=True,
    )
    return float(cost), confusion


# -----------------------------------------------
# 5. Sauvegarde du modèle sur disque
# -----------------------------------------------
def save_pipeline(
    pipeline: Pipeline,
    path: str | Path = "../model/model.pkl",
) -> Path:
    """
    Sauvegarde le pipeline entraîné dans un fichier .pkl avec joblib.
    (Ce n'est pas l'artefact MLflow, mais un backup pratique.)

    Le dossier parent est créé au besoin.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path