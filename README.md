# Credit Scoring – Home Credit Default Risk

Projet de scoring crédit basé sur le jeu de données **Home Credit Default Risk** (Kaggle).  
Objectif : construire un modèle de classification qui prédit la probabilité de défaut d’un client et l’utilise pour **accepter / refuser** automatiquement les demandes de crédit, en tenant compte :

- du **déséquilibre** entre bons et mauvais payeurs ;
- de l’**asymétrie de coût métier** : un faux négatif (mauvais payeur accepté) coûte environ **10×** plus cher qu’un faux positif (bon payeur refusé).

Le projet est entièrement traqué avec **MLflow** (expériences + modèle final) et prévoit un **serving via Docker**.

---

## 1. Objectif métier

On cherche à estimer pour chaque demande de crédit :

- une **probabilité de défaut** `P(default)` (TARGET = 1),
- une **décision** binaire à partir d’un **seuil métier optimisé**.

Enjeux principaux :

- Le dataset est **fortement déséquilibré** (~8 % de défauts, 92 % de bons payeurs).
- Un **faux négatif (FN)** (mauvais payeur accepté) est **beaucoup plus coûteux** qu’un faux positif (FP).
- Le modèle doit rester **interprétable** (importance globale des variables, explications locales par client).

La métrique métier utilisée est un **coût pondéré** :

- `cost_fn = 10` pour un FN (reflet du risque de perte financière),
- `cost_fp = 1` pour un FP (opportunité manquée).

Le but est de **minimiser le coût moyen** sur l’ensemble des clients et de choisir un **seuil de décision optimal** (≈ 0.10 dans la configuration actuelle).

---

## 2. Données

Les données proviennent de la compétition Kaggle **Home Credit Default Risk**.  
Les principales tables utilisées :

- `application_train.csv` / `application_test.csv` : demande actuelle de crédit (features principales, TARGET uniquement dans `train`).
- `bureau.csv` + `bureau_balance.csv` : crédits précédents chez d’autres organismes (agrégés par client).
- `previous_application.csv` : anciennes demandes chez Home Credit.
- `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` : historique des POS / cartes / remboursements.
- `HomeCredit_columns_description.csv` : dictionnaire des variables.

Les tables secondaires sont **agrégées par `SK_ID_CURR`** pour produire des **features synthétiques** (moyenne, min, max, somme, indicateurs catégoriels) via `src/data_prep.py`, puis mergées sur `application_train` / `application_test`.

---

## 3. Structure du dépôt

```text
credit-scoring/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── data/                  # (non versionné) CSV Kaggle
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── credit_card_balance.csv
│   ├── installments_payments.csv
│   └── HomeCredit_columns_description.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_explainability.ipynb
│   └── 04_mlflow_serving_test.ipynb
│
├── src/
│   ├── data_prep.py        # chargement, jointures, encodage
│   ├── model_utils.py      # (optionnel) helpers split/train/save
│   ├── metrics.py          # AUC, précision, rappel, coût métier, courbe coût vs seuil
│   └── explainability.py   # (optionnel) SHAP global/local
│
├── model/
│   ├── MLmodel
│   ├── conda.yaml
│   └── model.pkl           # modèle final (pipeline préprocessing + LightGBM)
│
├── reports/
│   ├── rapport_credit_scoring.pdf
│   └── figures/
│       ├── shap_global.png
│       ├── shap_local.png
│       └── courbe_cout_vs_seuil.png
│
└── mlruns/                 # dossier de tracking MLflow (non nécessaire dans Git)
