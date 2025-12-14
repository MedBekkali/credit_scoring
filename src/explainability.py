import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Tuple, Optional

#  Utilitaires pour récupérer les noms de variables

def get_feature_names_from_preprocessor(preprocessor, input_columns) -> np.ndarray:

    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array(list(input_columns))

#  SHAP global : importance des variables (LightGBM)
def shap_global_lightgbm(
    model,
    preprocessor,
    X: pd.DataFrame,
    max_samples: int = 5000,
    random_state: int = 42,
    save_path: Optional[str] = None,
):
    # Sous-échantillon pour garder un temps de calcul raisonnable
    n_samples = min(max_samples, len(X))
    X_sample = X.sample(n=n_samples, random_state=random_state)

    # Transformation par le préprocesseur (imputation + encodage)
    X_trans = preprocessor.transform(X_sample)
    # Explainer SHAP pour modèle de type arbre (LightGBM)
    explainer = shap.TreeExplainer(model)

    # Compatibilité ancienne / nouvelle API SHAP
    try:
        shap_values = explainer(X_trans)
        if hasattr(shap_values, "values"):
            shap_values_array = shap_values.values
        else:
            shap_values_array = np.array(shap_values)
    except TypeError:
        # Ancienne API : shap_values(X) renvoie une liste pour le binaire
        shap_values_list = explainer.shap_values(X_trans)
        if isinstance(shap_values_list, list):
            # on prend la classe 1 (défaut)
            shap_values_array = np.array(shap_values_list[1])
        else:
            shap_values_array = np.array(shap_values_list)

    feature_names = get_feature_names_from_preprocessor(preprocessor, X.columns)

    # Summary plot SHAP global (importance + direction)
    shap.summary_plot(
        shap_values_array,
        X_trans,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return shap_values_array, X_sample
#  SHAP local : explication pour un client
def shap_local_lightgbm(
    model,
    preprocessor,
    X_row: pd.Series | pd.DataFrame,
    save_path: Optional[str] = None,
    top_k: int = 20,
) -> Tuple[np.ndarray, float]:
    # S'assurer d'avoir un DataFrame avec une seule ligne
    if isinstance(X_row, pd.Series):
        X_row = X_row.to_frame().T

    X_trans = preprocessor.transform(X_row)
    explainer = shap.TreeExplainer(model)

    # Compatibilité API
    try:
        shap_values = explainer(X_trans)
        if hasattr(shap_values, "values"):
            shap_row = np.array(shap_values.values)[0]
            base_value = float(np.array(shap_values.base_values)[0])
        else:
            shap_row = np.array(shap_values)[0]
            base_value = float(explainer.expected_value)
    except TypeError:
        shap_values_list = explainer.shap_values(X_trans)
        if isinstance(shap_values_list, list):
            shap_row = np.array(shap_values_list[1])[0]
            base_value = float(explainer.expected_value[1])
        else:
            shap_row = np.array(shap_values_list)[0]
            base_value = float(explainer.expected_value)

    feature_names = get_feature_names_from_preprocessor(preprocessor, X_row.columns)

    # On affiche les top_k features (en valeur absolue)
    abs_vals = np.abs(shap_row)
    idx_sorted = np.argsort(-abs_vals)
    k = min(top_k, len(idx_sorted))
    idx_top = idx_sorted[:k]

    features_top = feature_names[idx_top]
    contrib_top = shap_row[idx_top]

    plt.figure(figsize=(8, 6))
    plt.barh(
        list(features_top[::-1]),
        list(contrib_top[::-1]),
    )
    plt.axvline(0, linestyle="--")
    plt.xlabel("Contribution SHAP à la probabilité de défaut (classe 1)")
    plt.title("Explication locale SHAP – client unique")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return shap_row, base_value