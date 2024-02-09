import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
from imblearn.over_sampling import SMOTE, ADASYN

from scipy.special import softmax

def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    print(output)
    display(HTML(output))


def over_sampling(X, y, method='SMOTE'):
    if method == 'SMOTE':
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    elif method == 'ADASYN':
        ada = ADASYN(random_state=42)
        X_res, y_res = ada.fit_resample(X, y)
    else:
        raise ValueError('Invalid method')
    return X_res, y_res

def oversamplig_dataframe(df, target, method='SMOTE'):
    X = df.drop(target, axis=1)
    y = df[target]
    X_res, y_res = over_sampling(X, y, method)
    df_res = pd.concat([X_res, y_res], axis=1)
    return df_res



def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")