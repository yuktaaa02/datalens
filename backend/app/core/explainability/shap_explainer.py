import numpy as np
import pandas as pd
import shap
from typing import List
from sklearn.pipeline import Pipeline
from app.schemas.explainability import FeatureImportance, ShapExplanationResult

class ShapExplainer:
    @classmethod
    def explain_winning_pipeline(
        cls,
        fitted_pipeline: Pipeline,
        X_sample: pd.DataFrame,
        top_k: int = 10
    ) -> ShapExplanationResult:
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        model = fitted_pipeline.named_steps["model"]

        # Transform features through fitted preprocessor
        X_trans = preprocessor.transform(X_sample)

        # Retrieve feature names out of ColumnTransformer
        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [f.split("__")[-1] for f in feature_names]
        except Exception:
            feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

        # Ensure dense array for SHAP
        if hasattr(X_trans, "toarray"):
            X_dense = X_trans.toarray()
        else:
            X_dense = np.asarray(X_trans)

        # Sample background up to 100 samples for speed
        if X_dense.shape[0] > 100:
            indices = np.random.choice(X_dense.shape[0], 100, replace=False)
            X_eval = X_dense[indices]
        else:
            X_eval = X_dense

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval, check_additivity=False)

        # Handle binary/multiclass classification vs regression shapes
        if isinstance(shap_values, list):
            # Multiclass: take mean absolute across classes
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            # Shape: (samples, features, classes)
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # Regression or binary single array: (samples, features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Pair names with scores and sort descending
        scores = []
        for name, val in zip(feature_names, mean_abs_shap):
            scores.append(FeatureImportance(
                feature_name=str(name),
                importance_score=round(float(val), 4)
            ))

        scores.sort(key=lambda x: x.importance_score, reverse=True)

        return ShapExplanationResult(
            top_features=scores[:top_k],
            explanation_type="TreeExplainer"
        )