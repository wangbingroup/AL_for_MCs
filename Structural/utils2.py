from sklearn.model_selection import KFold
import numpy as np

def cross_val(predictor, model_name, data, nflod):
    kf = KFold(n_splits=nflod, shuffle=True, random_state=42)
    scores = []

    for fold, (_, val_idx) in enumerate(kf.split(data)):
        val_fold = data.iloc[val_idx]
        
        y_pred = predictor.predict(val_fold.iloc[:, :-1], model=model_name)
        score = predictor.evaluate_predictions(
            y_true = val_fold.iloc[:, -1],
            y_pred = y_pred,
            auxiliary_metrics=True
        )
        # score = predictor.evaluate(val_fold)
        scores.append(score)

    mean_r2 = np.mean([s['r2'] for s in scores])
    mean_rmse = np.mean([-s['root_mean_squared_error'] for s in scores])
    mean_mse = np.mean([-s['mean_squared_error'] for s in scores])
    return {"r2": float(round(mean_r2, 4)),
            "rmse": float(round(mean_rmse, 4)),
            "mse": float(round(mean_mse, 4))}

def eval_test(predictor, model_name, data):
    # res = predictor.evaluate(data)
    y_pred = predictor.predict(data.iloc[:, :-1], model=model_name)
    res = predictor.evaluate_predictions(
        y_true = data.iloc[:, -1],
        y_pred = y_pred,
        auxiliary_metrics=True
    )
    return {"r2": float(round(res['r2'], 4)), 
            "rmse": float(round(-res['root_mean_squared_error'], 4)), 
            "mse": float(round(-res['mean_squared_error'], 4))
            }
