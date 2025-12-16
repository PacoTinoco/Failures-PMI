import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Utilidades ---------------------------

def cargar_y_preparar(path_excel: str = 'FRR-11.xlsx', sheet_name: str = 'Sheet2') -> pd.DataFrame:
    """Carga el Excel, normaliza encabezados, tipa columnas y calcula Producción.
    """
    df = pd.read_excel(path_excel, sheet_name=sheet_name, engine='openpyxl')
    # Normaliza nombres de columnas
    df.columns = [str(c).strip().replace('\n', ' ').replace('  ', ' ').strip() for c in df.columns]
    rename_map = {
        'Fecha': 'Fecha',
        '#': 'N',
        'RV2': 'RV2',
        'RV4': 'RV4',
        'RV5': 'RV5',
        'RV6': 'RV6',
        'Lim. Max': 'Lim_Max',
        'Lim. Min': 'Lim_Min',
        'V3/KDF': 'V3_KDF',
        'V2/V3': 'V2_V3',
        'Presion': 'Presion',
        'Angulo': 'Angulo',
        'Posisión': 'Posicion',
        'Falla Faltante': 'Falla_Faltante',
        'Falla Extra': 'Falla_Extra',
        'Falla Roto': 'Falla_Roto',
        'Falla Posicion': 'Falla_Posicion',
        'Vel': 'Vel',
        'min.': 'Min',
        'Faltante FRR': 'FRR_Faltante',
        'Extra FRR': 'FRR_Extra',
        'Roto FRR': 'FRR_Roto',
        'Posicion FRR': 'FRR_Posicion',
        'Total': 'Total_FRR',
    }
    df = df.rename(columns=rename_map)
    # Tipado
    if 'Fecha' in df.columns:
        try:
            df['Fecha'] = pd.to_datetime(df['Fecha'])
        except Exception:
            pass
    num_cols = [c for c in df.columns if c != 'Fecha']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    # Producción
    df['Produccion'] = df['Vel'] * df['Min']
    return df


def validar_frr(df: pd.DataFrame) -> dict:
    """Valida que FRR por tipo coincida con conteos/producción, y que Total sea la suma.
    Devuelve dict con banderas de consistencia.
    """
    # Recalcula FRR por tipo
    for falla, fr in [
        ('Falla_Faltante', 'FRR_Faltante'),
        ('Falla_Extra', 'FRR_Extra'),
        ('Falla_Roto', 'FRR_Roto'),
        ('Falla_Posicion', 'FRR_Posicion'),
    ]:
        df[f'{fr}_calc'] = (df[falla] / df['Produccion']) * 100
    df['Total_FRR_calc'] = df[['FRR_Faltante_calc', 'FRR_Extra_calc', 'FRR_Roto_calc', 'FRR_Posicion_calc']].sum(axis=1)
    # Diferencias
    for fr in ['FRR_Faltante', 'FRR_Extra', 'FRR_Roto', 'FRR_Posicion', 'Total_FRR']:
        df[f'{fr}_diff'] = df.get(f'{fr}_calc', df[fr]) - df[fr]
    tol = 1e-6
    return {
        'FRR_por_tipo_OK': bool(df[['FRR_Faltante_diff', 'FRR_Extra_diff', 'FRR_Roto_diff', 'FRR_Posicion_diff']].abs().le(tol).all().all()),
        'FRR_total_OK': bool(df['Total_FRR'].sub(df[['FRR_Faltante', 'FRR_Extra', 'FRR_Roto', 'FRR_Posicion']].sum(axis=1)).abs().le(tol).all()),
    }


def weighted_frr(group: pd.DataFrame) -> float:
    prod = float(group['Produccion'].sum())
    falls = float(group[['Falla_Faltante', 'Falla_Extra', 'Falla_Roto', 'Falla_Posicion']].sum().sum())
    return (falls / prod) * 100 if prod > 0 else np.nan


def resumen_basico(df: pd.DataFrame) -> dict:
    prod_total = float(df['Produccion'].sum())
    sum_falt = int(df['Falla_Faltante'].sum())
    sum_extra = int(df['Falla_Extra'].sum())
    sum_roto = int(df['Falla_Roto'].sum())
    sum_pos = int(df['Falla_Posicion'].sum())
    w_frr_total = (sum_falt + sum_extra + sum_roto + sum_pos) / prod_total * 100
    return {
        'n_filas': int(len(df)),
        'produccion_total': int(prod_total),
        'w_FRR_total_%': w_frr_total,
        'mins_dist': df['Min'].value_counts().sort_index().to_dict(),
    }

# --------------------------- Modelo ponderado (Ridge) ---------------------------

def estandarizar(X: pd.DataFrame):
    means = X.mean()
    stds = X.std().replace(0, np.nan)
    Xz = (X - means) / stds
    return Xz.fillna(0.0), means, stds


def fit_weighted_ridge(Xz: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """Resuelve (X^T W X + alpha I) beta = X^T W y"""
    W = np.diag(w)
    XtW = Xz.T @ W
    A = XtW @ Xz + alpha * np.eye(Xz.shape[1])
    b = XtW @ y
    beta = np.linalg.solve(A, b)
    return beta


def kfold_cv_alpha(Xz: pd.DataFrame, y: np.ndarray, w: np.ndarray, K: int = 5, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 2, 20)
    n = Xz.shape[0]
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    fold_size = int(math.ceil(n / K))
    folds = [idx[i * fold_size:(i + 1) * fold_size] for i in range(K)]
    folds[-1] = idx[(K - 1) * fold_size:]

    cv_mse = []
    for alpha in alphas:
        mse_sum, w_sum = 0.0, 0.0
        for k in range(K):
            val_idx = folds[k]
            tr_idx = np.setdiff1d(idx, val_idx)
            Xtr, ytr, wtr = Xz.iloc[tr_idx].values, y[tr_idx], w[tr_idx]
            Xval, yval, wval = Xz.iloc[val_idx].values, y[val_idx], w[val_idx]
            beta = fit_weighted_ridge(Xtr, ytr, wtr, alpha)
            yhat = Xval @ beta
            mse_sum += float(np.sum(wval * (yval - yhat) ** 2))
            w_sum += float(np.sum(wval))
        cv_mse.append(mse_sum / w_sum)
    best_i = int(np.argmin(cv_mse))
    return float(alphas[best_i]), pd.DataFrame({'alpha': alphas, 'cv_weighted_mse': cv_mse})


# --------------------------- Sensibilidad y combinaciones ---------------------------

def pred_lineal(X_row: pd.Series, means: pd.Series, stds: pd.Series, beta: np.ndarray, cols: list) -> float:
    xz = ((X_row[cols] - means[cols]) / stds[cols]).fillna(0.0).values
    return float(np.dot(xz, beta))


def sensibilidad_un_variable(X: pd.DataFrame, means: pd.Series, stds: pd.Series, beta: np.ndarray,
                             feature: str, valores: np.ndarray, baseline: str = 'median') -> pd.DataFrame:
    """Curva de sensibilidad (ceteris paribus) variando una sola variable y fijando el resto en baseline."""
    cols = list(X.columns)
    if baseline == 'median':
        base = X.median()
    elif baseline == 'mean':
        base = X.mean()
    else:
        # Espera un dict con valores
        base = pd.Series(baseline)
        for c in cols:
            if c not in base.index:
                base[c] = X[c].median()
    rows = []
    for v in valores:
        x_row = base.copy()
        x_row[feature] = v
        pred = pred_lineal(x_row, means, stds, beta, cols)
        rows.append({'Variable': feature, 'Valor': v, 'Pred_FRR_%': pred})
    return pd.DataFrame(rows)


def grid_search(X: pd.DataFrame, means: pd.Series, stds: pd.Series, beta: np.ndarray, espacios: dict,
                baseline: str = 'median') -> pd.DataFrame:
    """Explora combinaciones de valores (espacios) sobre un baseline para el resto de variables."""
    cols = list(X.columns)
    if baseline == 'median':
        base = X.median()
    elif baseline == 'mean':
        base = X.mean()
    else:
        base = pd.Series(baseline)
        for c in cols:
            if c not in base.index:
                base[c] = X[c].median()
    # Construye combinaciones
    from itertools import product
    keys = list(espacios.keys())
    combos = list(product(*[espacios[k] for k in keys]))
    rows = []
    for comb in combos:
        x_row = base.copy()
        for k, v in zip(keys, comb):
            x_row[k] = v
        pred = pred_lineal(x_row, means, stds, beta, cols)
        rows.append({**{k: comb[i] for i, k in enumerate(keys)}, 'Pred_FRR_%': pred})
    return pd.DataFrame(rows).sort_values('Pred_FRR_%')


# --------------------------- Graficación ---------------------------

def plot_pareto(df: pd.DataFrame, path_out: str = 'pareto_fallas.png'):
    serie = pd.Series({
        'Faltante': df['Falla_Faltante'].sum(),
        'Posición': df['Falla_Posicion'].sum(),
        'Roto': df['Falla_Roto'].sum(),
        'Extra': df['Falla_Extra'].sum(),
    }).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    serie.plot(kind='bar', color=['#4e79a7', '#f28e2c', '#e15759', '#76b7b2'], ax=ax)
    ax.set_title('Pareto de fallas (KDF11)')
    ax.set_xlabel('Tipo de falla')
    ax.set_ylabel('Conteo')
    for i, v in enumerate(serie.values):
        ax.text(i, v + max(serie.values) * 0.02, f"{int(v):,}", ha='center')
    plt.tight_layout()
    fig.savefig(path_out, dpi=160)


def plot_distrib_frr(df: pd.DataFrame, path_out: str = 'hist_total_frr.png'):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['Total_FRR'], bins=10, color='#59a14f', edgecolor='black')
    ax.axvline(df['Total_FRR'].mean(), color='red', linestyle='--', label=f"Media = {df['Total_FRR'].mean():.3f}%")
    ax.axvline(df['Total_FRR'].median(), color='orange', linestyle='--', label=f"Mediana = {df['Total_FRR'].median():.3f}%")
    ax.set_title('Distribución del FRR total por ventana')
    ax.set_xlabel('FRR total (%)')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    plt.tight_layout()
    fig.savefig(path_out, dpi=160)


# --------------------------- MAIN ---------------------------

def main():
    # 1) Carga y validación
    df = cargar_y_preparar()
    ok = validar_frr(df)
    print('Consistencia FRR por tipo:', ok['FRR_por_tipo_OK'])
    print('Consistencia FRR total   :', ok['FRR_total_OK'])

    # 2) Resumen básico
    info = resumen_basico(df)
    print('Filas:', info['n_filas'])
    print('Producción total:', info['produccion_total'])
    print('FRR ponderado total (%):', f"{info['w_FRR_total_%']:.3f}")

    # 3) FRR ponderado por parámetro (para encontrar rangos “dulces”)
    parametros = ['RV4', 'V3_KDF', 'Angulo', 'Presion', 'RV6', 'V2_V3']
    for p in parametros:
        tabla = df.groupby(p).apply(weighted_frr).sort_index()
        tabla.to_csv(f'weighted_FRR_by_{p}.csv')

    # 4) Modelo lineal ridge ponderado
    y = df['Total_FRR'].values
    w = df['Produccion'].values.astype(float)
    # Usa solo parámetros continuos disponibles
    param_cols = ['RV2', 'RV4', 'RV5', 'RV6', 'Lim_Max', 'Lim_Min', 'V3_KDF', 'V2_V3', 'Presion', 'Angulo', 'Posicion']
    X = df[param_cols].copy()
    # Quita columnas con desviación 0 o NaN
    stds0 = X.std(numeric_only=True)
    use_cols = [c for c in param_cols if c in X.columns and np.isfinite(stds0[c]) and stds0[c] > 0]
    X = X[use_cols]
    Xz, means, stds = estandarizar(X)
    best_alpha, cv_tabla = kfold_cv_alpha(Xz, y, w, K=5)
    beta = fit_weighted_ridge(Xz.values, y, w, best_alpha)
    coef = pd.Series(beta, index=Xz.columns).sort_values(key=lambda s: s.abs(), ascending=False)
    cv_tabla.to_csv('cv_results.csv', index=False)
    pd.Series(coef).to_csv('model_coefficients.csv')
    with open('model_info.txt', 'w') as f:
        f.write(f"best_alpha={best_alpha}\nfeatures_used={list(X.columns)}\n")

    # 5) Predicciones en datos observados y top mejores observadas
    yhat = Xz.values @ beta
    res = df[['Fecha', 'N', 'Total_FRR', 'Produccion']].copy()
    res['Pred_FRR'] = yhat
    res['Error'] = res['Total_FRR'] - res['Pred_FRR']
    res.sort_values('Total_FRR').head(10).to_csv('best_observed_windows.csv', index=False)

    # 6) Sensibilidades (una variable a la vez)
    sensibilidades = {}
    for feat in ['RV4', 'V3_KDF', 'Angulo', 'Presion', 'RV6', 'V2_V3']:
        if feat in X.columns:
            vals = np.sort(X[feat].unique())
            sens_df = sensibilidad_un_variable(X, means, stds, beta, feat, vals, baseline='median')
            sensibilidades[feat] = sens_df
            sens_df.to_csv(f'sens_{feat}.csv', index=False)

    # 7) Grid de combinaciones (definible por el usuario)
    # Espacios por defecto: tomar 3–5 valores por parámetro dentro del rango observado
    espacios = {}
    for feat in ['RV4', 'V3_KDF', 'Angulo', 'Presion', 'RV6', 'V2_V3']:
        if feat in X.columns:
            vals = np.sort(X[feat].unique())
            if len(vals) >= 5:
                qs = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0])
                # elegimos los más cercanos a los cuantiles
                elegidos = []
                for q in qs:
                    elegidos.append(vals[(np.abs(vals - q)).argmin()])
                espacios[feat] = np.unique(elegidos)
            else:
                espacios[feat] = vals
    grid_df = grid_search(X, means, stds, beta, espacios, baseline='median')
    grid_df.head(50).to_csv('best_predicted_combinations_grid.csv', index=False)

    # 8) Grid restringido (en rangos favorables detectados empíricamente)
    espacios_restrict = {}
    if 'RV4' in X.columns:
        espacios_restrict['RV4'] = [v for v in [0.175, 0.19, 0.20] if v in X['RV4'].unique()] or list(X['RV4'].unique())
    if 'V3_KDF' in X.columns:
        espacios_restrict['V3_KDF'] = [v for v in [0.941, 0.95, 0.954] if v in X['V3_KDF'].unique()] or list(X['V3_KDF'].unique())
    if 'Angulo' in X.columns:
        espacios_restrict['Angulo'] = [v for v in [8.8, 10.0] if v in X['Angulo'].unique()] or list(X['Angulo'].unique())
    if 'Presion' in X.columns:
        espacios_restrict['Presion'] = [v for v in [3.0] if v in X['Presion'].unique()] or list(X['Presion'].unique())
    if 'RV6' in X.columns:
        espacios_restrict['RV6'] = [v for v in [2.0, 2.2] if v in X['RV6'].unique()] or list(X['RV6'].unique())
    if 'V2_V3' in X.columns:
        # usar el mayor valor observado de V2/V3 si existe
        espacios_restrict['V2_V3'] = [float(X['V2_V3'].max())]

    grid_res = grid_search(X, means, stds, beta, espacios_restrict, baseline='median')
    grid_res.head(50).to_csv('best_predicted_combinations_restricted.csv', index=False)

    # 9) Gráficos rápidos
    plot_pareto(df, 'pareto_fallas_kdf11.png')
    plot_distrib_frr(df, 'hist_total_frr_kdf11.png')

    print('\nListo. Archivos generados:')
    for fn in [
        'cv_results.csv', 'model_coefficients.csv', 'model_info.txt',
        'best_observed_windows.csv', 'best_predicted_combinations_grid.csv', 'best_predicted_combinations_restricted.csv',
        'weighted_FRR_by_RV4.csv', 'weighted_FRR_by_V3_KDF.csv' if os.path.exists('weighted_FRR_by_V3_KDF.csv') else 'weighted_FRR_by_V3_KDF(no_existe).csv',
        'weighted_FRR_by_Angulo.csv', 'weighted_FRR_by_Presion.csv', 'weighted_FRR_by_RV6.csv', 'weighted_FRR_by_V2_V3.csv' if os.path.exists('weighted_FRR_by_V2_V3.csv') else 'weighted_FRR_by_V2_V3(no_existe).csv',
        'sens_RV4.csv', 'sens_V3_KDF.csv', 'sens_Angulo.csv', 'sens_Presion.csv', 'sens_RV6.csv', 'sens_V2_V3.csv',
        'pareto_fallas_kdf11.png', 'hist_total_frr_kdf11.png',
    ]:
        print(' -', fn)


if __name__ == '__main__':
    main()
