import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 5: Selekcja cech dla modeli liniowych

    Używamy zbioru danych **Hitters** z pakietu ISLR (dane baseballowe).
    Zawiera statystyki graczy i ich zarobki (Salary).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import itertools
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        Ridge,
        StandardScaler,
        cross_val_score,
        go,
        itertools,
        make_subplots,
        mean_squared_error,
        np,
        pd,
        px,
        r2_score,
        sm,
        smf,
        train_test_split,
    )


@app.cell
def _(pd):
    from statsmodels.datasets import get_rdataset

    hitters_df = get_rdataset("Hitters", package="ISLR").data
    hitters_df = hitters_df.dropna()
    print(f"Rozmiar zbioru po usunięciu NA: {hitters_df.shape}")
    hitters_df.head()
    return get_rdataset, hitters_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wybór najlepszego podzbioru (Best Subset Selection)

    W R używano funkcji `regsubsets()` z pakietu `leaps`. W Pythonie nie ma
    bezpośredniego odpowiednika (biblioteka mlxtend oferuje exhaustive search,
    ale jest bardzo wymagająca obliczeniowo). Implementujemy funkcjonalność
    ręcznie dla małej liczby predyktorów.

    Cechy selekcjonowane są niezależnie dla każdej liczby predyktorów — szukamy
    modelu o najlepszym dopasowaniu ($R^2$) dla każdej liczby zmiennych.
    """)
    return


@app.cell
def _(hitters_df, itertools, smf):
    X_cols = hitters_df.columns.drop('Salary')
    nvmax = 3  # Dla demonstracji — pełne wyszukiwanie dla 19 predyktorów byłoby bardzo wolne

    results = []
    for k in range(1, nvmax + 1):
        for combo in itertools.combinations(X_cols, k):
            formula = 'Salary ~ ' + ' + '.join(combo)
            try:
                model = smf.ols(formula, data=hitters_df).fit()
                results.append({
                    'n_vars': k,
                    'vars': combo,
                    'rsquared': model.rsquared,
                    'bic': model.bic,
                    'aic': model.aic,
                })
            except Exception:
                pass

    # Najlepsze modele dla każdej liczby predyktorów (według R²)
    best_models = {}
    for n in range(1, nvmax + 1):
        subset = [r for r in results if r['n_vars'] == n]
        best = max(subset, key=lambda x: x['rsquared'])
        best_models[n] = best

    for nvar, mdl in best_models.items():
        print(f"{nvar} zmiennych: {mdl['vars']}, BIC: {mdl['bic']:.1f}, R²: {mdl['rsquared']:.3f}")
    return X_cols, best, best_models, combo, formula, k, mdl, model, n, nvar, nvmax, results, subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Zadanie**: Wykonaj podobną analizę dla kryteriów $C_p$ i poprawionego $R^2$.
    Pamiętaj, że poprawione $R^2$ powinno być **zmaksymalizowane**.

    ## Selekcja krokowa (Sequential Feature Selection)

    W scikit-learn dostępna jest klasa `SequentialFeatureSelector` — odpowiednik
    `regsubsets(method="forward")` / `regsubsets(method="backward")` z R.
    """)
    return


@app.cell
def _(LinearRegression, SequentialFeatureSelector, hitters_df, pd):
    X_hitters = hitters_df.drop('Salary', axis=1)
    y_hitters = hitters_df['Salary']

    # Zmienne kategoryczne kodujemy jako dummy
    X_hitters_encoded = pd.get_dummies(X_hitters, drop_first=True)

    model_lr = LinearRegression()

    # Selekcja do przodu
    sfs_forward = SequentialFeatureSelector(
        model_lr,
        n_features_to_select=5,
        direction='forward',
        scoring='neg_mean_squared_error',
        cv=5
    )
    sfs_forward.fit(X_hitters_encoded, y_hitters)

    selected_forward = X_hitters_encoded.columns[sfs_forward.get_support()].tolist()
    print("Selekcja do przodu (5 zmiennych):", selected_forward)

    # Selekcja wstecz
    sfs_backward = SequentialFeatureSelector(
        model_lr,
        n_features_to_select=5,
        direction='backward',
        scoring='neg_mean_squared_error',
        cv=5
    )
    sfs_backward.fit(X_hitters_encoded, y_hitters)

    selected_backward = X_hitters_encoded.columns[sfs_backward.get_support()].tolist()
    print("Selekcja wstecz (5 zmiennych):", selected_backward)
    return (
        X_hitters,
        X_hitters_encoded,
        model_lr,
        selected_backward,
        selected_forward,
        sfs_backward,
        sfs_forward,
        y_hitters,
    )


@app.cell
def _(X_hitters_encoded, sfs_forward):
    # SFS jest transformerem — stosujemy go na danych
    X_selected = X_hitters_encoded.loc[:, sfs_forward.get_support()]
    X_selected.head()
    return (X_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Zadanie**: Które podzbiory predyktorów są optymalne według kryteriów BIC,
    $C_p$ i poprawionego $R^2$? Czy pokrywają się z wynikami selekcji krokowej?

    ## Wybór modelu metodą zbioru walidacyjnego

    Ważne: wszystkie aspekty dopasowania modelu — włącznie z selekcją zmiennych —
    muszą być przeprowadzone wyłącznie na **zbiorze uczącym**.
    """)
    return


@app.cell
def _(LinearRegression, X_hitters_encoded, mean_squared_error, np, train_test_split, y_hitters):
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_hitters_encoded, y_hitters, test_size=0.5, random_state=1
    )

    # Dla każdej możliwej liczby zmiennych (przy selekcji krokowej) szacujemy MSE testowe
    from sklearn.feature_selection import SequentialFeatureSelector as SFS2

    val_errors = []
    n_features_range = range(1, min(20, X_hitters_encoded.shape[1]) + 1)

    for n_feat in n_features_range:
        sfs_val = SFS2(LinearRegression(), n_features_to_select=n_feat,
                       direction='forward', scoring='neg_mean_squared_error', cv=3)
        sfs_val.fit(X_train_h, y_train_h)
        X_train_sel = X_train_h.loc[:, sfs_val.get_support()]
        X_test_sel = X_test_h.loc[:, sfs_val.get_support()]
        lr_val = LinearRegression().fit(X_train_sel, y_train_h)
        pred = lr_val.predict(X_test_sel)
        val_errors.append(mean_squared_error(y_test_h, pred))

    optimal_n = np.argmin(val_errors) + 1
    print(f"Optymalna liczba zmiennych: {optimal_n}, MSE testowe: {val_errors[optimal_n-1]:.0f}")
    return (
        SFS2,
        X_test_h,
        X_test_sel,
        X_train_h,
        X_train_sel,
        lr_val,
        n_feat,
        n_features_range,
        optimal_n,
        pred,
        sfs_val,
        val_errors,
        y_test_h,
        y_train_h,
    )


@app.cell
def _(go, n_features_range, optimal_n, val_errors):
    fig_val = go.Figure()
    fig_val.add_scatter(x=list(n_features_range), y=val_errors, mode='lines+markers', name='Val MSE')
    fig_val.add_vline(x=optimal_n, line_dash='dash', line_color='red',
                      annotation_text=f'Optimum: {optimal_n}')
    fig_val.update_layout(title='MSE testowe vs liczba zmiennych (metoda walidacyjna)',
                          xaxis_title='Liczba zmiennych', yaxis_title='MSE')
    fig_val.show()
    return (fig_val,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularyzacja

    ### Regresja grzbietowa (Ridge) i Lasso

    Obie metody dostępne w `sklearn.linear_model`. Argument `alpha` (scikit-learn)
    odpowiada $\lambda$ z wykładu — większe `alpha` = silniejsza regularyzacja.

    scikit-learn domyślnie standaryzuje zmienne w Ridge/Lasso, odpowiednio do
    zachowania `glmnet()` w R.
    """)
    return


@app.cell
def _(Lasso, Ridge, StandardScaler, X_test_h, X_train_h, mean_squared_error, np, r2_score, y_test_h, y_train_h):
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_h)
    X_test_scaled = scaler.transform(X_test_h)

    ridge_results = []
    lasso_results = []

    for alpha_val in alphas:
        ridge = Ridge(alpha=alpha_val)
        ridge.fit(X_train_scaled, y_train_h)
        y_pred_ridge = ridge.predict(X_test_scaled)
        ridge_results.append({
            'alpha': alpha_val,
            'mse': mean_squared_error(y_test_h, y_pred_ridge),
            'r2': r2_score(y_test_h, y_pred_ridge),
            'coef_norm': np.linalg.norm(ridge.coef_)
        })

        lasso = Lasso(alpha=alpha_val, max_iter=10000)
        lasso.fit(X_train_scaled, y_train_h)
        y_pred_lasso = lasso.predict(X_test_scaled)
        lasso_results.append({
            'alpha': alpha_val,
            'mse': mean_squared_error(y_test_h, y_pred_lasso),
            'r2': r2_score(y_test_h, y_pred_lasso),
            'coef_norm': np.linalg.norm(lasso.coef_),
            'n_nonzero': np.sum(lasso.coef_ != 0)
        })

    print("Ridge:")
    for r in ridge_results:
        print(f"  alpha={r['alpha']}: MSE={r['mse']:.0f}, R²={r['r2']:.3f}, ||coef||={r['coef_norm']:.1f}")

    print("\nLasso:")
    for r in lasso_results:
        print(f"  alpha={r['alpha']}: MSE={r['mse']:.0f}, R²={r['r2']:.3f}, niezerowych cech: {r['n_nonzero']}")
    return (
        X_test_scaled,
        X_train_scaled,
        alpha_val,
        alphas,
        lasso,
        lasso_results,
        r,
        ridge,
        ridge_results,
        scaler,
        y_pred_lasso,
        y_pred_ridge,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-validation dla optymalnego alpha

    W R: `cv.glmnet()`. W scikit-learn: `RidgeCV` lub `LassoCV`.
    """)
    return


@app.cell
def _(X_test_scaled, X_train_scaled, mean_squared_error, np, y_test_h, y_train_h):
    from sklearn.linear_model import RidgeCV, LassoCV

    alphas_cv = np.logspace(-3, 5, 100)

    ridge_cv = RidgeCV(alphas=alphas_cv, cv=10)
    ridge_cv.fit(X_train_scaled, y_train_h)
    print(f"Ridge optymalne alpha: {ridge_cv.alpha_:.4f}")
    y_pred_ridge_cv = ridge_cv.predict(X_test_scaled)
    print(f"Ridge MSE (opt alpha): {mean_squared_error(y_test_h, y_pred_ridge_cv):.0f}")

    lasso_cv = LassoCV(alphas=alphas_cv, cv=10, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train_h)
    print(f"\nLasso optymalne alpha: {lasso_cv.alpha_:.4f}")
    y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)
    print(f"Lasso MSE (opt alpha): {mean_squared_error(y_test_h, y_pred_lasso_cv):.0f}")
    print(f"Lasso niezerowych cech: {np.sum(lasso_cv.coef_ != 0)}")
    return (
        LassoCV,
        RidgeCV,
        alphas_cv,
        lasso_cv,
        ridge_cv,
        y_pred_lasso_cv,
        y_pred_ridge_cv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ścieżka regularyzacji Lasso — które cechy są zerowane?
    """)
    return


@app.cell
def _(Lasso, X_hitters_encoded, X_test_scaled, X_train_scaled, go, make_subplots, np, y_train_h):
    feature_names = X_hitters_encoded.columns.tolist()[:5]  # Pokazujemy pierwsze 5 cech
    alphas_path = np.logspace(-3, 4, 50)

    ridge_coefs_path = []
    lasso_coefs_path = []

    from sklearn.linear_model import Ridge as Ridge2

    for a in alphas_path:
        r2_model = Ridge2(alpha=a)
        r2_model.fit(X_train_scaled, y_train_h)
        ridge_coefs_path.append(r2_model.coef_[:5])

        l_model = Lasso(alpha=a, max_iter=10000)
        l_model.fit(X_train_scaled, y_train_h)
        lasso_coefs_path.append(l_model.coef_[:5])

    fig_path = make_subplots(rows=1, cols=2,
                              subplot_titles=("Ridge - ścieżka współczynników",
                                              "Lasso - ścieżka współczynników"))

    for i, feat in enumerate(feature_names):
        fig_path.add_trace(
            go.Scatter(x=alphas_path, y=[c[i] for c in ridge_coefs_path],
                       mode='lines', name=feat),
            row=1, col=1
        )
        fig_path.add_trace(
            go.Scatter(x=alphas_path, y=[c[i] for c in lasso_coefs_path],
                       mode='lines', name=feat, showlegend=False),
            row=1, col=2
        )

    fig_path.update_xaxes(type='log', title_text='Alpha')
    fig_path.update_yaxes(title_text='Wartość współczynnika')
    fig_path.update_layout(title='Wpływ regularyzacji na współczynniki modelu')
    fig_path.show()
    return (
        Ridge2,
        a,
        feat,
        feature_names,
        fig_path,
        i,
        l_model,
        lasso_coefs_path,
        r2_model,
        ridge_coefs_path,
        alphas_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Elastic-Net

    Elastic-Net to ważona suma penalizacji Ridge i Lasso:
    $$
    \text{penalizacja} = \alpha \cdot \text{Lasso} + (1 - \alpha) \cdot \text{Ridge}
    $$

    W scikit-learn: `ElasticNet(alpha=..., l1_ratio=...)` gdzie `l1_ratio` odpowiada
    wagowaniu między Lasso (l1_ratio=1) a Ridge (l1_ratio=0).

    **Zadanie**: Dopasuj model Elastic-Net dla wybranej wartości `l1_ratio` i
    przeprowadź podobną analizę jak dla Lasso.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
