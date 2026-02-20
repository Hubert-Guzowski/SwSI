import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 6: Modele nieliniowe

    Używamy zbioru danych **Wage** z pakietu ISLR (dane o zarobkach pracowników).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.datasets import get_rdataset
    from statsmodels.gam.api import GLMGam, BSplines
    from statsmodels.genmod.families import Gaussian
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from sklearn.preprocessing import PolynomialFeatures
    from patsy import dmatrix
    import plotly.express as px
    import plotly.graph_objects as go
    return (
        BSplines,
        Gaussian,
        GLMGam,
        PolynomialFeatures,
        dmatrix,
        get_rdataset,
        go,
        lowess,
        np,
        pd,
        px,
        sm,
        smf,
    )


@app.cell
def _(get_rdataset):
    wage_df = get_rdataset("Wage", package="ISLR").data
    wage_df.info()
    return (wage_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja wielomianowa

    ### Wielomiany ortogonalne

    Regresja wielomianowa stopnia 4 zmiennej `wage` względem `age`.

    W R używana jest funkcja `poly()` generująca wielomiany ortogonalne (Legendrea
    lub Czebyszewa). W Pythonie możemy użyć wielomianów Czebyszewa z NumPy:
    """)
    return


@app.cell
def _(np, sm, wage_df):
    from numpy.polynomial.chebyshev import chebvander

    age = wage_df['age'].values
    age_scaled = 2 * (age - age.min()) / (age.max() - age.min()) - 1

    X_ortho = chebvander(age_scaled, 4)[:, 1:]  # pomijamy stały składnik
    X_ortho_const = sm.add_constant(X_ortho)

    ortho_model = sm.OLS(wage_df['wage'], X_ortho_const).fit()
    print(ortho_model.summary())
    return X_ortho, X_ortho_const, age, age_scaled, chebvander, ortho_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Uwaga: współczynniki wielomianu różnią się w zależności od wybranej bazy
    > ortogonalnej (Czebyszewa vs Legendrea vs standardowa). Natomiast dokładność
    > dopasowania ($R^2$) jest taka sama dla wszystkich.

    ### Standardowa baza wielomianowa

    Dla standardowej bazy $X, X^2, X^3, X^4$ możemy użyć `PolynomialFeatures`
    z scikit-learn — wynik pokrywa się z `poly(age, 4, raw=TRUE)` z R:
    """)
    return


@app.cell
def _(PolynomialFeatures, sm, wage_df):
    X_age = wage_df[['age']].values
    y_wage = wage_df['wage'].values

    poly_transformer = PolynomialFeatures(degree=4)
    X_poly = poly_transformer.fit_transform(X_age)

    model_poly4 = sm.OLS(y_wage, X_poly).fit()
    print(model_poly4.summary())
    return X_age, X_poly, model_poly4, poly_transformer, y_wage


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dopasujemy teraz wielomian stopnia 3 (stopień 4 nie jest istotnie lepszy):
    """)
    return


@app.cell
def _(PolynomialFeatures, go, np, sm, wage_df, y_wage):
    from statsmodels.stats.anova import anova_lm

    X_poly3 = PolynomialFeatures(degree=3).fit_transform(wage_df[['age']].values)
    model_poly3 = sm.OLS(y_wage, X_poly3).fit()

    # Porównanie modeli ANOVA
    print("Test ANOVA (stopień 3 vs 4):")
    print(anova_lm(model_poly3, sm.OLS(y_wage, PolynomialFeatures(degree=4).fit_transform(wage_df[['age']].values)).fit()))

    # Wizualizacja
    age_grid = np.linspace(wage_df['age'].min(), wage_df['age'].max(), 200)
    X_grid_poly3 = PolynomialFeatures(degree=3).fit_transform(age_grid.reshape(-1, 1))
    pred_poly3 = model_poly3.get_prediction(X_grid_poly3).summary_frame()

    fig_poly = go.Figure()
    fig_poly.add_scatter(x=wage_df['age'], y=wage_df['wage'],
                         mode='markers', opacity=0.3, marker=dict(size=3, color='gray'),
                         name='Dane')
    fig_poly.add_scatter(x=age_grid, y=pred_poly3['mean'],
                         mode='lines', name='Regresja (stopień 3)', line=dict(color='red'))
    fig_poly.add_scatter(x=age_grid, y=pred_poly3['mean_ci_upper'],
                         mode='lines', name='CI górny', line=dict(color='red', dash='dash'))
    fig_poly.add_scatter(x=age_grid, y=pred_poly3['mean_ci_lower'],
                         mode='lines', name='CI dolny', line=dict(color='red', dash='dash'))
    fig_poly.update_layout(title='Regresja wielomianowa (stopień 3)',
                           xaxis_title='Age', yaxis_title='Wage')
    fig_poly.show()
    return (
        X_grid_poly3,
        X_poly3,
        age_grid,
        anova_lm,
        fig_poly,
        model_poly3,
        pred_poly3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja logistyczna wielomianowa

    Klasyfikacja: zarabiający dużo (`wage > 250`) vs mało zarabiający.
    """)
    return


@app.cell
def _(X_ortho_const, age_grid, age_scaled, chebvander, go, np, sm, wage_df):
    wage_df_local = wage_df.copy()
    wage_df_local['high_wage'] = (wage_df_local['wage'] > 250).astype(int)

    logit_model = sm.GLM(wage_df_local['high_wage'],
                         X_ortho_const,
                         family=sm.families.Binomial()).fit()
    print(logit_model.summary())

    # Predykcja na siatce
    age_min, age_max = wage_df['age'].min(), wage_df['age'].max()
    age_grid_scaled = 2 * (age_grid - age_min) / (age_max - age_min) - 1
    X_grid_ortho = chebvander(age_grid_scaled, 4)[:, 1:]
    X_grid_const = sm.add_constant(X_grid_ortho)

    pred_logit = logit_model.get_prediction(X_grid_const).summary_frame()

    fig_logit = go.Figure()
    fig_logit.add_scatter(
        x=wage_df['age'], y=wage_df_local['high_wage'],
        mode='markers', opacity=0.3, marker=dict(size=3, color='gray'), name='Dane'
    )
    fig_logit.add_scatter(
        x=age_grid, y=pred_logit['mean'],
        mode='lines', name='P(wage>250)', line=dict(color='red')
    )
    fig_logit.add_scatter(x=age_grid, y=pred_logit['mean_ci_upper'],
                          mode='lines', line=dict(color='red', dash='dash'), name='CI')
    fig_logit.add_scatter(x=age_grid, y=pred_logit['mean_ci_lower'],
                          mode='lines', line=dict(color='red', dash='dash'), showlegend=False)
    fig_logit.update_layout(title='P(wage > 250 | age)', xaxis_title='Age',
                            yaxis_title='Prawdopodobieństwo')
    fig_logit.show()
    return (
        X_grid_const,
        X_grid_ortho,
        age_grid_scaled,
        age_max,
        age_min,
        fig_logit,
        logit_model,
        pred_logit,
        wage_df_local,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funkcje schodkowe

    Przekształcamy zmienną numeryczną w kategorie (`pd.cut`), a następnie
    dopasowujemy regresję na kategoriach.
    """)
    return


@app.cell
def _(age_grid, go, smf, wage_df):
    wage_df_step = wage_df.copy()
    wage_df_step['age_cut'] = __import__('pandas').cut(wage_df_step['age'], bins=4)

    model_step = smf.ols('wage ~ C(age_cut)', data=wage_df_step).fit()
    print(model_step.summary())

    pred_step_grid = model_step.predict(
        __import__('pandas').DataFrame({'age_cut': __import__('pandas').cut(age_grid, bins=4)})
    )

    fig_step = go.Figure()
    fig_step.add_scatter(x=wage_df['age'], y=wage_df['wage'],
                         mode='markers', opacity=0.3, marker=dict(size=3, color='gray'), name='Dane')
    fig_step.add_scatter(x=age_grid, y=pred_step_grid, mode='lines',
                         name='Funkcja schodkowa', line=dict(color='red'))
    fig_step.update_layout(title='Regresja - funkcja schodkowa',
                           xaxis_title='Age', yaxis_title='Wage')
    fig_step.show()
    return fig_step, model_step, pred_step_grid, wage_df_step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funkcje sklejane (Splines)

    Bazę regresyjnych funkcji sklejanych budujemy przy pomocy biblioteki `patsy`
    (analogicznie do funkcji `bs()` z pakietu `splines` w R).

    ### Funkcje sklejane z ustalonymi węzłami
    """)
    return


@app.cell
def _(age_grid, dmatrix, go, np, sm, wage_df):
    fixed_knots = [25, 40, 60]

    bs_basis = dmatrix(
        "bs(age, knots=fixed_knots, degree=3, include_intercept=False)",
        {"age": wage_df['age'], "fixed_knots": fixed_knots},
        return_type='dataframe'
    )
    X_bs = sm.add_constant(bs_basis)
    fit_bs_knots = sm.OLS(wage_df['wage'], X_bs).fit()

    # Predykcja na siatce
    bs_grid = dmatrix(
        "bs(age, knots=fixed_knots, degree=3, include_intercept=False)",
        {"age": age_grid, "fixed_knots": fixed_knots},
        return_type='dataframe'
    )
    X_bs_grid = sm.add_constant(bs_grid)
    pred_bs = fit_bs_knots.get_prediction(X_bs_grid).summary_frame()

    fig_bs = go.Figure()
    fig_bs.add_scatter(x=wage_df['age'], y=wage_df['wage'],
                       mode='markers', opacity=0.3, marker=dict(size=3, color='gray'), name='Dane')
    fig_bs.add_scatter(x=age_grid, y=pred_bs['mean'], mode='lines',
                       name='Spline', line=dict(color='red'))
    for knot in fixed_knots:
        fig_bs.add_vline(x=knot, line_dash='dot', line_color='blue')
    fig_bs.update_layout(title=f'Funkcje sklejane (węzły: {fixed_knots})',
                         xaxis_title='Age', yaxis_title='Wage')
    fig_bs.show()
    return (
        X_bs,
        X_bs_grid,
        bs_basis,
        bs_grid,
        fig_bs,
        fit_bs_knots,
        fixed_knots,
        knot,
        pred_bs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem**: Sprawdź jak ustawienie węzłów wpływa na dopasowany model.

    ### Funkcje sklejane ze stałą liczbą stopni swobody
    """)
    return


@app.cell
def _(age_grid, dmatrix, go, sm, wage_df):
    bs_basis_df = dmatrix("bs(age, df=6, degree=3, include_intercept=False)",
                          {"age": wage_df['age']}, return_type='dataframe')
    X_bs_df = sm.add_constant(bs_basis_df)
    fit_bs_df = sm.OLS(wage_df['wage'], X_bs_df).fit()

    bs_grid_df = dmatrix("bs(age, df=6, degree=3, include_intercept=False)",
                         {"age": age_grid}, return_type='dataframe')
    X_bs_grid_df = sm.add_constant(bs_grid_df)
    pred_bs_df = fit_bs_df.get_prediction(X_bs_grid_df).summary_frame()

    fig_bs_df = go.Figure()
    fig_bs_df.add_scatter(x=wage_df['age'], y=wage_df['wage'],
                          mode='markers', opacity=0.3, marker=dict(size=3, color='gray'), name='Dane')
    fig_bs_df.add_scatter(x=age_grid, y=pred_bs_df['mean'], mode='lines',
                          name='Spline (df=6)', line=dict(color='red'))
    fig_bs_df.update_layout(title='Funkcje sklejane (6 stopni swobody)',
                            xaxis_title='Age', yaxis_title='Wage')
    fig_bs_df.show()
    return (
        X_bs_df,
        X_bs_grid_df,
        bs_basis_df,
        bs_grid_df,
        fig_bs_df,
        fit_bs_df,
        pred_bs_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problemy**:
    - Sprawdź jak liczba stopni swobody wpływa na dopasowany model.
    - Zbadaj funkcje sklejane innych stopni (parametr `degree` w `bs()`).

    ## Wygładzające funkcje sklejane

    Odpowiednik `smooth.spline()` z R. W Pythonie używamy `GLMGam` z `statsmodels`
    z `BSplines`.
    """)
    return


@app.cell
def _(BSplines, GLMGam, Gaussian, age_grid, go, np, wage_df):
    x_age = wage_df["age"].values
    y_wage_gam = wage_df["wage"].values

    gam_smooth = GLMGam(
        y_wage_gam,
        smoother=BSplines(x_age[:, None], df=[10], degree=[3]),
        exog=np.ones((len(x_age), 1)),
        family=Gaussian()
    ).fit()

    preds_smooth = gam_smooth.predict(
        exog=np.ones((len(age_grid), 1)),
        exog_smooth=age_grid[:, None]
    )

    fig_smooth = go.Figure()
    fig_smooth.add_scatter(x=x_age, y=y_wage_gam,
                           mode='markers', opacity=0.3, marker=dict(size=3, color='gray'), name='Dane')
    fig_smooth.add_scatter(x=age_grid, y=preds_smooth, mode='lines',
                           name='Wygładzająca funkcja sklejana', line=dict(color='red'))
    fig_smooth.update_layout(title='Wygładzająca funkcja sklejana (df=10)',
                             xaxis_title='Age', yaxis_title='Wage', yaxis=dict(range=[0, 200]))
    fig_smooth.show()
    return fig_smooth, gam_smooth, preds_smooth, x_age, y_wage_gam


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja lokalna (LOESS)

    Odpowiednik `loess()` z R. W `statsmodels` funkcja `lowess`.
    """)
    return


@app.cell
def _(age, go, lowess, wage_df):
    spans = [0.2, 0.5]
    colors_loess = ['red', 'blue']

    fig_loess = go.Figure()
    fig_loess.add_scatter(x=wage_df['age'], y=wage_df['wage'],
                          mode='markers', opacity=0.2, marker=dict(size=3, color='gray'), name='Dane')

    for span_val, color_val in zip(spans, colors_loess):
        loess_result = lowess(wage_df['wage'], wage_df['age'],
                              frac=span_val, it=0, return_sorted=True)
        fig_loess.add_scatter(x=loess_result[:, 0], y=loess_result[:, 1],
                              mode='lines', name=f's={span_val}',
                              line=dict(color=color_val, width=2))

    fig_loess.update_layout(title='Regresja lokalna LOESS',
                            xaxis_title='Age', yaxis_title='Wage')
    fig_loess.show()
    return color_val, colors_loess, fig_loess, loess_result, span_val, spans


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uogólnione modele addytywne (GAM)

    GAM pozwala łączyć różne nieparametryczne transformacje predyktorów w jednym
    modelu addytywnym.

    ### GAM metodą najmniejszych kwadratów
    """)
    return


@app.cell
def _(dmatrix, sm, wage_df):
    X_gam_ls = dmatrix(
        "bs(year, df=4, include_intercept=False) + bs(age, df=5, include_intercept=False) + C(education)",
        data=wage_df,
        return_type='dataframe'
    )
    gam_ls_model = sm.OLS(wage_df['wage'], X_gam_ls).fit()
    gam_ls_model.summary()
    return X_gam_ls, gam_ls_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GAM ze wygładzającymi funkcjami sklejanymi

    `GLMGam` z `statsmodels` obsługuje wygładzające spline'y na wielu zmiennych:
    """)
    return


@app.cell
def _(BSplines, GLMGam, Gaussian, dmatrix, sm, wage_df):
    y_gam = wage_df['wage']
    X_gam_lin = dmatrix("C(education)", data=wage_df, return_type='dataframe')
    x_smooth_vars = wage_df[['year', 'age']].values

    bs_gam = BSplines(x_smooth_vars, df=[5, 5], degree=[3, 3])
    gam_model = GLMGam(y_gam, exog=X_gam_lin, smoother=bs_gam).fit()
    gam_model.summary()
    return X_gam_lin, bs_gam, gam_model, x_smooth_vars, y_gam


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wykresy częściowej zależności (PDP)

    PDP (Partial Dependency Plots) są użyteczne przy interpretacji modeli nieliniowych.
    W scikit-learn dostępna jest klasa `PartialDependenceDisplay`.
    """)
    return


@app.cell
def _(PolynomialFeatures, go, np, sm, wage_df):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import PartialDependenceDisplay
    import matplotlib.pyplot as plt

    # Zakodowanie zmiennych kategorycznych
    wage_encoded = __import__('pandas').get_dummies(wage_df, columns=['education', 'jobclass', 'region', 'race', 'maritl', 'health', 'health_ins'])
    wage_encoded = wage_encoded.select_dtypes(include='number')

    X_pdp = wage_encoded.drop(columns=['wage', 'logwage'], errors='ignore')
    y_pdp = wage_encoded['wage']

    gbr_pdp = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gbr_pdp.fit(X_pdp, y_pdp)

    # PDP dla zmiennych age i year
    features_to_plot = ['age', 'year']
    feature_indices = [list(X_pdp.columns).index(f) for f in features_to_plot if f in X_pdp.columns]

    if feature_indices:
        fig_pdp, ax_pdp = plt.subplots(1, len(feature_indices), figsize=(12, 4))
        if len(feature_indices) == 1:
            ax_pdp = [ax_pdp]
        PartialDependenceDisplay.from_estimator(gbr_pdp, X_pdp, feature_indices, ax=ax_pdp)
        plt.suptitle('Wykresy częściowej zależności (PDP)')
        plt.tight_layout()
        plt.show()
    return (
        GradientBoostingRegressor,
        PartialDependenceDisplay,
        X_pdp,
        ax_pdp,
        feature_indices,
        features_to_plot,
        fig_pdp,
        gbr_pdp,
        plt,
        wage_encoded,
        y_pdp,
    )


if __name__ == "__main__":
    app.run()
