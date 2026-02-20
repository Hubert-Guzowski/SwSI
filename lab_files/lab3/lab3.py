import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 3: Regresja liniowa

    Będziemy pracować wykorzystując zbiór danych **California Housing**
    (https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

    Zadaniem jest przewidzenie mediany cen mieszkań w dystryktach stanu Kalifornia
    (wartość podana w 100 000 USD).
    """)
    return


@app.cell
def _():
    from sklearn.datasets import fetch_california_housing
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    return fetch_california_housing, go, np, pd, px, sm, smf


@app.cell
def _(fetch_california_housing):
    housing_df = fetch_california_housing(as_frame=True).frame
    housing_df.info()
    return (housing_df,)


@app.cell
def _(housing_df):
    housing_df['MedHouseVal']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prosta regresja liniowa

    Dopasowanie modelu liniowego:
    $$
      MedHouseVal = \beta_0 + \beta_1 \cdot MedInc + \epsilon
    $$

    W Pythonie używamy biblioteki `statsmodels`. Klasa `OLS` (Ordinary Least Squares)
    odpowiada funkcji `lm()` z R.
    """)
    return


@app.cell
def _(housing_df, sm):
    X_simple = sm.add_constant(housing_df[['MedInc']])
    y = housing_df['MedHouseVal']

    fit_simple = sm.OLS(y, X_simple).fit()
    fit_simple.summary()
    return X_simple, fit_simple, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Funkcja `summary()` zawiera:
    - estymaty współczynników (`coef`)
    - błędy standardowe (`std err`)
    - statystyki testowe i p-wartości
    - RSE (`sigma`) i $R^2$

    ### Przedziały ufności dla współczynników
    """)
    return


@app.cell
def _(fit_simple):
    fit_simple.conf_int()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Predykcja z przedziałami ufności
    """)
    return


@app.cell
def _(fit_simple, pd, sm):
    new_data = pd.DataFrame({'MedInc': [5, 10, 15]})
    new_X = sm.add_constant(new_data)

    # Przedziały ufności dla predykcji
    pred_ci = fit_simple.get_prediction(new_X).summary_frame(alpha=0.05)
    print("Przedziały ufności (confidence):")
    print(pred_ci[['mean', 'mean_ci_lower', 'mean_ci_upper']])

    # Przedziały predykcji (szersze - dla nowych obserwacji)
    print("\nPrzedziały predykcji:")
    print(pred_ci[['mean', 'obs_ci_lower', 'obs_ci_upper']])
    return new_X, new_data, pred_ci


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wykres prostej regresji
    """)
    return


@app.cell
def _(fit_simple, housing_df, np, px):
    x_grid = np.linspace(housing_df['MedInc'].min(), housing_df['MedInc'].max(), 200)

    import statsmodels.api as sm_local
    X_grid = sm_local.add_constant(x_grid)
    pred_grid = fit_simple.get_prediction(X_grid).summary_frame()

    fig_simple = px.scatter(
        housing_df.sample(2000, random_state=42),
        x='MedInc', y='MedHouseVal',
        opacity=0.3,
        title='Prosta regresja liniowa: MedHouseVal ~ MedInc',
        labels={'MedInc': 'Median Income', 'MedHouseVal': 'Median House Value'}
    )
    fig_simple.add_scatter(
        x=x_grid, y=pred_grid['mean'],
        mode='lines', name='Regresja', line=dict(color='red', width=2)
    )
    fig_simple.add_scatter(
        x=x_grid, y=pred_grid['mean_ci_upper'],
        mode='lines', name='CI górny', line=dict(color='red', dash='dash', width=1)
    )
    fig_simple.add_scatter(
        x=x_grid, y=pred_grid['mean_ci_lower'],
        mode='lines', name='CI dolny', line=dict(color='red', dash='dash', width=1)
    )
    fig_simple.show()
    return X_grid, fig_simple, pred_grid, sm_local, x_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wykresy diagnostyczne

    Wykresy residuów pozwalają sprawdzić założenia modelu liniowego.
    """)
    return


@app.cell
def _(fit_simple, housing_df, px):
    import numpy as np_diag
    from scipy import stats as scipy_stats

    fitted_vals = fit_simple.fittedvalues
    residuals = fit_simple.resid
    std_resid = fit_simple.get_influence().resid_studentized_internal

    fig_resid = px.scatter(
        x=fitted_vals, y=residuals,
        title='Residua vs. Wartości dopasowane',
        labels={'x': 'Wartości dopasowane', 'y': 'Residua'}
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    fig_resid.show()
    return (
        fitted_vals,
        fig_resid,
        np_diag,
        residuals,
        scipy_stats,
        std_resid,
    )


@app.cell
def _(fit_simple, housing_df, px, std_resid):
    # Identyfikacja obserwacji wpływowych (hat values / dźwignia)
    influence = fit_simple.get_influence()
    hat_values = influence.hat_matrix_diag

    fig_hat = px.scatter(
        x=range(len(hat_values)), y=hat_values,
        title='Hat values (dźwignia)',
        labels={'x': 'Indeks obserwacji', 'y': 'Hat value'}
    )
    max_hat_idx = hat_values.argmax()
    print(f"Obserwacja o największej dźwigni: indeks {max_hat_idx}, wartość {hat_values[max_hat_idx]:.4f}")
    fig_hat.show()
    return fig_hat, hat_values, influence, max_hat_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresja wielokrotna

    Model:
    $$
      Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon
    $$

    W `statsmodels` możemy używać interfejsu formuły (podobnego do R):
    """)
    return


@app.cell
def _(housing_df, smf):
    # Regresja z dwoma predyktorami
    fit_la = smf.ols('MedHouseVal ~ MedInc + AveRooms', data=housing_df).fit()
    fit_la.summary()
    return (fit_la,)


@app.cell
def _(housing_df, smf):
    # Regresja względem wszystkich zmiennych
    fit_all = smf.ols('MedHouseVal ~ .', data=housing_df).fit()
    fit_all.summary()
    return (fit_all,)


@app.cell
def _(housing_df, smf):
    # Regresja z jedną zmienną usuniętą
    fit_no_AveRooms = smf.ols('MedHouseVal ~ . - AveRooms', data=housing_df).fit()
    fit_no_AveRooms.summary()
    return (fit_no_AveRooms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interakcje między zmiennymi

    Składnik $X_1 \cdot X_2$ zaznaczamy przez `X1:X2` lub korzystamy ze skrótu
    `X1*X2` (czyli `X1 + X2 + X1:X2`):
    """)
    return


@app.cell
def _(housing_df, smf):
    fit_interact = smf.ols('MedHouseVal ~ MedInc * AveRooms', data=housing_df).fit()
    fit_interact.summary()
    return (fit_interact,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nieliniowe transformacje predyktorów

    Model kwadratowy:
    $$
      MedHouseVal = \beta_0 + \beta_1 \cdot MedInc + \beta_2 \cdot MedInc^2 + \epsilon
    $$
    """)
    return


@app.cell
def _(housing_df, smf):
    fit_l2 = smf.ols('MedHouseVal ~ MedInc + I(MedInc**2)', data=housing_df).fit()
    fit_l2.summary()
    return (fit_l2,)


@app.cell
def _(fit_l2, fit_simple, sm):
    # Porównanie modeli przez test ANOVA (test F)
    from statsmodels.stats.anova import anova_lm

    anova_result = anova_lm(fit_simple, fit_l2)
    print("Test ANOVA porównujący modele liniowy i kwadratowy:")
    print(anova_result)
    return anova_lm, anova_result


@app.cell
def _(housing_df, smf):
    # Regresja wielomianowa stopnia 3
    from sklearn.preprocessing import PolynomialFeatures
    import statsmodels.api as sm_poly

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(housing_df[['MedInc']])
    X_poly_df = __import__('pandas').DataFrame(X_poly, columns=[f'MedInc^{i+1}' for i in range(3)])
    X_poly_df = sm_poly.add_constant(X_poly_df)

    fit_l3 = sm_poly.OLS(housing_df['MedHouseVal'], X_poly_df).fit()
    fit_l3.summary()
    return PolynomialFeatures, X_poly, X_poly_df, fit_l3, poly, sm_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zmienne kategoryczne

    Nowy zbiór danych (https://www.statsmodels.org/stable/datasets/generated/fair.html)
    dotyczy predykcji zdrad w małżeństwach i zawiera dane kategoryczne.
    """)
    return


@app.cell
def _(sm):
    import statsmodels.api as sm_fair

    fair_df = sm_fair.datasets.fair.load_pandas().data
    fair_df.info()
    return fair_df, sm_fair


@app.cell
def _(fair_df):
    fair_df.nunique()
    return


@app.cell
def _(fair_df, pd):
    # Mapowanie kolumn kategorycznych (nunique < 7, z wyjątkiem age i children)
    categoricals = fair_df.drop(columns=['age', 'children']).loc[:, fair_df.nunique() < 7].astype('category')
    others = pd.concat([fair_df.loc[:, fair_df.nunique() >= 7], fair_df[['age', 'children']]], axis=1)
    cat_fair_df = pd.concat([others, categoricals], axis=1)
    cat_fair_df.info()
    return cat_fair_df, categoricals, others


@app.cell
def _(cat_fair_df, sm):
    # Model OLS - standardowe API
    X_fair = cat_fair_df[['age', 'occupation', 'yrs_married']]
    X_fair_const = sm.add_constant(X_fair)
    y_fair = cat_fair_df['affairs']

    model1 = sm.OLS(y_fair, X_fair_const).fit()
    model1.summary()
    return X_fair, X_fair_const, model1, y_fair


@app.cell
def _(cat_fair_df, smf):
    # Model OLS - API formuły (jak R, zmienne kategoryczne wykrywane automatycznie)
    model2 = smf.ols('affairs ~ C(age) + religious + yrs_married', data=cat_fair_df).fit()
    print(model2.summary())
    return (model2,)


@app.cell
def _(cat_fair_df, model2, px):
    cat_fair_df['predicted_model2'] = model2.predict(cat_fair_df)

    fig1 = px.scatter(
        cat_fair_df,
        x='affairs', y='predicted_model2',
        title='Actual vs Predicted Affairs (Formula API)',
        labels={'affairs': 'Actual Affairs', 'predicted_model2': 'Predicted Affairs'},
        trendline='ols'
    )
    fig1.show()
    return (fig1,)


@app.cell
def _(cat_fair_df, model2, px):
    cat_fair_df['residuals_model2'] = cat_fair_df['affairs'] - cat_fair_df['predicted_model2']

    fig2 = px.scatter(
        cat_fair_df,
        x='predicted_model2', y='residuals_model2',
        title='Residuals vs Fitted Values',
        labels={'predicted_model2': 'Predicted Values', 'residuals_model2': 'Residuals'}
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    fig2.show()
    return (fig2,)


@app.cell
def _(cat_fair_df, model1, model2, px):
    import pandas as pd_coef

    model1_coefs = model1.params.reset_index()
    model1_coefs.columns = ['Variable', 'Coefficient']
    model1_coefs['Model'] = 'Standard API'
    model1_coefs['Error'] = model1.bse.values
    model1_coefs.iloc[0, 0] = "Intercept"

    model2_coefs = model2.params.reset_index()
    model2_coefs.columns = ['Variable', 'Coefficient']
    model2_coefs['Model'] = 'Formula API'
    model2_coefs['Error'] = model2.bse.values

    combined_coefs = pd_coef.concat([model1_coefs, model2_coefs])

    fig3 = px.bar(
        combined_coefs,
        x='Variable', y='Coefficient',
        error_y='Error',
        color='Model',
        barmode='group',
        title='Porównanie współczynników modeli',
        labels={'Variable': 'Predyktor', 'Coefficient': 'Współczynnik'}
    )
    fig3.show()
    return combined_coefs, fig3, model1_coefs, model2_coefs, pd_coef


if __name__ == "__main__":
    app.run()
