import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 12: Modele efektów mieszanych

    Modele efektów mieszanych pozwalają uwzględnić strukturę grupową w danych,
    gdzie obserwacje należące do tej samej grupy są skorelowane.

    W Pythonie używamy `statsmodels.MixedLM` oraz `statsmodels.BinomialBayesMixedGLM`.

    ## Przykład: Badanie zaburzeń snu

    Zbiór `sleepstudy` zawiera wyniki badania wpływu deprywacji snu na czas
    reakcji u 18 osób przez 10 dni.

    - `Days` — liczba dni deprywacji snu (0–9)
    - `Reaction` — średni czas reakcji [ms]
    - `Subject` — identyfikator osoby badanej
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    import plotly.express as px
    import plotly.graph_objects as go
    return MixedLM, go, np, pd, px, sm, smf


@app.cell
def _(pd):
    from statsmodels.datasets import get_rdataset

    sleepstudy = get_rdataset("sleepstudy", package="lme4").data
    print(sleepstudy.head(10))
    print(sleepstudy.tail(10))
    print(f"\nLiczba badanych: {sleepstudy['Subject'].nunique()}")
    return get_rdataset, sleepstudy


@app.cell
def _(px, sleepstudy):
    fig_raw = px.scatter(sleepstudy, x='Days', y='Reaction',
                         title='Czas reakcji vs liczba dni deprywacji snu',
                         labels={'Days': 'Dni deprywacji', 'Reaction': 'Czas reakcji [ms]'})
    fig_raw.show()
    return (fig_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model liniowy najmniejszych kwadratów (bez efektów grupowych)

    Najpierw dopasujemy zwykłą regresję liniową — ignoruje ona strukturę grupową.
    """)
    return


@app.cell
def _(go, px, sleepstudy, smf):
    ss_lm_fit = smf.ols('Reaction ~ Days', data=sleepstudy).fit()
    print(ss_lm_fit.summary())

    # Prosta regresji na tle danych
    fig_lm = px.scatter(sleepstudy, x='Days', y='Reaction', color='Subject',
                        title='Regresja liniowa (bez efektów grupowych)',
                        labels={'Days': 'Dni', 'Reaction': 'Czas reakcji [ms]'})
    import numpy as np_lm
    days_grid = np_lm.linspace(0, 9, 50)
    fig_lm.add_scatter(x=days_grid, y=ss_lm_fit.predict({'Days': days_grid}),
                       mode='lines', name='Regresja OLS', line=dict(color='black', width=3))
    fig_lm.show()
    return days_grid, fig_lm, np_lm, ss_lm_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model losowych wyrazów wolnych (Random Intercept — RI)

    Równanie **poziomu 1** (obserwacji):
    $$
      Reaction_{i|s} = b_s + \epsilon_{i|s}, \quad \epsilon_{i|s} \sim N(0, \sigma)
    $$

    Równanie **poziomu 2** (osoby badanej):
    $$
      b_s = \beta_0 + u_s, \quad u_s \sim N(0, \sigma_u)
    $$

    Łącznie:
    $$
      Reaction_{i|s} = \beta_0 + u_s + \epsilon_{i|s}
    $$

    - **Efekt stały**: $\beta_0$
    - **Efekt losowy**: $u_s$ (różne wyrazy wolne dla każdej osoby)

    W `statsmodels`: `MixedLM(y, X, groups=groups)`
    """)
    return


@app.cell
def _(MixedLM, sleepstudy, smf):
    # Model RI: Reaction ~ 1, efekt losowy: wyraz wolny w grupach Subject
    ss_ri_fit = smf.mixedlm('Reaction ~ 1', data=sleepstudy, groups=sleepstudy['Subject']).fit()
    print(ss_ri_fit.summary())
    return (ss_ri_fit,)


@app.cell
def _(go, np, px, sleepstudy, ss_ri_fit):
    # Składowe wariancji
    sigma2_u = ss_ri_fit.cov_re.iloc[0, 0]  # wariancja efektu losowego
    sigma2 = ss_ri_fit.scale                  # wariancja resztkowa

    icc = sigma2_u / (sigma2_u + sigma2)
    print(f"sigma²_u (między grupami): {sigma2_u:.2f}")
    print(f"sigma² (w grupie):          {sigma2:.2f}")
    print(f"ICC (korelacja wewnątrzklasowa): {icc:.3f}")

    # Predykcje dla każdej osoby (z efektami losowymi)
    sleepstudy_ri = sleepstudy.copy()
    sleepstudy_ri['predicted_ri'] = ss_ri_fit.fittedvalues

    fig_ri = px.scatter(sleepstudy_ri, x='Days', y='Reaction', color='Subject',
                        title='Model RI - predykcje per osoba',
                        labels={'Days': 'Dni', 'Reaction': 'Czas reakcji [ms]'})
    for subj in sleepstudy['Subject'].unique():
        mask = sleepstudy_ri['Subject'] == subj
        fig_ri.add_scatter(
            x=sleepstudy_ri.loc[mask, 'Days'],
            y=sleepstudy_ri.loc[mask, 'predicted_ri'],
            mode='lines', showlegend=False, line=dict(width=1)
        )
    fig_ri.show()
    return (
        fig_ri,
        icc,
        mask,
        sigma2,
        sigma2_u,
        sleepstudy_ri,
        subj,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model losowych wyrazów wolnych i współczynników (RSI)

    Równanie poziomu 1 (dodajemy predyktor `Days`):
    $$
      Reaction_{i|s} = b_s + b'_s \cdot Days_{i|s} + \epsilon_{i|s}
    $$

    Na poziomie 2:
    $$
      b_s = \beta_0 + u_s, \quad b'_s = \beta_1 + v_s
    $$

    Łącznie:
    $$
      Reaction_{i|s} = \beta_0 + \beta_1 \cdot Days_{i|s} + [u_s + v_s \cdot Days_{i|s} + \epsilon_{i|s}]
    $$

    - Efekty stałe: $\beta_0$ (śr. czas reakcji), $\beta_1$ (śr. wzrost/dzień)
    - Efekty losowe: $u_s$, $v_s$ (indywidualne różnice między osobami)
    """)
    return


@app.cell
def _(MixedLM, sleepstudy, smf):
    # Model RSI: Reaction ~ Days, losowy wyraz wolny i współczynnik dla Days
    ss_rsi_fit = smf.mixedlm(
        'Reaction ~ Days',
        data=sleepstudy,
        groups=sleepstudy['Subject'],
        re_formula='~Days'  # Efekt losowy dla wyrazu wolnego i Days
    ).fit()
    print(ss_rsi_fit.summary())
    return (ss_rsi_fit,)


@app.cell
def _(go, sleepstudy, ss_rsi_fit):
    # Efekty stałe
    print("Efekty stałe:")
    print(f"  β₀ (intercept): {ss_rsi_fit.fe_params['Intercept']:.2f}")
    print(f"  β₁ (Days):      {ss_rsi_fit.fe_params['Days']:.2f}")

    # Predykcje z efektami losowymi
    sleepstudy_rsi = sleepstudy.copy()
    sleepstudy_rsi['predicted_rsi'] = ss_rsi_fit.fittedvalues

    fig_rsi = go.Figure()
    colors = __import__('plotly.express', fromlist=['colors']).colors.qualitative.Plotly

    for i_subj, subj_id in enumerate(sorted(sleepstudy['Subject'].unique())):
        mask_rsi = sleepstudy_rsi['Subject'] == subj_id
        color = colors[i_subj % len(colors)]
        fig_rsi.add_scatter(x=sleepstudy.loc[mask_rsi, 'Days'],
                             y=sleepstudy.loc[mask_rsi, 'Reaction'],
                             mode='markers', name=str(subj_id),
                             marker=dict(color=color))
        fig_rsi.add_scatter(x=sleepstudy_rsi.loc[mask_rsi, 'Days'],
                             y=sleepstudy_rsi.loc[mask_rsi, 'predicted_rsi'],
                             mode='lines', showlegend=False, line=dict(color=color))

    fig_rsi.update_layout(title='Model RSI - proste regresji per osoba',
                          xaxis_title='Dni', yaxis_title='Czas reakcji [ms]')
    fig_rsi.show()
    return (
        color,
        colors,
        fig_rsi,
        i_subj,
        mask_rsi,
        sleepstudy_rsi,
        subj_id,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model RIFS (ustalony slope, losowy intercept)

    Równanie poziomu 2 ma tylko losowy wyraz wolny:
    $$
      Reaction_{i|s} = \beta_0 + \beta_1 \cdot Days_{i|s} + [u_s + \epsilon_{i|s}]
    $$

    To model pośredni między OLS (brak efektów losowych) a RSI.
    """)
    return


@app.cell
def _(sleepstudy, smf):
    # Model RIFS: Reaction ~ Days, tylko losowy wyraz wolny
    ss_rifs_fit = smf.mixedlm(
        'Reaction ~ Days',
        data=sleepstudy,
        groups=sleepstudy['Subject']
        # re_formula domyślnie = ~1, czyli tylko wyraz wolny
    ).fit()
    print(ss_rifs_fit.summary())
    return (ss_rifs_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Porównanie modeli

    Używamy testu ilorazu wiarygodności (LRT) lub kryterium informacyjnego (AIC/BIC).
    """)
    return


@app.cell
def _(go, ss_lm_fit, ss_ri_fit, ss_rifs_fit, ss_rsi_fit):
    models_comparison = {
        'OLS': ss_lm_fit,
        'RI': ss_ri_fit,
        'RIFS': ss_rifs_fit,
        'RSI': ss_rsi_fit,
    }

    print("Porównanie modeli (AIC, BIC):")
    print(f"{'Model':<10} {'AIC':>10} {'BIC':>10} {'LogLik':>10}")
    print("-" * 45)

    for name, mdl in models_comparison.items():
        try:
            aic = mdl.aic
            bic = mdl.bic
            llf = mdl.llf
            print(f"{name:<10} {aic:>10.2f} {bic:>10.2f} {llf:>10.2f}")
        except Exception as e:
            print(f"{name:<10} Błąd: {e}")
    return models_comparison, name, mdl


@app.cell
def _(models_comparison, ss_rsi_fit, ss_ri_fit, ss_rifs_fit):
    # Porównanie współczynników stałych
    import pandas as pd_comp

    coefs = {}
    for name_c, mdl_c in models_comparison.items():
        try:
            coefs[name_c] = mdl_c.fe_params if hasattr(mdl_c, 'fe_params') else mdl_c.params
        except Exception:
            pass

    coef_df = pd_comp.DataFrame(coefs)
    print("Porównanie estymát efektów stałych:")
    print(coef_df.round(3))
    return coef_df, coefs, mdl_c, name_c, pd_comp


if __name__ == "__main__":
    app.run()
