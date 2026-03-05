import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 12: Modele efektów mieszanych — Zadania

    ## Przykład: Osiągnięcia graczy NBA

    Dane z sezonu zasadniczego 2022/23 NBA.
    Źródło: https://www.basketball-reference.com/leagues/NBA_2023_per_game.html

    Analizujemy skuteczność wykonywania rzutów wolnych w zależności od pozycji gracza.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import plotly.express as px
    import plotly.graph_objects as go
    return go, np, pd, px, sm, smf


@app.cell
def _(pd):
    nba_df = pd.read_csv("nba-players-2022-3.csv", sep=";")
    nba_df.head()
    return (nba_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Porządkowanie danych
    """)
    return


@app.cell
def _(nba_df, pd):
    # Usunięcie NA
    nba_df_clean = nba_df.dropna().copy()

    # Gracze na 2 pozycjach — przypisujemy im pierwszą pozycję
    nba_df_clean['Pos'] = nba_df_clean['Pos'].str.extract(r'^(\w+)')
    nba_df_clean['Pos'] = nba_df_clean['Pos'].astype('category')
    print(nba_df_clean['Pos'].value_counts())
    return (nba_df_clean,)


@app.cell
def _(nba_df_clean):
    # Gracze, którzy zmieniali klub w sezonie
    transferred = (
        nba_df_clean.groupby('Rk')['Tm']
        .count()
        .reset_index()
        .query('Tm > 1')['Rk']
        .values
    )

    # Dla transferowanych graczy pozostawiamy tylko wyniki sumaryczne (Tm == "TOT")
    nba_final = nba_df_clean[
        (~nba_df_clean['Rk'].isin(transferred)) |
        (nba_df_clean['Tm'] == 'TOT')
    ].copy()

    print(f"Liczba graczy: {len(nba_final)}")
    print(f"Liczba pozycji: {nba_final['Pos'].nunique()}")
    return nba_final, transferred


@app.cell
def _(nba_final, pd):
    # Estymata MLE skuteczności rzutów wolnych na pozycji
    ft_by_pos = nba_final.groupby('Pos', observed=True).apply(
        lambda g: pd.Series({'FT_Acc_MLE': g['FT'].sum() / g['FTA'].sum()})
    )
    print("Skuteczność MLE według pozycji:")
    print(ft_by_pos)

    total_acc = nba_final['FT'].sum() / nba_final['FTA'].sum()
    print(f"\nOgólna skuteczność MLE: {total_acc:.4f}")
    return ft_by_pos, total_acc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analiza wizualna
    """)
    return


@app.cell
def _(nba_final, px):
    fig_box = px.box(nba_final, x='Pos', y='FT%',
                     title='Skuteczność rzutów wolnych według pozycji',
                     labels={'Pos': 'Pozycja', 'FT%': 'Skuteczność rzutów wolnych'})
    fig_box.show()
    return (fig_box,)


@app.cell
def _(nba_final, px):
    nba_final_plot = nba_final.copy()
    nba_final_plot['PTS_per_G'] = nba_final_plot['PTS'] / nba_final_plot['G']

    fig_scatter = px.scatter(
        nba_final_plot, x='PTS_per_G', y='FT%',
        color='Pos',
        title='Skuteczność rzutów wolnych vs średnia punktów/mecz',
        labels={'PTS_per_G': 'Punkty na mecz', 'FT%': 'Skuteczność rz. wolnych'}
    )
    fig_scatter.show()
    return fig_scatter, nba_final_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model GLMM (Generalized Linear Mixed Model)

    Modelujemy skuteczność wykonywania rzutów wolnych używając GLMM z rozkładem
    dwumianowym (Binomial).

    Model wyjściowy (RI z 3 poziomami: rzut → gracz → pozycja):
    $$
      FT_{p|pos} \sim Binom(FTA_{p|pos}, h(\beta_0 + v_{pos} + u_{p|pos}))
    $$

    gdzie $h$ to funkcja logistyczna.

    W `statsmodels` najlepiej przybliżamy to przez `BinomialBayesMixedGLM` lub
    `GLMM` z `statsmodels.genmod`. Dla zagnieżdżonych efektów losowych używamy
    uproszczonego modelu z dwoma poziomami grupowania.
    """)
    return


@app.cell
def _(nba_final, np, sm, smf):
    # Przygotowanie danych
    nba_model_data = nba_final[['FT', 'FTA', 'Pos', 'Player']].copy()
    nba_model_data = nba_model_data[nba_model_data['FTA'] > 0]

    # Model GLMM z efektem losowym dla gracza (i jego pozycji)
    # Uproszczony model: 2-poziomowy (gracz w pozycji)
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        # Zakodowanie grupowania
        nba_model_data['Pos_enc'] = nba_model_data['Pos'].cat.codes
        nba_model_data['Player_enc'] = nba_model_data['Player'].astype('category').cat.codes

        endog = nba_model_data[['FT', 'FTA']].values
        exog = np.ones((len(nba_model_data), 1))  # tylko intercept jako efekt stały

        # Efekty losowe
        exog_vc = {
            'Pos': (nba_model_data['Pos_enc'].values, nba_model_data['Pos_enc'].nunique()),
        }

        glmm_model = BinomialBayesMixedGLM(endog, exog, exog_vc=exog_vc)
        glmm_result = glmm_model.fit_vb()
        print(glmm_result.summary())

    except Exception as e:
        print(f"Błąd GLMM: {e}")
        print("\nAlternatywnie: model logistyczny z cechą grupową jako zmienną stałą:")
        nba_model_data['success_rate'] = nba_model_data['FT'] / nba_model_data['FTA']

        simple_model = smf.ols('success_rate ~ C(Pos)', data=nba_model_data).fit()
        print(simple_model.summary())

        # Skuteczności według pozycji z modelu
        print("\nEfekty pozycji:")
        print(simple_model.params)
    return nba_model_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zestawienie estymat MLE z estymatami modelu

    Porównujemy skuteczności:
    - **MLE** (bezpośrednie z danych) — mogą być niestabilne dla graczy z małą liczbą rzutów
    - **Model** — stabilniejsze, "ściągane" w stronę średniej grupowej

    ### Zadanie

    Porównaj model RI z następującymi modelami GLMM:

    1. Uproszczony RI z efektem losowym tylko na poziomie pozycji
    2. Model z dodatkowym efektem stałym: średnia liczba punktów/mecz (PTS/G)
       i losowymi wyrazami wolnymi jak w modelu wyjściowym
    3. Model z dwoma efektami stałymi jak wyżej i z dodanym efektem losowym —
       PTS/G na poziomie pozycji

    **Uwaga**: W modelach uwzględniających PTS/G warto tę zmienną ustandaryzować.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
