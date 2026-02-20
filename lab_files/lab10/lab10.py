import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 10: Modele graficzne skierowane i nieskierowane

    Laboratorium podzielone jest na dwie części:
    - **Modele graficzne skierowane** (Bayesian Networks / Sieci Bayesowskie)
    - **Modele graficzne nieskierowane** (Markov Random Fields)

    Narzędzia te służą do rozpoznawania i analizy zależności probabilistycznych
    między zmiennymi.

    Przed wykonaniem upewnij się, że masz zainstalowane:
    ```
    pip install pgmpy networkx
    ```

    Dokumentacja:
    - **pgmpy**: https://pgmpy.org
    - **networkx**: https://networkx.org
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import networkx as nx
    import plotly.express as px
    import plotly.graph_objects as go
    return go, np, nx, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane: zbiór Titanic

    Używamy klasycznego zbioru Titanic (wbudowany w scikit-learn / seaborn).
    """)
    return


@app.cell
def _(pd):
    import seaborn as sns

    titanic_raw = sns.load_dataset('titanic')

    # Wybieramy kluczowe kolumny (jak w oryginalnym zbiorze R: Class, Sex, Age, Survived)
    titanic_df = titanic_raw[['pclass', 'sex', 'age', 'survived']].dropna().copy()
    titanic_df.columns = ['Class', 'Sex', 'Age', 'Survived']

    # Dyskretyzacja Age
    titanic_df['Age'] = (titanic_df['Age'] > 18).map({True: 'Adult', False: 'Child'})
    titanic_df['Class'] = titanic_df['Class'].astype(str)
    titanic_df['Sex'] = titanic_df['Sex'].str.capitalize()
    titanic_df['Survived'] = titanic_df['Survived'].map({1: 'Yes', 0: 'No'})

    print(titanic_df.head())
    print(titanic_df.dtypes)
    print(titanic_df.shape)
    return sns, titanic_df, titanic_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modele skierowane (Bayesian Networks)

    ## Nauka struktury sieci bayesowskiej

    Używamy algorytmu **Hill Climbing** z pakietu `pgmpy`.
    """)
    return


@app.cell
def _(nx, titanic_df):
    try:
        from pgmpy.estimators import HillClimbSearch, BicScore
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import MaximumLikelihoodEstimator

        # Nauka struktury metodą Hill Climbing
        hc = HillClimbSearch(titanic_df)
        bic = BicScore(titanic_df)
        bn_structure = hc.estimate(scoring_method=bic)

        print("Krawędzie nauczonej sieci bayesowskiej:")
        for edge in bn_structure.edges():
            print(f"  {edge[0]} -> {edge[1]}")

        # Wizualizacja struktury
        G_bn = nx.DiGraph()
        G_bn.add_edges_from(bn_structure.edges())

        # Rysowanie grafu
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(G_bn, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw(G_bn, pos, with_labels=True, node_color='lightcoral',
                node_size=2000, font_size=12, arrows=True,
                arrowsize=20, edge_color='gray')
        plt.title("Struktura Sieci Bayesowskiej - Titanic")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Pakiet pgmpy nie jest zainstalowany.")
        print("Instalacja: pip install pgmpy")
        bn_structure = None
    return (
        BayesianNetwork,
        BicScore,
        HillClimbSearch,
        MaximumLikelihoodEstimator,
        G_bn,
        bn_structure,
        hc,
        plt,
        pos,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dopasowanie parametrów sieci bayesowskiej
    """)
    return


@app.cell
def _(BayesianNetwork, MaximumLikelihoodEstimator, bn_structure, titanic_df):
    if bn_structure is not None:
        bn_model = BayesianNetwork(bn_structure.edges())
        bn_model.fit(titanic_df, estimator=MaximumLikelihoodEstimator)

        # Prawdopodobieństwa warunkowe dla zmiennej Survived
        print("CPD dla węzła 'Survived':")
        print(bn_model.get_cpds('Survived'))
    return (bn_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **1. Zinterpretuj prawdopodobieństwa warunkowe dla zmiennej "Survived".
    Jak płeć wpływa na szanse przeżycia?**

    ## Wnioskowanie w sieci bayesowskiej
    """)
    return


@app.cell
def _(bn_model, bn_structure):
    if bn_structure is not None:
        from pgmpy.inference import VariableElimination

        inference = VariableElimination(bn_model)

        # P(Survived=Yes | Sex=Female, Class=1)
        try:
            result1 = inference.query(
                variables=['Survived'],
                evidence={'Sex': 'Female', 'Class': '1'}
            )
            print("P(Survived=Yes | Female, 1st class):")
            print(result1)
        except Exception as e:
            print(f"Błąd wnioskowania: {e}")

        # P(Survived=Yes | Sex=Male, Class=3)
        try:
            result2 = inference.query(
                variables=['Survived'],
                evidence={'Sex': 'Male', 'Class': '3'}
            )
            print("\nP(Survived=Yes | Male, 3rd class):")
            print(result2)
        except Exception as e:
            print(f"Błąd wnioskowania: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modele nieskierowane (Markov Random Fields)

    ## Analiza korelacji

    Podstawą grafów nieskierowanych jest analiza korelacji między zmiennymi.
    """)
    return


@app.cell
def _(px, titanic_df):
    import seaborn as sns_corr

    # Kodowanie numeryczne zmiennych kategorycznych
    titanic_numeric = titanic_df.copy()
    titanic_numeric['Class'] = titanic_numeric['Class'].map({'1': 1, '2': 2, '3': 3})
    titanic_numeric['Sex'] = titanic_numeric['Sex'].map({'Male': 0, 'Female': 1})
    titanic_numeric['Age'] = titanic_numeric['Age'].map({'Child': 0, 'Adult': 1})
    titanic_numeric['Survived'] = titanic_numeric['Survived'].map({'No': 0, 'Yes': 1})

    cor_matrix = titanic_numeric.corr()
    print("Macierz korelacji:")
    print(cor_matrix.round(3))

    fig_corr = px.imshow(cor_matrix, text_auto='.2f',
                         title='Macierz korelacji - Titanic',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_corr.show()
    return cor_matrix, fig_corr, sns_corr, titanic_numeric


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Które pary zmiennych wykazują najsilniejsze korelacje? Jak można to
    zinterpretować w kontekście katastrofy Titanica?**

    ## Graf nieskierowany oparty na korelacjach
    """)
    return


@app.cell
def _(cor_matrix, nx, plt):
    threshold = 0.1
    adj_matrix = (cor_matrix.abs() > threshold).values
    __import__('numpy').fill_diagonal(adj_matrix, False)

    G_undirected = nx.from_numpy_array(adj_matrix)
    G_undirected = nx.relabel_nodes(G_undirected, dict(enumerate(cor_matrix.columns)))

    pos_und = nx.spring_layout(G_undirected, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G_undirected, pos_und, with_labels=True,
            node_color='lightblue', node_size=2000,
            font_size=12, edge_color='gray')
    plt.title(f"Graf nieskierowany (|korelacja| > {threshold})")
    plt.tight_layout()
    plt.show()
    return G_undirected, adj_matrix, pos_und, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kliki w grafie nieskierowanym
    """)
    return


@app.cell
def _(G_undirected, nx):
    max_cliques = list(nx.find_cliques(G_undirected))
    print(f"Liczba klik maksymalnych: {len(max_cliques)}")

    for i, clique in enumerate(max_cliques):
        print(f"Klika {i+1}: {', '.join(clique)}")
    return i, max_cliques


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3. Zinterpretuj znalezione kliki. Co oznacza, że zmienne tworzą klikę
    w grafie nieskierowanym?**

    ## Porównanie modeli

    ### Test niezależności warunkowej
    """)
    return


@app.cell
def _(titanic_numeric):
    from scipy.stats import chi2_contingency

    # Test chi-kwadrat dla par zmiennych
    pairs = [('Age', 'Class'), ('Sex', 'Survived'), ('Class', 'Survived')]

    for col1, col2 in pairs:
        contingency = __import__('pandas').crosstab(titanic_numeric[col1], titanic_numeric[col2])
        chi2, p_val, dof, _ = chi2_contingency(contingency)
        print(f"{col1} vs {col2}: chi²={chi2:.2f}, p={p_val:.4f}, {'istotne' if p_val < 0.05 else 'nieistotne'}")
    return chi2, chi2_contingency, col1, col2, contingency, dof, p_val, pairs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **4. Stwórz własną strukturę sieci bayesowskiej na podstawie wiedzy domenowej
    o katastrofie Titanica. Porównaj ją ze strukturą nauczoną automatycznie.**
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
