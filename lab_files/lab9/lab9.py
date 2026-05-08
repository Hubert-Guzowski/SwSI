import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 9: Modele graficzne — sieci bayesowskie, pola Markowa, grafy czynników

    Pełen rozkład łączny zmiennych dyskretnych rośnie wykładniczo wraz z liczbą zmiennych. Modele graficzne dostarczają kompaktowej reprezentacji rozkładu łącznego (jako iloczynu lokalnych czynników) oraz języka wnioskowania o niezależnościach warunkowych w oparciu o własności topologiczne grafu.

    Trzy reprezentacje, którymi zajmujemy się w tym laboratorium:

    - **Sieci bayesowskie** (skierowane) — naturalne, gdy istnieje porządek przyczynowy lub czasowy.
    - **Pola losowe Markowa** (MRF, nieskierowane) — naturalne przy zależnościach symetrycznych (piksele obrazu, atomy w sieci krystalicznej).
    - **Grafy czynników** (bipartytowe) — uogólnienie obu poprzednich reprezentacji, wygodne przy algorytmach przekazywania komunikatów.

    Podstawową lekturą jest podręcznik *Probabilistic Graphical Models: Principles and Techniques* Daphne Koller i Nira Friedmana (MIT Press, 2009): http://mcb111.org/w06/KollerFriedman.pdf. Rozdziały 3, 4 i 11 pokrywają — odpowiednio — sieci bayesowskie, pola Markowa i wnioskowanie dokładne. W tym laboratorium ograniczamy się do ujęcia intuicyjnego, a po formalne wyprowadzenia odsyłamy do podręcznika.

    Narzędzia:

    - **pgmpy** — https://pgmpy.org
    - **networkx** — https://networkx.org

    Zbiór: ten sam **Adult** z UCI, którego używaliśmy w lab8.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='pgmpy')

    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, TreeSearch
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination, BeliefPropagation

    return (
        BeliefPropagation,
        DiscreteBayesianNetwork,
        HillClimbSearch,
        TreeSearch,
        VariableElimination,
        mo,
        np,
        nx,
        pd,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dane: Adult (UCI)

    Wczytujemy zbiór bezpośrednio z UCI i sprowadzamy wszystkie zmienne do dyskretnych etykiet — pgmpy operuje na zmiennych dyskretnych, a ograniczenie liczby poziomów zapobiega nadmiernemu rozrostowi i trochę uprości nam interpretację.
    """)
    return


@app.cell
def _(pd):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]
    raw = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

    education_map = {
        'Preschool': 'No-HS', '1st-4th': 'No-HS', '5th-6th': 'No-HS', '7th-8th': 'No-HS',
        '9th': 'No-HS', '10th': 'No-HS', '11th': 'No-HS', '12th': 'No-HS',
        'HS-grad': 'HS',
        'Some-college': 'College', 'Assoc-voc': 'College', 'Assoc-acdm': 'College',
        'Bachelors': 'Bachelors',
        'Masters': 'Advanced', 'Prof-school': 'Advanced', 'Doctorate': 'Advanced',
    }
    marital_map = {
        'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
        'Married-spouse-absent': 'Other', 'Separated': 'Other',
        'Divorced': 'Other', 'Widowed': 'Other',
        'Never-married': 'Single',
    }
    occupation_map = {
        'Exec-managerial': 'White-collar', 'Prof-specialty': 'White-collar',
        'Tech-support': 'White-collar', 'Adm-clerical': 'White-collar', 'Sales': 'White-collar',
        'Craft-repair': 'Blue-collar', 'Machine-op-inspct': 'Blue-collar',
        'Transport-moving': 'Blue-collar', 'Handlers-cleaners': 'Blue-collar',
        'Farming-fishing': 'Blue-collar',
        'Other-service': 'Service', 'Priv-house-serv': 'Service', 'Protective-serv': 'Service',
        'Armed-Forces': 'Other',
    }

    adult = raw.dropna(subset=['occupation', 'workclass']).copy()
    adult = adult[['age', 'education', 'marital-status', 'occupation',
                   'relationship', 'sex', 'hours-per-week', 'income']]
    adult['age'] = pd.cut(adult['age'], bins=[0, 30, 50, 100],
                          include_lowest=True, labels=['Young', 'Mid', 'Senior']).astype(str)
    adult['hours'] = pd.cut(adult['hours-per-week'], bins=[0, 35, 45, 100],
                            include_lowest=True, labels=['Part', 'Full', 'Over']).astype(str)
    adult['education'] = adult['education'].map(education_map)
    adult['marital'] = adult['marital-status'].map(marital_map)
    adult['occupation'] = adult['occupation'].map(occupation_map)
    adult['income'] = adult['income'].map({'<=50K': 'low', '>50K': 'high'})
    adult = adult.drop(columns=['marital-status', 'hours-per-week'])
    adult = adult[['age', 'education', 'marital', 'occupation', 'relationship', 'sex', 'hours', 'income']]
    adult = adult.dropna()

    print(f"{adult.shape[0]} obserwacji, {adult.shape[1]} zmiennych")
    print("\nLiczba poziomów na zmienną:")
    for col in adult.columns:
        print(f"  {col:14s} {adult[col].nunique()}  ({sorted(adult[col].unique())})")
    return (adult,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sieci bayesowskie

    Sieć bayesowska to skierowany graf acykliczny, w którym każdy węzeł reprezentuje zmienną losową, a każda strzałka — bezpośrednią zależność warunkową. Rozkład łączny faktoryzuje się na iloczyn lokalnych rozkładów warunkowych „zmienna pod warunkiem swoich rodziców":

    $$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \mathrm{Pa}(X_i)).$$

    Każdemu węzłowi $X_i$ odpowiada **tablica rozkładów warunkowych** (CPD). Im więcej rodziców ma węzeł, tym większa tablica — stąd motywacja, by struktura sieci była rzadka.

    ## Uczenie struktury — Hill Climbing

    Pełna enumeracja grafów acyklicznych jest niewykonalna nawet dla niewielu węzłów, więc uczenie struktury realizujemy heurystycznie — przeszukujemy przestrzeń grafów lokalnymi modyfikacjami (dodaj, usuń lub odwróć krawędź), kierując się funkcją oceniającą karzącą za złożoność. Standardowym wyborem jest poznane wcześniej **BIC** (Bayesian Information Criterion).
    """)
    return


@app.cell
def _(HillClimbSearch, adult, time):
    t0 = time.perf_counter()
    hc = HillClimbSearch(adult)
    bn_struct_bic = hc.estimate(scoring_method='bic-d', show_progress=False)
    t_bic = time.perf_counter() - t0

    print(f"HC + BIC: {t_bic:.1f}s, {len(list(bn_struct_bic.edges()))} krawędzi")
    print("\nKrawędzie nauczonej sieci:")
    for src, dst in bn_struct_bic.edges():
        print(f"  {src} -> {dst}")
    return bn_struct_bic, hc


@app.cell
def _(adult, bn_struct_bic, nx, plt):
    G_bn = nx.DiGraph()
    G_bn.add_nodes_from(adult.columns.tolist())
    G_bn.add_edges_from(bn_struct_bic.edges())

    pos_bn = nx.spring_layout(G_bn, seed=42, k=1.6)
    fig_bn, ax_bn = plt.subplots(figsize=(10, 7))
    nx.draw(G_bn, pos_bn, with_labels=True, node_color='#fcb1a6', node_size=2400,
            font_size=11, arrows=True, arrowsize=22, edge_color='gray',
            connectionstyle='arc3,rad=0.05', ax=ax_bn)
    ax_bn.set_title("Sieć bayesowska — Adult (HC + BIC)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inna funkcja oceniająca — K2

    BIC nie jest jedyną sensowną funkcją oceny. **K2** (Cooper & Herskovits, 1992) to bayesowska funkcja oceniająca. W przeciwieństwie do BIC nie zawiera jawnego członu kary za złożoność, przez co zwykle dopuszcza nieco bogatsze grafy.
    """)
    return


@app.cell
def _(bn_struct_bic, hc):
    bn_struct_k2 = hc.estimate(scoring_method='k2', show_progress=False)
    edges_bic = set(bn_struct_bic.edges())
    edges_k2 = set(bn_struct_k2.edges())

    print(f"Liczba krawędzi: BIC = {len(edges_bic)}, K2 = {len(edges_k2)}")
    print("\nKrawędzie tylko w K2 (BIC ich nie wybrał):")
    for e in sorted(edges_k2 - edges_bic):
        print(f"  {e[0]} -> {e[1]}")
    print("\nKrawędzie tylko w BIC:")
    for e in sorted(edges_bic - edges_k2):
        print(f"  {e[0]} -> {e[1]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dopasowanie parametrów

    Mając strukturę, parametry lokalnych CPD estymujemy metodą największej wiarygodności — czyli relatywnymi częstościami w danych. `bn.fit(df)` korzysta z MLE domyślnie.
    """)
    return


@app.cell
def _(DiscreteBayesianNetwork, adult, bn_struct_bic):
    bn_model = DiscreteBayesianNetwork(bn_struct_bic.edges())
    bn_model.add_nodes_from(adult.columns.tolist())
    bn_model.fit(adult)

    print("CPD dla węzła 'income':")
    print(bn_model.get_cpds('income'))
    return (bn_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wizualizacja CPD

    Tabela liczb jest mało czytelna. Ten sam rozkład warunkowy łatwiej odczytać z wykresu — dla każdej kombinacji wartości rodziców pokazujemy słupkowy rozkład prawdopodobieństwa zmiennej. Aby wykres pozostał czytelny niezależnie od konkretnej nauczonej struktury, wybieramy węzeł z najmniejszą liczbą kombinacji rodziców.
    """)
    return


@app.cell
def _(bn_model, np, plt):
    from itertools import product

    def _combo_count(node):
        cpd = bn_model.get_cpds(node)
        parents = cpd.variables[1:]
        if not parents:
            return float('inf')
        total = 1
        for p in parents:
            total *= len(cpd.state_names[p])
        return total

    viz_node = min(bn_model.nodes(), key=_combo_count)
    cpd_viz = bn_model.get_cpds(viz_node)
    parents_viz = cpd_viz.variables[1:]
    target_states = cpd_viz.state_names[viz_node]
    parent_states = [cpd_viz.state_names[p] for p in parents_viz]

    values_viz = cpd_viz.values.reshape(len(target_states), -1)
    combos = list(product(*parent_states))
    labels_viz = ['\n'.join(c) for c in combos]

    x = np.arange(len(combos))
    width = 0.8 / len(target_states)
    fig_cpd, ax_cpd = plt.subplots(figsize=(max(8, 1.2 * len(combos)), 4.5))
    for _i, _st in enumerate(target_states):
        ax_cpd.bar(x + (_i - (len(target_states) - 1) / 2) * width,
                   values_viz[_i], width, label=f"{viz_node}={_st}")
    ax_cpd.set_xticks(x)
    ax_cpd.set_xticklabels(labels_viz, fontsize=9)
    ax_cpd.set_ylabel(f"P({viz_node} | rodzice)")
    ax_cpd.set_title(f"CPD węzła '{viz_node}' (rodzice: {', '.join(parents_viz)})")
    ax_cpd.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Otoczka Markowa

    Otoczka Markowa zmiennej $X$ to najmniejszy zbiór węzłów taki, że pod warunkiem otoczki $X$ jest niezależna od reszty grafu. W sieci bayesowskiej składa się z rodziców $X$, dzieci $X$ oraz pozostałych rodziców dzieci $X$. Znając wartości otoczki, dane spoza niej nie wnoszą nic nowego do predykcji $X$ — klasyfikator korzystający wyłącznie ze zmiennych otoczki Markowa zmiennej celu powinien osiągać dokładność porównywalną z klasyfikatorem korzystającym ze wszystkich predyktorów.
    """)
    return


@app.cell
def _(bn_model):
    mb_income = bn_model.get_markov_blanket('income')
    print(f"Otoczka Markowa zmiennej 'income': {sorted(mb_income)}")

    print("\nOtoczki Markowa pozostałych zmiennych:")
    for var in ['age', 'education', 'occupation', 'sex']:
        print(f"  {var:12s} {sorted(bn_model.get_markov_blanket(var))}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wnioskowanie — eliminacja zmiennych

    Mając kilka zmiennych w roli dowodów (`evidence`), chcemy wyznaczyć rozkład warunkowy interesującej nas zmiennej. Algorytm **eliminacji zmiennych** wyznacza go dokładnie, sumując kolejno zmienne spoza zapytania i dowodów. Kolejność eliminacji wpływa na koszt obliczeń.
    """)
    return


@app.cell
def _(VariableElimination, bn_model):
    inf = VariableElimination(bn_model)

    q1 = inf.query(['income'],
                   evidence={'sex': 'Female', 'education': 'Advanced'},
                   show_progress=False)
    print("P(income | Female, Advanced):")
    print(q1)

    q2 = inf.query(['income'],
                   evidence={'sex': 'Male', 'education': 'No-HS', 'hours': 'Part'},
                   show_progress=False)
    print("\nP(income | Male, No-HS, Part-time):")
    print(q2)

    q3 = inf.query(['occupation'],
                   evidence={'income': 'high', 'education': 'Bachelors'},
                   show_progress=False)
    print("\nP(occupation | high income, Bachelors):")
    print(q3)
    return (inf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Próbkowanie z rozkładu

    Sieć bayesowska reprezentuje pełny rozkład łączny, więc — odwracając kierunek użycia — możemy nią próbkować syntetyczne obserwacje. `simulate(N)` losuje wartości węzłów w porządku topologicznym: najpierw węzły bez rodziców z ich rozkładów brzegowych, potem kolejne — z odpowiednich CPD.
    """)
    return


@app.cell
def _(bn_model):
    sample = bn_model.simulate(n_samples=10, show_progress=False)
    print(sample.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 1

    **a)** Wyświetl CPD dla węzła z największą liczbą rodziców w nauczonej strukturze i porównaj liczbę parametrów tej tablicy z rozmiarem pełnej tablicy łącznej dla tych samych zmiennych.

    **b)** Sprawdź empirycznie własność otoczki Markowa: czy zapytanie `P(income | evidence)` daje ten sam wynik, gdy `evidence` zawiera wyłącznie otoczkę Markowa, jak wtedy gdy zawiera dodatkowo zmienne spoza niej?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pola losowe Markowa

    Czasem między zmiennymi nie ma naturalnej kolejności przyczynowej — np. piksele obrazu, sąsiadujące słowa w zdaniu, atomy w sieci krystalicznej. Wtedy model nieskierowany jest wygodniejszy. **Pole losowe Markowa** (MRF, Markov Random Field) ma postać

    $$P(\mathbf{x}) = \frac{1}{Z}\prod_{C\in\mathcal{C}}\phi_C(\mathbf{x}_C),$$

    gdzie $\mathcal{C}$ to zbiór klik grafu nieskierowanego, $\phi_C$ to nieujemne funkcje potencjału na konfiguracjach zmiennych w klice $C$, a $Z$ to stała normalizacyjna (Koller & Friedman, rozdz. 4).

    ## Moralizacja sieci bayesowskiej

    Do każdej sieci bayesowskiej można skonstruować równoważny MRF poprzez **moralizację**: usuwamy kierunki krawędzi i łączymy parami wszystkich rodziców każdego węzła (stąd potoczna nazwa: „ożeń rodziców"). Operacja zachowuje rozkład łączny, ale traci informację o niezależnościach kierunkowych — w szczególności o v-strukturach.
    """)
    return


@app.cell
def _(bn_model, nx, plt):
    mn = bn_model.to_markov_model()

    G_mrf = nx.Graph()
    G_mrf.add_nodes_from(mn.nodes())
    G_mrf.add_edges_from(mn.edges())

    fig_mrf, ax_mrf = plt.subplots(figsize=(10, 7))
    nx.draw(G_mrf, nx.spring_layout(G_mrf, seed=42, k=1.6),
            with_labels=True, node_color='#bca6fc', node_size=2400,
            font_size=11, edge_color='gray', ax=ax_mrf)
    ax_mrf.set_title("MRF zmoralizowany z BN — Adult")
    plt.tight_layout()
    plt.show()

    print(f"Liczba czynników w MRF: {len(mn.factors)}")
    print(f"Pierwsze trzy czynniki (zmienne, kardynalność):")
    for f in mn.factors[:3]:
        print(f"  vars={f.variables}  card={list(f.cardinality)}")
    return (mn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 2

    Znajdź w nauczonej sieci bayesowskiej węzeł z co najmniej dwoma rodzicami. Wykonaj test chi-kwadrat brzegowej niezależności tych rodziców (`scipy.stats.chi2_contingency`), a następnie powtórz test po warunkowaniu na wartościach wspólnego dziecka — czy wyniki się różnią?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grafy czynników

    Graf czynników to **bipartytowa** reprezentacja faktoryzacji rozkładu: dwa rodzaje węzłów — okrągłe **węzły zmiennych** i kwadratowe **węzły czynników** — przy czym krawędzie łączą czynnik z każdą zmienną, od której zależy. Reprezentacja ta nie wnosi nowej informacji względem BN czy MRF, ale ujawnia strukturę faktoryzacji w sposób jednoznaczny i jest naturalna dla algorytmów **przekazywania komunikatów** (sum-product, belief propagation; Koller & Friedman, rozdz. 11).

    ## Konwersja BN → graf czynników

    Każde CPD $P(X_i \mid \mathrm{Pa}(X_i))$ staje się czynnikiem $\phi_i$ obejmującym $X_i$ i jego rodziców.
    """)
    return


@app.cell
def _(bn_model, mn, nx, plt):
    fg = mn.to_factor_graph()

    var_nodes = [n for n in fg.nodes() if not str(n).startswith('phi_')]
    factor_nodes = [n for n in fg.nodes() if str(n).startswith('phi_')]

    pos_fg = nx.spring_layout(fg, seed=42, k=1.8)
    fig_fg, ax_fg = plt.subplots(figsize=(11, 7.5))
    nx.draw_networkx_nodes(fg, pos_fg, nodelist=var_nodes, node_color='#fcb1a6',
                           node_size=2200, node_shape='o', ax=ax_fg)
    nx.draw_networkx_nodes(fg, pos_fg, nodelist=factor_nodes, node_color='#d4d4d4',
                           node_size=900, node_shape='s', ax=ax_fg)
    nx.draw_networkx_edges(fg, pos_fg, edge_color='gray', ax=ax_fg)
    nx.draw_networkx_labels(fg, pos_fg, font_size=9, ax=ax_fg)
    ax_fg.set_title("Graf czynników z BN — kółka: zmienne, kwadraty: czynniki")
    ax_fg.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Liczba węzłów zmiennych: {len(var_nodes)}")
    print(f"Liczba węzłów czynników: {len(factor_nodes)}")
    print(f"Czynniki BN: {len(bn_model.get_cpds())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Belief Propagation — to samo zapytanie, inny algorytm

    Eliminacja zmiennych wyznacza rozkład brzegowy jednej zmiennej naraz; każde nowe zapytanie wymaga osobnego przebiegu. **Belief Propagation** w jednym przejściu wyznacza brzegi wszystkich zmiennych jednocześnie, kosztem zbudowania **drzewa złączeń** z grafu czynników. Dla pojedynczego zapytania VE bywa szybsze; przy wielu zapytaniach na tym samym modelu BP zyskuje przewagę dzięki amortyzacji kosztu kalibracji.

    Sprawdźmy, że obie metody dają identyczny wynik:
    """)
    return


@app.cell
def _(BeliefPropagation, bn_model, inf):
    bp = BeliefPropagation(bn_model)
    bp.calibrate()

    evidence = {'sex': 'Male', 'education': 'Bachelors'}
    q_ve = inf.query(['income'], evidence=evidence, show_progress=False)
    q_bp = bp.query(['income'], evidence=evidence, show_progress=False)

    print("Variable Elimination:")
    print(q_ve)
    print("\nBelief Propagation:")
    print(q_bp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 3

    Zmierz `time.perf_counter()` dla pojedynczego zapytania oraz dla 50 różnych zapytań na obiektach `VariableElimination` i `BeliefPropagation` (pamiętaj o `calibrate()` po utworzeniu BP) — czy widzisz spodziewaną amortyzację BP?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zestawienie

    | Cecha | Sieć bayesowska | MRF | Graf czynników |
    |---|---|---|---|
    | Graf | skierowany acykliczny | nieskierowany | bipartytowy |
    | Lokalne czynniki | $P(X_i\mid\mathrm{Pa}(X_i))$ — sumują się do 1 | $\phi_C(\mathbf{x}_C)$ — dowolne nieujemne | jak źródło konwersji |
    | Stała normalizacyjna | brak (faktoryzacja już daje rozkład) | $Z$ — wymaga policzenia | dziedziczona z reprezentacji |
    | Niezależności | d-separacja | separacja | jak MRF |
    | Naturalne kiedy | zależności kierunkowe / przyczynowe | symetryczne sąsiedztwa | algorytmy przekazywania komunikatów |
    | Uczenie struktury | HC + BIC/K2, PC, GES | rzadziej automatyczne | dziedziczone |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NBC i TAN — szczególne sieci bayesowskie

    Naiwny klasyfikator bayesowski (NBC) i jego rozszerzenie TAN (Tree-Augmented Naive Bayes) to sieci bayesowskie o ustalonej, prostej strukturze:

    - **NBC** — gwiazda wokół klasy: zmienna celu jest jedynym rodzicem wszystkich pozostałych. Założenie: cechy są warunkowo niezależne pod warunkiem klasy.
    - **TAN** — rozszerza NBC o drzewo nad cechami: każda cecha może mieć dodatkowo jednego rodzica wśród innych cech. Złagodzenie założenia o niezależności cech kosztem niewielu krawędzi.

    Strukturę TAN wyznacza algorytm Chow–Liu (drzewo o maksymalnej informacji wzajemnej między cechami) uzupełniony o krawędzie z klasy do wszystkich cech. W pgmpy realizuje to `TreeSearch` z `estimator_type='tan'`.
    """)
    return


@app.cell
def _(DiscreteBayesianNetwork, adult):
    features = [c for c in adult.columns if c != 'income']
    nbc = DiscreteBayesianNetwork([('income', col) for col in features])
    nbc.fit(adult)

    print("NBC — krawędzie:")
    print("\n".join(f"  {u} -> {v}" for u, v in nbc.edges()))
    return features, nbc


@app.cell
def _(DiscreteBayesianNetwork, TreeSearch, adult):
    ts = TreeSearch(adult)
    tan_dag = ts.estimate(estimator_type='tan', class_node='income', show_progress=False)
    tan = DiscreteBayesianNetwork(tan_dag.edges())
    tan.fit(adult)

    print("TAN — krawędzie:")
    print("\n".join(f"  {u} -> {v}" for u, v in tan.edges()))
    return (tan,)


@app.cell
def _(features, nbc, np, nx, plt, tan):
    fig_clf, axes_clf = plt.subplots(1, 2, figsize=(14, 6))

    G_nbc = nx.DiGraph(nbc.edges())
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    pos_nbc = {'income': (0.0, 0.0)}
    pos_nbc.update({
        feat: (np.cos(ang), np.sin(ang))
        for feat, ang in zip(features, angles)
    })
    nx.draw(G_nbc, pos_nbc, ax=axes_clf[0], with_labels=True,
            node_color='#fcd5a6', node_size=2000, font_size=9,
            arrows=True, arrowsize=15, edge_color='gray')
    axes_clf[0].set_title("NBC — gwiazda")

    G_tan = nx.DiGraph(tan.edges())
    pos_tan = nx.spring_layout(G_tan, seed=42, k=1.5)
    nx.draw(G_tan, pos_tan, ax=axes_clf[1], with_labels=True,
            node_color='#fcd5a6', node_size=2000, font_size=9,
            arrows=True, arrowsize=15, edge_color='gray')
    axes_clf[1].set_title("TAN — gwiazda + drzewo nad cechami")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zadania

    ## Zadanie 1 — domenowa sieć bayesowska vs nauczona

    Zaprojektuj ręcznie sieć bayesowską dla zbioru Adult opartą na wiedzy domenowej (np. `age -> marital`, `education -> occupation`, `occupation -> income`).

    **a)** Dopasuj parametry obu sieci (ręcznej i wyuczonej HC + BIC) i porównaj log-likelihood na zbiorze.

    **b)** Wykonaj klasyfikację `income` jako MAP zmiennej zapytania (`inf.map_query`) na zbiorze testowym i porównaj dokładność — czy ranking według log-likelihood pokrywa się z rankingiem dokładności klasyfikacji?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2 — otoczka Markowa w klasyfikacji

    Wytrenuj klasyfikator (np. `RandomForestClassifier`) zmiennej `income` na trzech wariantach predyktorów: (i) wszystkie zmienne, (ii) tylko otoczka Markowa, (iii) zmienne spoza otoczki. Porównaj dokładność testową oraz sprawdź, czy ranking ważności cech (`feature_importances_`) pokrywa się z otoczką Markowa.
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 3 — wnioskowanie na grafie czynników

    Skonstruuj graf czynników z MRF (`mn.to_factor_graph()`).

    **a)** Porównaj liczbę czynników w grafie z liczbą CPD w wyjściowej BN — skąd różnica?

    **b)** Uruchom `BeliefPropagation` zainicjalizowane na `bn_model` i porównaj wyniki z `BeliefPropagation` zainicjalizowanym na `mn` — czy zapytania `bp.query(['income'], evidence=...)` dają identyczny wynik?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
