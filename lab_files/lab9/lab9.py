import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 9: Modele graficzne — sieci bayesowskie, pola Markowa, grafy czynników

    Pełen rozkład łączny dla $n$ zmiennych dyskretnych rośnie wykładniczo z liczbą zmiennych — dla 14 zmiennych binarnych potrzebujemy $2^{14}=16384$ wpisów, a przy zmiennych o większej liczbie kategorii wzrost jest jeszcze szybszy. Modele graficzne pełnią dwie funkcje jednocześnie: dostarczają **kompaktowej reprezentacji** rozkładu łącznego (jako iloczynu lokalnych czynników, z których każdy odwołuje się do kilku zmiennych) oraz **języka wnioskowania** o niezależnościach warunkowych w oparciu o własności topologiczne grafu.

    Trzy uzupełniające się reprezentacje, które omawiamy w tym laboratorium:

    - **Sieci bayesowskie** (skierowane) — faktoryzacja $P(X_1,\dots,X_n)=\prod_i P(X_i\mid \mathrm{Pa}(X_i))$, naturalna wszędzie tam, gdzie istnieje porządek przyczynowy lub czasowy. Strzałkę $A\to B$ czyta się "wartość $B$ zależy od $A$".
    - **Pola losowe Markowa** (MRF, nieskierowane) — faktoryzacja przez potencjały na klikach $P(\mathbf{x})=\frac{1}{Z}\prod_C \phi_C(\mathbf{x}_C)$, naturalna, gdy zależności są symetryczne (np. piksele obrazu, atomy w sieci krystalicznej).
    - **Grafy czynników** (bipartytowe) — uogólnienie dwóch poprzednich reprezentacji, wygodne przy algorytmach przekazywania komunikatów.

    Podstawową lekturą jest podręcznik *Probabilistic Graphical Models: Principles and Techniques* Daphne Koller i Nira Friedmana (MIT Press, 2009): http://mcb111.org/w06/KollerFriedman.pdf. Rozdziały 3, 4 i 11 pokrywają — odpowiednio — sieci bayesowskie, pola Markowa i wnioskowanie dokładne. W tym laboratorium ograniczamy się do ujęcia intuicyjnego, a po formalne wyprowadzenia odsyłamy do podręcznika.

    Narzędzia:

    - **pgmpy** — https://pgmpy.org
    - **networkx** — https://networkx.org

    Zbiór: ten sam **Adult** z UCI, którego używaliśmy w lab8. Ponieważ pgmpy operuje na zmiennych dyskretnych, kolumny ciągłe (`age`, `hours-per-week`) zbinujemy, a kategoryczne o dużej liczbie poziomów skonsolidujemy do bardziej ogólnych etykiet — tak, by struktura sieci pozostała czytelna, a tablice rozkładów warunkowych nie urosły do nieczytelnych rozmiarów.
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

    from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination, BeliefPropagation

    return (
        BeliefPropagation,
        DiscreteBayesianNetwork,
        HillClimbSearch,
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

    Wczytujemy zbiór bezpośrednio z UCI, wybieramy osiem kolumn tworzących reprezentatywny przekrój "demografia → praca → dochód" i sprowadzamy wszystkie zmienne do dyskretnych etykiet:

    - `age` w trzech przedziałach: Young (<30), Mid (30–50), Senior (>50)
    - `hours` w trzech przedziałach: Part (<35), Full (35–45), Over (>45)
    - `education` skonsolidowane do pięciu poziomów (No-HS, HS, College, Bachelors, Advanced)
    - `marital` skonsolidowane do trzech kategorii (Married, Single, Other)
    - `occupation` skonsolidowane do czterech grup (White-collar, Blue-collar, Service, Other)
    - `relationship`, `sex` zostają w oryginale
    - `income` jako binarne (low/high)

    Konsolidacje są arbitralne, ale uzasadniają je dwa cele: ograniczenie rozmiaru tablic CPD (które rosną iloczynowo z kardynalnością rodziców) oraz czytelność grafu po nauczeniu struktury.
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

    Sieć bayesowska to skierowany graf acykliczny, w którym każdy węzeł reprezentuje zmienną losową, a każda strzałka — bezpośrednią zależność warunkową. Faktoryzacja jest prosta: rozkład łączny rozbija się na iloczyn lokalnych rozkładów warunkowych "zmienna pod warunkiem swoich rodziców":

    $$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \mathrm{Pa}(X_i)).$$

    Każdemu węzłowi $X_i$ odpowiada **tablica rozkładów warunkowych** (CPD) o rozmiarze $|X_i| \cdot \prod_{p \in \mathrm{Pa}(X_i)} |p|$. Stąd silna motywacja, by struktura miała niewielu rodziców przypadających na jeden węzeł — graf z krawędziami między wszystkimi parami daje tablice rozmiaru pełnego rozkładu łącznego, niwecząc tym samym wszelkie oszczędności.

    ## Uczenie struktury — Hill Climbing

    Liczba acyklicznych skierowanych grafów na $n$ węzłach rośnie superwykładniczo (dla $n=8$ to już ponad $7\cdot 10^{11}$), więc pełna enumeracja jest niewykonalna. Standardowy kompromis: **przeszukiwanie heurystyczne** w przestrzeni grafów z lokalnymi modyfikacjami (dodaj krawędź, usuń krawędź, odwróć krawędź) i **funkcja oceniająca** karząca za złożoność.

    BIC (Bayesian Information Criterion) ma postać $\log L(\mathcal{D};G) - \tfrac{1}{2} d(G)\log N$ — dopasowanie do danych minus kara proporcjonalna do liczby parametrów modelu i logarytmu liczby obserwacji. Jest to standardowy wybór; w pgmpy realizujemy go przekazując `scoring_method='bic-d'` (BIC dyskretny).
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
    Niektóre kierunki krawędzi mogą wydawać się odwrócone wobec intuicji (np. `education -> age` zamiast `age -> education`). HC z BIC nie jest w stanie odróżnić struktur z **tej samej klasy równoważności Markowa** — pary $A\to B$ i $A\leftarrow B$ generują ten sam zbiór niezależności warunkowych, gdy nie tworzą one v-struktury z trzecim węzłem. Rozróżnienie wymagałoby albo wiedzy domenowej, albo eksperymentów interwencyjnych (Koller & Friedman, rozdz. 3.4).

    ## Inna funkcja oceniająca — K2

    BIC nie jest jedyną sensowną funkcją oceny. **K2** (Cooper & Herskovits, 1992) to bayesowska funkcja oceniająca, zakładająca równe prawdopodobieństwo a priori dla każdej struktury i jednorodne priory Dirichleta na CPD. W przeciwieństwie do BIC nie zawiera jawnego członu kary za złożoność — kara wynika tu z samego rozkładu a priori — przez co K2 zwykle dopuszcza nieco bogatsze grafy.
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

    Mając strukturę, parametry lokalnych CPD szacujemy maksymalną wiarygodnością — czyli relatywnymi częstościami w danych. `bn.fit(df)` używa MLE domyślnie. Dla rzadkich kombinacji rodziców MLE wyznacza zera w CPD; w zastosowaniach produkcyjnych lepiej skorzystać z estymatora bayesowskiego z priorem Dirichleta, ale tutaj wystarczy MLE.
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
    ## Otoczka Markowa

    Otoczka Markowa zmiennej $X$ to najmniejszy zbiór węzłów, taki że pod warunkiem otoczki zmienna $X$ jest niezależna od reszty grafu. W sieci bayesowskiej składa się z trzech grup: **rodziców** $X$, **dzieci** $X$ i **innych rodziców dzieci** $X$ (potocznie nazywanych małżonkami). Praktyczna intuicja: jeśli mamy wartości otoczki, dane spoza otoczki nie wnoszą już nic nowego do predykcji $X$.

    Z tej własności wynika prosty wniosek dla klasyfikacji: model używający tylko zmiennych z otoczki Markowa zmiennej celu powinien dać porównywalną dokładność jak model używający wszystkich predyktorów. To stanowi przedmiot Zadania 2 na końcu laboratorium.
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

    Podstawowe pytanie do wytrenowanej sieci: mając kilka zmiennych w roli dowodów (`evidence`), jaki jest rozkład warunkowy interesującej nas zmiennej? Algorytm **eliminacji zmiennych** liczy go dokładnie, sumując kolejno zmienne nienależące do zapytania ani dowodów. Kolejność eliminacji wpływa na koszt: niewłaściwy wybór może prowadzić do czynników o ogromnym rozmiarze (znalezienie optymalnej kolejności jest NP-trudne, ale heurystyki radzą sobie dobrze dla rzadkich grafów).
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
    ### Ćwiczenie 1

    **a)** Wyświetl CPD dla wybranego węzła z większą liczbą rodziców niż `income` (np. dla `marital`, jeśli ma wielu rodziców w nauczonej strukturze; jeśli nie — wybierz inny). Ile parametrów liczy ta tablica? Porównaj z liczbą wpisów w pełnej tablicy łącznej dla tych samych zmiennych — ile rzędów wielkości oszczędności daje faktoryzacja?

    **b)** Sprawdź własność otoczki Markowa empirycznie: dla zapytania `P(income | evidence)` porównaj wynik dla `evidence` zawierającego wyłącznie zmienne z otoczki Markowa `income` względem `evidence` z dodatkowymi zmiennymi spoza otoczki. Czy odpowiedzi są identyczne (tylko gdy faktycznie obserwujesz wszystkie zmienne otoczki)?
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

    gdzie $\mathcal{C}$ to zbiór klik grafu nieskierowanego (najczęściej maksymalnych), $\phi_C$ to nieujemne funkcje potencjału na konfiguracjach zmiennych w klice $C$ (niekoniecznie prawdopodobieństwa!), a $Z$ to stała normalizacyjna — suma iloczynów po wszystkich konfiguracjach, w fizyce statystycznej zwana funkcją podziału. Szczegóły: Koller & Friedman, rozdz. 4.

    ## Naiwne podejście — graf z progowania korelacji

    Zanim przejdziemy do właściwego MRF, zwizualizujmy zależności między zmiennymi w najprostszy możliwy sposób: zakodujmy każdą zmienną liczbowo, policzmy macierz korelacji par i dodajmy krawędź, gdy $|\rho|>t$.
    """)
    return


@app.cell
def _(adult, np, nx, plt):
    adult_num = adult.copy()
    for col in adult_num.columns:
        adult_num[col] = adult_num[col].astype('category').cat.codes

    cor = adult_num.corr()
    threshold = 0.1
    adj = (cor.abs() > threshold).values.copy()
    np.fill_diagonal(adj, False)

    G_thr = nx.from_numpy_array(adj)
    G_thr = nx.relabel_nodes(G_thr, dict(enumerate(cor.columns)))

    fig_thr, ax_thr = plt.subplots(figsize=(8, 6))
    nx.draw(G_thr, nx.spring_layout(G_thr, seed=42, k=1.5),
            with_labels=True, node_color='#a6c8fc', node_size=2200,
            font_size=11, edge_color='gray', ax=ax_thr)
    ax_thr.set_title(f"Graf z progowania |korelacji| > {threshold}")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten wykres ma walor poglądowy, ale **nie** jest MRF. Brakuje mu dwóch elementów: po pierwsze, korelacja Pearsona jest miarą liniową, więc traci sens dla zmiennych kategorycznych zakodowanych arbitralnymi etykietami liczbowymi. Po drugie, graf nie zawiera żadnych potencjałów na klikach — definiuje wyłącznie relację sąsiedztwa, a nie rozkład. Właściwa konstrukcja MRF wygląda inaczej.

    ## Moralizacja sieci bayesowskiej

    Do każdej sieci bayesowskiej można skonstruować równoważny MRF poprzez **moralizację**: usuń kierunki krawędzi i połącz parami wszystkich rodziców każdego węzła (stąd potoczna nazwa: "ożeń rodziców"). Operacja gwarantuje, że zbiór niezależności zachowywanych przez MRF jest podzbiorem niezależności sieci bayesowskiej — możemy stracić informację o niezależnościach kierunkowych (w szczególności o v-strukturach), ale rozkład łączny pozostaje ten sam.
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
    ## Separacja vs d-separacja

    Niezależności warunkowe odczytuje się z grafu inaczej w MRF i w BN. W MRF działa zwykła **separacja**: $A\perp B\mid C$ wtedy i tylko wtedy, gdy każda ścieżka między dowolnym węzłem z $A$ a dowolnym węzłem z $B$ przechodzi przez $C$. W BN trzeba używać **d-separacji**, w której v-struktury (typu $A\to C\leftarrow B$) zachowują się odwrotnie: blokują ścieżkę gdy $C$ *nie* jest obserwowane, a otwierają gdy jest.

    Praktyczna konsekwencja moralizacji: dodanie krawędzi między rodzicami zmiennej zaciera subtelność v-struktury. Po moralizacji nie da się już odczytać z grafu, że dwóch rodziców dziecka byłoby brzegowo niezależnych — graf sugeruje, że pozostają w bezpośredniej relacji. To strata informacji, którą trzeba świadomie zaakceptować przy korzystaniu z reprezentacji nieskierowanej.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ćwiczenie 2

    **a)** Znajdź w nauczonej sieci bayesowskiej węzeł z co najmniej dwoma rodzicami i wskaż w jego strukturze v-strukturę (parę rodziców i ich wspólne dziecko). Po moralizacji rodzice tego węzła powinni być połączeni krawędzią — zweryfikuj to wizualnie, porównując graf skierowany i graf MRF.

    **b)** Wykonaj test chi-kwadrat brzegowej niezależności tych dwóch rodziców (`scipy.stats.chi2_contingency` na tabeli kontyngencji). Czy są brzegowo niezależne, jak sugerowałaby v-struktura przed obserwacją dziecka? Następnie warunkuj zbiór po wartościach dziecka i ponownie wykonaj test — czy wyniki się różnią? Dlaczego MRF zmoralizowany "ukrywa" tę różnicę?
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

    Graf czynników to **bipartytowa** reprezentacja faktoryzacji rozkładu: dwa rodzaje węzłów — okrągłe **węzły zmiennych** i kwadratowe **węzły czynników** — krawędzie zaś łączą czynnik z każdą zmienną, od której zależy. Graf czynników nie zawiera niczego nowego względem BN czy MRF; ujawnia natomiast strukturę faktoryzacji w sposób jednoznaczny — z samego BN czy MRF nie zawsze widać, na ilu czynnikach o jakim "kształcie" rozkłada się rozkład łączny.

    Powód praktyczny: algorytmy **przekazywania komunikatów** (sum-product, belief propagation) operują naturalnie na grafach czynników. Każdy węzeł — czy zmienna, czy czynnik — wysyła do sąsiadów lokalnie obliczone komunikaty, a po kilku iteracjach (dla drzew: jednym przejściu w obie strony) z komunikatów odczytuje się rozkłady brzegowe wszystkich zmiennych jednocześnie. W tym laboratorium ograniczamy się do uruchomienia gotowej implementacji z pgmpy, a po szczegóły algorytmu odsyłamy do Koller & Friedman, rozdz. 11.

    ## Konwersja BN → graf czynników

    Każde CPD $P(X_i \mid \mathrm{Pa}(X_i))$ staje się czynnikiem $\phi_i$ obejmującym $X_i$ i jego rodziców. Czynników jest tyle, ile węzłów w BN.
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

    Eliminacja zmiennych daje rozkład brzegowy jednej zmiennej naraz; każde nowe zapytanie wymaga osobnego przebiegu. **Belief Propagation** natomiast w jednym przejściu wyznacza brzegi wszystkich zmiennych jednocześnie, kosztem skompilowania **drzewa złączeń** (junction tree) z grafu czynników. Dla pojedynczego zapytania VE bywa szybsze; przy wielu zapytaniach na tym samym modelu BP zyskuje przewagę dzięki amortyzacji kosztu kalibracji.

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

    **a)** Sprawdź eksperymentalnie, kiedy BP zaczyna się opłacać. Zmierz `time.perf_counter()` dla pojedynczego zapytania (osobny obiekt VE i BP, jedno wywołanie `.query`) oraz dla 50 różnych zapytań (50 wywołań `.query` na tym samym obiekcie, ale z innymi dowodami). Pamiętaj, że `BeliefPropagation` wymaga `calibrate()` raz po utworzeniu. Czy widzisz spodziewaną amortyzację BP?

    **b)** Zaproponuj zapytanie, w którym zmiennymi pytanymi (`variables`) jest kilka węzłów jednocześnie (np. `['income', 'occupation']`). Czy obie metody nadal dają zgodne wyniki? Co się dzieje, gdy zmienne pytane nie są od siebie d-separowane?
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
    # Zadania

    ## Zadanie 1 — domenowa sieć bayesowska vs nauczona

    Zaprojektuj ręcznie sieć bayesowską dla zbioru Adult opartą wyłącznie na wiedzy domenowej (np. `age -> marital`, `education -> occupation`, `relationship -> marital`, `occupation -> income`, ...). Następnie:

    **a)** Dopasuj parametry (`bn.fit(adult)`) i policz log-likelihood na zbiorze (`bn.log_probability(adult)` lub iteracja po wierszach z mnożeniem CPD) dla tej struktury i dla struktury wyuczonej z HC + BIC. Która sieć ma wyższy log-likelihood i o ile? Czy to oznacza, że jest lepsza?

    **b)** Dla każdej z dwóch sieci wykonaj klasyfikację `income` jako MAP zmiennej zapytania przy zadanych pozostałych zmiennych jako dowodach (`inf.map_query(['income'], evidence=...)` na zbiorze testowym, np. losowym podziale 80/20). Porównaj accuracy. Zwróć uwagę, że BIC/log-likelihood porównuje jakość modelowania *rozkładu łącznego*, a nie jakość predykcji jednej zmiennej — wyniki nie muszą się pokrywać.
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

    Wykorzystując nauczoną sieć bayesowską:

    **a)** Wyznacz otoczkę Markowa zmiennej `income`. Wytrenuj prosty klasyfikator (np. `RandomForestClassifier` ze sklearn) na trzech wariantach predyktorów: (i) wszystkie zmienne, (ii) tylko otoczka Markowa, (iii) wszystkie zmienne *spoza* otoczki Markowa. Porównaj dokładność testową. Pamiętaj o jednolitym kodowaniu kategorycznym (`pd.get_dummies`).

    **b)** Sprawdź, czy ranking ważności cech z lasu losowego (`feature_importances_`) pokrywa się z otoczką Markowa. Jeśli nie — które zmienne las losowy uznaje za ważne, a sieć bayesowska klasyfikuje jako "poza otoczką"? Zinterpretuj.
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

    Skonstruuj graf czynników bezpośrednio z **MRF z moralizacji** (`mn.to_factor_graph()`) oraz z **BN po dodatkowym kroku** (`bn_model.to_markov_model().to_factor_graph()`).

    **a)** Porównaj liczbę czynników w obu grafach. Wyjaśnij wynik odwołując się do tego, że w pgmpy moralizacja BN dodaje po jednym czynniku za każde CPD (zamiast łączyć je w potencjały klikowe).

    **b)** Uruchom `BeliefPropagation` zainicjalizowane na `bn_model` i porównaj wyniki z `BeliefPropagation` zainicjalizowanym na `mn` (MRF). Czy zapytania `bp.query(['income'], evidence=...)` dają identyczny wynik? Co to mówi o tym, że MRF "pamięta" dokładnie ten sam rozkład łączny co BN, choć utracił informację o kierunkowych niezależnościach?
    """)
    return


@app.cell
def _():
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
