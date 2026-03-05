import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 1: Zadania powtórkowe ze statystyki
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from scipy import stats
    from pathlib import Path

    DATA_DIR = Path(__file__).parents[2] / "data"
    return DATA_DIR, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rozkłady prawdopodobieństwa w Python (scipy.stats)

    Na zajęciach ze statystyki będą nam potrzebne rozkłady prawdopodobieństwa. W Pythonie, moduł `scipy.stats` dostarcza obiekty reprezentujące rozkłady prawdopodobieństwa. Każdy rozkład ma następujące metody:

    - `.pdf(x)` lub `.pmf(x)` - gęstość (dla rozkładów ciągłych) lub funkcja prawdopodobieństwa (dla dyskretnych)
    - `.cdf(x)` - dystrybuanta (kumulatywna funkcja rozkładu)
    - `.ppf(q)` - funkcja kwantylowa (odwrotność dystrybuanty, percent point function)
    - `.rvs(size)` - losowanie zgodne z rozkładem (random variates)

    Najczęściej używane rozkłady:
    - `stats.norm` - rozkład normalny (Gaussa)
    - `stats.uniform` - rozkład jednostajny
    - `stats.binom` - rozkład dwumianowy
    - `stats.poisson` - rozkład Poissona
    - `stats.expon` - rozkład wykładniczy
    - `stats.gamma` - rozkład gamma
    - `stats.t` - rozkład t-Studenta
    - `stats.chi2` - rozkład chi-kwadrat
    - `stats.f` - rozkład F (Fishera-Snedecora)
    - `stats.geom` - rozkład geometryczny
    """)
    return


@app.cell
def _(np, stats):
    # Rozkład normalny N(0, 1)
    print(stats.norm.pdf(2.3))        # gęstość w punkcie 2.3
    print(stats.norm.cdf(2.3))        # P(X <= 2.3)
    print(stats.norm.ppf(0.975))      # kwantyl 97.5%

    # Losowanie z rozkładu normalnego
    x1 = stats.norm.rvs(size=10)      # 10 losowań z N(0, 1)
    print(f"Średnia: {np.mean(x1):.4f}")
    print(f"Wariancja: {np.var(x1, ddof=1):.4f}")  # ddof=1 dla wariancji próbkowej
    print(f"Odch. std.: {np.std(x1, ddof=1):.4f}")
    return


@app.cell
def _(np, stats):
    # Rozkład normalny z innymi parametrami N(1, 25)
    x2 = stats.norm.rvs(loc=1, scale=5, size=10)  # loc=średnia, scale=odch.std.
    print(f"Średnia: {np.mean(x2):.4f}")
    print(f"Wariancja: {np.var(x2, ddof=1):.4f}")
    print(f"Odch. std.: {np.std(x2, ddof=1):.4f}")
    return


@app.cell
def _(stats):
    # Rozkład Poissona
    print(stats.poisson.pmf(2, mu=1))        # P(X = 2) dla Poisson(1)
    print(stats.poisson.cdf(2, mu=1))        # P(X <= 2)
    print(stats.poisson.ppf(0.75, mu=1))     # kwantyl 75%
    print(stats.poisson.rvs(mu=1, size=10))  # 10 losowań
    return


@app.cell
def _(stats):
    # Rozkład dwumianowy Binom(n=10, p=0.3)
    print(stats.binom.pmf(3, n=10, p=0.3))       # P(X = 3)
    print(stats.binom.rvs(n=10, p=0.3, size=5))  # 5 losowań
    return


@app.cell
def _(stats):
    # Rozkład jednostajny U(0, 1)
    print(stats.uniform.rvs(size=5))                  # 5 losowań z U(0, 1)
    print(stats.uniform.rvs(loc=2, scale=3, size=5))  # 5 losowań z U(2, 5)
                                                       # scale = max - min
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dla osiągnięcia powtarzalności obliczeń z czynnikiem losowym stosuje się `np.random.seed()` lub `random_state`:
    """)
    return


@app.cell
def _(np, stats):
    # Bez ustawienia ziarna - różne wyniki
    print("Losowanie 1:", stats.norm.rvs(size=5))
    print("Losowanie 2:", stats.norm.rvs(size=5))

    # Z ustawionym ziarnem - powtarzalne wyniki
    np.random.seed(2020)
    print("Z ziarnem 1:", stats.norm.rvs(size=5))

    np.random.seed(2020)
    print("Z ziarnem 2:", stats.norm.rvs(size=5))  # Te same wartości co powyżej
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatywnie, można użyć `random_state` dla większej kontroli:
    """)
    return


@app.cell
def _(np, stats):
    # Tworzenie generatora z konkretnym ziarnem
    rng = np.random.default_rng(2020)
    print("Generator 1:", stats.norm.rvs(size=5, random_state=rng))

    # Ten sam generator daje kolejne wartości
    print("Generator 2:", stats.norm.rvs(size=5, random_state=rng))

    # Nowy generator z tym samym ziarnem
    rng2 = np.random.default_rng(2020)
    print("Nowy generator:", stats.norm.rvs(size=5, random_state=rng2))  # Te same co Generator 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 1

    Czas oczekiwania na pewne zdarzenie ma rozkład Gamma(3, r). Wykonano serię
    pomiarów i uzyskano czasy 1.4, 1.8, 1.4, 1.4 i 1.5. Oblicz estymatę
    największej wiarygodności parametru r.
    """)
    return


@app.cell
def _():
    # Rozwiązanie zadania 2
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 2

    Plik `goals.csv` zawiera dane o liczbie goli strzelonych przez pewną drużynę
    piłkarską w kolejnych meczach. Zakładamy, że liczba goli ma rozkład Poissona
    o nieznanej wartości λ. Wyznacz estymator największej wiarygodności parametru λ.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    goals_df = pd.read_csv(DATA_DIR / "goals.csv")
    print(goals_df.describe())
    goals_df
    return


@app.cell
def _():
    # Estymacja MLE dla rozkładu Poissona
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 3
    Ladislaus Josephovich Bortkiewicz (lub Władysław Bortkiewicz) w 1898 roku wykorzystał dane z Preussische Statistik o śmierciach w wyniku kopnięcia przez konia w 14 dywizjach wojska pruskiego do pokazania, że rozkład Poissona dobrze opisuje rzadkie, losowe zdarzenia w dużych populacjach. Chociaż zarzucono mu później celowe odrzucenie kilku dywizji dla lepszego dopasowania, otrzymany zbiór danych stał się klasycznym przykładem zastosowania tego rozkładu w statystyce i jednym z pierwszych empirycznych dowodów na jego przydatność w modelowaniu rzeczywistych danych.

    Zakładamy, że liczba ofiar w korpusach w danym roku ma rozkład Poissona o nieznanym parametrze λ. Korzystając z poniższych danych, wyznacz estymator największej wiarygodności parametru λ oraz porównaj teoretyczny rozkład Poissona z empirycznym rozkładem danych.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    kicks_df = pd.read_csv(DATA_DIR / "kicks.csv")
    kicks_df
    return


@app.cell
def _():
    # Estymacja MLE dla rozkładów Poissona
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 4

    Wyznacz przedziały ufności na poziomie 0.95 i 0.99 dla średniej wysokości
    drzew ze zbioru `trees`.

    Zbiór `trees` jest dostępny przez bibliotekę `statsmodels`:
    """)
    return


@app.cell
def _():
    from statsmodels.datasets import get_rdataset

    trees = get_rdataset("trees").data
    print(trees.describe())
    trees
    return


@app.cell
def _():
    # Przedziały ufności dla średniej wysokości
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 5

    Ustal minimalną liczebność próby dla oszacowania średniej wzrostu noworodków
    o rozkładzie N(μ, 1.5 cm). Zakładamy maksymalny błąd szacunku d = 0.5 cm
    oraz poziom ufności 0.99.
    """)
    return


@app.cell
def _():
    # Minimalna liczebność próby
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 6

    Automat produkuje blaszki o nominalnej grubości 0.04 mm. Wyniki pomiarów
    grubości losowej próby 25 blaszek zebrane są w pliku `blaszki.csv`. Czy można
    twierdzić, że blaszki są cieńsze niż 0.04 mm? Przyjmujemy rozkład normalny
    grubości blaszek oraz poziom istotności α = 0.01.
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    blaszki_df = pd.read_csv(DATA_DIR / "blaszki.csv")
    blaszki_df
    return


@app.cell
def _():
    # Test hipotezy jednostronnej (blaszki cieńsze niż 0.04mm)
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 7

    Spośród 97 chorych na pewną chorobę, losowo wybranym 51 pacjentom podano lek.
    Pozostałym 46 podano placebo. Po tygodniu 12 pacjentów, którym podano lek,
    oraz 5 spośród tych, którym podano placebo, poczuło się lepiej. Zweryfikuj
    hipotezę o braku wpływu podanego leku na samopoczucie pacjentów.
    """)
    return


@app.cell
def _():
    # Test niezależności / test chi-kwadrat dla tabeli kontyngencji
    # Uzupełnij kod poniżej
    ...
    return


if __name__ == "__main__":
    app.run()
