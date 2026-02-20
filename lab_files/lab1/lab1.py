import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


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
    from scipy import stats, optimize
    return np, optimize, pd, stats


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
    # Rozwiązanie zadania 1
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
def _(pd):
    goals_df = pd.read_csv("goals.csv")
    print(goals_df.describe())
    goals_df
    return (goals_df,)


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

    Wyznacz przedziały ufności na poziomie 0.95 i 0.99 dla średniej wysokości
    drzew ze zbioru `trees`.

    Zbiór `trees` jest dostępny przez bibliotekę `statsmodels`:
    """)
    return


@app.cell
def _(pd):
    from statsmodels.datasets import get_rdataset

    trees = get_rdataset("trees").data
    print(trees.describe())
    trees
    return get_rdataset, trees


@app.cell
def _():
    # Przedziały ufności dla średniej wysokości
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 4

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
    ## Zadanie 5

    Automat produkuje blaszki o nominalnej grubości 0.04 mm. Wyniki pomiarów
    grubości losowej próby 25 blaszek zebrane są w pliku `blaszki.csv`. Czy można
    twierdzić, że blaszki są cieńsze niż 0.04 mm? Przyjmujemy rozkład normalny
    grubości blaszek oraz poziom istotności α = 0.01.
    """)
    return


@app.cell
def _(pd):
    blaszki_df = pd.read_csv("blaszki.csv")
    blaszki_df
    return (blaszki_df,)


@app.cell
def _():
    # Test hipotezy jednostronnej (blaszki cieńsze niż 0.04mm)
    # Uzupełnij kod poniżej
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zadanie 6

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
