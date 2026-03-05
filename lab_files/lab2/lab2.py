import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 2: Naiwny klasyfikator bayesowski

    Klasyczny klasyfikator oparty o własności prawdopodobieństwa warunkowego,
    twierdzenie Bayesa i niezależność zmiennych. Przystępny opis wprowadzający
    można znaleźć w:
    https://www.alextsun.com/files/Prob_Stat_for_CS_Book.pdf
    (Chapter 9.3: The Naive Bayes Classifier)

    Pracujemy na wycinku korpusu maili z firmy Enron zebranych w trakcie śledztwa
    po upadku firmy (https://en.wikipedia.org/wiki/Enron_Corpus). Dane podzielone
    są na zbiór treningowy i testowy, z kolumnami `email` (treść) i `label` (spam/ham).
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    train_df = pd.read_csv("train_emails.csv")
    test_df = pd.read_csv("test_emails.csv")
    train_df.describe()
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementacja klasyfikatora

    W ramach laboratorium należy wypełnić brakujący kod w pliku `naive_bayes.py`
    i następnie wytrenować klasyfikator na zbiorze treningowym maili oraz
    przewidzieć odpowiedzi na zbiorze testowym.

    Oczekiwana dokładność powinna wynieść około **0.98**.
    """)
    return


@app.cell
def _():
    from naive_bayes import NaiveBayes
    return (NaiveBayes,)


@app.cell
def _(NaiveBayes, test_df, train_df):
    # Trenowanie klasyfikatora
    clf = NaiveBayes()
    clf.fit(train_df['email'], train_df['label'])

    # Predykcja na zbiorze testowym
    predictions = clf.predict(test_df['email'])

    accuracy = (predictions == test_df['label']).mean()
    print(f"Dokładność: {accuracy:.4f}")
    return accuracy, clf, predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ulepszenia klasyfikatora

    Po ukończeniu podstawowej implementacji można usprawnić klasyfikator przez:

    a) **Lepszą tokenizację** — niektóre słowa (stopwords) można pominąć
    b) **Wygładzenie Laplace'a** — odpowiednie traktowanie słów niewidzianych
       w zbiorze treningowym
    c) **Multinomial Naïve Bayes** — uwzględnienie liczby wystąpień wyrazów
       zamiast tylko ich obecności

    Jako że obecny klasyfikator radzi sobie bardzo dobrze, można utrudnić zadanie
    poprzez podział zbioru w proporcji **50:50** zamiast standardowego podziału.
    """)
    return


if __name__ == "__main__":
    app.run()
