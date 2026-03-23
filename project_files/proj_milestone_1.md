# Milestone 1 — Eksploracja danych i pierwsze rozwiązanie (20 pkt)

## Kontekst

Otrzymujesz dane z platformy OpenDota zawierające mecze gry **Dota 2** rozegrane na różnych poziomach rankingowych. Dota 2 posiada osiem rang (tierów): Herald, Guardian, Crusader, Archon, Legend, Ancient, Divine, Immortal — gdzie Herald to najniższy poziom, a Immortal najwyższy.

**Zadanie:** Na podstawie statystyk meczu przewidź, w której randze został rozegrany.

---

## Dane

Otrzymujesz dwa pliki:

- `competition_train.json` — zbiór treningowy (~2 900 meczów) z polem `tier` (etykieta do przewidywania)
- `competition_test.json` — zbiór testowy (800 meczów) **bez** pola `tier`

Każdy mecz to zagnieżdżony obiekt JSON zawierający m.in.:
- statystyki na poziomie meczu: `duration`, `first_blood_time`, `radiant_score`, `dire_score`, …
- tablicę `players` z 10 obiektami (po 5 graczy z każdej drużyny), z polami takimi jak `gold_per_min`, `xp_per_min`, `kills`, `deaths`, `assists`, `last_hits`, `hero_damage`, `net_worth`, `level`, `kda`, …
- draft meczu (`picks_bans`)

Przetworzenie surowego JSONa na tabelaryczną postać jest **integralną częścią zadania**.

---

## Zadanie

### 1. Przesłanie notebooka z eksplorację i modelem na Upel

Notebook powinien zawierać kolejno:

1. **Wczytanie i spłaszczenie danych** — przekształcenie zagnieżdżonego JSONa do postaci, w której jeden wiersz = jeden mecz
2. **Ekstrakcja cech** — wybierz i uzasadnij cechy, które Twoim zdaniem najlepiej odróżniają rangi; może to być np. agregacja statystyk graczy (średnie, sumy), cechy meczu jako całości lub cokolwiek innego, co wydaje Ci się sensowne
3. **Eksploracja i wizualizacja** — zbadaj rozkład wybranych cech w podziale na tiery; skomentuj co widzisz
4. **Model** — dopasuj co najmniej jeden model klasyfikacji (regresja logistyczna, GLM, drzewo decyzyjne lub ich kombinacja); oceń jakość na zbiorze walidacyjnym
5. **Predykcja i submission** — wygeneruj plik CSV z predykcjami na zbiorze testowym

i być przesłany w wyeksportowanym formacie html.

### 2. Przesłanie rozwiązania na Kaggle

Sposób formatowania opisany na stronie konkursu.

---

## Punktacja

| Element | Pkt |
|---------|-----|
| Wczytanie i spłaszczenie danych | 2 |
| Ekstrakcja sensownych cech (min. 7) | 5 |
| Eksploracja i wizualizacje z komentarzem | 4 |
| Poprawny pipeline ML (podział danych, model, predykcja) | 4 |
| Wynik na leaderboardzie Kaggle | 3 |
| Czytelność notebooka (opisy, reprodukowalność) | 2 |
| **Suma** | **20** |

### Progi za wynik na leaderboardzie

| Accuracy | Pkt |
|----------|-----|
| **≥ 24%** | **3** |
| 20–23% | 2 |
| < 20% | 1 |
| brak submission | 0 |

> Accuracy mierzona jest jako odsetek poprawnie sklasyfikowanych meczów spośród 800 w zbiorze testowym.