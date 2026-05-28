# Milestone 3 - Wyjaśnialność i wnioski o zbiorze (30 pkt)

## Kontekst

W Milestone 1 zbudowałeś pierwsze rozwiązanie oparte na modelach liniowych / drzewiastych i uzyskałeś względnie skromne wyniki. W Milestone 2 dodałeś modele boostingowe, rozszerzoną inżynierię cech i walidację krzyżową - a po drodze pojawiło się obserwacja, że dane były **zbierane partiami z różnym priorytetem dla rang w czasie**. To znacząco poprawiło wyniki numeryczne, ale w dyskusji uznaliśmy, że niekoniecznie czyni to model lepszym w sensie tego, co miał przewidywać (przebieg rozgrywki → ranga).

Czas to zweryfikować narzędziami wyjaśnialności.

**Zadanie:** Zastosuj metody wyjaśnialności do modeli z M1 oraz M2 i odpowiedz na pytanie: **który z modeli, w świetle swoich wyjaśnień, podejmuje decyzje w sposób bardziej sensowny merytorycznie?** Nie zakładamy z góry odpowiedzi - samodzielnie wyciągasz wnioski.

---

## Zadanie

Notebook powinien zawierać kolejno:

### 1. Odtworzenie modeli z M1 i M2

Wczytaj lub przetrenuj swoje modele z poprzednich kamieni milowych tak, aby były dostępne w jednym notebooku do analizy porównawczej. Dopuszczalne są **uproszczenia podyktowane wymaganiami metod XAI**, np.:

- ograniczenie do top-N najważniejszych cech (jeśli wybierzesz metodę słabo skalującą się z wymiarowością),
- dyskretyzacja zmiennych ciągłych (jeśli używasz sieci Bayesowskiej lub grafu Markova),
- użycie reprezentatywnej próbki danych zamiast całego zbioru treningowego.

Każde uproszczenie krótko uzasadnij i odnotuj jego potencjalny wpływ na wyjaśnienia.

### 2. SHAP - wyjaśnienia globalne i lokalne (obowiązkowo, dla obu modeli)

Dla **każdego z modeli (M1 i M2)** wykonaj:

- **wyjaśnienia globalne**: ranking ważności cech (np. `summary_plot`, `bar_plot`) - które cechy najsilniej wpływają na predykcje w skali całego zbioru,
- **wyjaśnienia lokalne**: co najmniej **3 przykłady pojedynczych meczów** (preferencyjnie z różnych rang) z rozbiciem wkładu poszczególnych cech (`force_plot`, `waterfall_plot` lub odpowiednik),
- **interakcje / zależności** (opcjonalnie): np. `dependence_plot` dla 1–2 najciekawszych cech.

Skomentuj wyniki: czy ranking ważnych cech w M1 i M2 się pokrywa? Czy któryś z modeli silnie polega na cechach, które *nie powinny* być predykcyjne dla rangi rozgrywki, gdyby dane były zbierane w sposób losowy?

### 3. Druga metoda wyjaśnialności (do wyboru, dla obu modeli)

Wybierz **co najmniej jedną** dodatkową metodę z listy:

- **LIME** - lokalne wyjaśnienia liniowe; dobre uzupełnienie dla wyjaśnień lokalnych SHAP,
- **Sieć Bayesowska** - nauczona ze zdyskretyzowanego zbioru (np. `pgmpy`); zinterpretuj strukturę grafu i warunkowe zależności wokół zmiennej `tier`,
- **Graf Markova / Markov Random Field** - modelowanie zależności między cechami; zinterpretuj sąsiedztwo `tier`,
- **Permutation importance / Partial Dependence / ICE** - jako uzupełnienie globalnych wyjaśnień SHAP.

Wybór uzasadnij (dlaczego ta metoda dla tego modelu?) i porównaj jej wyniki z wynikami SHAP - czy potwierdzają, czy uzupełniają, a może przeczą sobie?

### 4. Porównanie M1 vs M2 - która historia jest sensowniejsza?

To **kluczowa sekcja** tego kamienia milowego. Zestaw obok siebie wyjaśnienia dla obu modeli i odpowiedz (z dowodami z wykresów):

- Czy model M2 - mimo wyższego accuracy - opiera się na cechach, które dają się sensownie zinterpretować w kategoriach jakości gry (KDA, GPM, XPM, itd.)?
- Czy część jego ważnych cech wygląda raczej jak **proxy dla momentu / partii zbierania danych** (np. patch, wersja, daty, identyfikatory, cechy o nietypowym rozkładzie skorelowanym z rangą)?
- Jak na tym tle wypadają wyjaśnienia z M1? Czy są **bardziej zdroworozsądkowe**, mimo gorszego wyniku numerycznego?
- Co z tego wynika dla pytania, **który model jest „lepszy"**? Jak rozumiesz pojęcie lepszy w kontekście tego projektu po wykonaniu tej analizy?

### 5. Raport wniosków

Na końcu notebooka umieść **podsumowanie (0.5–1 strony tekstu)** zawierające:

- najważniejsze obserwacje z analizy XAI,
- ocenę który model dostarcza sensowniejszych wyjaśnień i dlaczego,
- refleksję, jak ta analiza zmienia (lub nie zmienia) Twoją ocenę z M2,
- ewentualne rekomendacje, co należałoby zmienić w zbiorze / pipeline, gdyby projekt miał być kontynuowany.

---

## Punktacja

| Element | Pkt |
|---------|-----|
| Odtworzenie modeli M1 i M2 z uzasadnieniem ewentualnych uproszczeń | 3 |
| SHAP - wyjaśnienia globalne i lokalne dla obu modeli, z komentarzem | 8 |
| Druga metoda XAI (LIME / BN / Markov / PD-ICE) dla obu modeli, z uzasadnieniem wyboru | 6 |
| Porównanie M1 vs M2 - dyskusja, która historia jest sensowniejsza, oparta o dowody z wyjaśnień | 6 |
| Raport końcowy z wnioskami i refleksją nad jakością modeli | 4 |
| Czytelność notebooka, opisy, reprodukowalność | 3 |
| **Suma** | **30** |

---

## Format oddania

Notebook wyeksportowany do **HTML** i przesłany na Upel. Brak nowego submission na Kaggle - focus tego etapu jest analityczny.

> Istotne: w tym etapie nie ma „dobrej" odpowiedzi w sensie metryki. Oceniana jest jakość rozumowania i dowodów, które przedstawiasz, a nie to, do jakich konkretnie wniosków dojdziesz.
