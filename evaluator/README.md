# Braint: Evaluator package

This package evaluates the performance of the classifier The results of each instance of `Scorer` are neatly printed on one line, so that each time the classifier is evaluated you can easily see how its performance improves (hopefully).

## 1. Metrics

Evaluator includes Macro and Micro F-scores, followed by Precision and Recall for each of the six emotions. All values are tab-seperated.

Example of output with two identical instance of `Scorer`:

```java
Fmac	Fmic	supP	supR	disP	disR	feaP	feaR	sadP	sadR	joyP	joyR	angP	angR
0.048   0.167	0.0	0.0 	0.0  	0.0	 0.0	 0.0	 0.0	 0.0    0.17	 1.0	0.0 	0.0
0.048   0.167	0.0	0.0 	0.0  	0.0	 0.0	 0.0	 0.0	 0.0	0.17	 1.0	0.0 	0.0
```

## 2. Usage & Requirements
+ Requirements: Python3
+ Usage: Execute `main.py` (from within the directory `evaluator`)
+ The files prediction and gold are assumed to be located in the directory `Braint/data/` (otherwise `main.py` must be changed accordingly).
  + prediction file (in our case `trial.csv`) contains a list of tab-separated predicted labels (emotion) **and** tweets.
  + gold file (`trial.labels`) contains a list of labels (**only**).


## 3. Data structures

The package includes these 4 classes:
- `Tweet` - Stores tweet text, predicted and gold labels (used by Corpus)
  - Arguments(3 strings): tweet text, gold label, predicted label
- `Corpus` - An iterator of Tweet instances
  - Arguments (2 strings):
      1. filepath of predicted labels and tweets (tab-separated)
      2. feilpath of gold labels
- `Scorer` - Calculates scores for an instance of Corpus
  - Argument (1): `Corpus` object
- `Result` - Shows scores in compact columns for any instances of Scorer
  - Argument (0): None
  - Use the method `.show()` to print a Scorer object
