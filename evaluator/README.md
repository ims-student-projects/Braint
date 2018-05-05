# Braint: Evaluator package

This package meant to evaluate the performance of the classifier during training and testing modes. The results are printed in compact one-liner, so that each time the classifier is evaluated you can easily see how its performance improves (hopefully it does).

## 1. Metrics

The evaluator includes Macro (`Fmac`) and Micro (`Fmic`) F-scores, followed by Precision (\*`P`) and Recall (\*`R`) for each of the six emotions. All values are tab-seperated.

Example of output with two identical instances of `Scorer`:

```java
Fmac	Fmic	supP	supR	disP	disR	feaP	feaR	sadP	sadR	joyP	joyR	angP	angR
0.048   0.167	0.0	0.0 	0.0  	0.0	 0.0	 0.0	 0.0	 0.0    0.17	 1.0	0.0 	0.0
0.048   0.167	0.0	0.0 	0.0  	0.0	 0.0	 0.0	 0.0	 0.0	0.17	 1.0	0.0 	0.0
```

(This is a case when a dummy prediction was used. All predictions were "joy".)

## 2. Usage & Requirements
+ Run with Python3
+ Can be executed with `main.py` (from within the directory `evaluator`), which is meant as an example of usage.
+ The prediction and gold files are assumed to be located in the directory `Braint/data/` (otherwise `main.py` must be changed accordingly).
  + prediction file (in our case `trial.csv`) contains a list of tab-separated predicted labels (emotion) **and** tweets.
  + gold file (`trial.labels`) contains a list of labels (**only**).


## 3. Data structures

The package includes these 4 classes:
- `Tweet` - Stores tweet text, predicted and gold labels (used by Corpus)
  - Arguments (3 strings): tweet text, gold label, predicted label
- `Corpus` - An iterator of Tweet instances
  - Arguments (2 strings):
      1. filepath of predicted labels and tweets (tab-separated)
      2. filepath of gold labels
- `Scorer` - Calculates scores for an instance of Corpus
  - Argument (1): `Corpus` object
- `Result` - Shows scores in compact columns for any instances of `Scorer`
  - Argument (0)
  - Use the method `.show()` to print a Scorer object
