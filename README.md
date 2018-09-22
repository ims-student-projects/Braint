# _BrainT_ : two models for Implicit Emotion Detection

# Introduction

This repository contains two models that we built in the context of [WASSA-2018 Implicit Emotion Shared Task](http://implicitemotions.wassa2018.com/) and the course Teamlab at the University of Stuttgart.

Here we only give a general overview of the task, the data set and the results. For the documentation of each models, please navigate to the corresponding subfolder:

* [mcPerceptron](mcPerceptron) (averaged multiclass perceptron)
* [deepLearning](deepLearning) (bi-directional LSTM with attention)

Our two models showed comparable results: F-Macro 0.63 and 0.64 for the multi-class Perceptron and the Deep Learning model respectively. We submitted the predictions of mcPerceptron model to IESA since at the time it performed (slightly) better than the DL model. Our [paper](http://185.203.116.239/publ/braint/braint_at_iest_2018.pdf) describing this model was accepted for the EMNLP 2018 conference.


# Task Description

Our task was to predict emotions in a large dataset of tweets annotated with distant supervision. The predicted emotion should have been `anger​`, `disgust`, `fear`, `joy`, `sadness` or `surprise​`. This was an _implicit_ emotion detection task, since the words actually expressing these six emotions (or synonyms) were masked in the train data.

An example of original tweet:

<div class="center">
<blockquote class="twitter-tweet" data-partner="tweetdeck"><p lang="en" dir="ltr">
I spent 24 hours with my boyfriend yet I was still sad when he dropped me off
</p>
&mdash; вяeadney|-/ (@katzlover64) <a href="https://twitter.com/katzlover64/status/894446290448285696">8:33 AM - 7 Aug 2017</a></blockquote>
</div>

The tweet in the dataset:

```
I spent 24 hours with my boyfriend yet I was still [#TRIGGERWORD#] when he dropped me off
```


## Datasets

The train and test datasets were provided by the [shared task](http://implicitemotions.wassa2018.com/data/ ) and contain

- `train-v3.csv` : 153,383
- `test-text-labels.csv` : 28,757

tweets. Both our models expect that these datasets  are located in the subfolder `data`.

## Class distribution in train data
The six emotions are more or less evenly distributed in the train data.


| surprise | anger	| disgust	| sad	| fear	| joy	| total |
|----------|--------|---------|-----|-------|-----|-------|
| 16.7% (25,565)	| 16.7% (25,562)	| 16.7% (25,558)	| 15.1% (23,165)	| 16.7% (25,575)	| 18.2% (27,958)	| 100.0% (153,383) |


# References

* Vachagan Gratian, Marina Haid. 2018 [BrainT at IEST 2018: Fine-tuning Multiclass Perceptron For Implicit
Emotion Classification](http://185.203.116.239/publ/braint/braint_at_iest_2018.pdf) (Our paper submitted to EMNLP 2018, describes mcPerceptron)
* [Our report for the Teamlab course describing both models](http://185.203.116.239/publ/braint/braint_final_report.pdf)
* Roman Klinger, Orphée de Clerq, Saif M Mohammad,
and Alexandra Balahur. 2018. [IEST: WASSA-2018 Implicit Emotions Shared Task](http://implicitemotions.wassa2018.com/paper/iest-description-2018.pdf)
