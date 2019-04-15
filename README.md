# Calibrating-Rough-Volatility-Models-with-Deep-Learning

This is a course project of the course « Machine Learning for Finance » at ENSAE ParisTech.

## Structure

- The notebook `SPX-Parser.ipynb` demonstrates how to parse the raw data retrieved from https://www.cboe.com, to get the joint distribution of moneyness $M$ and time to maturity $T$, and draw randomly 1000000 pairs from the estimated distribution.

- The notebook `Data-Generator.ipynb` demonstrates how to generate labeled dataset of Heston Model and rBergomi Model for training the IV prediction Neural Network.

- The notebook `Deep-Calibration.ipynb` demonstrates how to preprocess synthetic data, build and train Neural Networks, and use them to predict IV.

- The notebook `CNN-calibration.ipynb` demonstrates the whole pipeline of the second paper.

## Citation

This project aims to reimplement the methods and reproduce the results in the following two articles:

<pre><code>
@article{Deep-Calibration,
title = {Deep calibration of rough stochastic volatility models},
author = {Bayer, Christian and Stemper, Benjamin},
year = {2018},
month = {10}
}

@article{CNN-Calibration,
title = {Calibrating Rough Volatility Models: A Convolutional Neural Network Approach},
author = {Stone, Henry},
year = {2019},
month = {01},
journal = {SSRN Electronic Journal},
doi = {10.2139/ssrn.3327135}
}
</code></pre>
