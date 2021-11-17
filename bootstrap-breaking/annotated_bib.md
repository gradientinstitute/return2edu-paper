# Casual Bootstrapping (with duplicates)

## General thoughts

* papers are largely specific to the specific inner methods used
* subsampling (sampling a subset without replacement) is promising (apparently requires fewer assumptions)
  * I glanced at [@politisSubsampling1999] but it looked like a rabbit hole I could get lost in, and has no specific causal information. I think
  * non-stationary time series chapter isn't as promising as it looks; requires *eventual stationarity* e.g. Markov chain that's warming up, which must surely be violated in our case
    * I might investigate subsampling in the causal/matching literature

## High level annotated bibliography

* Investigate Bootstrapping w.r.t matching [@imbensNonparametricEstimationAverage2004]
  * Imbens review paper:
    * Bootstrapping is often used, poorly justified
  >  "Researchers have suggested several ways for estimating the variance of these average-treatment-effect estimators. One, more cumbersome approach requires esti- mating each component of the variance nonparametrically. A more common method relies on bootstrapping. A third alternative, developed by Abadie and Imbens (2002) for the matching estimator, requires no additional nonparametric estimation. There is, as yet, however, no consensus on which are the best estimation methods to apply in practice."
  > " There is little formal evidence specific for these estimators, but, given that the estimators are asymptotically linear, it is likely that boot- strapping will lead to valid standard errors and confidence intervals at least for the regression and propensity score methods. **Bootstrapping may be more complicated for matching estimators, as the process introduces discreteness in the distribution that will lead to ties in the matching algorithm. Subsampling (Politis and Romano, 1999) will still work in this setting...**"

* In particular, [@abadieFailureBootstrapMatching2006] says that for **nearest-neighbour** matching estimators, bootstrapping the ATE (at least by bootstrapping control/treated separately) is explicitly broken
  * Key  idea: discontinuity
* [@otsuBootstrapInferenceMatching2017] introduces a "wild bootstrapping" procedure, where treatments and covariates are kept the same, but **noise is added to the response $Y$ values**. This is designed to ameliorate difficulties with **nearest neighbour** estimators (but presumably could be used by others, just maybe not as useful). Intuition: only addding noise to the $Y$ values keeps the nearest neighbours constant
* [@adusumilliBootstrapInferencePropensity] has an algorithm where bootstrapped treatments are *sampled from $Binom(prop-score(X_i)$* which is interesting; explicitly requires prop-score estimation TODO come back to this one~!
* [@hillIntervalEstimationTreatment2006] simply re-samples groups separately, then performs everything the same. Didn't do inner CV/model selection/hyper-param tuning because it was 2006  and would have been too slow
  * Concerned that bootstrapping **overestimates** uncertainty
* Subsampling methods [@politisSubsampling1999] generally look like sampling **without replacement** $b$ samples where $b/n \rightarrow 0$ and $b \rightarrow \infty$
* [@zhangDesignedBootstrapCausal2021] appears to look at bootstrapping *after matching/designing a dataset*: TODO: is this useful?
  * idea: big data too big to do whole thing every time
