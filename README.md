# Different types of syntactic agreement recruit the same units within large language models

Using a functional localization approach inspired by cognitive neuroscience, we identify the LLM units most responsive to 67 English syntactic phenomena in seven open-weight models. These units are consistently recruited across sentence instances and causally support model performance. Critically, different types of syntactic agreement (e.g., subject-verb, anaphor, determiner-noun) recruit overlapping sets of units, suggesting that agreement constitutes a meaningful functional category in LLMs. Beyond English, this pattern holds in Russian and Chinese, and further, *across* languages: in a cross-lingual analysis of 57 languages, syntactically similar languages share more units for subject-verb agreement.

## Setup

1. Create virtual environment: `python -m venv .venv`
2. Activate environment: `. .venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`

## Repository Structure

* `english/`: Data and figures for the English experiments
  * `cross-validation/`: Main cross-validation experiment (Sec. 4.1, Fig. 1)
  * `ablation/`: Main zero-ablation experiment (Sec. 4.1, Fig. 2) + figure showing individual model scores (Appendix G, Fig. 18)
  * `cross-overlap/`: Within-category and cross-category overlaps between phenomena (Sec. 4.2, Figs. 3-5)
  * `0.5%/`: Experiments targeting top-0.5% of units (Appendix A, Figs. 9-10)
  * `5%/`: Experiments targeting top-5% of units (Appendix B, Figs. 11-12)
  * `finegrained/`: Experiments targeting MLP and attention submodules (Appendix C, Figs. 13-14)
  * `5-fold/`: Five-fold cross-validation (Appendix D, Fig. 15)
  * `generalization/`: Comparison of units localized on BLiMP versus other benchmarks (Appendix E, Fig. 16)
  * `mean-ablation/`: The mean ablation experiment (Appendix F, Fig. 17)
  * `scatterplot.png`/`pdf`: Correlation between cross-validation consistency and ablation effect (Appendix H, Fig. 19)
  * `cv-blimp-gemma/`: Cross-validation result for Gemma on BLiMP (upper subplot of Fig. 20)
* `multilingual/`: Data and figures for the multilingual experiments (Sec. 4.3)
  * `rublimp/`: Experiments with the RuBLiMP benchmark (left subplot of Fig.6, upper subplot of Fig. 7, middle subplot of Fig. 20)
  * `sling/`: Experiments with the SLING benchmark (right subplot of Fig. 6, lower subplot of Fig. 7, lower subplot of Fig. 20)
  * `multiblimp/`: Experiment with the MultiBLiMP benchmark (Fig. 8)
* `benchmarks/`: Code for converting minimal pair benchmarks into the appropriate format for unit localization
  * `processed/`: Converted data files
* `t-test/`: Utility for running one-sample and two-sample t-tests.

## Citation
