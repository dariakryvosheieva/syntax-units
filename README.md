# Different types of syntactic agreement recruit the same units within large language models

[TODO: preamble]

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
* `multilingual/`: Data and figures for the multilingual experiments (Section 4.3)
