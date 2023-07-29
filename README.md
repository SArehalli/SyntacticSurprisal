# Code for Arehalli, Dillon &amp; Linzen (2022) "Syntactic Surprisal From Neural Models Predicts, But Underestimates, Human Processing Difficulty From Syntactic Ambiguities".

Note that this repository corrects two errors in the analysis presented in the CoNLL paper which mistakenly excluded some filler words and represented some in-vocabulary words are out-of-vocabulary tokens. These errors are corrected in the latest arXiv version of the paper.


## Instructions to replicate

### Prepare additional files
Download [pretrained models]((https://drive.google.com/drive/folders/1nWodm9joib-D9aSQP4hMpVnTFMR5bMKT?usp=sharing) to a folder named "models"

Download [SAP Benchmark](https://github.com/caplabnyu/sapbenchmark/) data for the ClassicGP and Filler subsets, as well as the freqs_coca.csv (word frequencies taken from the COCA) to the analysis folder. 

Optionally download the SAP benchmark repo to sapbenchmark and the [CCGMultitask](https://github.com/SArehalli/CCGMultitask) codebase to CCGMultitask. However, these steps are not necessary; the specific files needed to run this analysis is provided here.

### Generate surprisals

Run get_synsurp.py to estimate surprisals (lexical and syntactic) from the models. An example of how to run this code is provided in scripts/template.sh, and running scripts/gen_scripts.py will generate scripts to run these analyses in parallel using SLURM.

surps/* contain the resulting surprisal estimates, while logs/* contain the command-line output of get_synsurp.py.

### Run analyses

analysis/analysis.R contains all the analyses presented in the paper. 

plots/* and pairwise_comps.log contain the resulting plots and statistical comparisons.

The surprisal-to-RT conversion models used in both the CoNLL and corrected analyses can be downloaded [here](https://drive.google.com/drive/folders/1nWodm9joib-D9aSQP4hMpVnTFMR5bMKT?usp=sharing) and should be placed in analysis/models/

