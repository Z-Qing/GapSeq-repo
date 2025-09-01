# SPIN-Seq
This repo stores some scripts used in the publication 'Single-molecule phenotyping and in-situ sequencing for mechanistic analysis of protein-DNA interactions and reactions'.
The functions include movie correction (drift correction (AIM method), channel alignment), localization count method, base calling method and etc.
It uses picasso (https://github.com/jungmannlab/picasso) as backend for localization detection & fitting. The predictive mean from a Bayesian online method (
https://doi.org/10.48550/arXiv.0710.3742) is used for PELT change point detection.  

Please notice that since it's tailored to our experiment, constants such as camera baseline, gain, ROI, penalty for PELT etc. need to be changed directly in the codes before they can be used in other experiments. 
We may provide a standalone software or napari plugin if it's a popular demand.
