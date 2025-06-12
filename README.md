# SPIN-Seq
This repo stores some scripts used in the publication 'Single-molecule phenotyping and in-situ sequencing for mechanistic analysis of protein-DNA interactions and reactions'.
The functions include movie correction (drift correction (AIM method), channel alignment), time series analysis using PELT algorithm, localization count method, etc.
It uses python 3.10, picasso (https://github.com/jungmannlab/picasso) as backend for localization detection & fitting and ruptures (https://github.com/deepcharles/ruptures) for change point detection.

Please notice that since it's tailored to our experiment, constants such as camera baseline, gain, ROI, etc. may need to be changed for use in other experiments. 
We may provide software or napari plugin if it's a popular demand.
