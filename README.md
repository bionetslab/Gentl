![Gentl](Gentl-icon.jpeg)

# Gentl
The source code for Gentl (<ins>GEN</ins>e<ins>T</ins>ic a<ins>L</ins>gorithm for predicting stage and grade from medical scans of patients with cancer) [[access preprint](doi-when-available)].

<!------------------>

# About

This is a repository that contains information on how to reproduce results corresponding to the *bladder cancer* case study reported in [Paper title](https://paper-doi-when-available).

<!------------------>

# Data

![fig2-bladder-paper](fig2-bladder-paper.png)

## Description

- As described in our [paper](https://www.mdpi.com/2072-6694/15/6/1673), the data used for our analyses comprised a total of 100 CT scans of the bladder, each from a patient with bladder cancer.

- Disease: urothelial carcinoma of the bladder

- Stages: Ta, Tis, T0, T1, T2, T3, T4

- Stage annotation technique: Performed manually by radiologists

For more details, interested readers are directed to the **Dataset** section of the [paper](https://www.mdpi.com/2072-6694/15/6/1673).

## Availability

Data will be made available under reasonable request to the corresponding author, <a href="mailto:suryadipto.sarkar@fau.de">Suryadipto Sarkar</a> (more contact details below).

<!------------------>

# Data preprocessing

## Otsu's thresholding

![github-otsus-thresholding](github-otsus-thresholding.jpg)


## Overall ROI bounding box selection

Done using hyperparameter that takes as input the choice of the user, as follows:

- Mode 1: Circum-rectangle

- Mode 2: Inner rectangle derived using elipse

- Mode 3 (default): In-rectangle


![github-overall-roi-bounding-box-selection](github-overall-roi-bounding-box-selection.jpg)




<!------------------>

# Installation

Install conda environment as follows (there also exists a requirements.txt)
```bash
conda create --name imaging_heterogeneity_study
conda activate imaging_heterogeneity_study
pip install scipy==1.10.1 numpy==1.23.5 squidpy==1.3.0 pandas==1.5.3 scikit-learn==1.2.2
```
*Note:* Additionally, modules *math* and *statistics* were used, however no installation is required as they are provided with Python by default.

<!------------------>

# Robustness testing

Pending

<!------------------>

# Scalability testing

Pending

<!------------------>

# Reproducing figures

Pending

<!------------------>

# Citing the work

## MLA

Will be made available upon publication.

## APA

Will be made available upon publication.

## BibTex

Will be made available upon publication.

<!------------------>

# Contact

&#x2709;&nbsp;&nbsp;suryadipto.sarkar@fau.de<br/>
&#x2709;&nbsp;&nbsp;ssarka34@asu.edu<br/>
&#x2709;&nbsp;&nbsp;ssarkarmanipal@gmail.com

<!------------------>

# Impressum

Suryadipto Sarkar ("Surya"), MS<br/><br/>
PhD Candidate<br/>
Biomedical Network Science Lab<br/>
Department of Artificial Intelligence in Biomedical Engineering (AIBE)<br/>
Friedrich-Alexander University Erlangen-Nürnberg (FAU)<br/>
Werner von Siemens Strasse<br/>
91052 Erlangen<br/><br/>
MS in CEN from Arizona State University, AZ, USA.<br/>
B.Tech in ECE from MIT Manipal, KA, India.
