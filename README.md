![Gentl](Gentl-icon.jpeg)

# Gentl
The source code for Gentl (<ins>GEN</ins>e<ins>T</ins>ic a<ins>L</ins>gorithm for predicting stage and grade from medical scans of patients with cancer) [[access preprint](doi-when-available)].

<!------------------>

# About

This is a repository that contains information on how to reproduce results corresponding to the *bladder cancer* case study reported in [Paper title](https://paper-doi-when-available).

<!------------------>

# Abstract

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

![github-otsus-thresholding](github-otsus-thresholding.jpeg)

<br/>

## Overall ROI bounding box selection

Done using hyperparameter that takes as input the choice of the user, as follows:

- Mode 1: Circum-rectangle

- Mode 2: Inner rectangle derived using elipse

- Mode 3 (default): In-rectangle

<br/>

![github-overall-roi-bounding-box-selection](github-overall-roi-bounding-box-selection.jpeg)


## Smaller ROI bounding box selection from healthy tissue

Smaller ROI bounding box selection is done using a sliding window implementation, from healthy areas of the tissue, binarized features extracted from which are subsequently used as the initial population for the genetic algorithm.

Note that the binarized feature extracted from the cancer ROI of the same image is subsequently used as the target (also known as, goal) for the genetic algorithm.

Reported results have used initial population size as {10, 20, 50, 100, 500, 1000}. Reported results have also healthy areas with constant size across the image samples, although this was not really necessary since we are using binarized GLCM texture features from the ROIs and not the ROIs themselves as input to the genetic algorithm.

<!------------------>

# Feature extraction

The following five GLCM features were extracted from the cancer ROI, as well as healthy ROIs from the same patient:

- Dissimilarity
- Correlation
- Energy
- Contrast
- Homogeneity,

using $20$ configurations ($4$ angles: $\{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4} \}$; $5$ distances: $\{1, 2, 3,5, 7\}$ pixels).

<!------------------>

# Feature binarization

- Performed on the cancer ROI using bimodal Gaussian mixture model (GMM) fitting using the ** package. All feature values that are closer in Euclidean distance to lower mean ($\mu_1$) is assigned a value of $0$, else $1$ if closer to higher mean ($\mu_2$).
- All healthy ROI feature values from the same image sample are assigned a value of $0$ if they are closer in Euclidean distance to the lower mean ($\mu_1$) obtained from the **cancer ROI above**, else assigned a value of $1$ if closer to $\mu_2$.

*Note*: Bimodal GMM fitting only done once per image, pertaining to the cancer ROI. Feature binarization of healthy ROIs performed based on mean values obtained from bimodal GMM fitting on the **cancer ROI** pertaining to the same image sample.

![github-gmm-individual-patients](github-gmm-individual-patients.jpeg)

<!------------------>

# Genetic algorithm

## General information about our implementation of the algorithm

- We perform the genetic algorithm on each sample image separately.


## An overview of the terms *gene*, *chromosome* and *population*

![github-gene-chromosome-population](github-gene-chromosome-population.jpeg)


## Algorithmic workflow

![github-pictorial-description-of-genetic-algorithm](github-pictorial-description-of-genetic-algorithm.jpeg)


### Step 1: Population initialization

- The initial population ($P$) comprises binarized GLCM features extracted from the healthy ROIs.

- Reported results include $P=\{10, 20, 50, 100, 500, 1000\}$.


### Step 2: Parent selection by fitness evaluation

- Fitness metric: Euclidean distance to target.
	- In our implementation target is binarized feature list from cancer ROI.

- Parent selection rate: $50\%$ of the population at the end of iteration *$i$* is retained as parents for iteration *$i+1$*. Therefore, list of selected parents contains top $50\%$ of the chromosomes closest to the target sequence.

### Step 3: Crossover (initial offspring generation)

![github-crossover](github-crossover.jpeg)

- For crossover between two parents:
	- The first parent ($p_1$) is always chosen from the top $50\%$ of chromosomes (that is, ones having least Euclidean distance to the target sequence).
	- The second parent ($p_2$) is chosen from the initial population at each iteration.

- Random portions of parents $p_1$ and $p_2$ constitute the respective offspring&mdash;with at least one gene compulsorily selected from each parent \{ $p_1$ , $p_2$ \}.

### Step 4: Mutation (final offspring generation)

![github-mutation](github-mutation.jpeg)

- Initial offspring $\overline{o_{1,2}}$ generated from parents $p_1$ and $p_2$ in step 3 (crossover) described above, undergoes mutation to give rise to final offspring $o_{1,2}$.

### Step 5: Replacement

- In this step, we replace the worst-performing individuals in the current population with new offspring, retaining the better-performing individuals.

- In the script */gentl/\_ga\_step5\_replacement.py*:
	- Input parameters:
    	- `population`: The current population.
    	- `new_generation`: The new generation of chromosomes.
    	- `goal`: The target sequence.

    - Returns:
    	- `best_individuals`: The updated population containing the best individuals.


<!------------------>

# Supplementary information

## Intermediate results

### Average distance

#### Contrast

| patient_id | average_distance   |
| ---------- | ------------------ |
| CT-073     | 0.2                |
| CT-101     | 0.63284271         |
| CT-080     | 0.7694452528530629 |
| CT-121     | 0.7742640687119285 |
| CT-052     | 0.8328427124746189 |
| CT-059     | 0.8901559309717175 |
| CT-160     | 0.9108666090903723 |
| CT-015     | 0.9121320343559642 |
| CT-156     | 0.91568542         |
| CT-098     | 0.9815772872090271 |
| CT-146     | 1.002287965327682  |
| CT-056     | 1.0096011838247807 |
| CT-062     | 1.1060477932315067 |
| CT-158     | 1.1292528739883945 |
| CT-137     | 1.1413849083443588 |
| CT-127     | 1.1522879653276819 |
| CT-155     | 1.154782367965915  |
| CT-143     | 1.154782367965915  |
| CT-144     | 1.1779874487228026 |
| CT-106     | 1.187625080440534  |
| CT-047     | 1.2229986434463367 |
| CT-097     | 1.2278174593052023 |
| CT-147     | 1.2620955864630137 |
| CT-111     | 1.2754930460845697 |
| CT-123     | 1.2888905057061257 |
| CT-119     | 1.3010225400620903 |
| CT-009     | 1.304782367965915  |
| CT-049     | 1.323898985338003  |
| CT-083     | 1.3276588132418277 |
| CT-118     | 1.3279874487228027 |
| CT-102     | 1.3424438962993999 |
| CT-124     | 1.3742276208189779 |
| CT-140     | 1.3986981268414573 |
| CT-172     | 1.4120955864630136 |
| CT-104     | 1.4163537981217507 |
| CT-134     | 1.4315408393160765 |
| CT-076     | 1.4365660924854933 |
| CT-151     | 1.4436471011197753 |
| CT-154     | 1.4605015257164469 |
| CT-115     | 1.4642613536202718 |
| CT-169     | 1.4678147442135454 |
| CT-174     | 1.4837430543662853 |
| CT-139     | 1.5097711732423809 |
| CT-128     | 1.5145899891012466 |
| CT-054     | 1.5760647524952613 |
| CT-055     | 1.5829762539992687 |
| CT-053     | 1.6089074649698802 |
| CT-126     | 1.6097711732423807 |
| CT-069     | 1.61299593         |
| CT-177     | 1.6192499177530748 |
| CT-138     | 1.6653369010646144 |
| CT-051     | 1.6781574381404027 |
| CT-016     | 1.6951811287236864 |
| CT-133     | 1.7149917060943376 |
| CT-142     | 1.7263366248097587 |
| CT-067     | 1.7310535577717268 |
| CT-120     | 1.7406911894894583 |
| CT-071     | 1.7438793109825916 |
| CT-131     | 1.7721462785280622 |
| CT-159     | 1.772839997382962  |
| CT-175     | 1.7887316184990887 |
| CT-100     | 1.7907642710015748 |
| CT-112     | 1.8083468607697515 |
| CT-074     | 1.809940921516318  |
| CT-087     | 1.8109741369190935 |
| CT-024     | 1.8360473380396065 |
| CT-176     | 1.838037899644188  |
| CT-060     | 1.8480774894986731 |
| CT-084     | 1.852299067538245  |
| CT-023     | 1.8591502947559342 |
| CT-086     | 1.8803229641539978 |
| CT-163     | 1.881126398140181  |
| CT-178     | 1.887409430067019  |
| CT-041     | 1.9227775357940833 |
| CT-108     | 1.9473468958809441 |
| CT-117     | 1.9666387188057086 |
| CT-107     | 1.9739153036836423 |
| CT-150     | 1.9811263981401808 |
| CT-025     | 1.9970182603999695 |
| CT-179     | 2.0134707268598873 |
| CT-012     | 2.017270217215393  |
| CT-136     | 2.024943061807678  |
| CT-184     | 2.027205254583095  |
| CT-164     | 2.0320865320308394 |
| CT-105     | 2.0496821701684773 |
| CT-167     | 2.0564970049106632 |
| CT-091     | 2.0594531576846435 |
| CT-129     | 2.0827057866181744 |
| CT-109     | 2.0923689516713786 |
| CT-029     | 2.101950061366295  |
| CT-165     | 2.10353842         |
| CT-090     | 2.116641747638225  |
| CT-065     | 2.1326012341313914 |
| CT-153     | 2.137425748412659  |
| CT-075     | 2.161485327899803  |
| CT-011     | 2.169945282685042  |
| CT-125     | 2.1983907312069575 |
| CT-099     | 2.200950624266585  |
| CT-042     | 2.256108936673326  |
| CT-096     | 2.3017741269350407 |

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
