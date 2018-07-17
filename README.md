# LASCAD
LASCAD: Language-Agnostic Software Categorization and Similar Application Detection

## Paper to reference:

If you use any of the source code, the datasets, or the results of the paper, please reference:
- Altarawy, Doaa, Hossameldin Shahin, Ayat Mohammed, and Na Meng. "Lascad: Language-agnostic software categorization and similar application detection." Journal of Systems and Software 142 (2018): 21-34.


## Abstract
Categorizing software and detecting similar programs are useful for various purposes including expertise sharing, program comprehension, and rapid prototyping. However, existing categorization and similar software detection tools are not sufficient. Some tools only handle applications written in certain languages or belonging to specific domains like Java or Android. Other tools require significant configuration effort due to their sensitivity to parameter settings, and may produce excessively large numbers of categories. In this paper, we present a more usable and reliable approach of Language-Agnostic Software Categorization and similar Application Detection (LASCAD). Our approach applies Latent Dirichlet Allocation (LDA) and hierarchical clustering to programs' source code in order to reveal which applications implement similar functionalities.
LASCAD is easier to use in cases when no domain-specific tool is available or when users want to find similar software across programming languages.

To evaluate LASCAD's capability of categorizing software, we used three labeled data sets: two sets from prior work and one larger set that we created with 103 applications implemented in 19 different languages. By comparing LASCAD with prior approaches on these data sets, we found LASCAD to be more usable and outperform existing tools. To evaluate LASCAD's capability of similar application detection, we reused our 103-application data set and a newly created unlabeled data set of 5,220 applications. The relevance scores of the Top-1 retrieved applications within these two data sets were 70% and 71%, respectively. Overall, LASCAD effectively categorizes and detects similar programs across languages.

## Data set:
The showcases data set of 103 processed source code applications is available at:
http://doi.org/10.5281/zenodo.1154941
