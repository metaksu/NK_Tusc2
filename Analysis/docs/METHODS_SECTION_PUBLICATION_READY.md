# Methods

## scRNA-seq data retrieval and preprocessing

Single-cell RNA sequencing data were retrieved from two complementary datasets obtained directly from the original authors as pre-processed expression matrices. Healthy blood NK cell reference data were obtained from Rebuffet et al., comprising 35,578 NK cells from peripheral blood mononuclear cells of healthy donors, profiled using 10x Genomics Chromium Single Cell 3' v2 chemistry. The original data were provided as a Seurat R object containing TPM-normalized expression data and converted to AnnData format using custom processing scripts that extracted expression matrices, cell metadata, and gene annotations. Tissue-resident and tumor-associated NK cell data were obtained from Tang et al., containing 142,304 NK cells from blood, normal tissues, and tumor microenvironments across 25 cancer types plus healthy donors, integrated from 64 source datasets predominantly using 10x Genomics platforms.

Expression matrices were processed using Scanpy (v1.11.0) in Python 3.13.2. Quality control metrics were calculated for each cell using `scanpy.pp.calculate_qc_metrics`, including: (1) mitochondrial gene percentage (genes starting with 'MT-'), (2) ribosomal gene percentage (genes starting with 'RPS' or 'RPL'), (3) hemoglobin gene percentage (genes matching pattern '^HB[^(P)]'), and (4) total unique molecular identifier (UMI) counts. 

Quality control filtering was applied using the enhanced_preprocessing_pipeline function with adaptive thresholds. Cells expressing fewer than 200 distinct genes were removed. Cells with mitochondrial gene percentage exceeding context-specific adaptive thresholds were filtered out. Consensus doublet detection was performed to identify and remove potential doublets. Genes detected in fewer than 10 cells were excluded from analysis using `scanpy.pp.filter_genes` with parameter `min_cells=10`.

For normalization, two distinct approaches were employed based on data characteristics. For the Tang dataset containing raw count data, normalization was performed using the `scanpy.pp.normalize_total` function with parameter `target_sum=1e4` to account for differences in sequencing depth between cells, followed by natural logarithm transformation using `scanpy.pp.log1p`. For the Rebuffet dataset containing TPM-normalized data, log-transformation was applied directly using `numpy.log1p`. 

Highly variable genes were identified using the Seurat method implemented in `scanpy.pp.highly_variable_genes` with parameter `flavor='seurat'`. The top 1,000 most variable genes were selected for downstream analysis while preserving the complete gene expression matrix in the `.raw` attribute to maintain access to all genes for signature scoring.

## Cell type annotation and subtype classification

The datasets contained pre-filtered NK cells as provided by the original authors, identified based on positive expression of CD56 (NCAM1) and/or CD16 (FCGR3A) with negative expression of CD3 complex genes. The original subtype annotations were preserved to maintain biological context.

For healthy blood NK cells, the Rebuffet classification system was employed, which identifies six developmental states: NK2 (regulatory), NKint (intermediate), NK1A (early mature), NK1B (intermediate mature), NK1C (mature cytotoxic), and NK3 (adaptive). For tissue-resident NK cells, the Tang classification system was used, resolving 14 distinct subtypes based on CD56/CD16 expression patterns and tissue-specific functional specializations.

Subtype assignments were validated through differential expression analysis using the Wilcoxon rank-sum test implemented in `scanpy.tl.rank_genes_groups` with parameter `method='wilcoxon'`. Marker genes were identified using criteria: adjusted p-value < 0.05, log fold change > 0.25, minimum percentage of expressing cells > 0.2. Cells with insufficient confidence in subtype assignment were classified as "Unassigned" and excluded from subtype-specific analyses.

## Batch-effect correction and dimensionality reduction

Prior to dimensionality reduction, gene expression values were scaled to unit variance using `scanpy.pp.scale` with parameter `max_value=10` to clip extreme outliers. Principal component analysis was performed on the matrix of highly variable genes using `scanpy.tl.pca` with parameter `svd_solver='arpack'` and `random_state=42` for reproducibility.

Datasets were analyzed separately to preserve biological differences between healthy blood and tissue contexts. This approach maintained the integrity of tissue-specific NK cell signatures while enabling robust comparative analysis.

Neighborhood graphs were constructed using `scanpy.pp.neighbors` with 15 principal components and `random_state=42`. UMAP dimensionality reduction was computed using `scanpy.tl.umap` with parameter `min_dist=0.3` and `random_state=42` for reproducible visualization.

## Functional signature scoring and analysis

Comprehensive gene sets were curated representing key NK cell functional programs and signaling pathways:

**Core functional signatures** comprised six manually curated gene sets reflecting fundamental NK cell programs: 

(1) **Activating Receptors** (20 genes): IL2RB, IL18R1, IL18RAP, NCR1, NCR2, NCR3, KLRK1, FCGR3A, CD226, KLRC2, CD244, SLAMF6, SLAMF7, CD160, KLRF1, KIR2DS1, KIR2DS2, KIR2DS4, KIR3DS1, ITGAL

(2) **Inhibitory Receptors** (18 genes): KLRC1, KIR2DL1, KIR2DL2, KIR2DL3, KIR3DL1, KIR3DL2, LILRB1, PDCD1, TIGIT, CTLA4, HAVCR2, LAG3, SIGLEC7, SIGLEC9, KLRG1, CD300A, LAIR1, CEACAM1

(3) **Cytotoxicity Machinery** (15 genes): PRF1, GZMA, GZMB, GZMH, GZMK, GZMM, NKG7, GNLY, SERPINB9, SRGN, FASLG, TNFSF10, LAMP1, CTSC, CTSW

(4) **Cytokine/Chemokine Production** (13 genes): IFNG, TNF, LTA, CSF2, IL10, IL32, XCL1, XCL2, CCL3, CCL4, CCL5, CXCL8, CXCL10

(5) **Exhaustion/Suppression Markers** (15 genes): PDCD1, HAVCR2, LAG3, TIGIT, KLRC1, KLRG1, CD96, LILRB1, ENTPD1, TOX, EGR2, MAF, PRDM1, HSPA1A, DNAJB1

(6) **Decidual NK Markers** (9 genes): CD9, ITGA1, KLRC2, KLRC3, HAVCR2, ITGB7, GNLY, VEGFA, IFNG

**Neurotransmitter receptor signatures** captured 17 receptor families representing potential neuroimmune interactions:

- **Acetylcholine Receptors** (10 genes): CHRNA2, CHRNA3, CHRNA4, CHRNA5, CHRNA7, CHRNB2, CHRNB4, CHRNE, CHRM3, CHRM5
- **Norepinephrine Receptors** (1 gene): ADRB2  
- **Dopamine Receptors** (5 genes): DRD1, DRD2, DRD3, DRD4, DRD5
- **Serotonin Receptors** (4 genes): HTR1A, HTR2A, HTR2C, HTR7
- **Substance P Receptors** (1 gene): TACR1
- **Estrogen Receptors** (3 genes): ESR1, ESR2, GPER1
- **Testosterone Receptors** (1 gene): NR3C4
- **Glutamate Receptors** (24 genes): GRIA1, GRIA2, GRIA3, GRIA4, GRIN1, GRIN2A, GRIN2B, GRIN2C, GRIN2D, GRIN3A, GRIN3B, GRIK1, GRIK2, GRIK3, GRIK4, GRIK5, GRM1, GRM2, GRM3, GRM4, GRM5, GRM6, GRM7, GRM8
- **GABA Receptors** (21 genes): GABRA1, GABRA2, GABRA3, GABRA4, GABRA5, GABRA6, GABRB1, GABRB2, GABRB3, GABRG1, GABRG2, GABRG3, GABRD, GABRE, GABRP, GABRR1, GABRR2, GABRR3, GABRQ, GABBR1, GABBR2
- **Histamine Receptors** (3 genes): HRH1, HRH2, HRH4
- **Cannabinoid Receptors** (4 genes): CNR1, CNR2, GPR18, GPR55
- **Opioid Receptors** (3 genes): OPRM1, OPRD1, OPRK1
- **Neuropeptide Y Receptors** (2 genes): NPY1R, NPY2R
- **Somatostatin Receptors** (2 genes): SSTR1, SSTR2
- **VIP Receptors** (2 genes): VIPR1, VIPR2
- **CGRP Receptors** (2 genes): CALCRL, RAMP1
- **Purinergic Receptors** (6 genes): P2RX1, P2RX4, P2RY2, P2RY11, ADORA2A, ADORA2B

**Interleukin downstream signaling signatures** encompassed seven critical cytokine pathways:

- **IL15/IL2 Downstream** (17 genes): STAT5A, STAT5B, AKT1, MTOR, HIF1A, MYC, BCL2, MCL1, CCND2, ID2, ZEB2, KLF2, GZMB, PRF1, IFNG, CISH, SOCS2
- **IL12 Downstream** (11 genes): STAT4, TBX21, EOMES, IFNG, IRF1, GZMB, PRF1, FASLG, IL12RB2, IL18R1, SOCS1
- **IL18 Downstream** (12 genes): NFKB1, MYD88, IRAK1, IRAK4, TNF, XCL1, CCL3, CCL4, IFNG, PRF1, STX11, NLRP3
- **IL21 Downstream** (12 genes): STAT1, STAT3, BCL6, ID2, SAMHD1, TBX21, EOMES, GZMA, GZMB, PRF1, IFNG, IL21R
- **IL10 Downstream** (6 genes): STAT3, SOCS3, BCL3, MAF, TGFB1, IL10RA
- **IL27 Downstream** (8 genes): STAT1, STAT3, ID2, IL10, LAG3, HAVCR2, PDCD1, TIGIT
- **IL33 Downstream** (12 genes): MYD88, IRAK1, IRAK4, NFKB1, GATA3, IL5, IL13, IFNG, GZMB, CSF2

**Metabolic signatures** were derived from MSigDB Hallmark gene sets for glycolysis and oxidative phosphorylation pathways, representing key metabolic programs that distinguish NK cell functional states.

**Dynamic developmental signatures** were generated empirically from both reference datasets using a data-driven approach with optimized gene selection criteria. Differential expression analysis was performed using `scanpy.tl.rank_genes_groups` with Wilcoxon rank-sum test and Benjamini-Hochberg multiple testing correction (`corr_method='benjamini-hochberg'`). Subtype-specific marker genes were identified using a highly selective gene selection algorithm that balances specificity with biological relevance for developmental NK cell subtypes. Selection criteria included: (1) minimum log fold change of 0.2 (approximately 1.15-fold upregulation), (2) minimum specificity ratio of 1.2 (20% higher expression in target vs. other subtypes), (3) expression in at least 10% of target subtype cells, and (4) no maximum overlap restriction to allow for shared expression patterns among related developmental stages. This relaxed approach recognizes that NK cell development represents a continuous process where related subtypes may share overlapping gene expression programs. For the healthy blood NK reference dataset, the top 30 most selective genes meeting these criteria were selected for each of the six Rebuffet subtypes: NK2 (regulatory CD56bright), NKint (intermediate CD56bright), NK1A (early mature CD56dim), NK1B (intermediate mature CD56dim), NK1C (mature cytotoxic CD56dim), and NK3 (adaptive/memory CD56dim). Similarly, dynamic signatures were generated for the 14 Tang subtypes representing tissue-resident and tumor-associated NK cell states, with additional subset-specific analysis for CD56+CD16- and CD56-CD16+ populations to capture the full spectrum of NK cell heterogeneity across tissue contexts.

Gene set scores were calculated using a custom AUCell-like scoring algorithm that provides superior cross-dataset robustness compared to standard methods. This algorithm ranks genes by expression within each cell and calculates the area under the recovery curve for each gene set, measuring how highly ranked the signature genes are relative to the total gene expression distribution. The method converts gene ranks to normalized scores where higher expression corresponds to higher scores, with final scores normalized to the range [0,1] for comparability across signatures. This approach is particularly robust to differences in normalization procedures, expression units, and dataset-specific technical variations that can affect cross-dataset analyses. For signatures with insufficient genes (< 5 genes available), cells were assigned `np.nan` values. All scoring utilized log-normalized expression values from the `.raw` attribute to ensure consistency across analyses and maintain access to the complete gene expression matrix.

## TUSC2 expression analysis

TUSC2 (Tumor Suppressor Candidate 2) expression was quantified using log-normalized UMI counts extracted from the `.raw` expression matrix to ensure consistency with signature scoring methodology. Two complementary analytical approaches were employed: (1) continuous expression analysis treating TUSC2 as a quantitative variable across all cells, and (2) binary classification approach dividing cells into TUSC2-positive (TUSC2+) versus TUSC2-negative (TUSC2-) populations using an expression threshold of 0.1 log-normalized counts.

Differential expression analysis between TUSC2+ and TUSC2- populations was performed using the Wilcoxon rank-sum test implemented in `scanpy.tl.rank_genes_groups` with default parameters. Genes were considered significantly differentially expressed at adjusted p-value < 0.05 (Benjamini-Hochberg correction) and absolute log fold change > 0.25. Results were validated using the Mann-Whitney U test implemented in SciPy.stats for methodological robustness and cross-validation of findings.

## Cross-context comparative analysis

Comparative analysis was performed between healthy blood and tissue-resident NK cells, with datasets analyzed separately to preserve context-specific biological differences. For each functional signature, mean score differences between contexts were calculated using non-parametric statistical tests.

Statistical significance was assessed using the Mann-Whitney U test with Benjamini-Hochberg correction for multiple testing to control false discovery rate at 5%. Effect sizes were quantified using Cohen's d with standard conventions (small: 0.2, medium: 0.5, large: 0.8).

Subtype-specific analyses were performed within each context separately, followed by cross-context comparison of functionally equivalent states. Marker gene expression validation was performed to confirm that observed differences reflected genuine biological variation.

## Visualization and statistical analysis

Data visualization was performed using matplotlib (v3.7+) and seaborn (v0.12+) with custom styling for publication-quality figures. All plots employed consistent color schemes and statistical annotation conventions. Statistical significance was assessed using non-parametric tests (Wilcoxon rank-sum, Mann-Whitney U, Kruskal-Wallis).

Multiple testing correction was systematically applied using the Benjamini-Hochberg method to control false discovery rate at 5% across all comparative analyses. For heatmap visualizations, expression values were standardized per gene across samples using the `standard_scale='var'` parameter in scanpy plotting functions. Dotplot visualizations simultaneously displayed mean expression levels and the percentage of cells expressing each gene within each subtype.

Statistical annotations followed standard conventions with significance levels indicated as: *p < 0.05, **p < 0.01, ***p < 0.001. All confidence intervals were calculated at 95% confidence level where applicable.

## Quality control and validation

Analytical validation was performed through marker gene expression validation for all subtype assignments and cross-dataset validation where overlapping populations existed between studies.

Technical robustness was assessed through sensitivity analysis using different parameter settings. All random processes employed fixed seeds (seed=42) for numpy, scanpy, and matplotlib to ensure complete reproducibility across computational environments and analysis runs.

The analysis pipeline included comprehensive logging and checkpoint functionality for troubleshooting, modification, and iterative development. Data integrity checks were implemented at each major processing step to identify potential issues early in the analytical workflow.

## Computational implementation

All analyses were implemented in Python 3.13.2 using Scanpy (v1.11.0) as the primary single-cell analysis framework. Key computational dependencies included pandas (v2.0+) for data manipulation, NumPy (v1.24+) for numerical operations, SciPy (v1.10+) for statistical functions and sparse matrix operations, matplotlib (v3.7+) and seaborn (v0.12+) for visualization and plotting. Random seed was systematically set to 42 for all stochastic processes including random sampling, clustering initialization, and UMAP embedding to ensure complete analytical reproducibility.

The custom AUCell-like scoring algorithm was implemented using NumPy operations with optimized sparse matrix handling. For each cell, genes were ranked by expression using `numpy.argsort` with descending order, signature gene positions were identified in the ranked list, and AUC-like scores were calculated as the complement of the average rank normalized by total gene number (1 - average_rank/total_genes). Score normalization was performed using min-max scaling to ensure [0,1] range comparability across signatures.

The analysis pipeline employed modular implementation with comprehensive logging and checkpoint functionality. Sparse matrix representations were utilized throughout via SciPy.sparse (csr_matrix format) to optimize memory usage and computational efficiency for large single-cell datasets. All analysis code is version-controlled using Git and documented with comprehensive inline comments and docstrings.

## Statistical analysis

Statistical analyses employed non-parametric methods. Differential expression testing used the Wilcoxon rank-sum test for two-group comparisons and Kruskal-Wallis test for multi-group comparisons. Correlation analyses employed Spearman rank correlation. Effect sizes were calculated using Cohen's d for continuous variables and odds ratios for categorical comparisons.

Multiple testing correction was applied using the Benjamini-Hochberg method to control false discovery rate at 5% across all comparative analyses. Statistical significance thresholds were set at α = 0.05 for all tests.

## Data availability

Input datasets were obtained directly from the original authors as pre-processed expression matrices. The underlying data are publicly available through the Gene Expression Omnibus (GEO) database under accession numbers GSE181433 (Rebuffet et al.) and GSE212890 (Tang et al.). All processed data objects, intermediate analysis files, and final results are available upon request.

## Code availability

All custom analysis code, processing scripts, and visualization functions developed for this study are available on GitHub at [repository URL to be added]. The repository includes comprehensive documentation and step-by-step tutorials enabling complete reproduction of all analyses. Code is provided under an open-source license with version control. 