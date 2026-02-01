# Methods Section: NK Cell Transcriptomics and TUSC2 Function Analysis

## Single-cell RNA-seq Data Sources and Integration

We integrated two complementary single-cell RNA sequencing datasets to comprehensively characterize NK cell subtypes across biological contexts. The primary reference dataset was obtained from Rebuffet et al. (2024), comprising healthy blood NK cells with well-defined developmental subtypes. The tissue-infiltrating NK cell data was obtained from Tang et al., representing a pan-cancer atlas of NK cells from normal and tumor tissues across multiple cancer types.

For the Rebuffet dataset, we utilized the processed TPM-normalized expression matrix (`PBMC_V2_VF1_AllGenes_NewNames_TPM_corrected.h5ad`) containing healthy blood NK cells with established subtype annotations (NK2, NKint, NK1A, NK1B, NK1C, NK3). The Tang dataset was accessed as a combined matrix (`comb_CD56_CD16_NK.h5ad`) containing 142,304 NK cells from blood (67,202 cells), tumor (34,900 cells), normal tissue (22,792 cells), and other tissues (17,410 cells) across 25 cancer types from 64 datasets.

## Data Preprocessing and Quality Control

### Enhanced Quality Control Pipeline

Data preprocessing was performed using an enhanced quality control pipeline implemented in Scanpy (version 1.11.0) with custom quality control functions. For datasets requiring quality assessment, we applied adaptive filtering based on mitochondrial gene percentage, ribosomal gene content, and hemoglobin gene percentage. Doublets were identified using consensus-based methods with removal of cells showing signatures of multiple cell types.

### Normalization Strategy

For the Rebuffet dataset containing TPM-normalized data, we applied log(TPM+1) transformation to ensure compatibility with downstream analyses. The Tang dataset underwent total normalization using `scanpy.pp.normalize_total` with `target_sum=1e4` followed by `scanpy.pp.log1p` transformation. All normalized data were stored in the `.raw` attribute before any scaling operations to preserve unscaled, log-normalized expression values for differential expression analyses.

### Gene Filtering and Data Structure

Genes expressed in fewer than 10 cells were filtered from the main expression matrix while preserving all genes in the `.raw` attribute. We applied systematic gene exclusion patterns to remove non-informative genes including ribosomal genes (^RP[SL]), mitochondrial genes (^MT-), hemoglobin genes (^HB[AB]), and immunoglobulin genes (^IG[HKL]) from marker gene analyses.

## Dimensionality Reduction and Visualization

### Highly Variable Gene Selection and Scaling

Highly variable genes were identified using the Seurat flavor method (`scanpy.pp.highly_variable_genes`) with `n_top_genes=1000` and `subset=False` to flag genes without filtering. Expression data were scaled using `scanpy.pp.scale` with `max_value=10` to clip extreme outliers while preserving biological variation.

### Principal Component Analysis and Neighborhood Graph Construction

Principal component analysis was performed using the ARPACK SVD solver with `random_state=42` for reproducibility. We retained 15 principal components based on variance ratio analysis and used these for neighborhood graph construction via `scanpy.pp.neighbors` with `n_pcs=15` and `random_state=42`.

### UMAP Embedding

Uniform Manifold Approximation and Projection (UMAP) was employed for two-dimensional visualization using `scanpy.tl.umap` with `min_dist=0.3` and `random_state=42`. These parameters were optimized to preserve both local and global structure while maintaining reproducibility across analyses.

## NK Cell Subtype Annotation and Classification

### Subtype Preservation Strategy

We maintained the original subtype annotations from each dataset to preserve biological context and avoid cross-dataset contamination. Rebuffet subtypes were preserved as ordered categories (NK2, NKint, NK1A, NK1B, NK1C, NK3) representing developmental progression from immature to mature NK cells. Tang subtypes were maintained as 14 distinct categories representing tissue-specific NK cell states.

### Tang Subtype Subset Analysis

For Tang data analysis, we implemented automatic subset generation based on CD56 and CD16 expression patterns. The analysis pipeline automatically detects Tang data and generates separate analytical tracks for CD56+CD16- (CD56bright) and CD56-CD16+ (CD56dim) NK cell populations using the `get_tang_subtype_subsets()` function, ensuring comprehensive coverage of NK cell heterogeneity.

## Functional Signature Score Calculation

### Gene Signature Curation

We compiled comprehensive gene signature libraries encompassing 84+ functionally distinct gene sets across multiple categories: developmental markers, cytotoxicity machinery, receptor expression, cytokine production, metabolic programs, and neurotransmitter signaling. Developmental signatures were generated dynamically from differential expression analysis of reference datasets using the top 50 upregulated genes per subtype.

### Signature Scoring Methodology

Functional signature scores were calculated using a custom AUCell-like scoring algorithm (`score_genes_aucell`) that provides superior cross-dataset robustness compared to standard methods. This algorithm ranks genes by expression within each cell and calculates the area under the recovery curve for each gene set, measuring how highly ranked the signature genes are relative to the total gene expression distribution. The method converts gene ranks to normalized scores where higher expression corresponds to higher scores, with final scores normalized to the range [0,1] for comparability across signatures. This approach is particularly robust to differences in normalization procedures, expression units, and dataset-specific technical variations that can affect cross-dataset analyses. A minimum threshold of 5 genes was required for valid score calculation; signatures with insufficient available genes were set to `np.nan`. All scoring utilized log-normalized expression data from the `.raw` attribute to ensure consistency across analyses.

### Dynamic Signature Generation

For developmental characterization, we implemented dynamic signature generation using differential expression analysis with optimized gene selection criteria. The `generate_rebuffet_developmental_signatures()` and `generate_tang_developmental_signatures()` functions perform subtype-specific differential expression analysis using the Wilcoxon rank-sum test with Benjamini-Hochberg correction, followed by highly selective gene selection using relaxed criteria appropriate for developmental NK cell subtypes. Selection criteria included: (1) minimum log fold change of 0.2 (approximately 1.15-fold upregulation), (2) minimum specificity ratio of 1.2 (20% higher expression in target vs. other subtypes), (3) expression in at least 10% of target subtype cells, and (4) no maximum overlap restriction to allow for shared expression patterns among related developmental stages. This approach recognizes that NK cell development represents a continuous process where related subtypes may share overlapping gene expression programs. The top 30 most selective genes meeting these criteria were selected per subtype.

## TUSC2 Expression Analysis Framework

### Binary Classification System

TUSC2 expression was classified using a binary threshold system with expression values extracted directly from `.raw.X` matrices. Cells were categorized as "TUSC2_Not_Expressed" or "TUSC2_Expressed" based on log-normalized expression exceeding `TUSC2_EXPRESSION_THRESHOLD_BINARY`. Binary classifications were stored as ordered categorical variables for consistent statistical analysis.

### Layered Analysis Approach

We implemented a five-layer analytical framework for TUSC2 functional characterization: (1) broad context expression patterns, (2) within-context subtype analysis, (3) functional signature impact assessment, (4) differential expression analysis, and (5) cross-context synthesis. This hierarchical approach enabled comprehensive evaluation of TUSC2's role across biological contexts and NK cell subtypes.

## Differential Expression Analysis

### Statistical Framework

Differential expression analysis was performed using `scanpy.tl.rank_genes_groups` with the Wilcoxon rank-sum test (`method='wilcoxon'`), Benjamini-Hochberg multiple testing correction (`corr_method='benjamini-hochberg'`), and percentage-based metrics (`pts=True`). All analyses utilized `use_raw=True` to ensure analysis of unscaled, log-normalized expression data.

### Gene Selection and Filtering

Differentially expressed genes were filtered using optimized criteria balancing specificity with biological relevance: adjusted p-value < 0.01 (stricter statistical threshold), log fold change > 0.5 (approximately 1.4-fold upregulation for high-confidence markers), and exclusion of non-informative gene patterns. The `select_optimal_subtype_markers()` function implemented intelligent marker selection with sophisticated conflict resolution to prevent marker overlap across subtypes while allowing for shared expression patterns among related developmental stages. A maximum of 4 highly selective markers per subtype were selected using composite scoring that incorporates statistical significance, effect size, and subtype specificity.

## Cross-Context Comparative Analysis

### Multi-Context Integration

Cross-context analysis integrated findings across blood, normal tissue, and tumor tissue environments using standardized analytical pipelines. Each context was processed independently using identical preprocessing and analysis parameters before comparative integration to maintain analytical consistency.

### Statistical Comparison Framework

Inter-context comparisons employed Mann-Whitney U tests for continuous variables and chi-square tests for categorical distributions. Multiple testing correction was applied using the Benjamini-Hochberg method across all comparisons within each analytical layer. Effect sizes were calculated as mean differences for continuous variables and reported with 95% confidence intervals.

## Visualization and Data Export

### Publication-Quality Figure Generation

All visualizations were generated at 300 DPI resolution using standardized color palettes and formatting. Heatmaps utilized the "icefire" diverging colormap centered at zero for fold-change visualization. Dynamic layout optimization was implemented through the `calculate_heatmap_layout()` function to accommodate variable signature name lengths and ensure optimal readability.

### Comprehensive Data Export

Analysis results were exported in GraphPad-compatible CSV format with complete statistical metadata including p-values, adjusted p-values, effect sizes, and sample sizes. Figure data and statistical results were systematically organized in hierarchical directory structures facilitating downstream analysis and manuscript preparation.

## Software and Computational Environment

### Core Dependencies

Analysis was performed using Python 3.13.2 with Scanpy 1.11.0, AnnData 0.11.3, pandas 2.0+, NumPy 1.24+, SciPy 1.10+, matplotlib 3.7+, and seaborn 0.12+. Statistical analyses utilized scikit-learn and statsmodels packages for advanced statistical methods. The custom AUCell-like scoring algorithm was implemented using NumPy operations with optimized sparse matrix handling, providing computational efficiency while maintaining theoretical foundations of the AUCell method.

### Hardware Acceleration

Computational analyses utilized GPU acceleration via PyTorch 2.6.0 with CUDA 12.4 support on NVIDIA RTX 4070 hardware (12.9 GB GPU memory). This configuration provided 10-100× acceleration for deep learning-based analyses while maintaining full reproducibility through fixed random seeds.

### Reproducibility Framework

All analyses incorporated comprehensive reproducibility measures including fixed random seeds (`RANDOM_SEED = 42`) applied to NumPy, Scanpy, PCA, UMAP, and neighborhood graph construction. Complete parameter documentation and version control ensured full analytical reproducibility across computational environments.

## Statistical Analysis

### Hypothesis Testing

Statistical comparisons employed non-parametric tests appropriate for single-cell expression data. Mann-Whitney U tests were used for continuous expression comparisons between groups, with Wilcoxon rank-sum tests for differential expression analysis. All tests utilized two-sided alternatives unless specifically indicated.

### Multiple Testing Correction

Multiple testing correction was systematically applied using the Benjamini-Hochberg false discovery rate (FDR) method implemented via `statsmodels.stats.multitest.multipletests`. Correction was applied within each analytical context to control family-wise error rates while maintaining statistical power for biological discovery.

### Effect Size Quantification

Effect sizes were calculated as mean differences for continuous variables and Cohen's d for standardized effect size interpretation. Significance was reported using standardized notation: * p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001, with "ns" indicating non-significant results.

### Sample Size Requirements

Statistical analyses required minimum sample sizes of 3 cells per group for valid comparisons. Analyses with insufficient sample sizes were excluded from statistical testing and marked as "N/A" in result tables. This conservative approach ensured robust statistical inference while acknowledging the inherent variability in single-cell datasets. 