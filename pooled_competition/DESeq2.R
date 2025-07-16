library("DESeq2")

counts <- read.csv("../data/pooled_comp/counts_db.csv", row.names='X')
counts[is.na(counts)] <- 0
samples <- c('T0A','T0B','T0C','T0D','T5A','T5B','T5C','T5D')
print(samples)
metadata <- read.csv("../data/pooled_comp/metadata.csv", row.names='sample')
dds <- DESeqDataSetFromMatrix(countData=counts, colData=metadata, design=~comp_pool + time)
        dds$time <- relevel(dds$time, ref='start')
#use precalculated normalization factors
normFactors <- read.csv('../data/pooled_comp/lin_log_normalization_factors.csv', header=FALSE)
colnames(normFactors) <- c('T0A','T0B','T0C','T0D','T5A','T5B','T5C','T5D')
normalizationFactors(dds) <- as.matrix(normFactors)
dds <- DESeq(dds)
res <- results(dds, alpha=0.01,  lfcThreshold=0.58, altHypothesis="greaterAbs")
write.csv(res, '../data/pooled_comp/deseq2_t0v5_l2fc058.csv')

#A threshold of 0.72% growth defect


