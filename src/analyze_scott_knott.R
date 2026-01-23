# Load libraries
library(ScottKnottESD)
library(readr)
library(ggplot2)
library(reshape2)

# 1. Load the wide-format data
model_performance <- read_csv("data/model_performance_wide.csv")

# 2. Apply Scott-Knott ESD
# Note: sk_esd identifies statistically distinct groups based on mean and effect size
sk_results <- sk_esd(model_performance)

# 3. Prepare ranking dataframe
sk_ranks <- data.frame(
  model = names(sk_results$groups),
  rank = paste0('Rank-', sk_results$groups)
)

# 4. Prepare data for visualization
# Melt the wide data back to long format for ggplot
plot_data <- melt(model_performance)
colnames(plot_data) <- c("variable", "value")

# Merge with the ranks
plot_data <- merge(plot_data, sk_ranks, by.x = 'variable', by.y = 'model')

# 5. Generate the Visualization (Replicating XAI4SE Style)
g <- ggplot(data = plot_data, aes(x = reorder(variable, -value, FUN = mean), y = value, fill = rank)) +
     geom_boxplot(outlier.size = 0.5, alpha = 0.8) +
     # Scott-Knott usually focuses on the 0.5-1.0 range for AUC
     coord_cartesian(ylim = c(0.5, 1.0)) + 
     facet_grid(~rank, scales = 'free_x', space = 'free_x') +
     scale_fill_brewer(palette = "RdYlBu", direction = 1) +
     ylab('AUC-ROC Score') + 
     xlab('Classifiers') + 
     ggtitle('RQ3: Statistical Ranking of Model Performance (Agents-Only)') + 
     theme_bw() +
     theme(
       text = element_text(size = 14),
       axis.text.x = element_text(angle = 45, hjust = 1),
       legend.position = 'none',
       strip.background = element_rect(fill = "grey90"),
       panel.spacing = unit(0, "lines")
     )

# Save the plot
ggsave("visuals/rq3_scott_knott_auc.pdf", g, width = 10, height = 6)

# Print the ranking summary to console
print("Final Scott-Knott Ranking:")
print(sk_ranks[order(sk_ranks$rank), ])