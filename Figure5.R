library(ggplot2)
library(dplyr)

# Load the active‑learning data
df <- read.csv("/Users/uzairkhan/Downloads/active_learning_data.csv", stringsAsFactors = FALSE)

# Print the exact kernel names from the CSV to confirm they match
cat("Kernel names in CSV:\n")
print(unique(df$kernel))

# Build color and linetype maps dynamically from the actual CSV names
kernel_names <- unique(df$kernel)

kernel_colors <- c("#2166AC", "#D6604D", "#4DAF4A", "#984EA3")
names(kernel_colors) <- kernel_names

kernel_linetypes <- c("solid", "solid", "solid", "solid")
names(kernel_linetypes) <- kernel_names

# Manuscript‑style plot
p <- ggplot(df, aes(x = training_points, y = mean_accuracy,
                    color = kernel, fill = kernel, linetype = kernel)) +
  geom_ribbon(aes(ymin = mean_accuracy - std_accuracy,
                  ymax = mean_accuracy + std_accuracy),
              alpha = 0.18, colour = NA) +
  geom_line(linewidth = 1.15) +
  scale_colour_manual(values = kernel_colors) +
  scale_fill_manual(values = kernel_colors) +
  scale_linetype_manual(values = kernel_linetypes) +
  scale_x_continuous(breaks = seq(5, 20, 5)) +
  labs(x = "Number of training points",
       y = "Accuracy",
       title = "Active Learning for HEO Crystal‑Structure Classification") +
  theme_classic(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.7),
    axis.line = element_blank(),
    axis.ticks = element_line(linewidth = 0.6, colour = "black"),
    axis.ticks.length = unit(4, "pt"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 17, margin = margin(b = 8)),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12, colour = "black")
  )

ggsave("/Users/uzairkhan/Downloads/active_learning_no_random_R.png", p, width = 8, height = 5.5, dpi = 300)