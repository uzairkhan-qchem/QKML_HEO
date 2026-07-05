# ============================================================
# Figure 4a:
#   Absolute test accuracy on the fluorite-vs-bixbyite task
# Figure 4b:
#   Relative accuracy, quantum/classical, with 95% Katz confidence intervals
# Figure 4c:
#   Faceted relative-accuracy panels, Adams-style
# Supplementary Figure S1:
#   Individual ratio panels, one classical baseline per panel
# ============================================================

# Install once if needed:
# install.packages(c("reticulate", "ggplot2", "dplyr", "tidyr", "purrr", "readr", "patchwork"))

library(reticulate)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(patchwork)

# ------------------------------------------------------------
# 1. Input/output paths
# ------------------------------------------------------------

input_file <- "/Users/uzairkhan/Downloads/heo_pooled_results_32.npz"
output_dir <- "/Users/uzairkhan/Downloads/heo_figures_Pooled_data"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# 2. Load NumPy pooled dataset through reticulate
# ------------------------------------------------------------

np <- import("numpy", convert = FALSE)
builtins <- import_builtins(convert = TRUE)

npz <- np$load(input_file, allow_pickle = TRUE)

train_sizes <- py_to_r(py_get_item(npz, "train_sizes"))
train_sizes <- as.integer(train_sizes)

# Pooled dataset structure:
# pooled is a Python dictionary:
# pooled[[model_name]][[N_train]] = vector of run-level accuracies
#
# In heo_pooled_results_32(1).npz, the expected top-level keys are:
#   train_sizes
#   pooled
pooled <- py_get_item(npz, "pooled")$item()

all_model_keys <- builtins$list(pooled$keys())

# Robustly identify model keys.
# This handles small spelling/Unicode differences such as "Cosine‑dist exp".
quantum_key <- all_model_keys[grepl("Quantum", all_model_keys)][1]
arbf_key    <- all_model_keys[grepl("Angular", all_model_keys)][1]
cde_key     <- all_model_keys[grepl("Cosine", all_model_keys)][1]
g1_key      <- all_model_keys[grepl("Gaussian", all_model_keys)][1]

if (any(is.na(c(quantum_key, arbf_key, cde_key, g1_key)))) {
  stop(
    paste0(
      "Could not robustly identify all model keys.\n",
      "Available keys are:\n",
      paste(all_model_keys, collapse = "\n")
    )
  )
}

# Helper: extract run-level accuracy vector for one model and one N_train
get_runs <- function(model_key, n_train) {
  model_dict <- py_get_item(pooled, model_key)
  vals <- py_get_item(model_dict, as.integer(n_train))
  as.numeric(py_to_r(vals))
}

# ------------------------------------------------------------
# 3. Convert pooled dataset into long tidy format
# ------------------------------------------------------------

model_info <- tibble(
  model_key = c(quantum_key, arbf_key, cde_key, g1_key),
  model_name = c(
    "Quantum (sim.)",
    "Angular RBF",
    "Cosine-distance exp.",
    "Gaussian RBF (l=1)"
  )
)

raw_df <- map_dfr(model_info$model_key, function(k) {
  map_dfr(train_sizes, function(n) {
    runs <- get_runs(k, n)

    tibble(
      model_key = k,
      N_train = n,
      run_id = seq_along(runs),
      accuracy = runs
    )
  })
}) %>%
  left_join(model_info, by = "model_key")

summary_df <- raw_df %>%
  group_by(model_key, model_name, N_train) %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    sd_accuracy = sd(accuracy, na.rm = TRUE),
    n_runs = n(),
    .groups = "drop"
  )

write_csv(
  raw_df,
  file.path(output_dir, "heo_pooled_run_level_accuracy_R.csv")
)

write_csv(
  summary_df,
  file.path(output_dir, "heo_pooled_accuracy_summary_R.csv")
)

# ------------------------------------------------------------
# 4. Adams-like ggplot theme with boxed axes
# ------------------------------------------------------------

theme_adams <- function(base_size = 13) {
  theme_classic(base_size = base_size) +
    theme(
      panel.grid = element_blank(),

      panel.border = element_rect(
        colour = "black",
        fill = NA,
        linewidth = 0.7
      ),

      axis.line = element_blank(),

      axis.ticks = element_line(
        linewidth = 0.6,
        colour = "black"
      ),
      axis.ticks.length = unit(4, "pt"),

      plot.title = element_text(
        size = 17,
        face = "plain",
        hjust = 0.5,
        margin = margin(b = 8)
      ),

      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12, colour = "black"),

      legend.title = element_blank(),
      legend.text = element_text(size = 12),
      legend.position = "bottom",

      strip.background = element_blank(),
      strip.text = element_text(size = 15)
    )
}

# ------------------------------------------------------------
# 5. Shared x-axis helpers
# ------------------------------------------------------------

x_min <- min(train_sizes)
x_max <- max(train_sizes)

x_breaks_full <- c(5, 10, 15, 20, 25, 29, 33)

scale_x_train_full <- function() {
  scale_x_continuous(
    limits = c(x_min, x_max),
    breaks = x_breaks_full,
    expand = expansion(mult = c(0, 0), add = c(0, 0))
  )
}

# For stable range: total N = 32, so N_train <= 27 leaves at least 5 test points
total_N <- 32L
stable_max <- min(x_max - 5, max(train_sizes))
stable_train_sizes <- train_sizes[train_sizes <= stable_max]
x_breaks_stable <- unique(c(seq(min(stable_train_sizes), max(stable_train_sizes), by = 4),
                            max(stable_train_sizes)))

scale_x_train_stable <- function() {
  scale_x_continuous(
    limits = c(min(stable_train_sizes), max(stable_train_sizes)),
    breaks = x_breaks_stable,
    expand = expansion(mult = c(0, 0), add = c(0, 0))
  )
}

# ------------------------------------------------------------
# 6. Color palettes
# ------------------------------------------------------------

model_palette <- c(
  "Quantum (sim.)" = "#0072B2",
  "Angular RBF" = "#D55E00",
  "Cosine-distance exp." = "#009E73",
  "Gaussian RBF (l=1)" = "#CC79A7"
)

ratio_palette <- c(
  "Quantum vs. Angular RBF" = "#D55E00",
  "Quantum vs. Cosine-distance exp." = "#009E73",
  "Quantum vs. Gaussian RBF (l=1)" = "#CC79A7"
)

# ------------------------------------------------------------
# 7. Figure 4a:
# Absolute accuracy multiple line chart with uncertainty bands
# ------------------------------------------------------------
#
# Title:
#   HEO Crystal-Structure Classification: Absolute Accuracy
#
# Description:
#   Mean test accuracy of Bernoulli Gaussian process classifiers as a
#   function of training-set size for the quantum kernel and three
#   classical baselines. Shaded bands indicate ±1 standard deviation over
#   75 train-test splits pooled across three random seeds.

fig4a <- ggplot(
  summary_df,
  aes(
    x = N_train,
    y = mean_accuracy,
    colour = model_name,
    fill = model_name,
    group = model_name
  )
) +
  geom_ribbon(
    aes(
      ymin = mean_accuracy - sd_accuracy,
      ymax = mean_accuracy + sd_accuracy
    ),
    alpha = 0.18,
    colour = NA
  ) +
  geom_line(linewidth = 1.15) +
  scale_colour_manual(values = model_palette) +
  scale_fill_manual(values = model_palette) +
  labs(
    title = "HEO Crystal-Structure Classification",
    x = "Number of training points",
    y = "Accuracy"
  ) +
  coord_cartesian(ylim = c(0.4, 0.75)) +
  scale_x_train_full() +
  theme_adams()

ggsave(
  filename = file.path(output_dir, "fig4a_heo_absolute_accuracy_R.png"),
  plot = fig4a,
  width = 8.0,
  height = 4.5,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "fig4a_heo_absolute_accuracy_R.pdf"),
  plot = fig4a,
  width = 8.0,
  height = 4.5
)

# ------------------------------------------------------------
# 8. Build quantum/classical ratio data with 95% Katz intervals
# ------------------------------------------------------------
#
# Katz confidence interval for a risk ratio:
#
#   RR = p_Q / p_C
#
#   SE(log RR) = sqrt(1/x_Q - 1/n_Q + 1/x_C - 1/n_C)
#
#   CI_95 = exp(log(RR) ± 1.96 * SE(log RR))
#
# Here, p_Q and p_C are pooled test accuracies.
# Because the .npz stores accuracies rather than explicit correct counts,
# correct counts are reconstructed as:
#
#   x = round(accuracy * n_test)
#
# where:
#
#   n_test = 32 - N_train

katz_ratio_ci <- function(q_success, q_trials, c_success, c_trials, z = 1.96) {

  # Boundary correction for zero or complete-success cells.
  # This keeps the log-risk-ratio CI finite in edge cases.
  boundary_case <- (
    q_success == 0 || c_success == 0 ||
      q_success == q_trials || c_success == c_trials
  )

  if (boundary_case) {
    q_success <- q_success + 0.5
    c_success <- c_success + 0.5
    q_trials <- q_trials + 1.0
    c_trials <- c_trials + 1.0
  }

  p_q <- q_success / q_trials
  p_c <- c_success / c_trials

  rr <- p_q / p_c

  se_log_rr <- sqrt(
    (1 / q_success) - (1 / q_trials) +
      (1 / c_success) - (1 / c_trials)
  )

  lower <- exp(log(rr) - z * se_log_rr)
  upper <- exp(log(rr) + z * se_log_rr)

  tibble(
    quantum_successes = q_success,
    quantum_trials = q_trials,
    classical_successes = c_success,
    classical_trials = c_trials,
    quantum_accuracy_pooled = p_q,
    classical_accuracy_pooled = p_c,
    mean_ratio = rr,
    katz_lower_95 = lower,
    katz_upper_95 = upper,
    se_log_ratio = se_log_rr
  )
}

make_katz_ratio_df <- function(classical_key, comparison_name) {
  map_dfr(train_sizes, function(n) {
    q <- get_runs(quantum_key, n)
    c <- get_runs(classical_key, n)

    m <- min(length(q), length(c))
    q <- q[seq_len(m)]
    c <- c[seq_len(m)]

    n_test <- total_N - as.integer(n)

    q_success <- sum(round(q * n_test), na.rm = TRUE)
    c_success <- sum(round(c * n_test), na.rm = TRUE)

    q_trials <- m * n_test
    c_trials <- m * n_test

    katz_ratio_ci(
      q_success = q_success,
      q_trials = q_trials,
      c_success = c_success,
      c_trials = c_trials
    ) %>%
      mutate(
        classical_key = classical_key,
        comparison = comparison_name,
        N_train = n,
        n_test = n_test,
        n_runs = m
      )
  })
}

ratio_summary_df <- bind_rows(
  make_katz_ratio_df(arbf_key, "Quantum vs. Angular RBF"),
  make_katz_ratio_df(cde_key,  "Quantum vs. Cosine-distance exp."),
  make_katz_ratio_df(g1_key,   "Quantum vs. Gaussian RBF (l=1)")
) %>%
  select(
    classical_key,
    comparison,
    N_train,
    n_test,
    n_runs,
    quantum_successes,
    quantum_trials,
    classical_successes,
    classical_trials,
    quantum_accuracy_pooled,
    classical_accuracy_pooled,
    mean_ratio,
    katz_lower_95,
    katz_upper_95,
    se_log_ratio
  )

write_csv(
  ratio_summary_df,
  file.path(output_dir, "heo_pooled_quantum_classical_ratio_katz_summary_R.csv")
)

# Optional run-level ratio table for diagnostics only.
# The Katz plots below use pooled count-based ratios, not mean split-wise ratios.
make_run_level_ratio_df <- function(classical_key, comparison_name) {
  map_dfr(train_sizes, function(n) {
    q <- get_runs(quantum_key, n)
    c <- get_runs(classical_key, n)

    m <- min(length(q), length(c))
    q <- q[seq_len(m)]
    c <- c[seq_len(m)]

    ratio <- q / c
    ratio <- ratio[is.finite(ratio)]

    tibble(
      classical_key = classical_key,
      comparison = comparison_name,
      N_train = n,
      run_id = seq_along(ratio),
      ratio = ratio
    )
  })
}

ratio_raw_df <- bind_rows(
  make_run_level_ratio_df(arbf_key, "Quantum vs. Angular RBF"),
  make_run_level_ratio_df(cde_key,  "Quantum vs. Cosine-distance exp."),
  make_run_level_ratio_df(g1_key,   "Quantum vs. Gaussian RBF (l=1)")
)

write_csv(
  ratio_raw_df,
  file.path(output_dir, "heo_pooled_run_level_quantum_classical_ratios_R.csv")
)

# ------------------------------------------------------------
# 9. Figure 4b:
# Relative accuracy multiple line chart with 95% Katz confidence intervals
# ------------------------------------------------------------
#
# Title:
#   Quantum Advantage in Few-Shot Classification
#
# Description:
#   Ratio of quantum-kernel test accuracy to classical-kernel test accuracy
#   as a function of training-set size. The horizontal dashed line marks
#   equality. Values above 1 indicate a quantum advantage. Shaded bands
#   represent 95% Katz confidence intervals.

fig4b <- ggplot(
  ratio_summary_df,
  aes(
    x = N_train,
    y = mean_ratio,
    colour = comparison,
    fill = comparison,
    group = comparison
  )
) +
  geom_hline(
    yintercept = 1,
    linewidth = 0.55,
    alpha = 0.7,
    linetype = "dashed"
  ) +
  geom_ribbon(
    aes(
      ymin = katz_lower_95,
      ymax = katz_upper_95
    ),
    alpha = 0.18,
    colour = NA
  ) +
  geom_line(linewidth = 1.15) +
  scale_colour_manual(values = ratio_palette) +
  scale_fill_manual(values = ratio_palette) +
  labs(
    title = "Quantum Accuracy Relative to Classical Baselines",
    x = "Number of training points",
    y = "Quantum / classical accuracy"
  ) +
  coord_cartesian(ylim = c(0.5, 1.5)) +
  scale_x_train_full() +
  theme_adams()

ggsave(
  filename = file.path(output_dir, "fig4b_heo_quantum_advantage_katz_ratio_R.png"),
  plot = fig4b,
  width = 9.0,
  height = 4.5,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "fig4b_heo_quantum_advantage_katz_ratio_R.pdf"),
  plot = fig4b,
  width = 9.0,
  height = 4.5
)

# ------------------------------------------------------------
# 10. Figure 4c:
# Faceted relative accuracy multiple line chart, Adams-style
# ------------------------------------------------------------
#
# Title:
#   Quantum Advantage Across Classical Baselines
#
# Description:
#   Three side-by-side panels showing the relative accuracy of the quantum
#   kernel against each classical baseline individually.

fig4c <- ggplot(
  ratio_summary_df,
  aes(
    x = N_train,
    y = mean_ratio,
    colour = comparison,
    fill = comparison,
    group = comparison
  )
) +
  geom_hline(
    yintercept = 1,
    linewidth = 0.55,
    alpha = 0.7,
    linetype = "dashed"
  ) +
  geom_ribbon(
    aes(
      ymin = katz_lower_95,
      ymax = katz_upper_95
    ),
    alpha = 0.22,
    colour = NA
  ) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ comparison, nrow = 1) +
  scale_colour_manual(values = ratio_palette) +
  scale_fill_manual(values = ratio_palette) +
  labs(
    title = "Quantum Advantage Across Classical Baselines",
    x = "Number of training points",
    y = "Quantum / classical accuracy"
  ) +
  coord_cartesian(ylim = c(0.5, 1.5)) +
  scale_x_train_full() +
  theme_adams() +
  theme(
    legend.position = "none"
  )

ggsave(
  filename = file.path(output_dir, "fig4c_heo_quantum_advantage_faceted_R.png"),
  plot = fig4c,
  width = 10.5,
  height = 3.8,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "fig4c_heo_quantum_advantage_faceted_R.pdf"),
  plot = fig4c,
  width = 10.5,
  height = 3.8
)

# ------------------------------------------------------------
# 11. Supplementary Figure S1:
# Individual ratio panels with 95% Katz confidence intervals
# ------------------------------------------------------------

save_single_ratio_panel <- function(comparison_name, panel_title, filename_stub) {

  plot_df <- ratio_summary_df %>%
    filter(comparison == comparison_name)

  if (nrow(plot_df) == 0) {
    stop(
      paste0(
        "No rows found for comparison_name = '", comparison_name, "'.\n",
        "Available comparison names are:\n",
        paste(unique(ratio_summary_df$comparison), collapse = "\n")
      )
    )
  }

  line_col <- ratio_palette[[comparison_name]]

  p <- plot_df %>%
    ggplot(
      aes(
        x = N_train,
        y = mean_ratio
      )
    ) +
    geom_hline(
      yintercept = 1,
      linewidth = 0.55,
      alpha = 0.7,
      linetype = "dashed"
    ) +
    geom_ribbon(
      aes(
        ymin = katz_lower_95,
        ymax = katz_upper_95
      ),
      alpha = 0.22,
      fill = line_col,
      colour = NA
    ) +
    geom_line(linewidth = 1.25, colour = line_col) +
    labs(
      title = panel_title,
      x = "Number of training points",
      y = "Quantum / classical accuracy"
    ) +
    coord_cartesian(ylim = c(0.5, 1.5)) +
    scale_x_train_full() +
    theme_adams() +
    theme(
      legend.position = "none"
    )

  ggsave(
    filename = file.path(output_dir, paste0(filename_stub, ".png")),
    plot = p,
    width = 4.4,
    height = 3.8,
    dpi = 300
  )

  ggsave(
    filename = file.path(output_dir, paste0(filename_stub, ".pdf")),
    plot = p,
    width = 4.4,
    height = 3.8
  )

  return(p)
}

s1_arbf <- save_single_ratio_panel(
  comparison_name = "Quantum vs. Angular RBF",
  panel_title = "Quantum vs. Angular RBF",
  filename_stub = "figS1a_heo_quantum_vs_angular_rbf_katz_R"
)

s1_cde <- save_single_ratio_panel(
  comparison_name = "Quantum vs. Cosine-distance exp.",
  panel_title = "Quantum vs. Cosine-distance exp.",
  filename_stub = "figS1b_heo_quantum_vs_cosine_distance_exp_katz_R"
)

s1_g1 <- save_single_ratio_panel(
  comparison_name = "Quantum vs. Gaussian RBF (l=1)",
  panel_title = "Quantum vs. Gaussian RBF (l=1)",
  filename_stub = "figS1c_heo_quantum_vs_gaussian_rbf_l1_katz_R"
)

figS1_combined <- s1_arbf + s1_cde + s1_g1 +
  plot_layout(ncol = 3)

ggsave(
  filename = file.path(output_dir, "figS1_combined_individual_ratio_panels_katz_R.png"),
  plot = figS1_combined,
  width = 12.4,
  height = 3.8,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "figS1_combined_individual_ratio_panels_katz_R.pdf"),
  plot = figS1_combined,
  width = 12.4,
  height = 3.8
)

# ------------------------------------------------------------
# 12. Optional stable-range figures
# N_train <= 27 leaves at least 5 test points out of total 32
# ------------------------------------------------------------

ratio_stable_df <- ratio_summary_df %>%
  filter(N_train %in% stable_train_sizes)

accuracy_stable_df <- summary_df %>%
  filter(N_train %in% stable_train_sizes)

# Stable-range relative accuracy multiple line chart
figS2_ratio_stable <- ggplot(
  ratio_stable_df,
  aes(
    x = N_train,
    y = mean_ratio,
    colour = comparison,
    fill = comparison,
    group = comparison
  )
) +
  geom_hline(
    yintercept = 1,
    linewidth = 0.55,
    alpha = 0.7,
    linetype = "dashed"
  ) +
  geom_ribbon(
    aes(
      ymin = katz_lower_95,
      ymax = katz_upper_95
    ),
    alpha = 0.18,
    colour = NA
  ) +
  geom_line(linewidth = 1.15) +
  scale_colour_manual(values = ratio_palette) +
  scale_fill_manual(values = ratio_palette) +
  labs(
    title = "Quantum/Classical Ratio: Stable Test-Set Range",
    x = "Number of training points",
    y = "Quantum / classical accuracy"
  ) +
  coord_cartesian(ylim = c(0, 2.0)) +
  scale_x_train_stable() +
  theme_adams()

ggsave(
  filename = file.path(output_dir, "figS2_heo_quantum_ratio_stable_range_katz_R.png"),
  plot = figS2_ratio_stable,
  width = 7.5,
  height = 4.5,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "figS2_heo_quantum_ratio_stable_range_katz_R.pdf"),
  plot = figS2_ratio_stable,
  width = 7.5,
  height = 4.5
)

# Stable-range absolute accuracy multiple line chart
figS3_accuracy_stable <- ggplot(
  accuracy_stable_df,
  aes(
    x = N_train,
    y = mean_accuracy,
    colour = model_name,
    fill = model_name,
    group = model_name
  )
) +
  geom_ribbon(
    aes(
      ymin = mean_accuracy - sd_accuracy,
      ymax = mean_accuracy + sd_accuracy
    ),
    alpha = 0.18,
    colour = NA
  ) +
  geom_line(linewidth = 1.15) +
  scale_colour_manual(values = model_palette) +
  scale_fill_manual(values = model_palette) +
  labs(
    title = "HEO Crystal-Structure Classification: Stable Test-Set Range",
    x = "Number of training points",
    y = "Test accuracy"
  ) +
  coord_cartesian(ylim = c(0, 1.25)) +
  scale_x_train_stable() +
  theme_adams()

ggsave(
  filename = file.path(output_dir, "figS3_heo_absolute_accuracy_stable_range_R.png"),
  plot = figS3_accuracy_stable,
  width = 7.2,
  height = 4.5,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "figS3_heo_absolute_accuracy_stable_range_R.pdf"),
  plot = figS3_accuracy_stable,
  width = 7.2,
  height = 4.5
)

# ------------------------------------------------------------
# 13. Print completion message
# ------------------------------------------------------------

message("Done. Figures and CSV summaries saved in: ", output_dir)
message("Detected model keys:")
message("  Quantum: ", quantum_key)
message("  Angular RBF: ", arbf_key)
message("  Cosine-distance exp.: ", cde_key)
message("  Gaussian RBF: ", g1_key)
