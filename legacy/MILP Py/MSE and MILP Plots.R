rm(list = ls(all = TRUE))
# packs
library(tidyverse)
library(xtable)

# Set to git folder
setwd("C:/Users/peram/Documents/test")

# Create an output directory
output_dir <- "C:/Users/peram/Documents/test/MILP Py/output_plots"

# Create the directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Post train residuals
df <- read_csv("Validation_Predictions_Clean_v2.csv")
colnames(df)
res <- df |> 
  mutate(residuals = actual_total_points - predicted_total_points) |> 
  mutate(ressqr = residuals^2)

# Create a residuals distribution plot by position
res <- res |> 
  mutate(residuals = actual_total_points - predicted_total_points) |> 
  mutate(ressqr = residuals^2)

# position enforce
res <- res |> 
  mutate(position = factor(position, levels = c("GK", "DEF", "MID", "FWD")))

# Dens plots
position_residuals_plot <- ggplot(res, aes(x = residuals, fill = position, color = position)) +
  geom_density(alpha = 0.2) +  # Changed to 0.2 to match EDA
  facet_wrap(~ position) +
  scale_color_brewer(palette = "Set1") +  # Added to match EDA
  scale_fill_brewer(palette = "Set1") +   # Added to match EDA
  labs(title = "Distribution of Prediction Residuals by Player Position",
       x = "Residuals (Actual - Predicted)",
       y = "Density") +
  theme_grey() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme(legend.position = "bottom")

print(position_residuals_plot)

# Save the plot to the output directory
ggsave(file.path(output_dir, "position_residuals_distribution.png"), 
       plot = position_residuals_plot, 
       width = 10, height = 8)

# Calculate MSE and RMSE by position
mse_rmse_by_position <- res |> 
  group_by(position) |> 
  summarize(
    n = n(),
    mean_residual = mean(residuals),  
    mse = mean(ressqr),
    rmse = sqrt(mean(ressqr)),
    mean_actual = mean(actual_total_points),
    mean_predicted = mean(predicted_total_points)
  ) |>
  ungroup()

# Calculate overall MSE and RMSE
overall_mse_rmse <- res |> 
  summarize(
    position = "ALL",
    n = n(),
    mean_residual = mean(residuals),  
    mse = mean(ressqr),
    rmse = sqrt(mean(ressqr)),
    mean_actual = mean(actual_total_points),
    mean_predicted = mean(predicted_total_points)
  )

# Combine position-specific and overall metrics
all_metrics <- bind_rows(mse_rmse_by_position, overall_mse_rmse) |>
  arrange(position != "ALL", mse)

# display
print(all_metrics)

# ovrlf table
metrics_table <- xtable::xtable(
  all_metrics,
  caption = "Post Training Metrics by Player Position",
  digits = c(0, 0, 0, 2, 2, 2, 2)
)
# Save the xtable output to a text file
capture.output(
  print(metrics_table, 
        include.rownames = FALSE,
        caption.placement = "top"),
  file = file.path(output_dir, "position_error_metrics.txt")
)

# Forecasted ----

setwd("C:/Users/peram/Documents/test/MILP Py")

# init frame
combined_data_lstm <- data.frame()

# Loop through the n files
for (i in 1:10) {
  file_name <- paste0("Squad Selection t-auto, W77-108,SHL", i, ".csv")
  temp_data <- read.csv(file_name)
  temp_data <- temp_data %>%
    select(gameweek, objective_gw, actual_total_points, alpha) %>%
    mutate(sub_horizon = paste0("SHL", i))
  combined_data_lstm <- rbind(combined_data_lstm, temp_data)
}

# FIX actual_total_points to be net pints
colnames(combined_data_lstm)
transfer_penalty <- 4
combined_data_lstm <- combined_data_lstm |> 
  mutate(actual_total_points = actual_total_points - alpha*transfer_penalty)

# Set Sub-horisnt order
horizon_numbers_lstm <- as.integer(stringr::str_extract(unique(combined_data_lstm$sub_horizon), "\\d+"))
ordered_levels_lstm <- paste0("SHL", sort(horizon_numbers_lstm))

combined_data_lstm <- combined_data_lstm %>%
  mutate(sub_horizon = factor(sub_horizon, levels = ordered_levels_lstm))


# Reshape for ggplot
plot_data_lstm <- combined_data_lstm %>%
  pivot_longer(cols = c(objective_gw, actual_total_points),
               names_to = "line_type",
               values_to = "points") %>%
  mutate(line_type = recode(line_type,
                            "objective_gw" = "Forecasted Squad Points",
                            "actual_total_points" = "Actual Squad Points"))  # FIXED: actual_lineup_points to actual_total_points

# Create the original line plot 
p1_lstm <- ggplot(plot_data_lstm, aes(x = gameweek, y = points, color = sub_horizon, linetype = line_type)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Gameweek Points by Sub-Horizon, Squads from forecasts with actual squad points",
       x = "Gameweek (GW)",
       y = "Points",
       color = "Sub-Horizon",
       linetype = "Line Type") +
  theme_grey() +
  theme(legend.position = "right")
print(p1_lstm)
ggsave(file.path(output_dir, "fantasy_points_plot_lstm.png"), plot = p1_lstm, width = 10, height = 6) # Increased width for legend

# CUMULATIVE PLOT FOR LSTM DATA ----
cumulative_plot_data_lstm <- combined_data_lstm %>%
  group_by(sub_horizon) %>%
  arrange(gameweek) %>%
  mutate(
    forecast_cumu = cumsum(objective_gw),
    actual_cumu = cumsum(actual_total_points)
  ) %>%
  ungroup()

# Create long format for cumulative plot with both metrics
cumulative_plot_data_lstm_long <- cumulative_plot_data_lstm %>%
  pivot_longer(
    cols = c(forecast_cumu, actual_cumu),
    names_to = "cumulative_type",
    values_to = "cumulative_points"
  ) %>%
  mutate(cumulative_type = recode(cumulative_type,
                                 "forecast_cumu" = "Cumulative Forecasted Squad Points",
                                 "actual_cumu" = "Cumulative Actual Squad Points"))

# Updated cumulative plot for LSTM
p_cumulative_lstm <- ggplot(cumulative_plot_data_lstm_long, 
       aes(x = gameweek, y = cumulative_points, color = sub_horizon, linetype = cumulative_type)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Cumulative Points by Sub-Horizon (LSTM)",
       x = "Gameweek (GW)",
       y = "Cumulative Points",
       color = "Sub-Horizon",
       linetype = "Point Type") +
  theme_grey() +
  theme(legend.position = "right")
print(p_cumulative_lstm)
ggsave(file.path(output_dir, "cumulative_fantasy_points_plot_lstm.png"), plot = p_cumulative_lstm, width = 10, height = 6)

# Extract end values for LSTM 
end_values_lstm <- cumulative_plot_data_lstm %>%
  group_by(sub_horizon) %>%
  filter(gameweek == max(gameweek)) %>%
  select(
    sub_horizon, 
    end_gameweek = gameweek, 
    final_forecasted_points = forecast_cumu,
    final_actual_points = actual_cumu
  ) %>%
  ungroup()
cat("\n--- End Cumulative Values for LSTM Data ---\n")

end_values_lstm_x <- print(xtable::xtable(end_values_lstm))


# Actual ----

combined_data_actual <- data.frame()

# Loop through the n files
for (i in 1:10) {
  file_name <- paste0("Squad Selection t-auto-actual, W77-108,SHL", i, ".csv")
  temp_data <- read.csv(file_name)
  temp_data <- temp_data %>%
    select(gameweek, actual_total_points, alpha) %>%
    mutate(sub_horizon = paste0("SHL", i))
  combined_data_actual <- rbind(combined_data_actual, temp_data)
}
# Hot fix 
combined_data_actual <- combined_data_actual |> 
  mutate(actual_total_points = actual_total_points - alpha*transfer_penalty)


# Extract numeric part and order
horizon_numbers_actual <- as.integer(stringr::str_extract(unique(combined_data_actual$sub_horizon), "\\d+"))
ordered_levels_actual <- paste0("SHL", sort(horizon_numbers_actual))

combined_data_actual <- combined_data_actual %>%
  mutate(sub_horizon = factor(sub_horizon, levels = ordered_levels_actual))


plot_data_actual <- combined_data_actual %>%
  pivot_longer(cols = actual_total_points,
               names_to = "line_type",
               values_to = "points") %>%
  mutate(line_type = recode(line_type,
                            "actual_total_points" = "Actual Points"))

p1_actual <- ggplot(plot_data_actual, aes(x = gameweek, y = points, color = sub_horizon, linetype = line_type)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Optimised Squad Points Gameweek by Sub-Horizon, with actual data",
       x = "Gameweek (GW)",
       y = "Points",
       color = "Sub-Horizon",
       linetype = "Line Type") +
  theme_grey() +
  theme(legend.position = "right")
print(p1_actual)
ggsave(file.path(output_dir, "fantasy_points_plot_actual.png"), plot = p1_actual, width = 10, height = 6)


#  CUMULATIVE PLOT FOR ACTUAL DATA ----
cumulative_plot_data_actual <- combined_data_actual %>% # Use already factor-ordered combined_data_actual
  group_by(sub_horizon) %>%
  arrange(gameweek) %>%
  mutate(forecast_cumu = cumsum(actual_total_points)) %>%
  ungroup()

p_cumulative_actual <- ggplot(cumulative_plot_data_actual, aes(x = gameweek, y = forecast_cumu, color = sub_horizon)) +
  geom_line(linewidth = 0.5) +
  labs(title = "Cumulative Optimal Squad Points by Sub-Horizon, with actual data",
       x = "Gameweek (GW)",
       y = "Cumulative Points",
       color = "Sub-Horizon") +
  theme_grey() +
  theme(legend.position = "right")
print(p_cumulative_actual)
ggsave(file.path(output_dir, "cumulative_fantasy_points_plot_actual.png"), plot = p_cumulative_actual, width = 10, height = 6)

end_values_actual <- cumulative_plot_data_actual %>%
  group_by(sub_horizon) %>%
  filter(gameweek == max(gameweek)) %>%
  select(sub_horizon, end_gameweek = gameweek, final_cumulative_points = forecast_cumu) %>%
  ungroup()
cat("\n--- End Cumulative Values for Actual Data ---\n")

end_values_actual_x <- print(xtable::xtable(end_values_actual))

# FORCED GAMECHIPS SUB in range(1,5) for predicted ----
combined_data_forced <- data.frame()

# Loop through the 5 files with forced gamechips
for (i in 1:5) {
  file_name <- paste0("Squad Selection t-auto-hard, W77-108,SHL", i, ".csv")
  temp_data <- read.csv(file_name)
  temp_data <- temp_data %>%
    select(gameweek, objective_gw, actual_total_points, alpha) %>%  # Added actual_total_points
    mutate(sub_horizon = paste0("SHL", i))
  combined_data_forced <- rbind(combined_data_forced, temp_data)
}
# Hot fix
combined_data_forced <- combined_data_forced |> 
  mutate(actual_total_points = actual_total_points - alpha*transfer_penalty)


# Set sub-horizon order for forced gamechips data
horizon_numbers_forced <- as.integer(stringr::str_extract(unique(combined_data_forced$sub_horizon), "\\d+"))
ordered_levels_forced <- paste0("SHL", sort(horizon_numbers_forced))

combined_data_forced <- combined_data_forced %>%
  mutate(sub_horizon = factor(sub_horizon, levels = ordered_levels_forced))

# Create line plot for forced gamechips data with both metrics
plot_data_forced <- combined_data_forced %>%
  pivot_longer(cols = c(objective_gw, actual_total_points),  # Added actual_total_points
               names_to = "line_type",
               values_to = "points") %>%
  mutate(line_type = recode(line_type,
                           "objective_gw" = "Forecasted Squad Points",
                           "actual_total_points" = "Actual Squad Points"))

p1_forced <- ggplot(plot_data_forced, aes(x = gameweek, y = points, color = sub_horizon, linetype = line_type)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Gameweek Points by Sub-Horizon (Forced Gamechips)",
       x = "Gameweek (GW)",
       y = "Points",
       color = "Sub-Horizon",
       linetype = "Line Type") +
  theme_grey() +
  theme(legend.position = "right")
print(p1_forced)
ggsave(file.path(output_dir, "fantasy_points_plot_forced.png"), plot = p1_forced, width = 10, height = 6)

# Create cumulative plot for forced gamechips data with both metrics
cumulative_plot_data_forced <- combined_data_forced %>%
  group_by(sub_horizon) %>%
  arrange(gameweek) %>%
  mutate(
    forecast_cumu = cumsum(objective_gw),
    actual_cumu = cumsum(actual_total_points)  # Added actual cumulative
  ) %>%
  ungroup()

# Create long format for cumulative plot
cumulative_plot_data_forced_long <- cumulative_plot_data_forced %>%
  pivot_longer(
    cols = c(forecast_cumu, actual_cumu),
    names_to = "cumulative_type",
    values_to = "cumulative_points"
  ) %>%
  mutate(cumulative_type = recode(cumulative_type,
                                 "forecast_cumu" = "Cumulative Forecasted Squad Points",
                                 "actual_cumu" = "Cumulative Actual Squad Points"))

p_cumulative_forced <- ggplot(cumulative_plot_data_forced_long, 
       aes(x = gameweek, y = cumulative_points, color = sub_horizon, linetype = cumulative_type)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Cumulative Points by Sub-Horizon (Forced Gamechips)",
       x = "Gameweek (GW)",
       y = "Cumulative Points",
       color = "Sub-Horizon",
       linetype = "Point Type") +
  theme_grey() +
  theme(legend.position = "right")
print(p_cumulative_forced)
ggsave(file.path(output_dir, "cumulative_fantasy_points_plot_forced.png"), plot = p_cumulative_forced, width = 10, height = 6)

# Extract end values for forced gamechips - include both metrics
end_values_forced <- cumulative_plot_data_forced %>%
  group_by(sub_horizon) %>%
  filter(gameweek == max(gameweek)) %>%
  select(
    sub_horizon, 
    end_gameweek = gameweek, 
    final_forecasted_points = forecast_cumu,
    final_actual_points = actual_cumu
  ) %>%
  ungroup()

# ACTUAL FORCED GAMECHIPS SUB in range(1,5)
combined_data_actual_forced <- data.frame()

# Loop through the 5 files with actual forced gamechips
for (i in 1:5) {
  file_name <- paste0("Squad Selection t-auto-forced-actual, W77-108,SHL", i, ".csv")
  temp_data <- read.csv(file_name)
  temp_data <- temp_data %>%
    select(gameweek, actual_total_points, alpha) %>%  # Only using actual points here
    mutate(sub_horizon = paste0("SHL", i))
  combined_data_actual_forced <- rbind(combined_data_actual_forced, temp_data)
}
# Hot fix
combined_data_actual_forced <- combined_data_actual_forced |> 
  mutate(actual_total_points = actual_total_points - alpha*transfer_penalty)

# Fix sub-horizon order for actual forced gamechips data
horizon_numbers_actual_forced <- as.integer(stringr::str_extract(unique(combined_data_actual_forced$sub_horizon), "\\d+"))
ordered_levels_actual_forced <- paste0("SHL", sort(horizon_numbers_actual_forced))

combined_data_actual_forced <- combined_data_actual_forced %>%
  mutate(sub_horizon = factor(sub_horizon, levels = ordered_levels_actual_forced))

# Create line plot for actual forced gamechips data
plot_data_actual_forced <- combined_data_actual_forced %>%
  pivot_longer(cols = actual_total_points,
               names_to = "line_type",
               values_to = "points") %>%
  mutate(line_type = recode(line_type, 
                           "actual_total_points" = "Actual Squad Points"))

p1_actual_forced <- ggplot(plot_data_actual_forced, aes(x = gameweek, y = points, color = sub_horizon, linetype = line_type)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Gameweek Points by Sub-Horizon (Actual Forced Gamechips)",
       x = "Gameweek (GW)",
       y = "Points",
       color = "Sub-Horizon",
       linetype = "Line Type") +
  theme_grey() +
  theme(legend.position = "right")
print(p1_actual_forced)
ggsave(file.path(output_dir, "fantasy_points_plot_actual_forced.png"), plot = p1_actual_forced, width = 10, height = 6)

# Create cumulative plot for actual forced gamechips data
cumulative_plot_data_actual_forced <- combined_data_actual_forced %>%
  group_by(sub_horizon) %>%
  arrange(gameweek) %>%
  mutate(actual_cumu = cumsum(actual_total_points)) %>%
  ungroup()

p_cumulative_actual_forced <- ggplot(cumulative_plot_data_actual_forced, aes(x = gameweek, y = actual_cumu, color = sub_horizon)) +
  geom_line(linewidth = 0.5) +
  labs(title = "Cumulative Actual Squad Points by Sub-Horizon (Actual Forced Gamechips)",
       x = "Gameweek (GW)",
       y = "Cumulative Actual Points",
       color = "Sub-Horizon") +
  theme_grey() +
  theme(legend.position = "right")
print(p_cumulative_actual_forced)
ggsave(file.path(output_dir, "cumulative_fantasy_points_plot_actual_forced.png"), plot = p_cumulative_actual_forced, width = 10, height = 6)

# Extract end values for actual forced gamechips
end_values_actual_forced <- cumulative_plot_data_actual_forced %>%
  group_by(sub_horizon) %>%
  filter(gameweek == max(gameweek)) %>%
  select(
    sub_horizon, 
    end_gameweek = gameweek, 
    final_actual_points = actual_cumu
  ) %>%
  ungroup()

# Update the combined comparison table to include the new model
all_end_values <- bind_rows(
  mutate(end_values_lstm, model_type = "LSTM"),
  mutate(end_values_actual %>% 
           rename(final_forecasted_points = final_cumulative_points) %>%
           mutate(final_actual_points = NA), 
         model_type = "Actual"),
  mutate(end_values_forced, model_type = "Forced"),
  # Add actual_forced with NA for forecasted points to match column structure
  mutate(end_values_actual_forced %>%
           mutate(final_forecasted_points = NA), 
         model_type = "Actual Forced")
) %>%
  # Reorder columns for better readability
  select(model_type, sub_horizon, end_gameweek, final_forecasted_points, final_actual_points)

# Create a comprehensive table
comprehensive_xtable <- xtable::xtable(
  all_end_values, 
  caption = "Comprehensive Comparison of All Models and Sub-Horizons",
  digits = c(0, 0, 0, 0, 0, 0)  # Format decimal places
)

# Save as a single text file w
capture.output(
  print(comprehensive_xtable, 
        include.rownames = FALSE,
        caption.placement = "bottom"), 
  file = file.path(output_dir, "comprehensive_model_comparison.txt")
)

# Export table
residual_xtable <- xtable(
  all_metrics, 
  caption = "Prediction Error Metrics by Position",
  digits = c(0, 0, 0, 2, 2, 2, 2,2)  # Added digit for new column
)

# Save the residual analysis to a separate file
capture.output(
  print(residual_xtable, 
        include.rownames = FALSE,
        caption.placement = "bottom"), 
  file = file.path(output_dir, "detailed_residual_analysis.txt")
  )
