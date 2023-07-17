###### Confidence threshold tests ######
# Here we empirically determine the optimal confidence threshold for bounding 
# bounding box outputs, based on the test image set (Supplemental Figure 5).


args <- commandArgs(trailingOnly=TRUE)


library(tidyverse)

###### Importing the data ######
# Hand counted validation data
hand_validations <- read.csv(file="./test_set_hand_counts.tsv",
                               sep = '\t',
                               header = TRUE)

colnames(hand_validations)[1] <- "image_name"
hand_validations$image_name <- as.character(hand_validations$image_name)

print(args)

newModelPredictions2018 = read.csv(
  #"Inference/testingSetWarmanPaperX/InferenceOutput-03.06.23-12.55PM-022-07.11_16.22_0306/InferenceOutput-03.06.23-12.55PM-022-07.11_16.22.csv",
  args[1],
  header = TRUE,
  sep = ","
)

colnames(newModelPredictions2018)[1] <- "EarName"
newModelPredictions2018$EarName <- as.character(newModelPredictions2018$EarName)



#print(newModelPredictions2018)

###### Summarizing fluorescent/non-fluorescent R-squared values at different confidence thresholds ######
### Function for fluorescent kernels
get_fluor_r_squared <- function(input_df, hand_counts) {
  # Sub function that calculates the R-squared for a single threshold
  get_single_fluor_r_squared <- function(threshold, input_df, hand_counts) {
    
    # Create a dataframe  
    df <- input_df %>% 
      filter(AverageEarScoreFluor > threshold)

    
    df <- left_join(df, hand_counts, by=c("EarName" = "image_name"))
    #df2 <- df[complete.cases(df), ]

    
    # Returns 0 when there are no bounding boxes that satisfy a threshold
    if (nrow(df) == 0) {
      output <- data.frame("threshold" = threshold, "r_squared" = 0)
      return(output)
      
      # Runs the regression if bounding boxes are present
    } else {
      regression_r <- summary(lm(PredictedFluor ~ GFP_hand, data = df))$adj.r.squared
      output <- data.frame("threshold" = threshold, "r_squared" = regression_r)
      return(output)
    }
  }

  # This sets up a list, which will become a dataframe after it's populated by 
  # the following for loop
  df_list_fluor <- list()

  i = 1
  for (x in seq(0.07, 0.13, by=0.01)) {
    #print(x)
    output <- get_single_fluor_r_squared(x, input_df, hand_counts)
    df_list_fluor[[i]] <- output
    i = i + 1
  }
  
  fluor_r_squared <- bind_rows(df_list_fluor)
  return(fluor_r_squared)
}

### Function for nonfluorescent kernels
get_nonfluor_r_squared <- function(input_df, hand_counts) {
  # Sub function that calculates the R-squared for a single threshold
  get_single_nonfluor_r_squared <- function(threshold, input_df, hand_counts) {
    df <- input_df %>% 
      filter(AverageEarScoreNonFluor > threshold)
    
    
    df <- left_join(df, hand_counts, by=c("EarName" = "image_name"))
    #df2 <- df[complete.cases(df), ]
    
    # Returns 0 when there are no bounding boxes that satisfy a threshold
    if (nrow(df) == 0) {
      output <- data.frame("threshold" = threshold, "r_squared" = 0)
      return(output)
      
      # Runs the regression if bounding boxes are present
    } else {
      regression_r <- summary(lm(PredictedNonFluor ~ wt_hand, data = df))$adj.r.squared
      output <- data.frame("threshold" = threshold, "r_squared" = regression_r)
      return(output)
    }
  }
  
  # This sets up a list, which will become a dataframe after it's populated by 
  # the following for loop
  df_list_nonfluor <- list()
  i = 1
  for (x in seq(0.07, 0.13, by=0.01)) {
    #print(x)
    output <- get_single_nonfluor_r_squared(x, input_df, hand_counts)
    df_list_nonfluor[[i]] <- output
    i = i + 1
  }
  
  nonfluor_r_squared <- bind_rows(df_list_nonfluor)
  return(nonfluor_r_squared)
}

### Calculating fluorescent R-squared values
#fluor_r_squared_2018_2019 <- get_fluor_r_squared(model_predictions_2018_2019, hand_validations)
#fluor_r_squared_2018 <- get_fluor_r_squared(model_predictions_2018, hand_validations)
#fluor_r_squared_2019 <- get_fluor_r_squared(model_predictions_2019, hand_validations)
fluor_r_squared_2018 <- get_fluor_r_squared(newModelPredictions2018, hand_validations)
#fluor_r_squared_2019 <- get_fluor_r_squared(newModelPredictions2019, hand_validations)
print(args[1])
print("Fluor")
print(fluor_r_squared_2018)

#print("Y 2019")
#print(fluor_r_squared_2019)



### Calculating nonfluorescent R-squared values
#nonfluor_r_squared_2018_2019 <- get_nonfluor_r_squared(model_predictions_2018_2019, hand_validations)
nonfluor_r_squared_2018 <- get_nonfluor_r_squared(newModelPredictions2018, hand_validations)
#nonfluor_r_squared_2019 <- get_nonfluor_r_squared(model_predictions_2019, hand_validations)

print("NonFluor")
print(nonfluor_r_squared_2018)

#print("Y 2019")
#print(nonfluor_r_squared_2019)



###### Calculating the optimal confidence thresholds ######
# I will define the optimal confidence threshold as the confidence level that 
# maximizes the combined fluorescent and non-fluorescent R-squared.
#joined_df_2018_2019 <- data.frame(threshold = fluor_r_squared_2018_2019$threshold,
#                                  fluor_r_squared = fluor_r_squared_2018_2019$r_squared,
#                                  nonfluor_r_squared = nonfluor_r_squared_2018_2019$r_squared)
#joined_df_2018_2019$sum_r_squared <- joined_df_2018_2019$fluor_r_squared + joined_df_2018_2019$nonfluor_r_squared
#joined_df_2018_2019 <- joined_df_2018_2019[complete.cases(joined_df_2018_2019), ]
#max_r_squared_2018_2019 <- joined_df_2018_2019$threshold[joined_df_2018_2019$sum_r_squared == max(joined_df_2018_2019$sum_r_squared)]

#joined_df_2018 <- data.frame(threshold = fluor_r_squared_2018$threshold,
#                             fluor_r_squared = fluor_r_squared_2018$r_squared,
#                             nonfluor_r_squared = nonfluor_r_squared_2018$r_squared)
#joined_df_2018$sum_r_squared <- joined_df_2018$fluor_r_squared + joined_df_2018$nonfluor_r_squared
#joined_df_2018 <- joined_df_2018[complete.cases(joined_df_2018), ]
#max_r_squared_2018 <- joined_df_2018$threshold[joined_df_2018$sum_r_squared == max(joined_df_2018$sum_r_squared)]

#joined_df_2019 <- data.frame(threshold = fluor_r_squared_2019$threshold,
#                             fluor_r_squared = fluor_r_squared_2019$r_squared,
#                             nonfluor_r_squared = nonfluor_r_squared_2019$r_squared)
#joined_df_2019$sum_r_squared <- joined_df_2019$fluor_r_squared + joined_df_2019$nonfluor_r_squared
#joined_df_2019 <- joined_df_2019[complete.cases(joined_df_2019), ]
#max_r_squared_2019 <- joined_df_2019$threshold[joined_df_2019$sum_r_squared == max(joined_df_2019$sum_r_squared)]


###### Plotting (Supplemental Figure 5) ######
# Function to make the data tidy for plotting
#tidy_data_for_plot <- function(input_df) {
#  to_be_tidied <- input_df[ , 1:3]
#  colnames(to_be_tidied) <- c("Threshold", "Fluorescent", "Non-fluorescent")
#  tidy_joined_df <- gather(to_be_tidied, "Class", "R_squared", 2:3)
#  return(tidy_joined_df)
#}

# Plotting 2018/2019 data
#plot_2018_2019 <- tidy_data_for_plot(joined_df_2018_2019)

#ggplot(plot_2018_2019, aes(x = Threshold, y = R_squared, group = Class)) +
#  geom_point(aes(shape = Class), size = 2) +
#  scale_shape_manual(values = c(19, 4)) +
#  geom_vline(xintercept = max_r_squared_2018_2019, linetype = 'dashed', size = 1) +
#  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
#  labs(title = 'Optimal confidence threshold 2018/2019', x = 'Threshold', y = 'Adj. R-squared') +
#  theme_bw() +
#  theme(axis.title = element_text(size = 24, face = 'bold'),
#        axis.text = element_text(size = 18, face = 'bold'),
#        plot.title = element_text(hjust = 0.5, size = 28, face = 'bold', margin = margin(0, 0, 10, 0)),
#        axis.title.x = element_text(margin = margin(10, 0, 0, 0)),
#        axis.title.y = element_text(margin = margin(0, 10, 0, 0)),
#        axis.line = element_line(size = 0, color = '#4D4D4D'),
#        axis.ticks = element_line(size = 0.75, color = '#4D4D4D'),
#        axis.ticks.length = unit(4, 'pt'),
#        plot.margin = margin(0.5, 0.5, 0.5, 0.5, 'cm'),
#        panel.border = element_rect(color = '#4D4D4D', size = 2, fill = NA),
#        panel.grid.major.y = element_line(size = 0.75),
#        panel.grid.minor.y = element_line(size = 0.5),
#        panel.grid.major.x = element_line(size = 0.75),
#        panel.grid.minor.x = element_line(size = 0.5),
#        legend.position = 'none')

#ggsave(filename = './plots/optimal_confidence_thresholds_2018_2019.png',
#       device = 'png',
#       width = 9,
#       height = 8,
#       dpi = 400,
#      units = 'in')
#
# Plotting 2018 data
#plot_2018 <- tidy_data_for_plot(joined_df_2018)

#ggplot(plot_2018, aes(x = Threshold, y = R_squared, group = Class)) +
#  geom_point(aes(shape = Class), size = 2) +
#  scale_shape_manual(values = c(19, 4)) +
#  geom_vline(xintercept = max_r_squared_2018, linetype = 'dashed', size = 1) +
#  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
#  labs(title = 'Optimal confidence threshold 2018', x = 'Threshold', y = 'Adj. R-squared') +
#  theme_bw() +
#  theme(axis.title = element_text(size = 24, face = 'bold'),
#        axis.text = element_text(size = 18, face = 'bold'),
#        plot.title = element_text(size = 28, face = 'bold', margin = margin(0, 0, 10, 0)),
#        axis.title.x = element_text(margin = margin(10, 0, 0, 0)),
#        axis.title.y = element_text(margin = margin(0, 10, 0, 0)),
#        axis.line = element_line(size = 0, color = '#4D4D4D'),
#        axis.ticks = element_line(size = 0.75, color = '#4D4D4D'),
#        axis.ticks.length = unit(4, 'pt'),
#        plot.margin = margin(0.5, 0.5, 0.5, 0.5, 'cm'),
#        panel.border = element_rect(color = '#4D4D4D', size = 2, fill = NA),
#        panel.grid.major.y = element_line(size = 0.75),
#        panel.grid.minor.y = element_line(size = 0.5),
#        panel.grid.major.x = element_line(size = 0.75),
#        panel.grid.minor.x = element_line(size = 0.5),
#        legend.position = 'none')

#ggsave(filename = './plots/optimal_confidence_thresholds_2018.png',
#       device = 'png',
#       width = 9,
#       height = 8,
#       dpi = 400,
#       units = 'in')

# Plotting 2019 data
#plot_2019 <- tidy_data_for_plot(joined_df_2019)

#ggplot(plot_2019, aes(x = Threshold, y = R_squared, group = Class)) +
#  geom_point(aes(shape = Class), size = 2) +
#  scale_shape_manual(values = c(19, 4)) +
#  geom_vline(xintercept = max_r_squared_2019, linetype = 'dashed', size = 1) +
#  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
#  labs(title = 'Optimal confidence threshold 2019', x = 'Threshold', y = 'Adj. R-squared') +
#  theme_bw() +
#  theme(axis.title = element_text(size = 24, face = 'bold'),
#        axis.text = element_text(size = 18, face = 'bold'),
#        plot.title = element_text(size = 28, face = 'bold', margin = margin(0, 0, 10, 0)),
#        axis.title.x = element_text(margin = margin(10, 0, 0, 0)),
#        axis.title.y = element_text(margin = margin(0, 10, 0, 0)),
#        axis.line = element_line(size = 0, color = '#4D4D4D'),
#        axis.ticks = element_line(size = 0.75, color = '#4D4D4D'),
#        axis.ticks.length = unit(4, 'pt'),
#        plot.margin = margin(0.5, 0.5, 0.5, 0.5, 'cm'),
#        panel.border = element_rect(color = '#4D4D4D', size = 2, fill = NA),
#        panel.grid.major.y = element_line(size = 0.75),
#        panel.grid.minor.y = element_line(size = 0.5),
#        panel.grid.major.x = element_line(size = 0.75),
#        panel.grid.minor.x = element_line(size = 0.5),
#        legend.position = 'right',
#        legend.text = element_text(size = 16, face = 'bold'),
#        legend.title = element_text(size = 18, face = 'bold'))

#ggsave(filename = './plots/optimal_confidence_thresholds_2019.png',
#       device = 'png',
#       width = 10,
#       height = 8,
#       dpi = 400,
#       units = 'in')