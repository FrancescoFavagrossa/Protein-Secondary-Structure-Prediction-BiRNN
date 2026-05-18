library(tidyverse)
library(keras3)
library(randomForest)
library(ggplot2)
library(tictoc)
library(gridExtra)
set.seed(123)
data_raw <- read.csv("/Project/2022-12-17-pdb-intersect-pisces_pc30_r2.5.csv")
times_list <- list()

seq_lengths <- nchar(data_raw$seq)
perc_under_500 <- mean(seq_lengths < 500) * 100

ggplot(data_raw, aes(x = nchar(seq))) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_vline(xintercept = 500, color = "red", linetype = "dashed", linewidth = 1.5) +
  annotate("text", x = 500, y = Inf, 
           label = paste0(round(perc_under_500, 1), "% < 500"), 
           color = "red", vjust = 2, size = 5, fontface = "bold") +
  labs(title = "Distribution of Protein Sequence Lengths",
       x = "Sequence Length (amino acids)",
       y = "Frequency") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

data <- data_raw %>%
  filter(
    nchar(seq) >= 30,
    nchar(seq) <= 500,
    nchar(seq) == nchar(sst3),
    has_nonstd_aa == "False"
  ) %>%
  select(seq, sst3, len = len_x)

aminoacids <- c('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
aa_to_int <- setNames(seq_along(aminoacids), aminoacids)

seq_to_integers <- function(seq, aa_to_int) {
  chars <- strsplit(seq, "")[[1]]
  integers <- sapply(chars, function(aa) if (aa %in% names(aa_to_int)) aa_to_int[aa] else 0)
  return(unname(integers))
}

maxlen <- 500
sequences <- lapply(data$seq, seq_to_integers,  aa_to_int = aa_to_int)
x_all <- pad_sequences(sequences, maxlen = maxlen, padding = "post", truncating = "post")

sst_mapping <- c('C' = 1, 'H' = 2, 'E' = 3) 

sst_sequences <- lapply(data$sst3, function(s) {
  unname(sst_mapping[strsplit(s, "")[[1]]])
})

y_all <- pad_sequences(sst_sequences, maxlen = maxlen, padding = "post", value = 0)
y_all_categorical <- keras3::to_categorical(y_all, num_classes = 4) 



### Unidirectional RNN ###
model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = 21, output_dim = 64, 
                  input_length = maxlen, mask_zero = TRUE) %>%
  layer_lstm(units = 64, return_sequences = TRUE, dropout = 0.2) %>%
  time_distributed(
    layer_dense(units = 4, activation = "softmax")
  )

model_lstm %>% compile(
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

tic("LSTM")
history_lstm <- model_lstm %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)
times_list$LSTM <- toc()

model_gru <- keras_model_sequential() %>%
  layer_embedding(input_dim = 21, output_dim = 64, 
                  input_length = maxlen, mask_zero = TRUE) %>%
  layer_gru(units = 64, return_sequences = TRUE, dropout = 0.2) %>%
  time_distributed(
    layer_dense(units = 4, activation = "softmax")
  )

model_gru %>% compile(
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

tic("GRU")
history_gru <- model_gru %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)
times_list$GRU <- toc()

### BiDirectional RNN ###

model_bilstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = 21, output_dim = 64, 
                  input_length = maxlen, mask_zero = TRUE) %>%
  layer_bidirectional(
    layer_lstm(units = 64, return_sequences = TRUE, dropout = 0.2)
  ) %>%
  time_distributed(
    layer_dense(units = 4, activation = "softmax")
  )

model_bilstm %>% compile(
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

tic("BILSTM")
history_bilstm <- model_bilstm %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)
times_list$BILSTM <- toc()

model_bigru <- keras_model_sequential() %>%
  layer_embedding(input_dim = 21, output_dim = 64, 
                  input_length = maxlen, mask_zero = TRUE) %>%
  layer_bidirectional(
    layer_gru(units = 64, return_sequences = TRUE, dropout = 0.2)
  ) %>%
  time_distributed(
    layer_dense(units = 4, activation = "softmax")
  )

model_bigru %>% compile(
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

tic("BIGRU")
history_bigru <- model_bigru %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)
times_list$BIGRU <- toc()

Accuracies <- data.frame(Model = c("LSTM", "GRU", "BILSTM", "BIGRU"), `Accuracy` = c(score_lstm$acc, score_gru$acc, score_bilstm$acc, score_bigru$acc), `Run Time (min)` = sapply(times_list, function(x) (x$toc - x$tic)/60), check.names = FALSE)

p1 <- ggplot(Accuracies, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")), 
            hjust = -0.1, size = 4) +
  coord_flip() +
  ylim(0, max(Accuracies$Accuracy) * 1.1) +
  labs(title = "Accuracy", x = "", y = "") +
  theme_minimal() +
  theme(legend.position = "none")

p2 <- ggplot(Accuracies, aes(x = reorder(Model, Accuracy), y = `Run Time (min)`, fill = Model)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = paste0(round(`Run Time (min)`, 0), "min")), 
            hjust = -0.1, size = 4) +
  coord_flip() +
  ylim(0, max(Accuracies$`Run Time (min)`) * 1.1) +
  labs(title = "Training Time", x = "", y = "") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 2, top = "Model Comparison")