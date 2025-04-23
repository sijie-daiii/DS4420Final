set.seed(123)

library(tidyverse)
library(caret)
library(Metrics)
library(nnet)

state_dir   <- "data/States"
species_csv <- "data/SpeciesList.csv"

# ---------- load ----------
state_files <- list.files(state_dir, "\\.csv$", full.names = TRUE)
raw_states  <- map_dfr(state_files, read.csv, check.names = FALSE)
names(raw_states) <- str_trim(names(raw_states))

count_cols <- names(raw_states)[str_detect(names(raw_states), "^Count\\d+")]
raw_states$total_sightings <- rowSums(raw_states[count_cols], na.rm = TRUE)

species_list <- read.csv(species_csv)
joined <- raw_states %>% left_join(species_list, by = "AOU")
if (!"SpeciesName" %in% names(joined)) joined$SpeciesName <- "Unknown"

df_all <- joined %>%
  select(any_of(c("RouteDataID","Year","AOU","total_sightings",
                  "SpeciesName","CountryNum","StateNum"))) %>%
  mutate(year_since_start = Year - min(Year, na.rm = TRUE))

# ---------- split ----------
idx <- sample.int(nrow(df_all), 0.8 * nrow(df_all))
df_train <- df_all[idx, ]
df_test  <- df_all[-idx, ]

log1p_vec <- function(x) log(x + 1)
prep_fac  <- function(d, ref = NULL) {
  d %>% mutate(
    log_sightings = log1p_vec(total_sightings),
    SpeciesName   = factor(SpeciesName,
                           levels = if (is.null(ref)) NULL else levels(ref$SpeciesName)),
    StateNum      = factor(StateNum,
                           levels = if (is.null(ref)) NULL else levels(ref$StateNum))
  )
}

df_train <- prep_fac(df_train)
df_test  <- prep_fac(df_test, df_train)

# ---------- dummy vars ----------
cat_vars <- c("SpeciesName","StateNum")
cat_vars <- cat_vars[sapply(df_train[cat_vars], nlevels) > 1]
form     <- as.formula(paste("~", paste(c("year_since_start", cat_vars), collapse = " + ")))

dmy     <- dummyVars(form, data = df_train, fullRank = TRUE)
train_x <- as.data.frame(predict(dmy, df_train));  train_y <- df_train$log_sightings
test_x  <- as.data.frame(predict(dmy, df_test));   obs     <- df_test$total_sightings

# ---------- fast grids ----------
ctrl_fast <- trainControl(method = "cv", number = 3, verboseIter = TRUE)

grid_single <- expand.grid(size = c(3, 7), decay = 1e-3)   
grid_multi  <- expand.grid(size = 5,        decay = 1e-3, bag = FALSE)  

nn_single <- train(train_x, train_y, method = "nnet", linout = TRUE,
                   trControl = ctrl_fast, tuneGrid = grid_single,
                   maxit = 500, trace = FALSE)

nn_multi  <- train(train_x, train_y, method = "avNNet", linout = TRUE,
                   trControl = ctrl_fast, tuneGrid = grid_multi,
                   maxit = 500, repeats = 1, trace = FALSE)

# ---------- evaluation ----------
metric <- function(o,p) c(RMSE = rmse(o,p), MAE = mae(o,p), R2 = cor(o,p)^2)
pred1  <- exp(predict(nn_single, test_x)) - 1
pred2  <- exp(predict(nn_multi,  test_x)) - 1
print(list(single = metric(obs, pred1), multi = metric(obs, pred2)))

# ---------- full prediction ----------
df_all2 <- prep_fac(df_all, df_train)
all_x   <- as.data.frame(predict(dmy, df_all2))
df_all2$pred <- exp(predict(nn_multi, all_x)) - 1

cat("Done")
