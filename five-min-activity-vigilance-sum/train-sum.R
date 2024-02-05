library(dplyr)
library(future)
library(purrr)
library(tibble)
library(lubridate)
library(purrr)

source("data-and-models.R")
print(Sys.time())
devel = FALSE

if (devel) {
  num_workers = 0
  train_batch_size = 4
  epochs = 10
} else {
  num_workers = 20
  train_batch_size = 32
  epochs = 30
}

fd = tibble(
  fn = system("ls actigraphy-day-data", intern = TRUE) |>
    sprintf("actigraphy-day-data/%s", ...=_),
  eid= fn |> 
    strsplit("/") |>
    map_chr(~.x[2]) |>
    strsplit("_") |>
    map_chr(~.x[1]) |>
    as.integer(),
  id = fn |>
    strsplit("/") |>
    map_chr(~.x[2]) |>
    gsub(".pq", "", x = _),
  sample = fn |>
    strsplit("/") |>
    map_chr(~.x[2]) |>
    strsplit("-") |>
    map_chr(~.x[2]) |>
    gsub(".pq", "", x = _)
)

x = readRDS("matched-vigilance.rds") |>
  filter(eid %in% fd$eid)

fd = left_join(fd, x, by = "eid")

saveRDS(fd$eid, "fn_inds.rds")

fd$train = TRUE

set.seed(1)
test = sample(unique(fd$eid), round(length(unique(fd$eid)) * 0.1))
fd$train[fd$eid %in% test] = FALSE

my_loss = function(input, target) {
  eps = 1e-8
  torch_mean(torch_sum(-target * log(input + eps), dim = 2))
}

model = DayByFiveMinVigilanceMod |>
  setup(
    loss = my_loss,
    optimizer = optim_adam
  ) |>
  set_opt_hparams(lr = 1e-8) |>
  fit(
    dataloader(
      DayByFiveMinAccelData(fd |> filter(train)), 
      batch_size = train_batch_size, 
      shuffle = TRUE,
      num_workers = num_workers,
      worker_packages = c("torch", "arrow", "dplyr", "lubridate", "purrr"),
      worker_globals = 
        c("StartTime", 
          "DayByFiveMinSSAccelData", 
          "FiveMinSSMod", 
          "FiveMinConv",
          "get_day_accel",
          "get_spec_sig")
    ),
    epochs = epochs,
    valid_data = dataloader(
      DayByFiveMinAccelData(fd |> filter(!train)),
      batch_size = 1,
      shuffle = FALSE,
      num_workers = num_workers,
      worker_packages = c("torch", "arrow", "dplyr", "lubridate", "purrr"),
      worker_globals = 
        c("StartTime", 
          "DayByFiveMinSSAccelData", 
          "FiveMinSSMod", 
          "FiveMinConv",
          "get_day_accel",
          "get_spec_sig"),
    ),
    callbacks = list(
      luz_callback_model_checkpoint("best-model.pt", save_best_only = TRUE)
    ),
    verbose = TRUE
  )

print(model)
luz_save(model, "trained-model.luz") 
print(Sys.time())
