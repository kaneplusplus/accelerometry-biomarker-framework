library(dplyr)
library(future)
library(purrr)
library(tibble)

source("data-and-models.R")

num_workers = 20
#num_workers = 2

fd = tibble(
  fn = system("ls actigraphy-day-data", intern = TRUE) |> 
    sprintf("actigraphy-day-data/%s", ...=_),
  eid= fn |> 
    strsplit("/") |>
    map_chr(~.x[2]) |>
    strsplit("_") |>
    map_chr(~.x[1]) |>
    gsub(".pq", "", x = _) |>
    as.integer()
)

fns = dir("actigraphy-data") |>
  strsplit("_") |>
  map_chr(~.x[1])

x = readRDS("matched-vigilance.rds") |>
  filter(eid %in% fns)

fd = left_join(fd, x, by = "eid")

saveRDS(fd$eid, "fn_inds.rds")

fd$train = TRUE


set.seed(1)
test = sample(unique(fd$eid), round(length(unique(fd$eid)) * 0.1))
fd$train[fd$eid %in% test] = FALSE

my_loss = function(input, target) {
  eps = 1e-8
  torch_mean(torch_sum(-input * log(target + eps), dim = 2))
}

model = DayByFiveMinVigilanceMod |>
  setup(
    loss = my_loss,
    optimizer = optim_adam
  ) 

records = lr_finder(
  object = model,
  data = dataloader(
    DayByFiveMinAccelData(fd |> filter(train)),
      batch_size = 32,
#      batch_size = 2,
      shuffle = TRUE,
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
  verbose = TRUE,
  start_lr = 1e-08,
  end_lr = 10
)

saveRDS(records, "records.rds")
