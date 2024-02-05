library(dplyr)
library(torch)
library(future)
library(purrr)
library(tibble)
library(furrr)
plan(multicore)

source("data-and-models.R")

devel = FALSE
if (devel) {
  num_workers = 4
} else {
  num_workers = 16
}

fd = tibble(
  fn = system("ls five-min-processed", intern = TRUE) |> 
    sprintf("five-min-processed/%s", ...=_),
  eid= fn |> 
    strsplit("/") |>
    map_chr(~.x[2]) |>
    strsplit("-") |>
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

fd$train = TRUE

set.seed(1)
test = sample(unique(fd$eid), round(length(unique(fd$eid)) * 0.1))
fd$train[fd$eid %in% test] = FALSE

fd$start_time = future_map(fd$fn, ~ head(read_parquet(.x))$time[1]) |>
  unlist() |> as.POSIXct(tz = "UTC", origin = "1970-01-01 UTC")

if (devel) {
  fd = fd |> head(10000)
}

model = luz_load("trained-model.luz")
model$model$to(device = "cuda")

dl = dataloader(
  FiveMinVigilanceData(fd, device = "cuda"),
  batch_size = 1,
  shuffle = FALSE,
  num_workers = num_workers,
  worker_packages = c("torch", "arrow", "dplyr", "lubridate"),
  worker_globals = c("StartTime", "FiveMinSS", "FiveMinSSMod", "FiveMinConv")
)

embedding = list()
print(length(dl))
loop(for (it in dl) {
  embedding = append(
    embedding, 
    model$model$embed(it$x$to(device = "cuda"))$to(device = "cpu") |>
      as.matrix() |>
      list()
  )
})

dl = dataloader(
  FiveMinVigilanceData(fd, device = "cuda"),
  batch_size = 4,
  shuffle = FALSE,
  num_workers = num_workers,
  worker_packages = c("torch", "arrow", "dplyr", "lubridate"),
  worker_globals = c("StartTime", "FiveMinSS", "FiveMinSSMod", "FiveMinConv")
)

preds = predict(model, dl)
ps = preds[,1] |> 
  torch_tensor(device = "cpu") |>
  as.numeric()

#table(fd$vigilant, ps)

fd$est = ps
fd$embedding = embedding
saveRDS(fd, "fd.rds")
