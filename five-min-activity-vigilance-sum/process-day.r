library(dplyr)
library(future)
library(purrr)
library(tibble)
library(foreach)
library(doMC)
library(tidyr)

registerDoMC(cores = 3)

source("data-and-models.R")
print(Sys.time())
devel = FALSE

if (devel) {
  num_workers = 0
  train_batch_size = 2
  epochs = 1
} else {
  num_workers = 25
  train_batch_size = 32
  epochs = 30
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

saveRDS(fd$eid, "fn_inds.rds")

fdg = left_join(
  fd |> 
    select(-fn) |>
    distinct(),
  fd |>
    select(eid, fn) |>
    group_by(eid) |>
    group_nest(),
  by = "eid"
)

comp_prop_range = function(md) {
  d = open_dataset(md$fn) |>
    mutate(day = day(time)) |>
    group_by(day) |> 
    summarize(
      n = n(),
      prop = n / (24 * 60 * 60 * 100)
    ) |>
    collect()
  tibble(
    min_prop = min(d$prop),
    max_prop = max(d$prop)
  )
}

fdg = bind_cols(  
  fdg,
  map_dfr(fdg$data, comp_prop_range)
)

# for each person.
registerDoMC(cores = 10)
foreach (i = seq_len(nrow(fdg))) %dopar% {
  ud = open_dataset(fdg$data[[i]]$fn) |>
    mutate(day = floor_date(time, unit = "hours")) |>
    collect() |>
    mutate(day = day - hours(hour(day))) |>
    group_by(day) |>
    group_nest() |>
    mutate(count = map_dbl(data, nrow)) |>
#    filter(count > 1e6) |>
    mutate(sample = seq_along(count))

  registerDoMC(cores = 3)
  # for each day
  foreach (j = seq_len(nrow(ud))) %dopar% {
    d = ud$data[[j]]
    df = ud$day[[j]]
    d_sample = ud$sample[j]
    ret = d |>
      mutate(time = round_date(time, seconds(1 / 30))) |>
      distinct() |>
      complete(
        time = 
          seq.POSIXt(
            df,
            df + days(1) - seconds(1/30),
            by = seconds(1) / 30
        )
      ) |> 
      fill(X, Y, Z, user, .direction = "downup")  |>
      mutate(sample = d_sample)
    ret$info = na.omit(unique(ret$info))
    ret$id = sprintf("%s-%02d", ret$info[1], ret$sample[1])
    write_parquet(ret, paste0("actigraphy-day-data/", ret$id[1], ".pq"))
  }
  print(i)
  NULL
}
