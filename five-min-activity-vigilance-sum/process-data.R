library(tibble)
library(future)
library(itertools)
library(foreach)
library(purrr)
library(devtools)
library(dplyr)
library(lubridate)
library(doFuture)
plan(multisession, workers= 25)
registerDoFuture()
#library(doMC)
#registerDoMC(cores = 10)

fns = dir("actigraphy-data") |>
  strsplit("_") |>
  map_chr(~.x[1])

x = readRDS("matched-vigilance.rds") |>
  filter(eid %in% fns)

time_iter = function(ts, dt) {
  i_ <- 1
  nextEl <- function() {
    got_i <- FALSE
    while(!got_i) {
      new_i <- ts >= ts[i_] + dt
      # If the new iterator is at the end of times then we are done.
      if (!any(new_i)) {
        stop("StopIteration", call. = FALSE)
      }
      new_i = min(which(new_i))
      # If we can't get 5*100*60 values then move on to the next window.
      if (new_i - i_ >= 30000) {
        got_i <- TRUE
      } else {
        i_ <<- new_i
      }
    }
    ret <- c(i_, min(i_ + 30000 - 1, new_i))
    i_ <<- new_i
    return(ret)
  }
  obj <- list(nextElem = nextEl)
  class(obj) <- c("abstractiter", "iter")
  return(obj)
}

x$path = file.path("actigraphy-data", sprintf("%s_90001_0_0.parquet", x$eid))

is = system(
  "ls five-min-processed | awk -F \"-\" '{print $1}' | sort | uniq", 
  intern = TRUE
)

x = x |> 
  filter(!(eid %in% is))

foreach (i = seq_along(x$path), .inorder = FALSE, .errorhandling = "remove",
         .multicombine = TRUE) %dopar% {

  library(arrow)
  # about 3 gigs
  d_time = (read_parquet(x$path[i], as_data_frame = FALSE) |>
    select(time) |> collect())$time
  gc()
  foreach (it = time_iter(d_time, minutes(5)), counter = icount(),
           .inorder = FALSE, .multicombine = TRUE) %do% {
    ds = 
      read_parquet(x$path[i], as_data_frame = FALSE)[it[1]:it[2],] |> 
        collect()
    gc()
#    ds = d[it[1]:it[2],] |> collect()
    ret = tibble(
      time = ds$time[seq(1, nrow(ds), length.out = 10000)],
      X = stats::filter(ds[['X']], c(1/3, 1/3, 1/3), method = "convolution",
                        circular = TRUE) |>
        (\(x) x[seq(1, length(x), length.out = 10000)])(),
      Y = stats::filter(ds[['Y']], c(1/3, 1/3, 1/3), method = "convolution",
                        circular = TRUE) |>
        (\(x) x[seq(1, length(x), length.out = 10000)])(),
      Z = stats::filter(ds[['Z']], c(1/3, 1/3, 1/3), method = "convolution",
                        circular = TRUE) |>
        (\(x) x[seq(1, length(x), length.out = 10000)])()
    )
    ret$user = ds$user[1]
    ret$info = ds$info[1]
    ret$sample = counter
    write_parquet(
      ret, 
      sprintf("five-min-processed/%s-%05d.pq", ret$user[1], ret$sample[1])
    )
    rm("ret")
    rm("ds")
    gc()
    NULL
  }
  rm("ds")
  gc()
  NULL
}

