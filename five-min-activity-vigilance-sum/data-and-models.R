library(luz)
library(torch)
library(arrow)
library(lubridate)

get_day_accel = function(fn) {
  open_dataset(fn) |>
    collect() |>
    mutate(five_min_int = as.integer(as.factor(floor_date(time, minutes(5)))))
}

get_spec_sig = function(x, cutoff = floor(nrow(x) / 2)) {
  torch_stack(
    list(
      Mod(fft(x$X)[seq_len(cutoff)]),
      Mod(fft(x$Y)[seq_len(cutoff)]),
      Mod(fft(x$Z)[seq_len(cutoff)])
    ),
    dim = 1
  )
}
  
DayByFiveMinSSAccelData = dataset(
  name = "DayByFiveMinSSAccelData",
  initialize = function(fd, device = "gpu") {
    self$fd = fd
    self$device = "gpu"
  },
  .getitem = function(i) {
    da = get_day_accel(self$fd$fn[i]) |>
      group_by(five_min_int) |>
      group_nest() |>
      mutate(ss = map(data, ~ get_spec_sig(.x, 4500)))
    ret = torch_stack(da$ss, dim = 1)
    gc()
    ret
  },
  .length = function() {
    nrow(self$fd)
  } 
)

DayByFiveMinAccelData = dataset(
  name = "DayByFiveMinAccelData",
  initialize = function(fd, device = "cuda") {
    self$d = fd
    self$device = "gpu"
  },
  .getitem = function(i) {
    self$st = StartTime(self$d)
    self$fmss = DayByFiveMinSSAccelData(self$d)
    y = c(self$d$vigilant[i] == "non-vigilant")
    y = c(y, 1. - y)
    input = self$fmss[i][1:288, 1:3, 1:4500]
    ret = list(
      x = list(input = input, st = self$st[i]),
      y = torch_tensor(y)
    )
    gc()
    ret
  },
  .length = function() {
    nrow(self$d)
  }
)

FiveMinSSMod = nn_module(
  "FiveMinSSMod",
  initialize = function(dev = "cuda") {
    self$dev = dev
    self$nn1 = nn_sequential(
      nn_linear(4500, 1024),
      nn_linear(1024, 256),
      nn_linear(256, 64),
    )
    self$nn2 = nn_sequential(
      nn_linear(4500, 1024),
      nn_linear(1024, 256),
      nn_linear(256, 64),
    )
    self$nn3 = nn_sequential(
      nn_linear(4500, 1024),
      nn_linear(1024, 256),
      nn_linear(256, 64),
    )
  },
  forward = function(x) { 
    if (length(x$shape) == 3) {
      ret = torch_cat(
        list(
          self$nn1(x[,1,]),
          self$nn2(x[,2,]),
          self$nn3(x[,3,])
        ),
        dim = 2
      ) |> torch_flatten(start_dim = 1)
    } else if (length(x$shape) == 4) {
      ret = torch_cat(
        list(
          self$nn1(x[,,1,]),
          self$nn2(x[,,2,]),
          self$nn3(x[,,3,])
        ),
        dim = 2
      ) |> torch_flatten(start_dim = 2)
    } else {
      stop("Bad shape")
    }
    gc()
    ret
  }
)

FiveMinConv = nn_module(
  "FiveMinConv",
  initialize = function(dev = "cuda") {
    self$nn = nn_sequential(
      nn_conv1d(in_channels = 3, 6, 5),
      nn_relu(),
      nn_max_pool1d(5),
      nn_conv1d(6, 20, 5),
      nn_relu(),
      nn_max_pool1d(5),
      nn_flatten(start_dim = 2),
      nn_linear(3580, 64),
      nn_flatten(start_dim = 1)
    )
  },
  forward = function(x) {
    if (length(x$shape) == 3) {
      ret = self$nn(x)
    } else if (length(x$shape) == 4) {
      ret = map(seq_len(x$shape[1]), ~ self$nn(x[.x,,,])) |> 
        torch_stack(dim = 1)
    }
    gc()
    ret
  }
)

StartTime = dataset(
  name = "StartTime",
  initialize = function(fd, device = "cuda") {
    self$fd = fd
    self$device = "cuda"
  },
  .getitem = function(i) {
    d = arrow::read_parquet(self$fd$fn[i])
    dt = d$time[1]
    hour = as.numeric(difftime(dt, as.POSIXct(as.Date(dt)), units = "hours"))
    wd = wday(dt)
    torch_tensor(c(wd, hour))
  },
  .length = function() {
    nrow(self$fd)
  }
)

DayByFiveMinVigilanceMod = nn_module(
  "FiveMinVigilanceMod",
  initialize = function(dev = "cuda") {
    self$dev = dev
    self$fmm_mod = FiveMinSSMod("cuda")$to(device = dev)
    self$fmc_mod = FiveMinConv("cuda")$to(device = dev)
    self$nn = nn_linear(73730, 2)
  },
  forward = function(x) { 
    xr = torch_cat(
      list(
        x$st,
        self$fmm_mod$forward(x$input),
        self$fmc_mod$forward(x$input)
      ),
      dim = 2
    )
    self$nn(xr) |>
      nnf_softmax(dim = 2)
  }
)
