library(testthat)

Sys.setenv(
  KERAS_BACKEND = "tensorflow",
  MPLCONFIGDIR = file.path(tempdir(), "matplotlib"),
  XDG_CACHE_HOME = file.path(tempdir(), "cache")
)

project_root <- normalizePath(".", mustWork = TRUE)
rmd_path <- file.path(project_root, "run_yfinance_lstm.Rmd")

load_rmd_functions <- function() {
  if (!file.exists(rmd_path)) {
    return(FALSE)
  }

  source_path <- tempfile(fileext = ".R")
  knitr::purl(rmd_path, output = source_path, quiet = TRUE)
  source_env <- new.env(parent = globalenv())
  source_env$params <- list(
    target = "Open",
    tickers = "TSLA,AMZN,NVDA",
    horizon = 20L,
    run_pipeline = FALSE,
    pivot_order = 2L,
    pivot_min_prominence = 0
  )
  old_wd <- setwd(project_root)
  on.exit(setwd(old_wd), add = TRUE)
  sys.source(source_path, envir = source_env)
  list2env(as.list(source_env, all.names = TRUE), envir = globalenv())
  TRUE
}

required_functions <- c(
  "validate_target",
  "construct_lstm_data",
  "fit_minmax_scaler",
  "transform_minmax",
  "inverse_target",
  "prepare_lstm_data",
  "detect_pivots",
  "compare_pivots",
  "business_days",
  "build_model",
  "history_epochs",
  "forecast_future",
  "save_forecast_pivots"
)

if (!load_rmd_functions()) {
  for (function_name in required_functions) {
    assign(
      function_name,
      local({
        missing_name <- function_name
        function(...) stop(sprintf("%s is not implemented", missing_name), call. = FALSE)
      }),
      envir = globalenv()
    )
  }
}

test_that("validate_target accepts supported price fields", {
  expect_identical(
    vapply(c("Open", "Close", "High", "Low"), validate_target, character(1)),
    c(Open = "Open", Close = "Close", High = "High", Low = "Low")
  )
})

test_that("validate_target rejects non-price fields", {
  expect_error(validate_target("Volume"), "target must be one of")
})

test_that("construct_lstm_data uses the selected target column", {
  values <- matrix(seq(0, 29), nrow = 5, ncol = 6, byrow = TRUE)
  sequences <- construct_lstm_data(values, sequence_size = 2L, target_idx = 4L)

  expect_identical(dim(sequences$x), c(3L, 2L, 6L))
  expect_equal(sequences$y, c(15, 21, 27))
  expect_equal(sequences$x[1, , ], values[1:2, ])
})

test_that("min-max scaling is fitted per feature and can invert the target", {
  values <- matrix(
    c(10, 100, 5, 20, 18, 1000,
      20, 300, 5, 40, 38, 2000),
    nrow = 2,
    byrow = TRUE,
    dimnames = list(NULL, c("Open", "High", "Low", "Close", "Adj Close", "Volume"))
  )
  scaler <- fit_minmax_scaler(values)
  scaled <- transform_minmax(values, scaler)

  expect_equal(unname(scaled[1, ]), rep(0, 6))
  expect_equal(unname(scaled[2, c(1, 2, 4, 5, 6)]), rep(1, 5))
  expect_equal(unname(scaled[, 3]), c(0, 0))
  expect_equal(inverse_target(c(0, 1), scaler, target_idx = 4L), c(20, 40))
})

test_that("prepare_lstm_data keeps lookback context across chronological splits", {
  dates <- as.Date("2026-01-01") + 0:99
  feature_values <- matrix(seq_len(600), nrow = 100, ncol = 6)
  data <- data.frame(Date = dates, feature_values, check.names = FALSE)
  names(data)[-1] <- c("Open", "High", "Low", "Close", "Adj Close", "Volume")

  prepared <- prepare_lstm_data(data, target = "Close", lookback = 5L)

  expect_identical(dim(prepared$train$x), c(65L, 5L, 6L))
  expect_identical(dim(prepared$validate$x), c(15L, 5L, 6L))
  expect_identical(dim(prepared$test$x), c(15L, 5L, 6L))
  expect_equal(prepared$validate$dates[[1]], dates[[71]])
  expect_equal(prepared$test$dates[[1]], dates[[86]])
})

test_that("detect_pivots finds strict local highs and lows", {
  dates <- as.Date("2026-01-01") + 0:4
  pivots <- detect_pivots(dates, c(10, 12, 9, 11, 8), pivot_order = 1L)

  expect_equal(pivots$pivot_type, c("high", "low", "high"))
  expect_equal(pivots$Date, dates[2:4])
})

test_that("compare_pivots matches the nearest unused pivot of the same type", {
  predicted <- data.frame(
    Date = as.Date(c("2026-01-02", "2026-01-04")),
    price = c(3, 5),
    pivot_type = c("high", "high")
  )
  actual <- data.frame(
    Date = as.Date(c("2026-01-03", "2026-01-10")),
    price = c(4, 6),
    pivot_type = c("high", "high")
  )

  compared <- compare_pivots(predicted, actual, max_date_gap = 1L)

  expect_true(compared$matched[[1]])
  expect_equal(compared$date_error_days[[1]], 1)
  expect_false(compared$matched[[2]])
  expect_true(any(is.na(compared$predicted_date)))
})

test_that("business_days excludes weekends", {
  expect_equal(
    business_days(as.Date("2026-10-02"), 3L),
    as.Date(c("2026-10-05", "2026-10-06", "2026-10-07"))
  )
})

test_that("build_model creates four LSTM/dropout blocks and one output", {
  model <- build_model(c(5L, 6L))
  layer_classes <- vapply(model$layers, function(layer) class(layer)[[1]], character(1))

  expect_length(model$layers, 9L)
  expect_equal(sum(grepl("lstm", layer_classes, ignore.case = TRUE)), 4L)
  expect_equal(sum(grepl("dropout", layer_classes, ignore.case = TRUE)), 4L)
  expect_equal(as.integer(model$output_shape[[2]]), 1L)
})

test_that("history_epochs counts epochs rather than metric rows", {
  history <- list(metrics = list(
    loss = c(0.3, 0.2, 0.1),
    val_loss = c(0.4, 0.3, 0.2)
  ))

  expect_equal(history_epochs(history), 3L)
})

test_that("forecast_future recursively produces weekday target forecasts", {
  model <- keras3::keras_model_sequential(input_shape = c(2L, 6L)) |>
    keras3::layer_flatten() |>
    keras3::layer_dense(
      units = 1L,
      kernel_initializer = "zeros",
      bias_initializer = "zeros"
    )
  values <- matrix(
    c(10, 20, 5, 12, 12, 100,
      20, 30, 10, 22, 22, 200,
      30, 40, 15, 32, 32, 300),
    nrow = 3,
    byrow = TRUE,
    dimnames = list(NULL, c("Open", "High", "Low", "Close", "Adj Close", "Volume"))
  )
  scaler <- fit_minmax_scaler(values)
  recent_data <- data.frame(Date = as.Date(c("2026-10-01", "2026-10-02")), values[2:3, ], check.names = FALSE)

  forecast <- forecast_future(model, recent_data, scaler, target = "Open", horizon = 2L, lookback = 2L)

  expect_equal(forecast$Date, as.Date(c("2026-10-05", "2026-10-06")))
  expect_equal(forecast$Open, c(10, 10))
})

test_that("save_forecast_pivots writes a date-windowed CSV", {
  output_dir <- tempfile("forecast-pivots-")
  dir.create(output_dir)
  forecast <- data.frame(
    Date = as.Date("2026-10-05") + 0:4,
    Close = c(10, 12, 9, 11, 8)
  )

  output <- save_forecast_pivots(
    forecast,
    ticker = "TSLA",
    target = "Close",
    pivot_order = 1L,
    report_dir = output_dir
  )

  expect_true(file.exists(output))
  expect_match(basename(output), "tsla_close_forecast_pivots_2026-10-05_to_2026-10-09\\.csv")
  expect_equal(nrow(read.csv(output)), 3L)
})
