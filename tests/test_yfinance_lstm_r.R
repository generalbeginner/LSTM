library(testthat)

Sys.setenv(
  KERAS_BACKEND = "tensorflow",
  MPLCONFIGDIR = file.path(tempdir(), "matplotlib"),
  XDG_CACHE_HOME = file.path(tempdir(), "cache")
)

project_root <- normalizePath(".", mustWork = TRUE)
rmd_path <- file.path(project_root, "LSTM_functions.Rmd")

load_rmd_functions <- function() {
  if (!file.exists(rmd_path)) {
    return(FALSE)
  }

  source_path <- tempfile(fileext = ".R")
  knitr::purl(rmd_path, output = source_path, quiet = TRUE)
  source_env <- new.env(parent = globalenv())
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
  "build_outcome_frames",
  "detect_pivots",
  "compare_pivots",
  "build_pivot_analysis",
  "business_days",
  "build_model",
  "history_epochs",
  "forecast_future",
  "forecast_pivots",
  "plot_outcomes",
  "plot_pivots",
  "plot_pivot_date_distribution"
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

test_that("build_outcome_frames keeps actual and predicted values separate", {
  predictions <- list(
    train = list(
      dates = as.Date(c("2026-01-01", "2026-01-02")),
      actual = c(10, 11),
      predicted = c(9.5, 10.5)
    ),
    validate = list(
      dates = as.Date("2026-01-03"),
      actual = 12,
      predicted = 11.5
    ),
    test = list(
      dates = as.Date("2026-01-04"),
      actual = 13,
      predicted = 12.5
    )
  )

  outcomes <- build_outcome_frames(predictions, ticker = "GC=F", target = "Close")

  expect_named(
    outcomes$actual_outcomes,
    c("ticker", "target", "split", "Date", "actual")
  )
  expect_named(
    outcomes$predicted_outcomes,
    c("ticker", "target", "split", "Date", "predicted")
  )
  expect_equal(outcomes$actual_outcomes$actual, c(10, 11, 12, 13))
  expect_equal(outcomes$predicted_outcomes$predicted, c(9.5, 10.5, 11.5, 12.5))
  expect_equal(outcomes$actual_outcomes$split, c("train", "train", "validate", "test"))
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
  expect_equal(compared$date_difference_days[[1]], -1L)
  expect_equal(compared$date_error_days[[1]], 1)
  expect_false(compared$matched[[2]])
  expect_true(any(is.na(compared$predicted_date)))
})

test_that("build_pivot_analysis stores pivot dates and date differences", {
  dates <- as.Date("2026-01-01") + 0:4
  actual_outcomes <- data.frame(
    ticker = "EURUSD=X", target = "Close", split = "test",
    Date = dates, actual = c(1, 3, 1, 3, 1)
  )
  predicted_outcomes <- data.frame(
    ticker = "EURUSD=X", target = "Close", split = "test",
    Date = dates, predicted = c(1, 2, 4, 1, 3)
  )

  pivots <- build_pivot_analysis(
    actual_outcomes,
    predicted_outcomes,
    pivot_order = 1L,
    max_date_gap = 2L
  )

  expect_named(
    pivots,
    c("actual_pivots", "predicted_pivots", "pivot_date_differences")
  )
  expect_equal(unique(pivots$actual_pivots$source), "Actual")
  expect_equal(unique(pivots$predicted_pivots$source), "Predicted")
  expect_true(any(pivots$pivot_date_differences$matched))
  expect_true(all(
    pivots$pivot_date_differences$date_difference_days[
      pivots$pivot_date_differences$matched
    ] == 1L
  ))
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

test_that("forecast_pivots returns pivots without writing a file", {
  forecast <- data.frame(
    Date = as.Date("2026-10-05") + 0:4,
    Close = c(10, 12, 9, 11, 8)
  )

  pivots <- forecast_pivots(
    forecast,
    target = "Close",
    pivot_order = 1L
  )

  expect_equal(nrow(pivots), 3L)
  expect_named(pivots, c("Date", "price", "pivot_type"))
})

test_that("plot_outcomes creates separate actual and predicted line traces", {
  actual <- data.frame(
    ticker = "TSLA", target = "Close", split = c("train", "test"),
    Date = as.Date(c("2026-01-01", "2026-01-02")), actual = c(10, 12)
  )
  predicted <- data.frame(
    ticker = "TSLA", target = "Close", split = c("train", "test"),
    Date = as.Date(c("2026-01-01", "2026-01-02")), predicted = c(9, 11)
  )

  figure <- plot_outcomes(actual, predicted, ticker = "TSLA", target = "Close")
  traces <- plotly::plotly_build(figure)$x$data

  expect_s3_class(figure, "plotly")
  expect_equal(length(traces), 2L)
  expect_equal(vapply(traces, `[[`, character(1), "mode"), c("lines", "lines"))
})

test_that("plot_pivots renders hoverable marker traces for both sources", {
  dates <- as.Date("2026-01-01") + 0:4
  actual <- data.frame(
    ticker = "TSLA", target = "Close", split = "test",
    Date = dates, actual = c(1, 3, 1, 3, 1)
  )
  predicted <- data.frame(
    ticker = "TSLA", target = "Close", split = "test",
    Date = dates, predicted = c(1, 2, 4, 1, 3)
  )
  pivots <- build_pivot_analysis(actual, predicted, pivot_order = 1L)

  figure <- plot_pivots(
    actual, predicted,
    pivots$actual_pivots, pivots$predicted_pivots,
    ticker = "TSLA", target = "Close"
  )
  traces <- plotly::plotly_build(figure)$x$data
  marker_traces <- Filter(function(trace) identical(trace$mode, "markers"), traces)
  actual_markers <- Filter(function(trace) grepl("^Actual", trace$name), marker_traces)
  predicted_markers <- Filter(function(trace) grepl("^Predicted", trace$name), marker_traces)

  expect_s3_class(figure, "plotly")
  expect_equal(length(marker_traces), 4L)
  expect_true(all(vapply(marker_traces, function(trace) {
    all(trace$hoverinfo == "text")
  }, logical(1))))
  expect_true(all(vapply(actual_markers, function(trace) {
    grepl("-open$", trace$marker$symbol)
  }, logical(1))))
  expect_true(all(vapply(predicted_markers, function(trace) {
    !grepl("-open$", trace$marker$symbol)
  }, logical(1))))
})

test_that("plot_pivot_date_distribution charts matched signed differences", {
  differences <- data.frame(
    ticker = "TSLA", target = "Close",
    split = c("test", "test", "test", "test"),
    pivot_type = c("high", "high", "low", "low"),
    predicted_date = as.Date(c("2026-01-03", "2026-01-05", "2026-01-07", NA)),
    true_date = as.Date(c("2026-01-02", "2026-01-06", "2026-01-05", "2026-01-10")),
    date_difference_days = c(1L, -1L, 2L, NA_integer_),
    date_error_days = c(1L, 1L, 2L, NA_integer_),
    predicted_price = c(10, 11, 9, NA_real_),
    true_price = c(10.5, 10.5, 9.5, 8.5),
    matched = c(TRUE, TRUE, TRUE, FALSE)
  )

  figure <- plot_pivot_date_distribution(
    differences, ticker = "TSLA", target = "Close"
  )
  traces <- plotly::plotly_build(figure)$x$data

  expect_s3_class(figure, "plotly")
  expect_equal(length(traces), 2L)
  expect_true(all(vapply(traces, function(trace) {
    identical(trace$type, "histogram")
  }, logical(1))))
  expect_equal(
    sort(unlist(lapply(traces, `[[`, "x"), use.names = FALSE)),
    c(-1L, 1L, 2L)
  )
})

test_that("plot_pivot_date_distribution has an explicit unmatched-only state", {
  differences <- data.frame(
    ticker = "TSLA", target = "Close", split = "test",
    pivot_type = "high", predicted_date = as.Date(NA),
    true_date = as.Date("2026-01-10"),
    date_difference_days = NA_integer_, date_error_days = NA_integer_,
    predicted_price = NA_real_, true_price = 10,
    matched = FALSE
  )

  figure <- plot_pivot_date_distribution(
    differences, ticker = "TSLA", target = "Close"
  )
  built <- plotly::plotly_build(figure)
  trace <- built$x$data[[1L]]

  expect_equal(trace$type, "scatter")
  expect_equal(trace$mode, "markers")
  expect_equal(trace$hoverinfo, "skip")
  expect_false(trace$showlegend)
  expect_length(trace[["x"]], 0L)
  expect_match(built$x$layout$annotations[[1L]]$text, "No matched pivot")
})
