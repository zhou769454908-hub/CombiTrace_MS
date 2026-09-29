#!/usr/bin/env Rscript

# Official MS2Quant prediction bridge for CombiTrace-MS.
# This script invokes the user-installed KruveLab/MS2Quant package. It does not
# contain or redistribute the pretrained model itself.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: ms2quant_bridge.R input.csv output.csv metadata.json MeCN|MeOH pH")
}
input_path <- args[[1]]
output_path <- args[[2]]
metadata_path <- args[[3]]
organic_modifier <- args[[4]]
pH_aq <- as.numeric(args[[5]])
if (is.na(pH_aq)) {
  stop("pH must be numeric")
}

suppressPackageStartupMessages(library(MS2Quant))
suppressPackageStartupMessages(library(tibble))

input <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("row_id", "SMILES", "organic_percentage")
missing_cols <- setdiff(required, names(input))
if (length(missing_cols) > 0) {
  stop(paste("Missing input columns:", paste(missing_cols, collapse = ", ")))
}

extract_prediction <- function(obj, expected_n) {
  if (is.null(obj$chemicals_predicted_IEs)) {
    stop("MS2Quant_predict_IE() result does not contain chemicals_predicted_IEs")
  }
  df <- as.data.frame(obj$chemicals_predicted_IEs, stringsAsFactors = FALSE)
  if (nrow(df) != expected_n) {
    stop(paste0("MS2Quant returned ", nrow(df), " rows for ", expected_n, " input SMILES"))
  }
  nms <- names(df)
  numeric_names <- nms[vapply(df, is.numeric, logical(1))]
  preferred <- nms[grepl("pred.*ie|ie.*pred|predicted.*ion|ionization.*eff|logie", nms, ignore.case = TRUE)]
  preferred <- preferred[preferred %in% numeric_names]
  if (length(preferred) == 0) {
    # Keep this fallback auditable. The selected column is written to output.
    if (length(numeric_names) == 0) {
      stop(paste("No numeric prediction column found. Available columns:", paste(nms, collapse = ", ")))
    }
    # In current MS2Quant releases the predicted IE column is the final numeric
    # result column when no preferred name is found.
    selected <- tail(numeric_names, 1)
  } else {
    selected <- preferred[[1]]
  }
  list(values = as.numeric(df[[selected]]), column = selected, raw_names = nms)
}

results <- list()
groups <- unique(input$organic_percentage)
for (g in groups) {
  idx <- which(input$organic_percentage == g)
  sub <- input[idx, , drop = FALSE]
  status <- rep("PREDICTED", nrow(sub))
  reason <- rep("", nrow(sub))
  values <- rep(NA_real_, nrow(sub))
  selected_column <- rep("", nrow(sub))
  tryCatch({
    chemicals <- tibble::tibble(SMILES = as.character(sub$SMILES))
    prediction <- MS2Quant::MS2Quant_predict_IE(
      chemicals_for_IE_prediction = chemicals,
      organic_modifier = organic_modifier,
      organic_percentage = as.numeric(g),
      pH_aq = pH_aq
    )
    extracted <- extract_prediction(prediction, nrow(sub))
    values <- extracted$values
    selected_column[] <- extracted$column
  }, error = function(e) {
    status[] <<- "FAILED"
    reason[] <<- conditionMessage(e)
  })
  results[[length(results) + 1]] <- data.frame(
    row_id = sub$row_id,
    organic_percentage = sub$organic_percentage,
    predicted_logIE = values,
    prediction_column = selected_column,
    status = status,
    reason = reason,
    stringsAsFactors = FALSE
  )
}
output <- do.call(rbind, results)
write.csv(output, output_path, row.names = FALSE, fileEncoding = "UTF-8")

metadata <- list(
  R_version = paste(R.version$major, R.version$minor, sep = "."),
  MS2Quant_version = as.character(utils::packageVersion("MS2Quant")),
  MS2Quant_path = find.package("MS2Quant"),
  organic_modifier = organic_modifier,
  pH_aq = pH_aq,
  input_rows = nrow(input),
  predicted_rows = sum(is.finite(output$predicted_logIE)),
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
# Avoid an additional JSON dependency.
escape_json <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", as.character(x))
  x <- gsub('"', '\\"', x, fixed = TRUE)
  x
}
json_parts <- c()
for (nm in names(metadata)) {
  value <- metadata[[nm]]
  if (is.numeric(value)) {
    json_parts <- c(json_parts, paste0('"', nm, '":', value))
  } else {
    json_parts <- c(json_parts, paste0('"', nm, '":"', escape_json(value), '"'))
  }
}
writeLines(paste0("{", paste(json_parts, collapse = ","), "}"), metadata_path, useBytes = TRUE)
