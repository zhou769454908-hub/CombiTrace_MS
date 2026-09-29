# Install/update the official public KruveLab MS2Quant package.
# This follows the installation command published in the official repository.
# Use a 64-bit R installation. rJava/rcdk generally require compatible 64-bit Java.
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

devtools::install_github(
  "kruvelab/MS2Quant",
  ref = "main",
  INSTALL_opts = "--no-multiarch",
  upgrade = "never"
)

suppressPackageStartupMessages(library(MS2Quant))
cat("MS2Quant installed. Version:", as.character(utils::packageVersion("MS2Quant")), "\n")
ns <- asNamespace("MS2Quant")
cat("MS2Quant_predict_IE available:", exists("MS2Quant_predict_IE", envir = ns, inherits = FALSE), "\n")

# Minimal official-predictor smoke test. Failure here is reported but does not
# delete the installed package; it usually points to Java/rcdk configuration.
tryCatch({
  chemicals <- tibble::tibble(SMILES = c("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
  result <- MS2Quant::MS2Quant_predict_IE(
    chemicals_for_IE_prediction = chemicals,
    organic_modifier = "MeCN",
    organic_percentage = 20,
    pH_aq = 2.7
  )
  cat("Official predictor smoke test: OK\n")
}, error = function(e) {
  cat("Official predictor smoke test: FAILED\n")
  cat(conditionMessage(e), "\n")
  quit(status = 2)
})
