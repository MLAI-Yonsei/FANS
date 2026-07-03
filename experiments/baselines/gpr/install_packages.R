# 필요한 패키지와 버전 정의
options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- list(
  GPArotation = "2025.3-1",
  here = "1.0.2",
  jsonlite = "2.0.0",
  lattice = "0.22-7",
  lmtest = "0.9-40",
  Matrix = "1.7-4",
  mnormt = "2.1.1",
  mvtnorm = "1.3-3",
  nlme = "3.1-168",
  png = "0.1-8",
  psych = "2.5.6",
  quadprog = "1.5-8",
  rappdirs = "0.3.3",
  Rcpp = "1.1.0",
  RcppTOML = "0.2.3",
  reticulate = "1.44.1",
  rlang = "1.1.6",
  rprojroot = "2.1.1",
  withr = "3.0.2",
  zoo = "1.8-14",
  tmvnsim = "1.0-2"
)

# remotes 설치
if (!require("remotes")) install.packages("remotes")

# 각 패키지 설치
for (pkg in names(packages)) {
  cat(sprintf("Installing %s version %s...\n", pkg, packages[[pkg]]))
  tryCatch({
    remotes::install_version(pkg, version = packages[[pkg]], repos = "https://cloud.r-project.org")
  }, error = function(e) {
    cat(sprintf("Failed to install %s: %s\n", pkg, e$message))
  })
}

# 설치 확인
cat("\n=== Installation Summary ===\n")
installed <- installed.packages()
for (pkg in names(packages)) {
  if (pkg %in% installed[, "Package"]) {
    version <- installed[pkg, "Version"]
    status <- ifelse(version == packages[[pkg]], "✓ Correct", sprintf("✗ Wrong (installed: %s)", version))
    cat(sprintf("%s: %s\n", pkg, status))
  } else {
    cat(sprintf("%s: ✗ Not installed\n", pkg))
  }
}