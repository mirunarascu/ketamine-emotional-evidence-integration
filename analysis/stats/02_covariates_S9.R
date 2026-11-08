# Supplementary Table S9: drug effects with the preregistered covariates, visit order and
# gender (in person only).
#
#   Rscript analysis/stats/02_covariates_S9.R
#
# Every in-person drug model of Tables S1-S4 and the rating-change models is refitted with
# "+ order_c + gender_c", keeping the random-effects structure the model has in those
# tables. The BW-2 paired t-test becomes a linear model of the ketamine - placebo
# difference with the same covariates (its intercept is the drug effect).
#
#   order_c   +0.5 ketamine at the first visit, -0.5 placebo first
#   gender_c  +0.5 female, -0.5 male
#
# Inputs:  output/stats/in_person/all_rows.csv (run 01_supplement_tables.R first),
#          the tables in output/tables/in_person/, data/subjects_in_person.csv
# Outputs: output/stats/in_person/S9_rows.csv and table_S9.txt

# ---- To run this script line by line (e.g. in RStudio): uncomment the 2 lines below,
# ---- set REPO_DIR to the folder of this repository, and comment out the "terminal" lines.
# REPO_DIR <- "~/path/to/ketamine-emotional-evidence-integration"
# source(file.path(REPO_DIR, "analysis", "stats", "stats_functions.R"))

# ---- Terminal (default): Rscript analysis/stats/02_covariates_S9.R
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
                 "stats_functions.R"))

COHORT <- "in_person"
DIMS <- c("valence", "arousal")
CAP <- function(x) paste0(toupper(substr(x, 1, 1)), substring(x, 2))
COVARIATES <- "order_c + gender_c"

main <- read.csv(file.path(repo_dir(), "output", "stats", COHORT, "all_rows.csv"), stringsAsFactors = FALSE)

# covariates per participant
subj <- read.csv(file.path(data_dir(), "subjects_in_person.csv"), stringsAsFactors = FALSE)
first <- subj[subj$visit == 1, ]
order_c <- setNames(ifelse(first$drug == "ketamine", 0.5, -0.5), first$subID)
gender_c <- setNames(ifelse(first$gender == "F", 0.5, -0.5), first$subID)
add_covariates <- function(x) {
  x$order_c <- order_c[as.character(x$subID)]
  x$gender_c <- gender_c[as.character(x$subID)]
  x
}

rows <- list()
main$name <- sub(" \\(Poisson.*\\)$", "", main$model)   # rating-change names end with the dispersion ratio

refit <- function(tab, main_name, fixed, data, labels, family = NULL) {
  # refit one model of Table <tab> with the covariates, at its random-effects structure
  m <- main[main$table == tab & main$name == main_name & main$term %in% names(labels), ]
  stopifnot(nrow(m) == length(labels))
  fit <- fit_model(paste(fixed, "+", COVARIATES), unique(m$random), add_covariates(data), family)
  r <- model_rows(fit, labels)
  r$p_without_covariates <- m$p[match(r$term, m$term)]
  rows[[length(rows) + 1]] <<- data.frame(table = "S9", model = sprintf("%s + visit order + gender (Table %s)", main_name, tab), r)
}

DRUG <- c(drug_c = "Drug: ketamine - placebo")
DRUG_VOL <- c("drug_c:volatility_c" = "Drug : volatility")


# ------------------------------------------------------------------ kernels (S3, S3b)
for (kernel in c("facewise", "timebinned")) {
  k <- code_factors(read_output_table(COHORT, paste0(kernel, "_condition")))
  tab <- c(facewise = "S3", timebinned = "S3b")[[kernel]]
  for (d in DIMS) refit(tab, sprintf("%s IKS ~ drug x volatility", CAP(d)), "iks ~ drug_c * volatility_c",
                        k[k$dimension == d, ], c(DRUG, DRUG_VOL))
  for (d in DIMS) {
    # BW-2, stable blocks: ketamine - placebo difference ~ covariates
    x <- k[k$dimension == d & k$volatility == "stable", ]
    x <- x[order(x$subID), ]
    w <- add_covariates(data.frame(subID = unique(x$subID),
                                   diff = x$bw2[x$drug_label == "ketamine"] - x$bw2[x$drug_label == "placebo"]))
    fit <- lm(as.formula(paste("diff ~", COVARIATES)), data = w)
    co <- summary(fit)$coefficients
    ci <- confint(fit)
    m <- main[main$table == tab & main$name == sprintf("%s BW-2, stable blocks: paired t-test", CAP(d)), ]
    rows[[length(rows) + 1]] <- data.frame(
      table = "S9", model = sprintf("%s BW-2, stable blocks: ketamine - placebo difference ~ visit order + gender (Table %s)", CAP(d), tab),
      predictor = "Drug: ketamine - placebo", term = "paired_t",
      b = co[1, 1], se = co[1, 2], ci_low = ci[1, 1], ci_high = ci[1, 2], df_num = NA,
      df_den = fit$df.residual, stat_type = "t", stat = co[1, 3], p = co[1, 4],
      inference = "linear model of the paired difference", random = "", singular = NA,
      ladder = "", n_obs = nrow(w), n_subjects = nrow(w), p_without_covariates = m$p)
  }
}


# ------------------------------------------------------------------ RT (S2)
cp <- change_points(COHORT)
rt <- cp[cp$detected_correctly == 1, ]
for (d in DIMS) refit("S2", sprintf("%s logRT ~ drug x volatility + CP magnitude + CP direction", CAP(d)),
                      "log_rt ~ drug_c * volatility_c + cp_magnitude_c + cp_direction_c",
                      rt[rt$dimension == d, ], c(DRUG, DRUG_VOL))


# ------------------------------------------------------------------ rating error (S1)
re <- code_factors(read_output_table(COHORT, "rating_error_block_emotion"))
for (d in DIMS) refit("S1", sprintf("%s %s signed RE ~ drug", CAP(MIDDLE[[d]]), d), "RE_signed ~ drug_c",
                      middle_level(re, d), DRUG)
for (d in DIMS) refit("S1", sprintf("%s RE ~ drug x emotion level", CAP(d)), "RE_flipped ~ drug_c * emotion_c",
                      extreme_levels(re, d), c(DRUG, "drug_c:emotion_c" = "Drug : emotion level"))


# ------------------------------------------------------------------ CP accuracy (S4)
for (d in DIMS) refit("S4", sprintf("%s CP accuracy ~ drug x volatility + CP magnitude + CP direction", CAP(d)),
                      "detected_correctly ~ drug_c * volatility_c + cp_magnitude_c + cp_direction_c",
                      cp[cp$dimension == d, ], c(DRUG, DRUG_VOL), binomial())


# ------------------------------------------------------------------ rating changes (main text)
rc <- code_factors(read_output_table(COHORT, "rating_changes_block"))
for (d in DIMS) refit("rating_changes", sprintf("%s rating changes ~ drug x volatility", CAP(d)),
                      "n_rating_changes ~ drug_c * volatility_c", rc[rc$dimension == d, ],
                      c(DRUG, DRUG_VOL), poisson())


write_tables(do.call(rbind, rows), file.path(repo_dir(), "output", "stats", COHORT), "S9_rows.csv")
