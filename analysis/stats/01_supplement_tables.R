# Mixed-effects models of Supplementary Tables S1-S8 and the rating-change models in
# the main text. Every model reported in the paper is fitted here, in table order.
#
#   Rscript analysis/stats/01_supplement_tables.R --cohort in_person    # Tables S1-S4 + rating changes
#   Rscript analysis/stats/01_supplement_tables.R --cohort online       # Tables S5-S8
#
# Inputs (output/tables/<cohort>/, from analysis/behaviour and analysis/kernels):
#   cp_table.csv, rating_error_block_emotion.csv, rating_changes_block.csv,
#   facewise_condition.csv, timebinned_condition.csv
# Outputs (output/stats/<cohort>/):
#   all_rows.csv        every model row: b, SE, 95% CI, df, statistic, p, random effects
#                       used and the structures tried, n, APA text
#   table_<id>.txt      the same rows grouped by model, one file per table
#
# How models are fitted and tested: see the top of stats_functions.R.
# Models are fitted separately for valence and arousal blocks.

# ---- To run this script line by line (e.g. in RStudio): uncomment the 3 lines below,
# ---- set REPO_DIR to the folder of this repository, and comment out the "terminal" lines.
# REPO_DIR <- "~/path/to/ketamine-emotional-evidence-integration"
# COHORT <- "in_person"                          # or "online"
# source(file.path(REPO_DIR, "analysis", "stats", "stats_functions.R"))

# ---- Terminal (default): Rscript analysis/stats/01_supplement_tables.R --cohort in_person
args <- commandArgs(trailingOnly = TRUE)
COHORT <- args[match("--cohort", args) + 1]
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
                 "stats_functions.R"))

stopifnot(COHORT %in% c("in_person", "online"))
HAS_DRUG <- yaml::read_yaml(file.path(repo_dir(), "config", "analysis_config.yaml"))$cohorts[[COHORT]]$has_drug
DIMS <- c("valence", "arousal")
CAP <- function(x) paste0(toupper(substr(x, 1, 1)), substring(x, 2))
TAB <- if (HAS_DRUG) c(re = "S1", rt = "S2", kernel = "S3", acc = "S4") else
                     c(re = "S5", rt = "S6", kernel = "S7", acc = "S8")

# Model results table
rows <- list()
# Add new row to model results table
add <- function(table, model, r) rows[[length(rows) + 1]] <<- data.frame(table = table, model = model, r)

# Row labels
INT <- c("(Intercept)" = "Intercept")
DRUG <- c(drug_c = "Drug: ketamine - placebo")
VOL <- c(volatility_c = "Volatility: volatile - stable")
MAG <- c(cp_magnitude_c = "CP magnitude")
DIR <- c(cp_direction_c = "CP direction: up - down")
DRUG_VOL <- c("drug_c:volatility_c" = "Drug : volatility")


# ================================================================== data
cp <- change_points(COHORT)               # analysable change points (accuracy)
rt <- cp[cp$detected_correctly == 1, ]    # RT: correctly detected change points
rt$dimension_c <- ifelse(rt$dimension == "arousal", 0.5, -0.5)
re <- code_factors(read_output_table(COHORT, "rating_error_block_emotion"))
extremes <- function(d) extreme_levels(re, d)
middle <- function(d) middle_level(re, d)


# ================================================================== rating error (S1 / S5)
# DV: signed rating error per block and emotion level, flipped for positive valence
# and high arousal so that positive = less extreme than the true mean.

# Signed RE ~ emotion level (restricted to extreme levels: pos/neg valence, high/low arousal)
for (d in DIMS) {
  emo <- c(emotion_c = sprintf("Emotion level: %s - %s", EXTREMES[[d]][2], EXTREMES[[d]][1]))
  fit <- fit_model("RE_flipped ~ emotion_c", ladder("emotion_c"), extremes(d))
  add(TAB[["re"]], sprintf("%s RE ~ emotion level", CAP(d)), model_rows(fit, c(INT, emo)))
}
# Signed RE ~ 1 + (1|subID) for intermediate levels only
for (d in DIMS) {
  # middle level (neutral valence, medium arousal) rating bias
  fit <- fit_model("RE_signed ~ 1", "(1 | subID)", middle(d))
  add(TAB[["re"]], sprintf("%s %s signed RE ~ 1", CAP(MIDDLE[[d]]), d), model_rows(fit, INT))
}
if (HAS_DRUG) {
  for (d in DIMS) {
    # Signed RE ~ drug for intermediate levels only
    fit <- fit_model("RE_signed ~ drug_c", ladder("drug_c"), middle(d))
    add(TAB[["re"]], sprintf("%s %s signed RE ~ drug", CAP(MIDDLE[[d]]), d), model_rows(fit, c(INT, DRUG)))
  }
  # Signed RE ~ drug x emotion level (extreme levels only, flipped for positive valence and high arousal); 
  # follow-up: ketamine - placebo at each extreme
  for (d in DIMS) {
      emo <- c(emotion_c = sprintf("Emotion level: %s - %s", EXTREMES[[d]][2], EXTREMES[[d]][1]))
      fit <- fit_model("RE_flipped ~ drug_c * emotion_c", ladder("emotion_c"), extremes(d))
      name <- sprintf("%s RE ~ drug x emotion level", CAP(d))
      add(TAB[["re"]], name, model_rows(fit, c(INT, DRUG, emo, "drug_c:emotion_c" = "Drug : emotion level")))
      if (term_p_value(fit, "drug_c:emotion_c") < .05)
        add(TAB[["re"]], name, follow_up(fit, "drug_c", "emotion_c", "ketamine - placebo",
                                          CAP(EXTREMES[[d]])))
  }
  
  # DV: absolute RE per block, all emotion levels
  re_block <- code_factors(read_output_table(COHORT, "rating_error_block"))
  for (d in DIMS) {
    fit <- fit_model("RE_abs ~ drug_c", ladder("drug_c"), re_block[re_block$dimension == d, ])
    add("absolute_RE", sprintf("%s absolute RE ~ drug", CAP(d)), model_rows(fit, c(INT, DRUG)))
  }
}


# ================================================================== change points: RT (S2 / S6) and accuracy (S4 / S8)
# One row per change point. RT = log time to the first correct rating change (linear
# mixed model); accuracy = correctly detected or not (logistic mixed model).
cp_models <- function(tab, data, dv, dv_name, family) {
  by_dim <- function(d) data[data$dimension == d, ]
  # logRT/accuracy ~ rating dimesion
  if (dv_name == "logRT") {
    fit <- fit_model(paste(dv, "~ dimension_c"), ladder("dimension_c"), data)
    add(tab, "logRT ~ dimension", model_rows(fit, c(INT, dimension_c = "Dimension: arousal - valence")))
  }
  # logRT/accuracy ~ cp magnitude
  for (d in DIMS) {
    fit <- fit_model(paste(dv, "~ cp_magnitude_c"), ladder("cp_magnitude_c"), by_dim(d), family)
    add(tab, sprintf("%s %s ~ CP magnitude", CAP(d), dv_name), model_rows(fit, c(INT, MAG)))
  }
  # logRT/accuracy ~ cp direction + cp magnitude
  for (d in DIMS) {
    fit <- fit_model(paste(dv, "~ cp_direction_c + cp_magnitude_c"), ladder("cp_direction_c"), by_dim(d), family)
    add(tab, sprintf("%s %s ~ CP direction + CP magnitude", CAP(d), dv_name), model_rows(fit, c(INT, DIR, MAG)))
  }

  # logRT/accuracy ~ emotion level reached (separately for downward and upward change points)
  reached <- list(valence = list(down = c("negative", "neutral"), up = c("neutral", "positive")),
                  arousal = list(down = c("low", "medium"), up = c("medium", "high")))
  # CP magnitude < 3, because the largest jumps can only reach the extremes. Control with all magnitudes included
  versions <- if (dv_name == "logRT" && HAS_DRUG) c("; CP magnitude < 3", "; all magnitudes") else "; CP magnitude < 3"
  for (version in versions) for (dir in c("down", "up")) for (d in DIMS) {
    lv <- reached[[d]][[dir]]
    x <- data[data$dimension == d & data$cp_direction == dir & data$emotion_level %in% lv, ]
    if (version == "; CP magnitude < 3") x <- x[x$cp_magnitude < 3, ]
    x$emotion_c <- ifelse(x$emotion_level == lv[2], 0.5, -0.5)
    # arousal jumps below 3 all have magnitude 2, so magnitude is only a covariate when it varies
    has_mag <- length(unique(x$cp_magnitude)) > 1
    fit <- fit_model(paste(dv, "~ emotion_c", if (has_mag) "+ cp_magnitude_c"), ladder("emotion_c"), x, family)
    add(tab, sprintf("%s %s ~ emotion level%s - %sward CP%s", CAP(d), dv_name,
                     if (has_mag) " + CP magnitude" else "", dir, version),
        model_rows(fit, c(INT, emotion_c = sprintf("Emotion level: %s - %s", lv[2], lv[1]), if (has_mag) MAG)))
  }

  if (HAS_DRUG && dv_name == "logRT") {
    for (d in DIMS) {
      fit <- fit_model(paste(dv, "~ volatility_c + cp_magnitude_c"), ladder(c("volatility_c", "cp_magnitude_c")),
                       by_dim(d))
      add(tab, sprintf("%s logRT ~ volatility + CP magnitude", CAP(d)), model_rows(fit, c(INT, VOL, MAG)))
    }
    # logRT ~ volatility x CP magnitude (Fig S3D)
    # Follow-up: volatile - stable at each magnitude
    for (d in DIMS) {
      x <- by_dim(d)
      # Magnitude as a sum-coded factor
      x$cp_magnitude_f <- factor(x$cp_magnitude)
      contrasts(x$cp_magnitude_f) <- contr.sum(nlevels(x$cp_magnitude_f))
      fit <- fit_model("log_rt ~ volatility_c * cp_magnitude_f + cp_direction_c",
                       c("(1 + volatility_c + cp_magnitude_c | subID)", "(1 + volatility_c + cp_magnitude_c || subID)",
                         "(1 + volatility_c || subID)", "(1 | subID)"), x)
      name <- sprintf("%s logRT ~ volatility x CP magnitude (factor) + CP direction", CAP(d))
      a <- anova(fit, ddf = "Kenward-Roger")
      r1 <- model_rows(fit, c(INT, VOL, DIR))
      multi <- c(cp_magnitude_f = "CP magnitude", "volatility_c:cp_magnitude_f" = "Volatility : CP magnitude")
      rf <- r1[rep(1, 2), ]
      rf[, c("b", "se", "ci_low", "ci_high")] <- NA
      rf$predictor <- unname(multi); rf$term <- names(multi)
      rf$df_num <- a[names(multi), "NumDF"]; rf$df_den <- a[names(multi), "DenDF"]
      rf$stat <- a[names(multi), "F value"]; rf$p <- a[names(multi), "Pr(>F)"]
      rf$inference <- "type-III F, Kenward-Roger (lmerTest)"
      add(tab, name, rbind(r1, rf))
      if (a["volatility_c:cp_magnitude_f", "Pr(>F)"] < .05) {
        emm <- emmeans(fit, ~ volatility_c | cp_magnitude_f, at = list(volatility_c = c(-0.5, 0.5)),
                       lmer.df = "kenward-roger")
        s <- as.data.frame(summary(contrast(emm, method = list(`volatile - stable` = c(-1, 1)), by = "cp_magnitude_f"),
                                   by = NULL, adjust = "holm", infer = c(TRUE, TRUE)))
        add(tab, name, data.frame(predictor = sprintf("CP magnitude %s: volatile - stable", s$cp_magnitude_f),
                                  term = "", b = s$estimate, se = s$SE, ci_low = s$lower.CL, ci_high = s$upper.CL,
                                  df_num = NA, df_den = s$df, stat_type = "t", stat = s$t.ratio, p = s$p.value,
                                  inference = "emmeans follow-up, Holm (model scale, Kenward-Roger df)",
                                  random = "", singular = NA, ladder = "", n_obs = NA, n_subjects = NA))
      }
    }
  }
  # logRT/accuracy ~ drug x volatility + CP magnitude + CP direction
  for (d in DIMS) {
    if (HAS_DRUG) {
      fit <- fit_model(paste(dv, "~ drug_c * volatility_c + cp_magnitude_c + cp_direction_c"),
                       c("(1 + drug_c * volatility_c | subID)", "(1 + drug_c + volatility_c | subID)",
                         "(1 + volatility_c | subID)", "(1 | subID)"), by_dim(d), family)
      name <- sprintf("%s %s ~ drug x volatility + CP magnitude + CP direction", CAP(d), dv_name)
      add(tab, name, model_rows(fit, c(INT, DRUG, VOL, MAG, DIR, DRUG_VOL)))
      fu <- drug_by_volatility_follow_ups(fit)
      if (!is.null(fu)) add(tab, name, fu)
    } else {
      fit <- fit_model(paste(dv, "~ volatility_c + cp_magnitude_c + cp_direction_c"),
                       c("(1 + volatility_c | subID)", "(1 | subID)"), by_dim(d), family)
      add(tab, sprintf("%s %s ~ volatility + CP magnitude + CP direction", CAP(d), dv_name),
          model_rows(fit, c(INT, VOL, MAG, DIR)))
    }
  }
}
cp_models(TAB[["rt"]], rt, "log_rt", "logRT", NULL)
cp_models(TAB[["acc"]], cp, "detected_correctly", "CP accuracy", binomial())


# ================================================================== kernels (S3, S3b / S7)
# One row per subject x drug x dimension x volatility (mean of 3 blocks).
# IKS = beta(-1) - beta(-2); BW-2 = beta(-2).
kernel_models <- function(tab, kernel) {
  k <- code_factors(read_output_table(COHORT, paste0(kernel, "_condition")))
  by_dim <- list(valence = k[k$dimension == "valence", ], arousal = k[k$dimension == "arousal", ])
  if (HAS_DRUG) {
    # IKS ~ drug x volatility
    fits <- fit_both_dims_same_structure("iks ~ drug_c * volatility_c",
              c("(1 + drug_c * volatility_c | subID)", "(1 + drug_c + volatility_c | subID)",
                "(1 + volatility_c | subID)", "(1 | subID)"), by_dim)
    for (d in DIMS) {
      name <- sprintf("%s IKS ~ drug x volatility", CAP(d))
      add(tab, name, model_rows(fits[[d]], c(INT, DRUG, VOL, DRUG_VOL)))
      fu <- drug_by_volatility_follow_ups(fits[[d]])
      if (!is.null(fu)) add(tab, name, fu)
    }
    # BW-2 in stable blocks, ketamine vs placebo, paired t-test
    for (d in DIMS) {
      x <- by_dim[[d]][by_dim[[d]]$volatility == "stable", ]
      x <- x[order(x$subID), ]
      add(tab, sprintf("%s BW-2, stable blocks: paired t-test", CAP(d)),
          data.frame(predictor = "Drug: ketamine - placebo",
                     paired_t_test(x$bw2[x$drug_label == "ketamine"], x$bw2[x$drug_label == "placebo"])))
    }
  } else {
    fits <- fit_both_dims_same_structure("iks ~ volatility_c", c("(1 + volatility_c | subID)", "(1 | subID)"), by_dim)
    for (d in DIMS) add(tab, sprintf("%s IKS ~ volatility", CAP(d)), model_rows(fits[[d]], c(INT, VOL)))
  }
}
kernel_models(TAB[["kernel"]], "facewise")         # face-wise linear kernel (main text)
if (HAS_DRUG) kernel_models("S3b", "timebinned")   # time-binned logistic kernel (preregistered)


# ================================================================== rating changes (main text, in person)
# Number of rating changes per block, Poisson mixed model, drug x volatility.
if (HAS_DRUG) {
  rc <- code_factors(read_output_table(COHORT, "rating_changes_block"))
  for (d in DIMS) {
    fit <- fit_model("n_rating_changes ~ drug_c * volatility_c",
                     c("(1 + drug_c * volatility_c | subID)", "(1 + volatility_c | subID)", "(1 | subID)"),
                     rc[rc$dimension == d, ], poisson())
    dispersion <- sum(residuals(fit, type = "pearson")^2) / df.residual(fit)
    name <- sprintf("%s rating changes ~ drug x volatility (Poisson, dispersion ratio %.2f)", CAP(d), dispersion)
    add("rating_changes", name, model_rows(fit, c(INT, DRUG, VOL, DRUG_VOL)))
    fu <- drug_by_volatility_follow_ups(fit)
    if (!is.null(fu)) add("rating_changes", name, fu)
  }
}


# ================================================================== write
write_tables(do.call(rbind, rows), file.path(repo_dir(), "output", "stats", COHORT))
