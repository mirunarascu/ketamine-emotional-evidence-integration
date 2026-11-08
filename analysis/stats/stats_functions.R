# Functions used by the statistics scripts in this folder.
#
# Inference (as in the Methods):
#   linear mixed models (logRT, rating error, IKS)
#       Kenward-Roger t tests from lmerTest, reported as F(1, df) = t^2.
#       95% CI = b +/- t(.975, df) * SE, with the same Kenward-Roger df.
#   logistic / Poisson mixed models (CP accuracy, rating changes)
#       each predictor: type-III likelihood-ratio test, chi2(1), by refitting the
#       model without that predictor. Intercept: Wald z. 95% CI = b +/- 1.96 * SE.
#   follow-ups of a significant interaction (p < .05)
#       emmeans, Holm across the 2 tests, Bonferroni-adjusted CIs, Kenward-Roger df.
#       For logistic / Poisson models the differences are on the response scale
#       (probability / number of rating changes).
#
# Random effects:
#   each model is fitted with a list of random-effects structures, from the maximal one
#   down to (1 | subID). The first structure whose fit is not singular is used. If even
#   (1 | subID) is singular it is used anyway (between-subject variance ~ 0) and the
#   row is marked singular.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

# Kenward-Roger df in emmeans even for the large CP tables (the default limit would
# switch to asymptotic df without saying so).
emm_options(lmerTest.limit = 1e6, pbkrtest.limit = 1e6)

FIT_CONTROL <- list(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))


# ------------------------------------------------------------------ data

repo_dir <- function() {
  # repo root: REPO_DIR if it was set (running line by line, see the top of each script),
  # else from the location of the script being run with Rscript (analysis/<folder>/<script>.R)
  if (exists("REPO_DIR", envir = globalenv())) return(normalizePath(get("REPO_DIR", envir = globalenv())))
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
  normalizePath(file.path(dirname(f), "..", ".."))
}

data_dir <- function() {
  # shared data folder: $PEDUKS_DATA_DIR if set, else data/ in the repository (as in utils/config.py)
  d <- Sys.getenv("PEDUKS_DATA_DIR")
  if (!nzchar(d)) d <- file.path(repo_dir(), "data")
  if (!dir.exists(d)) stop("data folder ", d, " not found (see Data in README.md)")
  d
}

read_output_table <- function(cohort, name) {
  read.csv(file.path(repo_dir(), "output", "tables", cohort, paste0(name, ".csv")),
           stringsAsFactors = FALSE)
}

code_factors <- function(df) {
  # +/- 0.5 coding, so a coefficient is (second level - first level) and main effects
  # are averaged over the other factors.
  df$volatility_c <- ifelse(df$volatility == "volatile", 0.5, -0.5)
  df$drug_c <- ifelse(df$drug_label == "ketamine", 0.5, -0.5)   # online: all -0.5, unused
  if ("cp_direction" %in% names(df)) df$cp_direction_c <- ifelse(df$cp_direction == "up", 0.5, -0.5)
  # CP magnitude centred on the middle magnitude: 2 for valence, 3 for arousal
  if ("cp_magnitude" %in% names(df))
    df$cp_magnitude_c <- df$cp_magnitude - ifelse(df$dimension == "valence", 2, 3)
  df$subID <- factor(df$subID)
  df
}


change_points <- function(cohort) {
  # analysable change points, with log RT of the first correct rating change
  cp <- code_factors(read_output_table(cohort, "cp_table"))
  cp <- cp[as.logical(cp$admissible), ]
  cp$log_rt <- log(cp$rt_first_correct_s)              # NA when not detected
  cp
}

# Emotion levels used in the rating-error models
EXTREMES <- list(valence = c("negative", "positive"), arousal = c("low", "high"))
MIDDLE <- c(valence = "neutral", arousal = "medium")

extreme_levels <- function(re, d) {
  # blocks x the two extreme emotion levels; emotion_c = +0.5 for positive / high
  x <- re[re$dimension == d & re$emotion_level %in% EXTREMES[[d]], ]
  x$emotion_c <- ifelse(x$emotion_level == EXTREMES[[d]][2], 0.5, -0.5)
  x
}

middle_level <- function(re, d) re[re$dimension == d & re$emotion_level == MIDDLE[[d]], ]


# ------------------------------------------------------------------ fitting

ladder <- function(slopes) {
  # Standard list of random-effects structures for the given slopes:
  # correlated -> uncorrelated (||) -> drop slopes from the end -> (1 | subID).
  if (!length(slopes)) return("(1 | subID)")
  n <- length(slopes)
  c(sprintf("(1 + %s | subID)", paste(slopes, collapse = " + ")),
    vapply(n:1, function(k) sprintf("(1 + %s || subID)", paste(slopes[1:k], collapse = " + ")), ""),
    "(1 | subID)")
}

fit_model <- function(fixed, random, data, family = NULL) {
  # fixed:  "log_rt ~ drug_c * volatility_c + ..."
  # random: random-effects structures to try, in order (see ladder()).
  # family: NULL for a linear mixed model, else binomial() / poisson().
  # Returns the fit, with the structure used and the steps taken in attr "ladder".
  steps <- character(0)
  # Run the ladder: try each random-effects structure in order and keep the first fit that
  # neither fails nor is singular
  for (i in seq_along(random)) {
    f <- as.formula(paste(fixed, "+", random[i]))
    fit <- tryCatch(suppressMessages(suppressWarnings(
      if (is.null(family)) lmerTest::lmer(f, data = data, REML = TRUE,
                                          control = do.call(lmerControl, FIT_CONTROL))
      else glmer(f, data = data, family = family, control = do.call(glmerControl, FIT_CONTROL)))),
      error = function(e) NULL)
    if (is.null(fit)) { steps <- c(steps, paste(random[i], "failed")); next }
    singular <- isSingular(fit, tol = 1e-4)
    if (singular && i < length(random)) { steps <- c(steps, paste(random[i], "singular")); next }
    attr(fit, "random") <- random[i]
    attr(fit, "steps") <- steps                        # structures tried before this one
    attr(fit, "ladder") <- paste(c(steps, paste(random[i], if (singular) "singular, used" else "used")),
                                 collapse = " | ")
    return(fit)
  }
  stop("no random-effects structure could be fitted: ", fixed)
}

fit_both_dims_same_structure <- function(fixed, random, data_by_dim) {
  # Kernel models (IKS): valence and arousal are tested with the same random-effects
  # structure, the simpler of the two reached by fitting each dimension on its own.
  # The ladder note keeps both passes, so every structure tried is recorded.
  first <- lapply(data_by_dim, function(d) fit_model(fixed, random, d))
  k <- max(match(sapply(first, attr, "random"), random))
  out <- lapply(names(data_by_dim), function(d) {
    fit <- fit_model(fixed, random[k:length(random)], data_by_dim[[d]])
    own <- attr(first[[d]], "random")
    moved <- if (match(own, random) < k) paste(own, "not singular, but the other dimension needed a simpler structure")
    attr(fit, "ladder") <- paste(c(attr(first[[d]], "steps"), moved, attr(fit, "ladder")), collapse = " | ")
    fit
  })
  setNames(out, names(data_by_dim))
}


# ------------------------------------------------------------------ inference

lrt_chi2 <- function(fit, term) {
  # Type-III likelihood-ratio test of one predictor of a logistic / Poisson mixed model:
  # refit the model without that predictor (all other terms, including interactions, kept)
  # and compare how well the two models fit
  reduced <- suppressMessages(suppressWarnings(glmer(
    update(formula(fit), paste(". ~ . -", term)), # same formula without `term`
    data = fit@frame, family = family(fit), # same rows and family as the full model
    control = do.call(glmerControl, FIT_CONTROL))))
  chi2 <- max(0, 2 * (as.numeric(logLik(fit)) - as.numeric(logLik(reduced))))
  c(chi2 = chi2, p = pchisq(chi2, 1, lower.tail = FALSE))
}

model_rows <- function(fit, labels) {
  # Export one row per fixed effect, in the order of `labels` (c(term = "label shown")).
  if (inherits(fit, "lmerMod")) {
    # Linear mixed model. lmerTest gives t tests with Satterthwaite df by default, so
    # Kenward-Roger df are requested explicitly. Columns of co: Estimate, Std. Error, df,
    # t value, Pr(>|t|).
    co <- summary(fit, ddf = "Kenward-Roger")$coefficients[names(labels), , drop = FALSE]
    df <- co[, "df"]
    out <- data.frame(b = co[, 1], se = co[, 2],
                      ci_low = co[, 1] - qt(.975, df) * co[, 2],
                      ci_high = co[, 1] + qt(.975, df) * co[, 2],
                      df_num = 1, df_den = df, stat_type = "F", stat = co[, "t value"]^2,
                      p = co[, "Pr(>|t|)"], inference = "Kenward-Roger (lmerTest)")
  } else {
    # Logistic or Poisson mixed model. Columns of co: Estimate, Std. Error, z value, Pr(>|z|).
    # b, SE and the 95% CI are Wald (b +/- 1.96 * SE); the intercept is tested with Wald z.
    co <- summary(fit)$coefficients[names(labels), , drop = FALSE]
    out <- data.frame(b = co[, 1], se = co[, 2],
                      ci_low = co[, 1] - 1.96 * co[, 2], ci_high = co[, 1] + 1.96 * co[, 2],
                      df_num = NA, df_den = NA, stat_type = "z", stat = co[, 3], p = co[, 4],
                      inference = "Wald z", row.names = names(labels))
    # In all predictors except the intercept the Wald test is replaced by a likelihood-ratio
    # test, chi2(1), comparing the model with and without that predictor (see lrt_chi2).
    for (tm in setdiff(names(labels), "(Intercept)")) {
      lr <- lrt_chi2(fit, tm)
      out[tm, c("df_num", "stat_type", "stat", "p", "inference")] <-
        list(1, "chi2", lr[["chi2"]], lr[["p"]], "likelihood-ratio test (type III)")
    }
  }
  data.frame(predictor = unname(labels), term = names(labels), out,
             random = attr(fit, "random"), singular = isSingular(fit, tol = 1e-4),
             ladder = attr(fit, "ladder"), n_obs = nobs(fit),
             n_subjects = nlevels(droplevels(fit@frame$subID)), row.names = NULL)
}

term_p_value <- function(fit, term) model_rows(fit, setNames(term, term))$p

follow_up <- function(fit, effect, within, effect_name, within_names) {
  # Simple effects of `effect` (coded -0.5/+0.5) within each level of `within`,
  # e.g. ketamine - placebo within stable and within volatile.
  # Holm across the 2 tests; CIs Bonferroni-adjusted; Kenward-Roger df for LMMs.
  at <- setNames(list(c(-0.5, 0.5), c(-0.5, 0.5)), c(effect, within))
  emm <- if (inherits(fit, "lmerMod"))
    emmeans(fit, as.formula(paste("~", effect, "*", within)), at = at, lmer.df = "kenward-roger")
  else regrid(emmeans(fit, as.formula(paste("~", effect, "*", within)), at = at))  # response scale
  s <- as.data.frame(summary(contrast(emm, method = setNames(list(c(-1, 1)), effect_name), by = within),
                             by = NULL, adjust = "holm", infer = c(TRUE, TRUE)))
  lmm <- inherits(fit, "lmerMod")
  data.frame(predictor = sprintf("%s: %s", within_names, effect_name), term = "",
             b = s$estimate, se = s$SE,
             ci_low = if (lmm) s$lower.CL else s$asymp.LCL,
             ci_high = if (lmm) s$upper.CL else s$asymp.UCL,
             df_num = NA, df_den = if (lmm) s$df else NA,
             stat_type = if (lmm) "t" else "z", stat = if (lmm) s$t.ratio else s$z.ratio,
             p = s$p.value,
             inference = if (lmm) "emmeans follow-up, Holm (model scale)"
                         else "emmeans follow-up, Holm (response scale)",
             random = "", singular = NA, ladder = "", n_obs = NA, n_subjects = NA)
}

drug_by_volatility_follow_ups <- function(fit) {
  # Both follow-up families of a drug x volatility interaction, only if p < .05.
  if (term_p_value(fit, "drug_c:volatility_c") >= .05) return(NULL)
  rbind(follow_up(fit, "drug_c", "volatility_c", "ketamine - placebo", c("Stable", "Volatile")),
        follow_up(fit, "volatility_c", "drug_c", "volatile - stable", c("Placebo", "Ketamine")))
}

paired_t_test <- function(x, y) {
  # two-sided paired t-test of x - y, as a table row
  tt <- t.test(x, y, paired = TRUE)
  data.frame(term = "paired_t", b = unname(tt$estimate), se = unname(tt$stderr),
             ci_low = tt$conf.int[1], ci_high = tt$conf.int[2], df_num = NA,
             df_den = unname(tt$parameter), stat_type = "t", stat = unname(tt$statistic),
             p = tt$p.value, inference = "paired t-test", random = "", singular = NA,
             ladder = "", n_obs = length(x), n_subjects = length(x))
}


# ------------------------------------------------------------------ output

apa_text <- function(r) {
  # APA-style text for one row, e.g. "b = 0.077, 95% CI [0.015, 0.139], F(1, 84) = 6.03, p = .016"
  f3 <- function(x) formatC(round(x, 3), format = "f", digits = 3)
  df_txt <- function(d) if (abs(d - round(d)) < 0.05) sprintf("%d", round(d)) else sprintf("%.1f", d)
  est <- if (is.na(r$b)) "" else
    sprintf("%s = %s, 95%% CI [%s, %s], ",
            if (r$term == "paired_t") "M_diff" else if (grepl("response scale", r$inference)) "D" else "b",
            f3(r$b), f3(r$ci_low), f3(r$ci_high))
  st <- switch(r$stat_type,
               F = sprintf("F(%s, %s) = %.2f", r$df_num, df_txt(r$df_den), r$stat),
               chi2 = sprintf("chi2(1) = %.2f", r$stat),
               t = sprintf("t(%s) = %.2f", df_txt(r$df_den), r$stat),
               z = sprintf("z = %.2f", r$stat))
  p_name <- if (grepl("Holm", r$inference)) "p_Holm" else "p"
  p_txt <- if (r$p < .001) "< .001" else paste("=", sub("^0", "", formatC(r$p, format = "f", digits = 3)))
  paste0(est, st, ", ", p_name, " ", p_txt)
}

write_tables <- function(rows, out_dir, csv_name = "all_rows.csv") {
  # <csv_name>: every number, one row per predictor / follow-up.
  # table_<id>.txt: the same rows laid out as in the supplementary tables.
  rows$apa <- vapply(seq_len(nrow(rows)), function(i) apa_text(rows[i, ]), "")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(rows, file.path(out_dir, csv_name), row.names = FALSE)
  for (tab in unique(rows$table)) {
    x <- rows[rows$table == tab, ]
    lines <- character(0)
    for (m in unique(x$model)) {
      y <- x[x$model == m, ]
      re <- if (y$random[1] == "") "" else
        sprintf("   random effects: %s%s", y$random[1], if (isTRUE(y$singular[1])) " (singular)" else "")
      txt <- y$apa
      if ("p_without_covariates" %in% names(y))         # Table S9 only
        txt <- sprintf("%s   (without covariates: p = %s)", txt, sub("^0", "", sprintf("%.3f", y$p_without_covariates)))
      lines <- c(lines, "", paste0(m, re), sprintf("    %-45s %s", y$predictor, txt))
    }
    writeLines(lines[-1], file.path(out_dir, sprintf("table_%s.txt", tab)))
  }
  message(sprintf("%d rows in %d tables -> %s", nrow(rows), length(unique(rows$table)), out_dir))
}
