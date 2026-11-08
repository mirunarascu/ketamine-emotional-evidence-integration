# Questionnaire statistics (Supplementary Results, Fig S6), in-person sample.
#
#   Rscript analysis/stats/04_questionnaire_stats.R
#
# BPRS and CADSS-6 (before, during and after infusion): scores are at or near the floor
#   except during ketamine, so ketamine and placebo are compared at each timepoint with a
#   paired Wilcoxon signed-rank test, Holm-corrected across the timepoints that can be
#   tested (BPRS: during only, because all before/after pairs are equal).
#   Effect size r = z / sqrt(n pairs), with z recovered from the two-sided p value.
# PANAS negative and positive affect (before and after infusion): linear mixed model
#   score ~ drug * timepoint + visit + (1 | subject) + (1 | subject:drug)
#   (random intercept per participant and per infusion visit), drug and visit coded
#   -0.5 / +0.5, Kenward-Roger F tests. Follow-up: ketamine - placebo at each
#   timepoint (emmeans, Kenward-Roger df), Holm-corrected across the 2 timepoints.
#
# Missing data: one participant's after-infusion CADSS-6 at the placebo visit is missing,
#   so the CADSS "after" test uses the 28 participants with both visits (n_pairs = 28 in
#   the output). All other questionnaire timepoints are complete (29 pairs).
#
# Input:   data/questionnaires_in_person.csv
# Outputs: output/stats/in_person/questionnaires_wilcoxon.csv   (BPRS, CADSS)
#          output/stats/in_person/questionnaires_panas_lmm.csv  (PANAS)

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
# ---- To run this script line by line (e.g. in RStudio): uncomment the 2 lines below,
# ---- set REPO_DIR to the folder of this repository, and comment out the "terminal" lines.
# REPO_DIR <- "~/path/to/ketamine-emotional-evidence-integration"
# source(file.path(REPO_DIR, "analysis", "stats", "stats_functions.R"))

# ---- Terminal (default): Rscript analysis/stats/04_questionnaire_stats.R
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
                 "stats_functions.R"))
out_dir <- file.path(repo_dir(), "output", "stats", "in_person")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(file.path(data_dir(), "questionnaires_in_person.csv"), stringsAsFactors = FALSE)
d$subID <- factor(d$subID)


# ------------------------------------------------------------------ BPRS, CADSS: paired Wilcoxon
wilcoxon <- list()
for (m in c("BPRS", "CADSS")) {
  w <- list()
  for (tp in c("before", "during", "after")) {
    x <- d[d$measure == m & d$timepoint == tp, ]
    pair <- merge(x[x$drug == "ketamine", c("subID", "score")], x[x$drug == "placebo", c("subID", "score")],
                  by = "subID", suffixes = c("_ket", "_pla"))      # participants with both visits
    diff <- pair$score_ket - pair$score_pla
    p <- if (all(diff == 0)) NA else                           # e.g. BPRS before: all pairs equal
      suppressWarnings(wilcox.test(pair$score_ket, pair$score_pla, paired = TRUE, exact = FALSE))$p.value
    z <- abs(qnorm(p / 2))
    w[[tp]] <- data.frame(measure = m, timepoint = tp, n_pairs = nrow(pair),
                          mean_ketamine = mean(pair$score_ket), mean_placebo = mean(pair$score_pla),
                          n_higher_ketamine = sum(diff > 0), n_higher_placebo = sum(diff < 0),
                          n_ketamine_above_0 = sum(pair$score_ket > 0),  # CADSS: any dissociative symptom
                          max_ketamine = max(pair$score_ket),
                          z = z, r = z / sqrt(nrow(pair)), p = p)
  }
  w <- do.call(rbind, w)
  w$p_holm <- p.adjust(w$p, method = "holm")                     # across timepoints (NA ignored)
  wilcoxon[[m]] <- w
}
wilcoxon <- do.call(rbind, wilcoxon)
write.csv(wilcoxon, file.path(out_dir, "questionnaires_wilcoxon.csv"), row.names = FALSE)
print(format(wilcoxon, digits = 3), row.names = FALSE)


# ------------------------------------------------------------------ PANAS: mixed model
panas <- list()
for (m in c("PANAS_negative", "PANAS_positive")) {
  x <- d[d$measure == m, ]
  x$drug <- factor(x$drug, levels = c("placebo", "ketamine"))
  x$visit <- factor(x$visit, levels = c(1, 2))
  x$timepoint <- factor(x$timepoint, levels = c("before", "after"))
  contrasts(x$drug) <- c(-0.5, 0.5)
  contrasts(x$visit) <- c(-0.5, 0.5)
  contrasts(x$timepoint) <- contr.sum(2)

  fit <- lmer(score ~ drug * timepoint + visit + (1 | subID) + (1 | subID:drug), data = x)
  a <- anova(fit, type = 3, ddf = "Kenward-Roger")
  rows <- data.frame(measure = m, effect = rownames(a), b = NA, ci_low = NA, ci_high = NA,
                     df1 = a$NumDF, df2 = a$DenDF, stat_type = "F", stat = a$`F value`, p = a$`Pr(>F)`, p_holm = NA)

  # b for the interaction: (after - before on ketamine) - (after - before on placebo)
  em <- emmeans(fit, ~ drug * timepoint, lmer.df = "kenward-roger")                         # pla-before, ket-before, pla-after, ket-after
  ix <- summary(contrast(em, method = list(`change ketamine - change placebo` = c(1, -1, -1, 1))), infer = TRUE)
  rows[rows$effect == "drug:timepoint", c("b", "ci_low", "ci_high")] <- c(ix$estimate, ix$lower.CL, ix$upper.CL)

  # ketamine - placebo at each timepoint, Holm across the 2 timepoints (CIs unadjusted)
  ct <- summary(contrast(emmeans(fit, ~ drug | timepoint, lmer.df = "kenward-roger"), method = list(`ketamine - placebo` = c(-1, 1))),
                infer = TRUE, adjust = "none")
  rows <- rbind(rows, data.frame(measure = m, effect = paste0("ketamine - placebo, ", ct$timepoint),
                                 b = ct$estimate, ci_low = ct$lower.CL, ci_high = ct$upper.CL,
                                 df1 = NA, df2 = ct$df, stat_type = "t", stat = ct$t.ratio, p = ct$p.value,
                                 p_holm = p.adjust(ct$p.value, method = "holm")))
  panas[[m]] <- rows
}
panas <- do.call(rbind, panas)
write.csv(panas, file.path(out_dir, "questionnaires_panas_lmm.csv"), row.names = FALSE)
print(format(panas, digits = 3), row.names = FALSE)
