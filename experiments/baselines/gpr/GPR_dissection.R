#!/usr/bin/env Rscript

# =========================================

library(mvtnorm)
library(psych)
library(quadprog)
library(Matrix)
library(lmtest)
library(reticulate)
library(jsonlite)

Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")

source("Fun20210910-1.R")

cat("=================================================================\n")
cat("Sequential Shift Detection - All Nodes\n")
cat("=================================================================\n\n")

np <- import("numpy", convert = FALSE)
node_sizes <- c("nodes_20")
graph_types <- c("SF")
data_base <- "/mlainas/statduck/data_small"
results_base <- "results/shift_detection"
threshold <- 10^(-5)
dir.create(results_base, showWarnings = FALSE, recursive = TRUE)

total_tasks <- 0
total_completed <- 0
total_failed <- 0
start_time <- Sys.time()

cat(sprintf("Started at: %s\n\n", format(start_time, "%Y-%m-%d %H:%M:%S")))

for (node_size in node_sizes) {
  for (graph_type in graph_types) {
    cat(sprintf("==========================================\n"))
    cat(sprintf("Processing: %s / %s\n", node_size, graph_type))
    cat(sprintf("==========================================\n\n"))
    
    data_dir <- file.path(data_base, node_size, graph_type)
    results_dir <- file.path(results_base, node_size, graph_type)
    dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
    
    for (dataset_idx in 1:30) {
      metadata_file <- file.path(data_dir, sprintf("metadata_%d.json", dataset_idx))
      
      if (!file.exists(metadata_file)) {
        cat(sprintf("  [Dataset %d] Metadata not found, skipping\n", dataset_idx))
        next
      }
      
      metadata <- fromJSON(metadata_file)
      shifted_nodes <- metadata$shifted_nodes
      
      if (length(shifted_nodes) == 0) {
        cat(sprintf("  [Dataset %d] No shifted nodes, skipping\n", dataset_idx))
        next
      }
      
      cat(sprintf("  [Dataset %d] Processing %d shifted nodes: %s\n", 
                  dataset_idx, length(shifted_nodes), paste(shifted_nodes, collapse=", ")))
      
      tryCatch({
        data_env1 <- as.data.frame(py_to_r(np$load(file.path(data_dir, sprintf("data_env1_%d.npy", dataset_idx)))))
        data_env2 <- as.data.frame(py_to_r(np$load(file.path(data_dir, sprintf("data_env2_%d.npy", dataset_idx)))))
        adj_matrix <- py_to_r(np$load(file.path(data_dir, sprintf("adj_%d.npy", dataset_idx))))
        
        colnames(data_env1) <- paste0("X", 0:(ncol(data_env1) - 1))
        colnames(data_env2) <- paste0("X", 0:(ncol(data_env2) - 1))
      }, error = function(e) {
        cat(sprintf("  [Dataset %d] Error loading data: %s\n", dataset_idx, as.character(e)))
        return(NULL)
      })
      
      shift_types <- metadata$shift_types
      if(is.null(shift_types)) shift_types <- list()
      
      all_shifted <- unique(c(shifted_nodes, as.integer(names(shift_types))))
      function_shifted <- c()
      noise_shifted <- c()
      
      for(node in all_shifted) {
        node_key <- as.character(node)
        labels <- shift_types[[node_key]]
        if(is.null(labels)) next
        if(!is.list(labels) && !is.vector(labels)) labels <- list(labels)
        labels <- labels[!sapply(labels, is.null)]
        has_noise_shift <- any(grepl("^noise_", labels))
        if(has_noise_shift) {
          noise_shifted <- c(noise_shifted, node)
        } else {
          function_shifted <- c(function_shifted, node)
        }
      }
      
      for (node_idx in shifted_nodes) {
        total_tasks <- total_tasks + 1
        result_file <- file.path(results_dir, sprintf("result_dataset_%d_node_%d.json", dataset_idx, node_idx))

        if (file.exists(result_file)) {
          cat(sprintf("    [Node %d] Already processed, skipping\n", node_idx))
          total_completed <- total_completed + 1
          next
        }

        node_start <- Sys.time()
        if(node_idx %in% function_shifted) {
          true_shift <- "function"
        } else if(node_idx %in% noise_shifted) {
          true_shift <- "noise"
        } else {
          true_shift <- "none"
        }

        result <- list(
          node_size = node_size,
          graph_type = graph_type,
          dataset = dataset_idx,
          node = node_idx,
          true_shift = true_shift,
          status = "processing",
          timestamp_start = format(node_start, "%Y-%m-%d %H:%M:%S")
        )

        elapsed <- NA
        tryCatch({
          parents <- which(adj_matrix[, node_idx + 1] == 1) - 1

          if(length(parents) == 0) {
            result$status <- "no_parents"
            result$shift_detected <- "no_parents"
            elapsed <- as.numeric(difftime(Sys.time(), node_start, units="secs"))
            cat(sprintf("    [Node %d] No parents, skipping (%.1fs)\n", 
                        node_idx, elapsed))
          } else {
            node_col <- paste0("X", node_idx)
            parent_cols <- paste0("X", parents)

            Y_env1 <- data_env1[[node_col]]
            Y_env2 <- data_env2[[node_col]]
            Y_combined <- c(Y_env1, Y_env2)

            X_env1 <- as.matrix(data_env1[, parent_cols, drop=FALSE])
            X_env2 <- as.matrix(data_env2[, parent_cols, drop=FALSE])
            X_combined <- rbind(X_env1, X_env2)

            Z <- c(rep(1, length(Y_env1)), rep(2, length(Y_env2)))

            X_combined <- as.matrix(X_combined)
            X_combined <- t((t(X_combined) - colMeans(X_combined)) / apply(X_combined, 2, sd))
            Y_combined <- (Y_combined - mean(Y_combined)) / sd(Y_combined)
            kernel.list <- list(name = "Gaussian", alpha = 0.5, singular = 10^(-5))
            kernel.result <- kernel.matrix(X_combined, kernel.list = kernel.list, cov.mat = NULL)

            est.H0 <- mle.H0(Y_combined, kernel.result, threshold)
            result$H0 <- list(loglik = est.H0$log.lik, delta2 = est.H0$delta2, sigma2 = est.H0$sigma02)
            est.H1.hetero <- mle.H0.heterogeneous(Y_combined, kernel.result, Z, threshold)
            result$H1_hetero <- list(loglik = est.H1.hetero$log.lik, delta2 = est.H1.hetero$delta2, 
                                      sigma2_groups = as.list(est.H1.hetero$sigma2.groups))

            var.equal <- TRUE
            cov.alter <- cov.structure(Z, kernel.result, var.equal)
            est.H1 <- mle.H1(Y_combined, cov.alter, var.equal, est.H0, threshold)
            result$H1 <- list(loglik = est.H1$log.lik, converged = est.H1$lik.diff <= threshold)

            delta_hetero <- est.H1.hetero$log.lik - est.H0$log.lik
            delta_H1 <- est.H1$log.lik - est.H0$log.lik
            result$shift_detected <- if(delta_H1 > delta_hetero) "function" else "noise"
            result$correct <- (result$shift_detected == true_shift)
            result$status <- "completed"

            elapsed <- as.numeric(difftime(Sys.time(), node_start, units="secs"))
            check_mark <- if(result$correct) "✓" else "✗"
            cat(sprintf("    [Node %d] %s %s -> %s (%.1fs) %s\n", 
                        node_idx, result$status, true_shift, result$shift_detected, elapsed, check_mark))
            total_completed <- total_completed + 1
          }
        }, error = function(e) {
          result$status <<- "error"
          result$error_message <<- as.character(e)
          elapsed <<- as.numeric(difftime(Sys.time(), node_start, units="secs"))
          cat(sprintf("    [Node %d] ERROR: %s\n", node_idx, as.character(e)))
          total_failed <- total_failed + 1
        })

        result$timestamp_end <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
        result$elapsed_time <- elapsed
        write_json(result, result_file, pretty = TRUE, auto_unbox = TRUE)
      }
      
      cat("\n")
    }
  }
}

end_time <- Sys.time()
total_elapsed <- as.numeric(difftime(end_time, start_time, units="secs"))

cat("=================================================================\n")
cat("FINAL SUMMARY\n")
cat("=================================================================\n")
cat(sprintf("Total tasks: %d\n", total_tasks))
cat(sprintf("Completed: %d\n", total_completed))
cat(sprintf("Failed: %d\n", total_failed))
cat(sprintf("Total time: %.1f seconds (%.1f minutes)\n", total_elapsed, total_elapsed/60))
if (total_completed > 0) {
  cat(sprintf("Average time per task: %.1f seconds\n", total_elapsed/total_completed))
}
cat(sprintf("Finished at: %s\n", format(end_time, "%Y-%m-%d %H:%M:%S")))
cat("=================================================================\n")

# Slack 알림 발송
slack_message <- sprintf(
  "🔬 *GPR Shift Detection 완료*\\n\\n• 총 작업: %d\\n• 완료: %d\\n• 실패: %d\\n• 총 소요시간: %.1f분\\n• 평균 작업당 시간: %.1f초\\n\\n⏰ 완료 시각: %s",
  total_tasks, total_completed, total_failed, 
  total_elapsed/60, 
  ifelse(total_completed > 0, total_elapsed/total_completed, 0),
  format(end_time, "%Y-%m-%d %H:%M:%S")
)

slack_webhook <- "https://hooks.slack.com/services/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"
slack_payload <- sprintf('{"text": "%s"}', slack_message)
system(sprintf('curl -s -X POST -H "Content-Type: application/json" -d \'%s\' %s', slack_payload, slack_webhook))
cat("\n✉️ Slack 알림 발송 완료\n")
