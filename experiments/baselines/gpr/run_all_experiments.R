#!/usr/bin/env Rscript

# GPR Dissection - All Experiments Runner
# Runs 10 experiments: nodes_10/20/30/40/50 × SF/ER

library(jsonlite)

node_sizes <- c("nodes_10", "nodes_20", "nodes_30", "nodes_40", "nodes_50")
graph_types <- c("SF", "ER")
pause_minutes <- 10

slack_webhook <- "https://hooks.slack.com/services/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"

send_slack <- function(msg) {
  payload <- sprintf('{"text": "%s"}', gsub('"', '\\"', msg))
  system(sprintf('curl -s -X POST -H "Content-Type: application/json" -d \'%s\' %s', payload, slack_webhook))
}

show_system_status <- function() {
  cat("\n📊 System Status:\n")
  system("free -h | head -2")
  cat("\nCPU Load: ")
  system("uptime | awk -F'load average:' '{print $2}'")
  cat("\n")
}

total_experiments <- length(node_sizes) * length(graph_types)
current_exp <- 0

cat("=================================================================\n")
cat(sprintf("GPR All Experiments Runner - %d experiments\n", total_experiments))
cat(sprintf("Started: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat("=================================================================\n")

send_slack(sprintf("🚀 *GPR 전체 실험 시작*\\n• 총 %d개 실험\\n• 시작: %s", 
                   total_experiments, format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

for (ns in node_sizes) {
  for (gt in graph_types) {
    current_exp <- current_exp + 1
    
    cat(sprintf("\n[%d/%d] %s / %s\n", current_exp, total_experiments, ns, gt))
    show_system_status()
    
    # Write temp config and run GPR_dissection.R with modified params
    exp_start <- Sys.time()
    
    # Create temp script with current params
    script_content <- sprintf('
node_sizes <- c("%s")
graph_types <- c("%s")
source("Fun20210910-1.R")
', ns, gt)
    
    # Run the main script with modified params
    system(sprintf('cd /home/statduck/fans/experiments/baselines/gpr && Rscript -e \'
node_sizes <- c("%s")
graph_types <- c("%s")
source("GPR_dissection_core.R")
\'', ns, gt))
    
    exp_elapsed <- as.numeric(difftime(Sys.time(), exp_start, units="mins"))
    
    send_slack(sprintf("✅ *[%d/%d] %s / %s 완료*\\n• 소요: %.1f분", 
                       current_exp, total_experiments, ns, gt, exp_elapsed))
    
    # Pause between experiments (except last one)
    if (current_exp < total_experiments) {
      cat(sprintf("\n⏸️  Pausing %d minutes before next experiment...\n", pause_minutes))
      Sys.sleep(pause_minutes * 60)
    }
  }
}

cat("\n=================================================================\n")
cat("ALL EXPERIMENTS COMPLETED!\n")
cat(sprintf("Finished: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat("=================================================================\n")

send_slack(sprintf("🎉 *GPR 전체 실험 완료!*\\n• 총 %d개 실험 완료\\n• 종료: %s", 
                   total_experiments, format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

