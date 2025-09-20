//! GPU Diagnostics and Memory Monitoring
//!
//! This module provides comprehensive GPU memory monitoring, process tracking,
//! and real-time diagnostics for GPU-accelerated model loading and inference.
//!
//! # Features
//!
//! - Real-time GPU memory monitoring
//! - Process-level GPU memory usage tracking
//! - Memory evolution monitoring during model loading
//! - GPU utilization and temperature monitoring
//! - Multi-GPU support
//! - Memory fragmentation analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::time::interval;

/// GPU memory information in MB
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuMemoryInfo {
    /// GPU device ID
    pub device_id: u32,
    /// GPU name/model
    pub gpu_name: String,
    /// Total memory in MB
    pub total_mb: u32,
    /// Used memory in MB
    pub used_mb: u32,
    /// Free memory in MB
    pub free_mb: u32,
    /// Memory utilization percentage (0-100)
    pub utilization_percent: f32,
    /// Temperature in Celsius
    pub temperature_c: Option<u32>,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: Option<u32>,
    /// Timestamp when this measurement was taken
    pub timestamp: u64,
}

/// Process using GPU memory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuProcess {
    /// Process ID
    pub pid: u32,
    /// Process type (C = Compute, G = Graphics)
    pub process_type: String,
    /// Process name/command
    pub process_name: String,
    /// GPU memory usage in MB
    pub gpu_memory_mb: u32,
}

/// Complete GPU status including memory and processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    /// Memory information
    pub memory: GpuMemoryInfo,
    /// Processes using GPU
    pub processes: Vec<GpuProcess>,
    /// Driver version
    pub driver_version: Option<String>,
    /// CUDA version
    pub cuda_version: Option<String>,
}

/// Memory usage snapshot for tracking evolution over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Memory used in MB
    pub used_mb: u32,
    /// Memory free in MB
    pub free_mb: u32,
    /// Optional label for this snapshot (e.g., "model_loading_start")
    pub label: Option<String>,
}

/// Memory evolution tracker for monitoring memory usage over time
#[derive(Debug, Clone)]
pub struct MemoryEvolution {
    /// Device ID being monitored
    pub device_id: u32,
    /// Memory snapshots over time
    pub snapshots: Vec<MemorySnapshot>,
    /// Start time of monitoring
    pub start_time: Instant,
}

/// GPU diagnostics manager
pub struct GpuDiagnostics {
    /// Interval for periodic monitoring
    monitoring_interval: Duration,
    /// Broadcast sender for real-time updates
    update_sender: broadcast::Sender<GpuStatus>,
    /// Memory evolution trackers by device ID
    memory_trackers: Arc<Mutex<HashMap<u32, MemoryEvolution>>>,
    /// Whether monitoring is currently active
    is_monitoring: Arc<Mutex<bool>>,
}

impl Default for GpuMemoryInfo {
    fn default() -> Self {
        Self {
            device_id: 0,
            gpu_name: "Unknown".to_string(),
            total_mb: 0,
            used_mb: 0,
            free_mb: 0,
            utilization_percent: 0.0,
            temperature_c: None,
            gpu_utilization_percent: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

impl MemoryEvolution {
    /// Create a new memory evolution tracker
    pub fn new(device_id: u32) -> Self {
        Self {
            device_id,
            snapshots: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Add a memory snapshot
    pub fn add_snapshot(&mut self, used_mb: u32, free_mb: u32, label: Option<String>) {
        let timestamp_ms = self.start_time.elapsed().as_millis() as u64;
        self.snapshots.push(MemorySnapshot {
            timestamp_ms,
            used_mb,
            free_mb,
            label,
        });
    }

    /// Get memory usage delta from start
    pub fn memory_delta_mb(&self) -> Option<i32> {
        if let (Some(first), Some(last)) = (self.snapshots.first(), self.snapshots.last()) {
            Some(last.used_mb as i32 - first.used_mb as i32)
        } else {
            None
        }
    }

    /// Get peak memory usage
    pub fn peak_memory_mb(&self) -> Option<u32> {
        self.snapshots.iter().map(|s| s.used_mb).max()
    }

    /// Get memory usage at specific label
    pub fn memory_at_label(&self, label: &str) -> Option<&MemorySnapshot> {
        self.snapshots
            .iter()
            .find(|s| s.label.as_deref() == Some(label))
    }
}

impl GpuDiagnostics {
    /// Create a new GPU diagnostics manager
    pub fn new(monitoring_interval: Duration) -> Self {
        let (update_sender, _) = broadcast::channel(100);

        Self {
            monitoring_interval,
            update_sender,
            memory_trackers: Arc::new(Mutex::new(HashMap::new())),
            is_monitoring: Arc::new(Mutex::new(false)),
        }
    }

    /// Create with default 1-second monitoring interval
    pub fn new_default() -> Self {
        Self::new(Duration::from_secs(1))
    }

    /// Get a receiver for real-time GPU status updates
    pub fn subscribe(&self) -> broadcast::Receiver<GpuStatus> {
        self.update_sender.subscribe()
    }

    /// Start periodic monitoring
    pub async fn start_monitoring(&self) {
        let mut is_monitoring = self.is_monitoring.lock().unwrap();
        if *is_monitoring {
            return; // Already monitoring
        }
        *is_monitoring = true;
        drop(is_monitoring);

        let sender = self.update_sender.clone();
        let interval_duration = self.monitoring_interval;
        let is_monitoring = self.is_monitoring.clone();

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                interval.tick().await;

                // Check if monitoring should continue
                {
                    let monitoring = is_monitoring.lock().unwrap();
                    if !*monitoring {
                        break;
                    }
                }

                // Get GPU status
                if let Ok(status) = get_gpu_status(0).await {
                    // Ignore send errors (no receivers)
                    let _ = sender.send(status);
                }
            }
        });
    }

    /// Stop periodic monitoring
    pub fn stop_monitoring(&self) {
        let mut is_monitoring = self.is_monitoring.lock().unwrap();
        *is_monitoring = false;
    }

    /// Start memory evolution tracking for a specific device
    pub fn start_memory_tracking(&self, device_id: u32, initial_label: Option<String>) {
        let mut trackers = self.memory_trackers.lock().unwrap();
        let mut evolution = MemoryEvolution::new(device_id);

        // Add initial snapshot
        if let Ok(memory) = get_gpu_memory_info(device_id) {
            evolution.add_snapshot(memory.used_mb, memory.free_mb, initial_label);
        }

        trackers.insert(device_id, evolution);
    }

    /// Add a labeled snapshot to memory tracking
    pub fn add_memory_snapshot(&self, device_id: u32, label: Option<String>) {
        let mut trackers = self.memory_trackers.lock().unwrap();
        if let Some(evolution) = trackers.get_mut(&device_id) {
            if let Ok(memory) = get_gpu_memory_info(device_id) {
                evolution.add_snapshot(memory.used_mb, memory.free_mb, label);
            }
        }
    }

    /// Get memory evolution for a device
    pub fn get_memory_evolution(&self, device_id: u32) -> Option<MemoryEvolution> {
        let trackers = self.memory_trackers.lock().unwrap();
        trackers.get(&device_id).cloned()
    }

    /// Stop memory tracking for a device
    pub fn stop_memory_tracking(&self, device_id: u32) -> Option<MemoryEvolution> {
        let mut trackers = self.memory_trackers.lock().unwrap();
        trackers.remove(&device_id)
    }

    /// Display memory evolution summary
    pub fn display_memory_evolution(&self, device_id: u32) -> Option<String> {
        let evolution = self.get_memory_evolution(device_id)?;

        let mut output = String::new();
        output.push_str(&format!("Memory Evolution for GPU {}\n", device_id));
        output.push_str(&format!(
            "Duration: {:.1}s\n",
            evolution.start_time.elapsed().as_secs_f64()
        ));

        if let Some(delta) = evolution.memory_delta_mb() {
            output.push_str(&format!("Memory delta: {:+} MB\n", delta));
        }

        if let Some(peak) = evolution.peak_memory_mb() {
            output.push_str(&format!("Peak memory: {} MB\n", peak));
        }

        output.push_str("\nTimeline:\n");
        for snapshot in &evolution.snapshots {
            let label = snapshot.label.as_deref().unwrap_or("measurement");
            output.push_str(&format!(
                "  {:6.1}s: {} MB used ({}) {}\n",
                snapshot.timestamp_ms as f64 / 1000.0,
                snapshot.used_mb,
                label,
                if snapshot.label.is_some() {
                    "[LABEL]"
                } else {
                    ""
                }
            ));
        }

        Some(output)
    }
}

/// Get GPU memory information using nvidia-smi
pub fn get_gpu_memory_info(device_id: u32) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.memory,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
            &format!("--id={}", device_id),
        ])
        .output()?;

    let output_str = String::from_utf8(output.stdout)?;
    let line = output_str.trim();

    if line.is_empty() {
        return Err("No GPU found with the specified device ID".into());
    }

    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() < 8 {
        return Err("Invalid nvidia-smi output format".into());
    }

    let device_id = parts[0].parse()?;
    let gpu_name = parts[1].to_string();
    let used_mb = parts[2].parse()?;
    let free_mb = parts[3].parse()?;
    let total_mb = parts[4].parse()?;
    let utilization_percent = parts[5].parse().unwrap_or(0.0);
    let temperature_c = parts[6].parse().ok();
    let gpu_utilization_percent = parts[7].parse().ok();

    Ok(GpuMemoryInfo {
        device_id,
        gpu_name,
        total_mb,
        used_mb,
        free_mb,
        utilization_percent,
        temperature_c,
        gpu_utilization_percent,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

/// Get processes using GPU memory
pub fn get_gpu_processes() -> Result<Vec<GpuProcess>, Box<dyn std::error::Error>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()?;

    let output_str = String::from_utf8(output.stdout)?;
    let mut processes = Vec::new();

    for line in output_str.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(", ").collect();
        if parts.len() >= 2 {
            if let (Ok(pid), Ok(gpu_memory_mb)) = (parts[0].parse(), parts[1].parse()) {
                // Try to get process name
                let process_name = get_process_name(pid).unwrap_or_else(|| format!("PID {}", pid));

                processes.push(GpuProcess {
                    pid,
                    process_type: "C".to_string(), // Compute
                    process_name,
                    gpu_memory_mb,
                });
            }
        }
    }

    Ok(processes)
}

/// Get complete GPU status
pub async fn get_gpu_status(device_id: u32) -> Result<GpuStatus, Box<dyn std::error::Error>> {
    let memory = get_gpu_memory_info(device_id)?;
    let processes = get_gpu_processes().unwrap_or_default();

    // Get driver and CUDA versions
    let (driver_version, cuda_version) = get_driver_versions();

    Ok(GpuStatus {
        memory,
        processes,
        driver_version,
        cuda_version,
    })
}

/// Get process name by PID
fn get_process_name(pid: u32) -> Option<String> {
    let output = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output()
        .ok()?;

    let name = String::from_utf8(output.stdout).ok()?;
    Some(name.trim().to_string())
}

/// Get NVIDIA driver and CUDA versions
fn get_driver_versions() -> (Option<String>, Option<String>) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let driver_version = output
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    // CUDA version from nvidia-smi header
    let cuda_output = Command::new("nvidia-smi").output();
    let cuda_version = cuda_output
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|output| {
            // Extract CUDA version from first line
            output
                .lines()
                .next()?
                .split("CUDA Version: ")
                .nth(1)?
                .split_whitespace()
                .next()
                .map(|s| s.to_string())
        });

    (driver_version, cuda_version)
}

/// Convenience function to monitor model loading memory evolution
pub async fn monitor_model_loading<F, Fut, T>(
    device_id: u32,
    model_name: &str,
    loading_fn: F,
) -> Result<(T, MemoryEvolution), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, Box<dyn std::error::Error>>>,
{
    let diagnostics = GpuDiagnostics::new_default();

    // Start tracking
    diagnostics.start_memory_tracking(device_id, Some(format!("{}_start", model_name)));
    println!(
        "Starting memory monitoring for {} on GPU {}",
        model_name, device_id
    );

    // Monitor during loading
    let loading_result = {
        diagnostics.add_memory_snapshot(device_id, Some("before_loading".to_string()));

        let result = loading_fn().await;

        diagnostics.add_memory_snapshot(device_id, Some("after_loading".to_string()));
        result
    };

    // Get final evolution
    let evolution = diagnostics
        .stop_memory_tracking(device_id)
        .ok_or("No memory tracking data")?;

    // Display summary
    if let Some(summary) = diagnostics.display_memory_evolution(device_id) {
        println!("{}", summary);
    }

    match loading_result {
        Ok(result) => Ok((result, evolution)),
        Err(e) => {
            diagnostics.add_memory_snapshot(device_id, Some("loading_failed".to_string()));
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_evolution() {
        let mut evolution = MemoryEvolution::new(0);
        evolution.add_snapshot(1000, 23000, Some("start".to_string()));
        evolution.add_snapshot(5000, 19000, Some("loading".to_string()));
        evolution.add_snapshot(4500, 19500, Some("end".to_string()));

        assert_eq!(evolution.memory_delta_mb(), Some(3500));
        assert_eq!(evolution.peak_memory_mb(), Some(5000));
        assert!(evolution.memory_at_label("loading").is_some());
    }

    #[tokio::test]
    async fn test_gpu_memory_info() {
        // This test will only work if nvidia-smi is available
        match get_gpu_memory_info(0) {
            Ok(info) => {
                assert!(info.total_mb > 0);
                assert!(info.used_mb <= info.total_mb);
                assert!(info.free_mb <= info.total_mb);
            }
            Err(_) => {
                // Skip test if no GPU available
                println!("GPU not available for testing");
            }
        }
    }
}
