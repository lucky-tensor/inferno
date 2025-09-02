//! SWIM Protocol Optimizations for 10,000+ Node Clusters
//!
//! This module contains specific optimizations for massive scale SWIM deployments.
//! These optimizations are critical for achieving the performance characteristics
//! needed to support 10,000+ AI inference nodes efficiently.

use super::swim::{SwimMember, MemberState, GossipUpdate};
use super::errors::{ServiceDiscoveryError, ServiceDiscoveryResult};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, instrument};

/// Optimized member storage for massive scale
pub struct CompactMemberStorage {
    /// Primary member storage - optimized for fast lookups
    members: BTreeMap<u32, CompactMember>,
    /// Address to ID mapping for reverse lookups
    address_map: HashMap<std::net::SocketAddr, u32>,
    /// Node ID string to hash mapping
    id_map: HashMap<String, u32>,
    /// Free ID pool for reuse
    free_ids: VecDeque<u32>,
    /// Next available ID
    next_id: u32,
}

/// Highly optimized member representation
#[repr(packed)]
#[derive(Debug, Clone, Copy)]
pub struct CompactMember {
    /// Compact member ID (4 bytes vs 24+ for String)
    pub id: u32,
    /// Socket address (28 bytes for IPv6)
    pub addr: std::net::SocketAddr,
    /// State packed into single byte
    pub state: u8, // MemberState as u8
    /// Incarnation number
    pub incarnation: u32,
    /// Metrics port
    pub metrics_port: u16,
    /// Node flags (type, load balancer, etc.)
    pub flags: u8,
    /// Last probe time (seconds since epoch)
    pub last_probe: u32,
    /// State change time (seconds since epoch)
    pub state_change: u32,
}

impl CompactMember {
    /// Creates compact member from SwimMember
    pub fn from_swim_member(member: &SwimMember, compact_id: u32) -> Self {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
            
        let mut flags = 0u8;
        flags |= (member.node_type as u8) << 0; // Bits 0-1: node type
        flags |= (member.is_load_balancer as u8) << 2; // Bit 2: load balancer
        
        Self {
            id: compact_id,
            addr: member.addr,
            state: member.state as u8,
            incarnation: member.incarnation,
            metrics_port: member.metrics_port,
            flags,
            last_probe: now_secs,
            state_change: now_secs,
        }
    }
    
    /// Converts to full SwimMember
    pub fn to_swim_member(&self, node_id: &str) -> SwimMember {
        let node_type = match self.flags & 0x03 {
            0 => super::types::NodeType::Proxy,
            1 => super::types::NodeType::Backend,
            2 => super::types::NodeType::Governator,
            _ => super::types::NodeType::Backend,
        };
        
        let is_load_balancer = (self.flags & 0x04) != 0;
        let state = match self.state {
            0 => MemberState::Alive,
            1 => MemberState::Suspected,
            2 => MemberState::Dead,
            3 => MemberState::Left,
            _ => MemberState::Alive,
        };
        
        SwimMember {
            id: self.id,
            node_id: node_id.to_string(),
            addr: self.addr,
            state,
            incarnation: self.incarnation,
            metrics_port: self.metrics_port,
            node_type,
            is_load_balancer,
            last_probe_time: Instant::now(), // Approximation
            state_change_time: Instant::now(), // Approximation
            failed_probe_count: 0,
            suspicion_timeout: None,
        }
    }
    
    /// Gets memory size of this member
    pub const fn memory_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl CompactMemberStorage {
    /// Creates new optimized storage
    pub fn new() -> Self {
        Self {
            members: BTreeMap::new(),
            address_map: HashMap::new(),
            id_map: HashMap::new(),
            free_ids: VecDeque::new(),
            next_id: 1,
        }
    }
    
    /// Adds member with automatic ID assignment
    pub fn add_member(&mut self, member: SwimMember) -> u32 {
        let compact_id = self.allocate_id();
        let compact_member = CompactMember::from_swim_member(&member, compact_id);
        
        self.members.insert(compact_id, compact_member);
        self.address_map.insert(member.addr, compact_id);
        self.id_map.insert(member.node_id, compact_id);
        
        compact_id
    }
    
    /// Removes member and frees ID
    pub fn remove_member(&mut self, compact_id: u32) -> Option<CompactMember> {
        if let Some(member) = self.members.remove(&compact_id) {
            self.address_map.remove(&member.addr);
            self.id_map.retain(|_, &mut v| v != compact_id);
            self.free_ids.push_back(compact_id);
            Some(member)
        } else {
            None
        }
    }
    
    /// Gets member by compact ID
    pub fn get_member(&self, compact_id: u32) -> Option<&CompactMember> {
        self.members.get(&compact_id)
    }
    
    /// Gets member by address
    pub fn get_member_by_addr(&self, addr: std::net::SocketAddr) -> Option<&CompactMember> {
        self.address_map.get(&addr)
            .and_then(|&id| self.members.get(&id))
    }
    
    /// Gets all alive members efficiently
    pub fn get_alive_members(&self) -> Vec<u32> {
        self.members
            .iter()
            .filter(|(_, member)| member.state == MemberState::Alive as u8)
            .map(|(&id, _)| id)
            .collect()
    }
    
    /// Gets member count by state
    pub fn count_by_state(&self) -> (usize, usize, usize, usize) {
        let mut counts = (0, 0, 0, 0); // alive, suspected, dead, left
        
        for member in self.members.values() {
            match member.state {
                0 => counts.0 += 1, // Alive
                1 => counts.1 += 1, // Suspected
                2 => counts.2 += 1, // Dead
                3 => counts.3 += 1, // Left
                _ => {}
            }
        }
        
        counts
    }
    
    /// Gets total memory usage
    pub fn memory_usage(&self) -> usize {
        let member_memory = self.members.len() * CompactMember::memory_size();
        let address_map_memory = self.address_map.len() * 
            (std::mem::size_of::<std::net::SocketAddr>() + std::mem::size_of::<u32>());
        let id_map_memory = self.id_map.iter()
            .map(|(k, _)| k.len() + std::mem::size_of::<u32>())
            .sum::<usize>();
        
        member_memory + address_map_memory + id_map_memory
    }
    
    /// Allocates next available ID
    fn allocate_id(&mut self) -> u32 {
        if let Some(id) = self.free_ids.pop_front() {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            id
        }
    }
}

/// Optimized gossip buffer for high-throughput scenarios
pub struct HighThroughputGossipBuffer {
    /// Circular buffer for recent updates
    buffer: Vec<Option<GossipUpdate>>,
    /// Current write position
    write_pos: usize,
    /// Buffer capacity
    capacity: usize,
    /// Update generation counter
    generation: u32,
}

impl HighThroughputGossipBuffer {
    /// Creates new high-throughput buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            write_pos: 0,
            capacity,
            generation: 1,
        }
    }
    
    /// Adds update to buffer (overwrites oldest)
    pub fn add_update(&mut self, mut update: GossipUpdate) {
        update.generation = self.generation;
        self.generation = self.generation.wrapping_add(1);
        
        self.buffer[self.write_pos] = Some(update);
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }
    
    /// Gets recent updates for gossip
    pub fn get_recent_updates(&self, count: usize) -> Vec<GossipUpdate> {
        let mut updates = Vec::with_capacity(count);
        let mut pos = self.write_pos;
        
        for _ in 0..count.min(self.capacity) {
            pos = if pos == 0 { self.capacity - 1 } else { pos - 1 };
            
            if let Some(ref update) = self.buffer[pos] {
                updates.push(update.clone());
                if updates.len() >= count {
                    break;
                }
            }
        }
        
        updates
    }
    
    /// Clears old updates
    pub fn cleanup_old_updates(&mut self, max_age: Duration) {
        let now = Instant::now();
        for slot in &mut self.buffer {
            if let Some(ref update) = slot {
                // In a real implementation, we'd track update timestamps
                // For now, just demonstrate the pattern
                if update.generation < self.generation.saturating_sub(1000) {
                    *slot = None;
                }
            }
        }
    }
}

/// Network-efficient message packing
pub struct MessagePacker {
    /// Compression threshold
    compression_threshold: usize,
    /// Maximum message size
    max_message_size: usize,
}

impl MessagePacker {
    /// Creates new message packer
    pub fn new(compression_threshold: usize, max_message_size: usize) -> Self {
        Self {
            compression_threshold,
            max_message_size,
        }
    }
    
    /// Packs multiple gossip updates into single message
    pub fn pack_gossip_updates(&self, updates: Vec<GossipUpdate>) -> ServiceDiscoveryResult<Vec<u8>> {
        // Serialize updates
        let serialized = bincode::serialize(&updates)
            .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))?;
        
        // Compress if over threshold
        if serialized.len() > self.compression_threshold {
            let compressed = zstd::encode_all(&serialized[..], 3)
                .map_err(|e| ServiceDiscoveryError::CompressionError(e.to_string()))?;
            
            if compressed.len() <= self.max_message_size {
                Ok(compressed)
            } else {
                // Split message if too large even after compression
                self.split_large_message(updates)
            }
        } else {
            Ok(serialized)
        }
    }
    
    /// Unpacks gossip message
    pub fn unpack_gossip_message(&self, data: &[u8]) -> ServiceDiscoveryResult<Vec<GossipUpdate>> {
        // Try decompression first
        if let Ok(decompressed) = zstd::decode_all(data) {
            bincode::deserialize(&decompressed)
                .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))
        } else {
            // Direct deserialization
            bincode::deserialize(data)
                .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))
        }
    }
    
    /// Splits large messages into chunks
    fn split_large_message(&self, updates: Vec<GossipUpdate>) -> ServiceDiscoveryResult<Vec<u8>> {
        // For now, just return the first half
        // In production, this would split into multiple messages
        let half_size = updates.len() / 2;
        let first_half = updates.into_iter().take(half_size).collect();
        
        let serialized = bincode::serialize(&first_half)
            .map_err(|e| ServiceDiscoveryError::SerializationError(e.to_string()))?;
            
        Ok(serialized)
    }
}

/// Adaptive timeout calculator for network conditions
pub struct AdaptiveTimeoutCalculator {
    /// Recent RTT measurements
    rtt_samples: VecDeque<Duration>,
    /// Sample window size
    window_size: usize,
    /// Current average RTT
    avg_rtt: Duration,
    /// RTT variance
    rtt_variance: Duration,
}

impl AdaptiveTimeoutCalculator {
    /// Creates new calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            rtt_samples: VecDeque::with_capacity(window_size),
            window_size,
            avg_rtt: Duration::from_millis(50),
            rtt_variance: Duration::from_millis(10),
        }
    }
    
    /// Records new RTT sample
    pub fn record_rtt(&mut self, rtt: Duration) {
        if self.rtt_samples.len() >= self.window_size {
            self.rtt_samples.pop_front();
        }
        self.rtt_samples.push_back(rtt);
        
        self.recalculate_stats();
    }
    
    /// Gets adaptive probe timeout
    pub fn get_probe_timeout(&self) -> Duration {
        // Conservative timeout: average + 3 * variance
        self.avg_rtt + Duration::from_nanos((self.rtt_variance.as_nanos() * 3) as u64)
    }
    
    /// Gets adaptive suspicion timeout
    pub fn get_suspicion_timeout(&self) -> Duration {
        // Longer timeout for suspicion: 10 * probe timeout
        self.get_probe_timeout() * 10
    }
    
    /// Recalculates average and variance
    fn recalculate_stats(&mut self) {
        if self.rtt_samples.is_empty() {
            return;
        }
        
        // Calculate average
        let sum: u64 = self.rtt_samples.iter().map(|d| d.as_nanos() as u64).sum();
        self.avg_rtt = Duration::from_nanos(sum / self.rtt_samples.len() as u64);
        
        // Calculate variance
        let variance_sum: u64 = self.rtt_samples
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as i64 - self.avg_rtt.as_nanos() as i64;
                (diff * diff) as u64
            })
            .sum();
        
        let variance = variance_sum / self.rtt_samples.len() as u64;
        self.rtt_variance = Duration::from_nanos((variance as f64).sqrt() as u64);
    }
}

/// Performance monitoring for 10k scale
pub struct ScalePerformanceMonitor {
    /// Operation timing samples
    operation_times: HashMap<String, VecDeque<Duration>>,
    /// Member count history
    member_count_history: VecDeque<(Instant, usize)>,
    /// Network throughput tracking
    network_stats: NetworkStats,
}

#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    pub bytes_sent_per_second: f64,
    pub bytes_received_per_second: f64,
    pub messages_sent_per_second: f64,
    pub messages_received_per_second: f64,
    pub packet_loss_rate: f64,
}

impl ScalePerformanceMonitor {
    /// Creates new performance monitor
    pub fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
            member_count_history: VecDeque::new(),
            network_stats: NetworkStats::default(),
        }
    }
    
    /// Records operation timing
    #[instrument(skip(self))]
    pub fn record_operation(&mut self, operation: &str, duration: Duration) {
        let samples = self.operation_times.entry(operation.to_string()).or_insert_with(VecDeque::new);
        
        if samples.len() >= 1000 {
            samples.pop_front();
        }
        samples.push_back(duration);
    }
    
    /// Records member count change
    pub fn record_member_count(&mut self, count: usize) {
        if self.member_count_history.len() >= 100 {
            self.member_count_history.pop_front();
        }
        self.member_count_history.push_back((Instant::now(), count));
    }
    
    /// Gets operation statistics
    pub fn get_operation_stats(&self, operation: &str) -> Option<OperationStats> {
        self.operation_times.get(operation).map(|samples| {
            if samples.is_empty() {
                return OperationStats::default();
            }
            
            let sum: u64 = samples.iter().map(|d| d.as_nanos() as u64).sum();
            let avg = Duration::from_nanos(sum / samples.len() as u64);
            
            let mut sorted_samples: Vec<_> = samples.iter().copied().collect();
            sorted_samples.sort();
            
            let p50 = sorted_samples[samples.len() / 2];
            let p95 = sorted_samples[(samples.len() * 95) / 100];
            let p99 = sorted_samples[(samples.len() * 99) / 100];
            
            OperationStats {
                count: samples.len(),
                average: avg,
                p50,
                p95,
                p99,
                min: *sorted_samples.first().unwrap(),
                max: *sorted_samples.last().unwrap(),
            }
        })
    }
    
    /// Checks if performance is degraded
    pub fn is_performance_degraded(&self) -> bool {
        // Check probe timeout performance
        if let Some(stats) = self.get_operation_stats("probe_timeout") {
            if stats.average > Duration::from_millis(100) {
                return true;
            }
        }
        
        // Check member addition performance
        if let Some(stats) = self.get_operation_stats("add_member") {
            if stats.p95 > Duration::from_millis(10) {
                return true;
            }
        }
        
        // Check network performance
        if self.network_stats.packet_loss_rate > 0.05 { // > 5% loss
            return true;
        }
        
        false
    }
    
    /// Gets scaling recommendations
    pub fn get_scaling_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check current member count
        if let Some((_, current_count)) = self.member_count_history.back() {
            if *current_count > 5000 {
                recommendations.push("Consider splitting cluster into multiple regions".to_string());
            }
            
            if *current_count > 1000 && self.network_stats.packet_loss_rate > 0.01 {
                recommendations.push("Increase gossip compression threshold".to_string());
                recommendations.push("Reduce gossip fanout to manage network load".to_string());
            }
        }
        
        // Check operation performance
        if let Some(stats) = self.get_operation_stats("consensus_resolution") {
            if stats.p95 > Duration::from_millis(50) {
                recommendations.push("SWIM migration is required for this scale".to_string());
            }
        }
        
        recommendations
    }
}

#[derive(Debug, Clone, Default)]
pub struct OperationStats {
    pub count: usize,
    pub average: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub min: Duration,
    pub max: Duration,
}

/// Memory pool for reducing allocations at scale
pub struct MemoryPool<T> {
    /// Available items
    available: VecDeque<T>,
    /// Factory function for creating new items
    factory: Box<dyn Fn() -> T + Send + Sync>,
    /// Maximum pool size
    max_size: usize,
}

impl<T> MemoryPool<T> {
    /// Creates new memory pool
    pub fn new<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            available: VecDeque::new(),
            factory: Box::new(factory),
            max_size,
        }
    }
    
    /// Gets item from pool or creates new one
    pub fn get(&mut self) -> T {
        self.available.pop_front().unwrap_or_else(|| (self.factory)())
    }
    
    /// Returns item to pool
    pub fn put(&mut self, item: T) {
        if self.available.len() < self.max_size {
            self.available.push_back(item);
        }
        // Otherwise drop the item to prevent unbounded growth
    }
    
    /// Gets current pool size
    pub fn size(&self) -> usize {
        self.available.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_discovery::{NodeType, PeerInfo};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::SystemTime;

    #[test]
    fn test_compact_member_storage() {
        let mut storage = CompactMemberStorage::new();
        
        // Add members
        let mut member_ids = Vec::new();
        for i in 0..1000 {
            let member = create_test_swim_member(i);
            let id = storage.add_member(member);
            member_ids.push(id);
        }
        
        // Verify all members present
        assert_eq!(storage.members.len(), 1000);
        
        // Test memory efficiency
        let memory_usage = storage.memory_usage();
        let per_member_usage = memory_usage / 1000;
        
        // Should be much less than unoptimized SwimMember
        assert!(per_member_usage < 200, "Per-member usage too high: {} bytes", per_member_usage);
        
        // Test lookups
        let first_member = storage.get_member(member_ids[0]).unwrap();
        assert_eq!(first_member.id, member_ids[0]);
        
        // Test alive member filtering
        let alive = storage.get_alive_members();
        assert_eq!(alive.len(), 1000);
    }
    
    #[test]
    fn test_high_throughput_gossip_buffer() {
        let mut buffer = HighThroughputGossipBuffer::new(100);
        
        // Add updates
        for i in 0..150 {
            let update = GossipUpdate {
                node_id: i,
                state: MemberState::Alive,
                incarnation: 1,
                member_info: None,
                generation: 0,
            };
            buffer.add_update(update);
        }
        
        // Get recent updates
        let recent = buffer.get_recent_updates(50);
        assert_eq!(recent.len(), 50);
        
        // Should be most recent updates
        assert!(recent.iter().any(|u| u.node_id >= 100));
    }
    
    #[test]
    fn test_message_packer() {
        let packer = MessagePacker::new(500, 1400);
        
        // Create test updates
        let updates: Vec<GossipUpdate> = (0..10).map(|i| GossipUpdate {
            node_id: i,
            state: MemberState::Alive,
            incarnation: 1,
            member_info: None,
            generation: i,
        }).collect();
        
        // Pack updates
        let packed = packer.pack_gossip_updates(updates.clone()).unwrap();
        
        // Unpack and verify
        let unpacked = packer.unpack_gossip_message(&packed).unwrap();
        assert_eq!(unpacked.len(), updates.len());
    }
    
    #[test]
    fn test_adaptive_timeout_calculator() {
        let mut calculator = AdaptiveTimeoutCalculator::new(10);
        
        // Record RTT samples
        for i in 1..=20 {
            calculator.record_rtt(Duration::from_millis(i * 5));
        }
        
        let probe_timeout = calculator.get_probe_timeout();
        let suspicion_timeout = calculator.get_suspicion_timeout();
        
        // Timeouts should be reasonable
        assert!(probe_timeout > Duration::from_millis(10));
        assert!(probe_timeout < Duration::from_secs(1));
        assert!(suspicion_timeout > probe_timeout);
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(|| vec![0u8; 1024], 10);
        
        // Get items
        let item1 = pool.get();
        let item2 = pool.get();
        assert_eq!(item1.len(), 1024);
        
        // Return items
        pool.put(item1);
        pool.put(item2);
        assert_eq!(pool.size(), 2);
        
        // Get item from pool
        let item3 = pool.get();
        assert_eq!(item3.len(), 1024);
        assert_eq!(pool.size(), 1);
    }

    fn create_test_swim_member(id: usize) -> SwimMember {
        let peer_info = PeerInfo {
            id: format!("test-node-{}", id),
            address: format!("127.0.0.{}:8000", (id % 254) + 1),
            metrics_port: 9090,
            node_type: NodeType::Backend,
            is_load_balancer: false,
            last_updated: SystemTime::now(),
        };
        
        SwimMember::from_peer_info(peer_info).unwrap()
    }
}