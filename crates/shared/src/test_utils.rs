//! Test Utilities
//!
//! Common test utilities shared across all test modules to avoid port conflicts
//! and provide consistent testing infrastructure.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU16, Ordering};

/// Global port counter to ensure unique ports across all tests
static GLOBAL_TEST_PORT_COUNTER: AtomicU16 = AtomicU16::new(50000);

/// Generate a unique random port address for testing to avoid conflicts
/// 
/// This function uses a global atomic counter starting at port 50000
/// and wrapping back to 50000 at 65000 to ensure unique port allocation
/// across all test files and concurrent test execution.
pub fn get_random_port_addr() -> SocketAddr {
    let port = GLOBAL_TEST_PORT_COUNTER.fetch_add(1, Ordering::SeqCst);
    let port = if port > 64000 { 
        GLOBAL_TEST_PORT_COUNTER.store(50000, Ordering::SeqCst);
        50000 
    } else { 
        port 
    };
    
    format!("127.0.0.1:{}", port).parse().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_ports() {
        let port1 = get_random_port_addr();
        let port2 = get_random_port_addr();
        assert_ne!(port1.port(), port2.port());
    }

    #[test]
    fn test_port_range() {
        let addr = get_random_port_addr();
        let port = addr.port();
        assert!(port >= 50000);
        assert!(port <= 64000);
    }
}