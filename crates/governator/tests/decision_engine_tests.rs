use inferno_governator::decision_engine::{DecisionEngine, ResourceDecision};

#[tokio::test]
async fn test_decision_engine_scale_up_low_performance() {
    let engine = DecisionEngine::new(100.0);

    // Low performance should trigger scale up regardless of cost
    let decision = engine.make_decision(50.0, 0.3).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleUp);

    let decision = engine.make_decision(200.0, 0.4).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleUp);
}

#[tokio::test]
async fn test_decision_engine_scale_down_high_cost_good_performance() {
    let engine = DecisionEngine::new(100.0);

    // High cost with good performance should scale down
    let decision = engine.make_decision(150.0, 0.9).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleDown);

    let decision = engine.make_decision(120.0, 0.81).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleDown);
}

#[tokio::test]
async fn test_decision_engine_no_change_scenarios() {
    let engine = DecisionEngine::new(100.0);

    // Low cost, adequate performance
    let decision = engine.make_decision(50.0, 0.7).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);

    // Cost at threshold, adequate performance
    let decision = engine.make_decision(100.0, 0.6).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);

    // High cost but performance not high enough for scale down
    let decision = engine.make_decision(150.0, 0.7).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);
}

#[tokio::test]
async fn test_decision_engine_edge_cases() {
    let engine = DecisionEngine::new(100.0);

    // Exactly at performance threshold for scale up
    let decision = engine.make_decision(50.0, 0.5).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);

    // Just below performance threshold for scale up
    let decision = engine.make_decision(50.0, 0.49).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleUp);

    // Exactly at cost threshold
    let decision = engine.make_decision(100.0, 0.9).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);

    // Just above cost threshold with high performance (needs > 0.8)
    let decision = engine.make_decision(100.1, 0.81).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleDown);
}

#[tokio::test]
async fn test_decision_engine_different_cost_thresholds() {
    // Test with very low threshold
    let engine = DecisionEngine::new(10.0);
    let decision = engine.make_decision(50.0, 0.9).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleDown);

    // Test with very high threshold
    let engine = DecisionEngine::new(1000.0);
    let decision = engine.make_decision(500.0, 0.9).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);
}

#[tokio::test]
async fn test_decision_engine_extreme_values() {
    let engine = DecisionEngine::new(100.0);

    // Zero performance
    let decision = engine.make_decision(50.0, 0.0).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleUp);

    // Perfect performance
    let decision = engine.make_decision(200.0, 1.0).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleDown);

    // Zero cost
    let decision = engine.make_decision(0.0, 0.6).await.unwrap();
    assert_eq!(decision, ResourceDecision::NoChange);

    // Very high cost
    let decision = engine.make_decision(999999.0, 0.9).await.unwrap();
    assert_eq!(decision, ResourceDecision::ScaleDown);
}

#[test]
fn test_resource_decision_debug_and_partial_eq() {
    // Test Debug trait
    let scale_up = ResourceDecision::ScaleUp;
    let debug_output = format!("{:?}", scale_up);
    assert_eq!(debug_output, "ScaleUp");

    // Test PartialEq trait
    assert_eq!(ResourceDecision::ScaleUp, ResourceDecision::ScaleUp);
    assert_eq!(ResourceDecision::ScaleDown, ResourceDecision::ScaleDown);
    assert_eq!(ResourceDecision::NoChange, ResourceDecision::NoChange);

    assert_ne!(ResourceDecision::ScaleUp, ResourceDecision::ScaleDown);
    assert_ne!(ResourceDecision::ScaleUp, ResourceDecision::NoChange);
    assert_ne!(ResourceDecision::ScaleDown, ResourceDecision::NoChange);
}
