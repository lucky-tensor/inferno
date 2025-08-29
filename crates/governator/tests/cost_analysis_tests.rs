use inferno_governator::cost_analysis::{CostAnalysis, CostAnalyzer};

#[tokio::test]
async fn test_cost_analyzer_creation() {
    let providers = vec!["aws".to_string(), "gcp".to_string()];
    let analyzer = CostAnalyzer::new(providers.clone());

    // We can't directly access the providers field since it's private,
    // but we can test the behavior through the analyze_costs method
    let analysis = analyzer.analyze_costs().await.unwrap();

    // Based on the implementation, total_cost should be providers.len() * 100.0
    assert_eq!(analysis.total_cost, 200.0);
    assert_eq!(analysis.recommendations.len(), 2);
}

#[tokio::test]
async fn test_cost_analyzer_empty_providers() {
    let analyzer = CostAnalyzer::new(vec![]);
    let analysis = analyzer.analyze_costs().await.unwrap();

    assert_eq!(analysis.total_cost, 0.0);
    assert!(analysis.recommendations.is_empty());
}

#[tokio::test]
async fn test_cost_analyzer_single_provider() {
    let analyzer = CostAnalyzer::new(vec!["aws".to_string()]);
    let analysis = analyzer.analyze_costs().await.unwrap();

    assert_eq!(analysis.total_cost, 100.0);
    assert_eq!(analysis.recommendations.len(), 1);
    assert!(analysis.recommendations[0].contains("aws"));
}

#[tokio::test]
async fn test_cost_analyzer_multiple_providers() {
    let providers = vec![
        "aws".to_string(),
        "gcp".to_string(),
        "azure".to_string(),
        "digitalocean".to_string(),
    ];
    let analyzer = CostAnalyzer::new(providers.clone());
    let analysis = analyzer.analyze_costs().await.unwrap();

    assert_eq!(analysis.total_cost, 400.0);
    assert_eq!(analysis.recommendations.len(), 4);

    // Verify each provider has a recommendation
    for provider in &providers {
        assert!(analysis
            .recommendations
            .iter()
            .any(|r| r.contains(provider)));
    }
}

#[tokio::test]
async fn test_cost_analysis_recommendation_format() {
    let analyzer = CostAnalyzer::new(vec!["test-provider".to_string()]);
    let analysis = analyzer.analyze_costs().await.unwrap();

    assert_eq!(analysis.recommendations.len(), 1);
    assert_eq!(
        analysis.recommendations[0],
        "Consider optimizing test-provider usage"
    );
}

#[test]
fn test_cost_analysis_debug_format() {
    let analysis = CostAnalysis {
        total_cost: 150.5,
        recommendations: vec!["test recommendation".to_string()],
    };

    let debug_output = format!("{:?}", analysis);
    assert!(debug_output.contains("150.5"));
    assert!(debug_output.contains("test recommendation"));
}
