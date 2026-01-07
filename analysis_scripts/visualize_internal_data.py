
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_logs():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'ppt_assets_advanced')
    log_path = os.path.join(output_dir, 'internal_decision_log.csv')
    
    if not os.path.exists(log_path):
        print(f"No log file found at {log_path}")
        return

    df = pd.read_csv(log_path)
    
    # Setup style
    sns.set_theme(style="whitegrid")
    
    # 1. Strategy Distribution
    plt.figure(figsize=(10, 6))
    strategy_counts = df['best_strategy'].value_counts()
    sns.barplot(x=strategy_counts.values, y=strategy_counts.index, palette="viridis")
    plt.title('Agent Strategy Distribution (Internal Decision)', fontsize=14)
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_distribution_internal.png'))
    plt.close()
    
    # 2. Decision Confidence across strategies (using score or spread as proxy if confidence column isn't clean)
    # The logs have 'chosen_score'. Let's see score distribution by strategy.
    plt.figure(figsize=(12, 6))
    # Filter strategies with < 2 occurrences to clean up plot
    top_strategies = df['best_strategy'].value_counts()
    main_strategies = top_strategies[top_strategies > 1].index
    df_filtered = df[df['best_strategy'].isin(main_strategies)]
    
    sns.boxplot(data=df_filtered, x='chosen_score', y='best_strategy', palette="coolwarm")
    plt.title('Decision Score Range by Strategy', fontsize=14)
    plt.xlabel('Internal Score (Efficiency Metric)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_scores_internal.png'))
    plt.close()
    
    # 3. Candidate Count vs Score Spread (Complexity Analysis)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='candidate_count', y='score_spread', hue='step_type', s=100, alpha=0.7)
    plt.title('Decision Complexity: Candidates vs Score Spread', fontsize=14)
    plt.xlabel('Number of Candidates Considered')
    plt.ylabel('Score Spread (Max - Min)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'decision_complexity_internal.png'))
    plt.close()
    
    print(f"Internal visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_logs()
