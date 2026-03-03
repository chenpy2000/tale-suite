import pandas as pd


def compute_token_efficiency(total_tokens: int, score: int) -> float:
    """
    Computes the operational efficiency of the agent by calculating the number of tokens
    consumed per game score point.
    """
    if score == 0:
        return 0.0
    return total_tokens / score


def compute_doom_loop_count(rollouts_df: pd.DataFrame, threshold: int = 3) -> int:
    """
    Computes the number of times the agent repeated identical commands that failed sequentially
    for strictly greater than the threshold number of times.
    """
    if rollouts_df.empty:
        return 0

    doom_loop_count = 0
    current_action_feedback = None
    current_streak = 0

    for _, row in rollouts_df.iterrows():
        action = row.get("Action", "")
        feedback = row.get("Feedback", "")
        action_feedback_pair = (action, feedback)

        if action_feedback_pair == current_action_feedback:
            current_streak += 1
        else:
            if current_streak > threshold:
                doom_loop_count += 1
            current_action_feedback = action_feedback_pair
            current_streak = 1

    # Check if the episode ended on a doom loop
    if current_streak > threshold:
        doom_loop_count += 1

    return doom_loop_count
