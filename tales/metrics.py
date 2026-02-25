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
    current_action = None
    current_streak = 0

    for _, row in rollouts_df.iterrows():
        action = row.get("Action", "")
        feedback = row.get("Feedback", "")

        # Consider a failed command if it produces no effect or is explicitly rejected.
        # This logic can be refined based on the environments. In AlfWorld, failure is "Nothing happens."
        # In ScienceWorld/TextWorld it can be things like "You can't do that" or "I don't understand".
        is_failure = (
            "Nothing happens" in feedback
            or "You can't" in feedback
            or "I don't understand" in feedback
            or "recognize" in feedback
        )

        if is_failure:
            if action == current_action:
                current_streak += 1
            else:
                if current_streak > threshold:
                    doom_loop_count += 1
                current_action = action
                current_streak = 1
        else:
            if current_streak > threshold:
                doom_loop_count += 1
            current_action = None
            current_streak = 0

    # Check if the episode ended on a doom loop
    if current_streak > threshold:
        doom_loop_count += 1

    return doom_loop_count
