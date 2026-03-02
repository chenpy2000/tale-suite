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
    Counts repeated no-progress action loops.
    A loop is either:
    1) same action repeated above `threshold` while feedback indicates rejection/no progress, or
    2) an alternating ABAB pattern with no progress.
    """
    if rollouts_df.empty:
        return 0

    doom_loop_count = 0
    current_action = None
    current_streak = 0

    previous_feedback = ""
    previous_score = None
    actions = []
    stalls = []

    for _, row in rollouts_df.iterrows():
        action = row.get("Action", "")
        feedback = row.get("Feedback", "")
        score = row.get("Score", None)
        normalized_feedback = " ".join(str(feedback).strip().lower().split())
        actions.append(str(action))

        explicit_failure = (
            "nothing happens" in normalized_feedback
            or "you can't" in normalized_feedback
            or "i don't understand" in normalized_feedback
            or "invalid" in normalized_feedback
            or "failed" in normalized_feedback
            or "not possible" in normalized_feedback
        )
        no_progress = bool(previous_feedback) and normalized_feedback == previous_feedback
        no_progress = no_progress and score == previous_score
        is_failure = explicit_failure or no_progress
        stalls.append(is_failure)

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

        previous_feedback = normalized_feedback
        previous_score = score

    # Check if the episode ended on a doom loop
    if current_streak > threshold:
        doom_loop_count += 1

    # Detect alternating ABAB no-progress loops.
    i = 0
    while i + 3 < len(actions):
        a, b, c, d = actions[i : i + 4]
        if a and b and a != b and a == c and b == d and all(stalls[i : i + 4]):
            doom_loop_count += 1
            i += 4
            continue
        i += 1

    return doom_loop_count
