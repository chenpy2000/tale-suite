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
    current_action_feedback = None
    current_streak = 0

    previous_feedback = ""
    previous_score = None
    actions = []
    stalls = []

    for _, row in rollouts_df.iterrows():
        action = str(row.get("Action", "") or "").strip()
        feedback = str(row.get("Feedback", "") or "").strip()
        score = row.get("Score", None)

        normalized_feedback = " ".join(feedback.lower().split())
        action_feedback_pair = (action.lower(), normalized_feedback)

        if action_feedback_pair == current_action_feedback:
            current_streak += 1
        else:
            if current_streak > threshold:
                doom_loop_count += 1
            current_action_feedback = action_feedback_pair
            current_streak = 1

        score_unchanged = previous_score is None or score == previous_score
        feedback_unchanged = (
            previous_feedback != "" and normalized_feedback == previous_feedback
        )
        stalls.append(score_unchanged and feedback_unchanged)
        actions.append(action.lower())

        previous_feedback = normalized_feedback
        previous_score = score

    if current_streak > threshold:
        doom_loop_count += 1

    i = 0
    while i + 3 < len(actions):
        a, b, c, d = actions[i : i + 4]
        if a and b and a != b and a == c and b == d and all(stalls[i : i + 4]):
            doom_loop_count += 1
            i += 4
            continue
        i += 1

    return doom_loop_count
