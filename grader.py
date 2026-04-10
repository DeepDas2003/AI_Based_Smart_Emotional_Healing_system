def grade(history, final_emotion, total_reward):
    steps = len(history)
    if steps == 0:
        return 0.0

    score = 0.0

    if final_emotion == "neutral":
        score += 0.5

    score += 0.2 * max(0, 1 - steps / 12)

    neutral_count = sum(1 for h in history if h["new"] == "neutral")
    score += 0.2 * (neutral_count / steps)

    score += 0.1 * min(total_reward / 5, 1)

    return round(min(max(score, 0.0), 1.0), 3)
