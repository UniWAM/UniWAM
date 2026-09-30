"""Small dependency-free helpers for LIBERO-plus evaluation."""


def deterministic_episode_seed(base_seed: int, task_id: int, episode_index: int) -> int:
    """Make policy noise independent of worker scheduling and server topology."""
    modulus = 2**31 - 1
    return int(
        ((int(base_seed) + 1) * 1_000_003 + int(task_id) * 10_007 + int(episode_index) * 101)
        % modulus
    )
