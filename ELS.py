import numpy as np


def simulate_longest_loss_streak(n_bets, win_rate):
    """
    Simulate one sequence of n_bets and return the longest run of consecutive losses.

    Parameters:
      n_bets (int): Number of bets in the simulation.
      win_rate (float): The probability of a winning bet (strike rate).

    Returns:
      int: The length of the longest consecutive loss sequence.
    """
    # Generate outcomes: True for win, False for loss.
    outcomes = np.random.rand(n_bets) < win_rate
    max_streak = 0
    current_streak = 0

    # Track the longest streak of losses.
    for outcome in outcomes:
        if outcome:  # Reset streak when you have a win.
            current_streak = 0
        else:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
    return max_streak


def run_simulations(n_bets, win_rate, n_simulations):
    """
    Run multiple simulations and compute the expected maximum losing streak.

    Parameters:
      n_bets (int): Number of bets per simulation.
      win_rate (float): The probability of winning a bet.
      n_simulations (int): Number of simulation runs.

    Returns:
      float: The expected (average) longest losing streak.
      list: List of longest streaks from each simulation.
    """
    longest_streaks = [simulate_longest_loss_streak(n_bets, win_rate) for _ in range(n_simulations)]
    expected_longest_streak = np.mean(longest_streaks)
    return expected_longest_streak, longest_streaks


def main():
    # Prompt the user for the strike rate.
    try:
        win_rate = float(input("Enter the strike rate (e.g. 0.3 for a 30% win rate): "))
        if not (0 <= win_rate <= 1):
            raise ValueError("The strike rate must be between 0 and 1.")
    except ValueError as e:
        print("Invalid input for strike rate:", e)
        return

    # Prompt for number of bets per simulation; default to 1000 if left blank.
    try:
        n_bets_input = input("Enter the number of bets per simulation (default 1000): ")
        n_bets = int(n_bets_input) if n_bets_input.strip() else 1000
    except ValueError:
        print("Invalid input for number of bets.")
        return

    # Prompt for number of simulation runs; default to 10000 if left blank.
    try:
        n_simulations_input = input("Enter the number of simulations (default 10000): ")
        n_simulations = int(n_simulations_input) if n_simulations_input.strip() else 10000
    except ValueError:
        print("Invalid input for number of simulations.")
        return

    expected_streak, all_streaks = run_simulations(n_bets, win_rate, n_simulations)
    print(f"\nAfter {n_simulations} simulations of {n_bets} bets each with a strike rate of {win_rate:.2f}:")
    print(f"  The expected (average) longest losing streak is approximately: {expected_streak:.2f}")

    # Display additional statistics.
    print("\nAdditional statistics on longest losing streaks:")
    print(f"  Minimum streak: {min(all_streaks)}")
    print(f"  Maximum streak: {max(all_streaks)}")
    print(f"  Median streak : {np.median(all_streaks)}")


if __name__ == "__main__":
    main()
