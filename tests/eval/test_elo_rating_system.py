import pytest
import math
from evals.elo_rating import EloRatingSystem # Corrected class name

# --- Test Cases ---

def test_elo_system_initialization():
    """Test EloRatingSystem initialization."""
    # Test with default values
    elo_system_default = EloRatingSystem()
    assert elo_system_default.k_factor == 32
    assert elo_system_default.initial_rating == 1200
    assert elo_system_default.min_rating == 100
    assert elo_system_default.max_rating == 3000

    # Test with custom values
    k_custom = 24
    initial_custom = 1500
    elo_system_custom = EloRatingSystem(k_factor=k_custom, initial_rating=initial_custom)
    assert elo_system_custom.k_factor == k_custom
    assert elo_system_custom.initial_rating == initial_custom

def test_elo_expected_score():
    """Test the expected_score calculation."""
    elo_system = EloRatingSystem()

    # Case 1: Equal ratings
    r_a1, r_b1 = 1200, 1200
    e_a1 = elo_system.expected_score(r_a1, r_b1)
    assert math.isclose(e_a1, 0.5), f"Expected 0.5 for equal ratings, got {e_a1}"

    # Case 2: Player A higher rated
    r_a2, r_b2 = 1200, 1000
    e_a2 = elo_system.expected_score(r_a2, r_b2)
    # E_A = 1 / (1 + 10^((1000 - 1200) / 400)) = 1 / (1 + 10^(-200/400)) = 1 / (1 + 10^(-0.5))
    # 10^(-0.5) = 1 / sqrt(10) ≈ 0.31622776601683794
    # E_A = 1 / (1 + 0.31622776601683794) ≈ 0.7597469266479578
    expected_e_a2 = 1.0 / (1.0 + math.pow(10, (r_b2 - r_a2) / 400.0))
    assert math.isclose(e_a2, expected_e_a2, abs_tol=1e-10), f"Expected {expected_e_a2} for 1200 vs 1000, got {e_a2}"

    # Case 3: Player A lower rated
    r_a3, r_b3 = 1000, 1200
    e_a3 = elo_system.expected_score(r_a3, r_b3)
    # E_A = 1 / (1 + 10^((1200 - 1000) / 400)) = 1 / (1 + 10^(200/400)) = 1 / (1 + 10^(0.5))
    # 10^(0.5) = sqrt(10) ≈ 3.1622776601683795
    # E_A = 1 / (1 + 3.1622776601683795) ≈ 0.24025307334520423
    expected_e_a3 = 1.0 / (1.0 + math.pow(10, (r_b3 - r_a3) / 400.0))
    assert math.isclose(e_a3, expected_e_a3, abs_tol=1e-10), f"Expected {expected_e_a3} for 1000 vs 1200, got {e_a3}"

def test_elo_update_ratings_logic():
    """Test the core update_ratings logic with various scenarios."""
    k_factor = 32
    elo_system = EloRatingSystem(k_factor=k_factor)

    # Scenario 1: Player A (1200) beats Player B (1000). S_A = 1.0
    r_a, r_b = 1200, 1000
    e_a = elo_system.expected_score(r_a, r_b) # ~0.75973
    e_b = 1 - e_a # ~0.24027

    new_r_a1, new_r_b1 = elo_system.update_ratings(r_a, r_b, score_a=1.0)
    expected_change_a1 = k_factor * (1.0 - e_a)
    expected_change_b1 = k_factor * (0.0 - e_b) # (1.0 - score_a) is 0 for Player B

    assert math.isclose(new_r_a1, r_a + expected_change_a1)
    assert math.isclose(new_r_b1, r_b + expected_change_b1)
    assert new_r_a1 > r_a
    assert new_r_b1 < r_b
    assert (r_a + expected_change_a1 - r_a) < (k_factor / 2) # Change for higher rated winning is < K/2

    # Scenario 2: Player B (1000) beats Player A (1200) - "upset". S_A = 0.0
    r_a, r_b = 1200, 1000
    e_a = elo_system.expected_score(r_a, r_b) # ~0.75973
    e_b = 1 - e_a # ~0.24027

    new_r_a2, new_r_b2 = elo_system.update_ratings(r_a, r_b, score_a=0.0)
    expected_change_a2 = k_factor * (0.0 - e_a)
    expected_change_b2 = k_factor * (1.0 - e_b) # (1.0 - score_a) is 1 for Player B

    assert math.isclose(new_r_a2, r_a + expected_change_a2)
    assert math.isclose(new_r_b2, r_b + expected_change_b2)
    assert new_r_a2 < r_a
    assert new_r_b2 > r_b
    assert (r_b + expected_change_b2 - r_b) > (k_factor / 2) # Change for lower rated winning is > K/2 (if E_B was < 0.5)

    # Scenario 3: Players with equal ratings (1200 vs 1200) draw. S_A = 0.5
    r_a_eq, r_b_eq = 1200, 1200
    e_a_eq = elo_system.expected_score(r_a_eq, r_b_eq) # 0.5

    new_r_a3, new_r_b3 = elo_system.update_ratings(r_a_eq, r_b_eq, score_a=0.5)
    assert math.isclose(new_r_a3, r_a_eq) # No change
    assert math.isclose(new_r_b3, r_b_eq) # No change

    # Scenario 4: Player A (1200) draws with Player B (1000). S_A = 0.5
    r_a, r_b = 1200, 1000
    e_a = elo_system.expected_score(r_a, r_b) # ~0.75973
    e_b = 1 - e_a # ~0.24027

    new_r_a4, new_r_b4 = elo_system.update_ratings(r_a, r_b, score_a=0.5)
    expected_change_a4 = k_factor * (0.5 - e_a) # Negative change for A
    expected_change_b4 = k_factor * (0.5 - e_b) # Positive change for B (0.5 - score_a_for_b = 0.5 - 0.5 = 0, this is wrong)
                                               # score_b = 1 - score_a = 0.5
                                               # change_b = k_factor * (score_b - e_b)

    assert math.isclose(new_r_a4, r_a + expected_change_a4)
    assert math.isclose(new_r_b4, r_b + k_factor * (0.5 - e_b)) # Corrected expectation for B
    assert new_r_a4 < r_a
    assert new_r_b4 > r_b

def test_elo_rating_bounds():
    """Test that ratings are clamped within min_rating and max_rating."""
    elo_system = EloRatingSystem(k_factor=32)
    elo_system.min_rating = 100
    elo_system.max_rating = 300

    # Test min bound
    # Player A (very low, e.g., 10) loses to Player B (e.g., 120)
    # Expected score for A is very low, actual is 0. Change should be small negative.
    # If r_a + change_a goes below min_rating, it should be clamped.
    # Let's make r_a just above min_rating, and make it lose to a much higher rated player
    r_a_low = 105
    r_b_high = 295
    # E_A = 1 / (1 + 10^((295-105)/400)) = 1 / (1 + 10^(190/400)) = 1 / (1 + 10^0.475) approx 1 / (1+2.98) = 0.25
    # Change_A = 32 * (0 - 0.25) = -8. Rating would be 105 - 8 = 97. Should be clamped to 100.
    new_r_a_min, _ = elo_system.update_ratings(r_a_low, r_b_high, score_a=0.0)
    assert new_r_a_min == elo_system.min_rating

    # Test max bound
    # Player A (very high) beats Player B (very low)
    r_a_high = 295
    r_b_low = 105
    # E_A = 1 / (1 + 10^((105-295)/400)) = 1 / (1 + 10^(-190/400)) = 1 / (1 + 10^-0.475) approx 1 / (1+0.335) = 0.75
    # Change_A = 32 * (1 - 0.75) = 8. Rating would be 295 + 8 = 303. Should be clamped to 300.
    new_r_a_max, _ = elo_system.update_ratings(r_a_high, r_b_low, score_a=1.0)
    assert new_r_a_max == elo_system.max_rating


def test_elo_get_rating_class():
    """Test the descriptive rating classes."""
    elo_system = EloRatingSystem()
    assert elo_system.get_rating_class(700) == "Beginner"
    assert elo_system.get_rating_class(900) == "Novice"
    assert elo_system.get_rating_class(1100) == "Amateur"
    assert elo_system.get_rating_class(1300) == "Intermediate"
    assert elo_system.get_rating_class(1500) == "Advanced"
    assert elo_system.get_rating_class(1700) == "Expert"
    assert elo_system.get_rating_class(1900) == "Master"
    assert elo_system.get_rating_class(2100) == "Grandmaster"
    assert elo_system.get_rating_class(2300) == "Super Grandmaster"

def test_elo_state_management():
    """Test get_state and load_state methods."""
    elo_system = EloRatingSystem(k_factor=20, initial_rating=1000)
    elo_system.min_rating = 50
    elo_system.max_rating = 2500

    state = elo_system.get_state()
    assert state['k_factor'] == 20
    assert state['initial_rating'] == 1000
    assert state['min_rating'] == 50
    assert state['max_rating'] == 2500

    new_elo_system = EloRatingSystem() # Default values
    new_elo_system.load_state(state)
    assert new_elo_system.k_factor == 20
    assert new_elo_system.initial_rating == 1000
    assert new_elo_system.min_rating == 50
    assert new_elo_system.max_rating == 2500

    # Test loading partial state
    partial_state = {'k_factor': 16}
    default_system = EloRatingSystem()
    default_system.load_state(partial_state)
    assert default_system.k_factor == 16
    assert default_system.initial_rating == 1200 # Should remain default
