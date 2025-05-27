import math
from typing import Tuple, Dict, Any


class EloRatingSystem:
    """
    Elo rating system for competitive multi-agent evaluation.
    
    Implements the standard Elo rating algorithm with configurable K-factor
    and support for draws in zero-sum games.
    """
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1200):
        """
        Initialize Elo rating system.
        
        Args:
            k_factor: Maximum rating change per game (higher = more volatile)
            initial_rating: Starting rating for new players/models
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        
        # Rating bounds for stability
        self.min_rating = 100
        self.max_rating = 3000
        
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B
            
        Returns:
            Expected score (0.0 to 1.0) for player A
        """
        rating_diff = rating_b - rating_a
        return 1.0 / (1.0 + math.pow(10, rating_diff / 400.0))
    
    def update_ratings(self, rating_a: float, rating_b: float, score_a: float) -> Tuple[float, float]:
        """
        Update ratings based on game result.
        
        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            score_a: Actual score for player A (1.0 = win, 0.5 = draw, 0.0 = loss)
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        # Calculate rating changes
        change_a = self.k_factor * (score_a - expected_a)
        change_b = self.k_factor * ((1.0 - score_a) - expected_b)
        
        # Apply changes with bounds
        new_rating_a = max(self.min_rating, min(self.max_rating, rating_a + change_a))
        new_rating_b = max(self.min_rating, min(self.max_rating, rating_b + change_b))
        
        return new_rating_a, new_rating_b
    
    def get_rating_class(self, rating: float) -> str:
        """
        Get descriptive class for a rating.
        
        Args:
            rating: Elo rating
            
        Returns:
            Rating class string
        """
        if rating < 800:
            return "Beginner"
        elif rating < 1000:
            return "Novice"
        elif rating < 1200:
            return "Amateur"
        elif rating < 1400:
            return "Intermediate"
        elif rating < 1600:
            return "Advanced"
        elif rating < 1800:
            return "Expert"
        elif rating < 2000:
            return "Master"
        elif rating < 2200:
            return "Grandmaster"
        else:
            return "Super Grandmaster"
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'k_factor': self.k_factor,
            'initial_rating': self.initial_rating,
            'min_rating': self.min_rating,
            'max_rating': self.max_rating
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization."""
        self.k_factor = state.get('k_factor', self.k_factor)
        self.initial_rating = state.get('initial_rating', self.initial_rating)
        self.min_rating = state.get('min_rating', self.min_rating)
        self.max_rating = state.get('max_rating', self.max_rating)