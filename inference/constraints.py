"""Constraint extraction from backstory text."""
import re
from typing import List
from .models import Constraint


class ConstraintExtractor:
    """Extract constraints (traits, beliefs, vows) from backstory text."""
    
    TRAIT_PATTERNS = [
        r'(\w+)\s+(was|is|became|remained)\s+(\w+)',
        r'(\w+)\s+(never|always|often|rarely)\s+(\w+)',
        r'(\w+)\s+(vowed|promised|swore)\s+to\s+(\w+)',
        r'(\w+)\s+(feared|hated|loved)\s+(\w+)',
        r'(\w+)\s+(believed|thought|considered)\s+that\s+([^.]+)',
    ]
    
    def __init__(self):
        self.categories = {
            'trait': ['was', 'is', 'became', 'remained', 'never', 'always'],
            'vow': ['vowed', 'promised', 'swore'],
            'fear': ['feared', 'hated', 'dreaded'],
            'belief': ['believed', 'thought', 'considered'],
        }
    
    def extract(self, backstory_text: str) -> List[Constraint]:
        """Extract constraints from backstory text."""
        constraints = []
        sentences = re.split(r'[.!?]\s+', backstory_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            for pattern in self.TRAIT_PATTERNS:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    category = self._classify_category(match.group(0))
                    polarity = self._classify_polarity(match.group(0))
                    
                    constraint = Constraint(
                        text=match.group(0),
                        category=category,
                        polarity=polarity,
                        confidence=0.7
                    )
                    constraints.append(constraint)
            
            if any(word in sentence.lower() for word in ['never', 'always', 'refused', 'avoided']):
                constraint = Constraint(
                    text=sentence,
                    category='commitment',
                    polarity='negative' if 'never' in sentence.lower() else 'positive',
                    confidence=0.6
                )
                constraints.append(constraint)
        
        return constraints

    def _classify_category(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['vowed', 'promised', 'swore']):
            return 'vow'
        elif any(word in text_lower for word in ['feared', 'hated', 'dreaded']):
            return 'fear'
        elif any(word in text_lower for word in ['believed', 'thought', 'considered']):
            return 'belief'
        return 'trait'

    def _classify_polarity(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['never', 'avoided', 'refused', 'feared', 'hated', 'no']):
            return 'negative'
        elif any(word in text_lower for word in ['always', 'loved', 'vowed to', 'promised to']):
            return 'positive'
        return 'neutral'
