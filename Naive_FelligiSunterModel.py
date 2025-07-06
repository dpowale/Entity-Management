import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Tuple, Optional
from collections import defaultdict
import itertools
import re
from datetime import datetime, timedelta
import warnings

class FelligiSunterModel:
    """
    Implementation of the Felligi-Sunter model for probabilistic record linkage
    optimized for person identification records with name, DOB, phone, address, and government IDs.
    """
    
    def __init__(self):
        """Initialize the Felligi-Sunter model with specialized comparison functions."""
        self.comparison_functions = {}
        self.m_probs = {}  # m-probabilities (agreement given match)
        self.u_probs = {}  # u-probabilities (agreement given non-match)
        self.match_weights = {}  # log-likelihood ratios for agreement
        self.non_match_weights = {}  # log-likelihood ratios for disagreement
        self.prior_match_prob = 0.01  # Prior probability of a match
        self.is_trained = False
        
        # Initialize specialized comparison functions
        self._setup_comparison_functions()
    
    def _setup_comparison_functions(self):
        """Setup specialized comparison functions for each field type."""
        self.comparison_functions = {
            'name': self.name_comparison,
            'dob': self.dob_comparison,
            'phone': self.phone_comparison,
            'address': self.address_comparison,
            'government_id': self.government_id_comparison
        }
    
    def name_comparison(self, name1, name2) -> int:
        """
        Advanced name comparison with multiple similarity levels.
        Returns: 0 = no match, 1 = partial match, 2 = strong match
        """
        if pd.isna(name1) or pd.isna(name2):
            return 0
        
        name1 = str(name1).strip().lower()
        name2 = str(name2).strip().lower()
        
        if name1 == name2:
            return 2  # Exact match
        
        # Split names into components
        parts1 = [part for part in re.split(r'[,\s]+', name1) if part]
        parts2 = [part for part in re.split(r'[,\s]+', name2) if part]
        
        # Check for exact component matches
        exact_matches = 0
        for part1 in parts1:
            for part2 in parts2:
                if part1 == part2 and len(part1) > 2:  # Ignore short components
                    exact_matches += 1
        
        # Strong match if multiple exact component matches
        if exact_matches >= 2:
            return 2
        
        # Calculate Jaro-Winkler similarity
        similarity = self._jaro_winkler_similarity(name1, name2)
        
        if similarity >= 0.85:
            return 2  # Strong match
        elif similarity >= 0.7:
            return 1  # Partial match
        else:
            return 0  # No match
    
    def dob_comparison(self, dob1, dob2) -> int:
        """
        Date of birth comparison with tolerance for data entry errors.
        Returns: 0 = no match, 1 = close match, 2 = exact match
        """
        if pd.isna(dob1) or pd.isna(dob2):
            return 0
        
        try:
            # Parse dates - handle various formats
            date1 = self._parse_date(dob1)
            date2 = self._parse_date(dob2)
            
            if date1 is None or date2 is None:
                return 0
            
            # Exact match
            if date1 == date2:
                return 2
            
            # Check for common data entry errors
            diff = abs((date1 - date2).days)
            
            # Same year and month, different day (typo in day)
            if date1.year == date2.year and date1.month == date2.month and diff <= 2:
                return 1
            
            # Same year and day, different month (typo in month)
            if date1.year == date2.year and date1.day == date2.day and abs(date1.month - date2.month) <= 1:
                return 1
            
            # Same month and day, different year (typo in year)
            if date1.month == date2.month and date1.day == date2.day and abs(date1.year - date2.year) <= 1:
                return 1
            
            # Within 1 year (major typo but potentially same person)
            if diff <= 365:
                return 1
            
            return 0
            
        except Exception:
            return 0
    
    def phone_comparison(self, phone1, phone2) -> int:
        """
        Phone number comparison with normalization.
        Returns: 0 = no match, 1 = partial match, 2 = exact match
        """
        if pd.isna(phone1) or pd.isna(phone2):
            return 0
        
        # Normalize phone numbers
        norm_phone1 = self._normalize_phone(str(phone1))
        norm_phone2 = self._normalize_phone(str(phone2))
        
        if not norm_phone1 or not norm_phone2:
            return 0
        
        # Exact match
        if norm_phone1 == norm_phone2:
            return 2
        
        # Partial match - last 7 digits (local number)
        if len(norm_phone1) >= 7 and len(norm_phone2) >= 7:
            if norm_phone1[-7:] == norm_phone2[-7:]:
                return 1
        
        # Partial match - last 10 digits (missing country code)
        if len(norm_phone1) >= 10 and len(norm_phone2) >= 10:
            if norm_phone1[-10:] == norm_phone2[-10:]:
                return 1
        
        return 0
    
    def address_comparison(self, addr1, addr2) -> int:
        """
        Address comparison with normalization and component matching.
        Returns: 0 = no match, 1 = partial match, 2 = strong match
        """
        if pd.isna(addr1) or pd.isna(addr2):
            return 0
        
        # Normalize addresses
        norm_addr1 = self._normalize_address(str(addr1))
        norm_addr2 = self._normalize_address(str(addr2))
        
        if norm_addr1 == norm_addr2:
            return 2
        
        # Extract components
        components1 = self._extract_address_components(norm_addr1)
        components2 = self._extract_address_components(norm_addr2)
        
        # Check for matches in key components
        matches = 0
        total_components = 0
        
        for key in ['street_number', 'street_name', 'city', 'state', 'zip']:
            if components1.get(key) and components2.get(key):
                total_components += 1
                if components1[key] == components2[key]:
                    matches += 1
        
        if total_components == 0:
            return 0
        
        match_ratio = matches / total_components
        
        if match_ratio >= 0.8:
            return 2  # Strong match
        elif match_ratio >= 0.5:
            return 1  # Partial match
        else:
            return 0
    
    def government_id_comparison(self, id1, id2) -> int:
        """
        Government ID comparison (SSN, driver's license, etc.).
        Returns: 0 = no match, 1 = partial match, 2 = exact match
        """
        if pd.isna(id1) or pd.isna(id2):
            return 0
        
        # Normalize IDs (remove spaces, dashes, etc.)
        norm_id1 = re.sub(r'[^\w]', '', str(id1)).upper()
        norm_id2 = re.sub(r'[^\w]', '', str(id2)).upper()
        
        if norm_id1 == norm_id2:
            return 2
        
        # Partial match for long IDs (e.g., last 4 digits of SSN)
        if len(norm_id1) >= 4 and len(norm_id2) >= 4:
            if norm_id1[-4:] == norm_id2[-4:]:
                return 1
        
        return 0
    
    def _jaro_winkler_similarity(self, s1, s2, prefix_scale=0.1):
        """Calculate Jaro-Winkler similarity between two strings."""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Calculate Jaro similarity
        jaro_sim = self._jaro_similarity(s1, s2)
        
        if jaro_sim < 0.7:
            return jaro_sim
        
        # Calculate common prefix length (up to 4 characters)
        prefix_len = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
        
        # Apply Winkler modification
        return jaro_sim + (prefix_len * prefix_scale * (1 - jaro_sim))
    
    def _jaro_similarity(self, s1, s2):
        """Calculate Jaro similarity between two strings."""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        match_window = max(len(s1), len(s2)) // 2 - 1
        if match_window < 0:
            match_window = 0
        
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len(s1)):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len(s2))
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len(s1) + matches / len(s2) + 
               (matches - transpositions / 2) / matches) / 3
        
        return jaro
    
    def _parse_date(self, date_str):
        """Parse date string in various formats."""
        if isinstance(date_str, datetime):
            return date_str.date()
        
        date_str = str(date_str).strip()
        
        # Common date formats
        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m/%d/%y', '%d/%m/%y',
            '%Y%m%d', '%m%d%Y', '%d%m%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        return None
    
    def _normalize_phone(self, phone):
        """Normalize phone number by removing non-digits."""
        digits = re.sub(r'\D', '', phone)
        
        # Remove country code if present
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        
        return digits if len(digits) >= 7 else None
    
    def _normalize_address(self, address):
        """Normalize address string."""
        # Convert to lowercase and remove extra spaces
        addr = re.sub(r'\s+', ' ', address.lower().strip())
        
        # Common abbreviations
        abbreviations = {
            'street': 'st', 'avenue': 'ave', 'boulevard': 'blvd',
            'road': 'rd', 'drive': 'dr', 'lane': 'ln', 'court': 'ct',
            'place': 'pl', 'apartment': 'apt', 'suite': 'ste'
        }
        
        for full, abbr in abbreviations.items():
            addr = re.sub(r'\b' + full + r'\b', abbr, addr)
        
        return addr
    
    def _extract_address_components(self, address):
        """Extract address components (street number, name, city, state, zip)."""
        components = {}
        
        # Extract ZIP code
        zip_match = re.search(r'\b\d{5}(?:-\d{4})?\b', address)
        if zip_match:
            components['zip'] = zip_match.group()
            address = address.replace(zip_match.group(), '').strip()
        
        # Extract state (2-letter abbreviation)
        state_match = re.search(r'\b[A-Z]{2}\b', address.upper())
        if state_match:
            components['state'] = state_match.group().lower()
            address = address.replace(state_match.group().lower(), '').strip()
        
        # Extract street number
        street_num_match = re.search(r'^\d+', address)
        if street_num_match:
            components['street_number'] = street_num_match.group()
            address = address.replace(street_num_match.group(), '').strip()
        
        # Remaining parts
        parts = [part.strip() for part in address.split(',')]
        
        if len(parts) >= 1:
            components['street_name'] = parts[0]
        if len(parts) >= 2:
            components['city'] = parts[1]
        
        return components
    
    def generate_comparison_vector(self, record1: pd.Series, record2: pd.Series, 
                                 fields: List[str]) -> List[int]:
        """Generate comparison vector for a pair of records."""
        vector = []
        
        for field in fields:
            if field in self.comparison_functions:
                comparison_result = self.comparison_functions[field](
                    record1[field], record2[field]
                )
            else:
                comparison_result = 0
            
            vector.append(comparison_result)
        
        return vector
    
    def estimate_parameters_em(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             fields: List[str], max_iterations: int = 100,
                             tolerance: float = 1e-6, max_pairs: int = 50000) -> None:
        """
        Estimate m and u probabilities using Expectation-Maximization algorithm.
        """
        print(f"Starting EM algorithm with {len(df1)} x {len(df2)} comparisons...")
        
        # Generate comparison vectors
        comparison_vectors = []
        record_pairs = []
        
        # Sample pairs to avoid memory issues
        total_pairs = len(df1) * len(df2)
        if total_pairs > max_pairs:
            pairs_sample = np.random.choice(total_pairs, size=max_pairs, replace=False)
            for pair_idx in pairs_sample:
                i = pair_idx // len(df2)
                j = pair_idx % len(df2)
                record_pairs.append((i, j))
        else:
            for i in range(len(df1)):
                for j in range(len(df2)):
                    record_pairs.append((i, j))
        
        print(f"Generating comparison vectors for {len(record_pairs)} pairs...")
        
        for i, j in record_pairs:
            vector = self.generate_comparison_vector(df1.iloc[i], df2.iloc[j], fields)
            comparison_vectors.append(tuple(vector))
        
        # Count unique comparison patterns
        pattern_counts = defaultdict(int)
        for vector in comparison_vectors:
            pattern_counts[vector] += 1
        
        print(f"Found {len(pattern_counts)} unique comparison patterns")
        
        # Initialize parameters for multi-level comparisons
        num_fields = len(fields)
        
        # Initialize m and u probabilities for each field and comparison level
        m_probs = {}
        u_probs = {}
        
        for field_idx, field in enumerate(fields):
            m_probs[field_idx] = {0: 0.1, 1: 0.3, 2: 0.9}  # Higher levels = higher m prob
            u_probs[field_idx] = {0: 0.8, 1: 0.15, 2: 0.05}  # Higher levels = lower u prob
        
        # EM iterations
        for iteration in range(max_iterations):
            old_m_probs = {k: v.copy() for k, v in m_probs.items()}
            old_u_probs = {k: v.copy() for k, v in u_probs.items()}
            
            # E-step: Calculate posterior probabilities
            posterior_probs = {}
            
            for pattern, count in pattern_counts.items():
                # Calculate likelihood of pattern given match
                match_likelihood = 1.0
                for field_idx, level in enumerate(pattern):
                    match_likelihood *= m_probs[field_idx][level]
                
                # Calculate likelihood of pattern given non-match
                non_match_likelihood = 1.0
                for field_idx, level in enumerate(pattern):
                    non_match_likelihood *= u_probs[field_idx][level]
                
                # Calculate posterior probability of match
                numerator = self.prior_match_prob * match_likelihood
                denominator = (numerator + 
                             (1 - self.prior_match_prob) * non_match_likelihood)
                
                if denominator > 0:
                    posterior_probs[pattern] = numerator / denominator
                else:
                    posterior_probs[pattern] = 0.0
            
            # M-step: Update parameters
            for field_idx in range(num_fields):
                for level in [0, 1, 2]:
                    m_numerator = 0.0
                    m_denominator = 0.0
                    u_numerator = 0.0
                    u_denominator = 0.0
                    
                    for pattern, count in pattern_counts.items():
                        posterior_prob = posterior_probs[pattern]
                        
                        if pattern[field_idx] == level:
                            m_numerator += count * posterior_prob
                            u_numerator += count * (1 - posterior_prob)
                        
                        m_denominator += count * posterior_prob
                        u_denominator += count * (1 - posterior_prob)
                    
                    # Update parameters with smoothing
                    if m_denominator > 0:
                        m_probs[field_idx][level] = max(0.01, min(0.99, m_numerator / m_denominator))
                    if u_denominator > 0:
                        u_probs[field_idx][level] = max(0.01, min(0.99, u_numerator / u_denominator))
            
            # Check convergence
            total_change = 0
            for field_idx in range(num_fields):
                for level in [0, 1, 2]:
                    total_change += abs(m_probs[field_idx][level] - old_m_probs[field_idx][level])
                    total_change += abs(u_probs[field_idx][level] - old_u_probs[field_idx][level])
            
            if total_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Store results
        self.m_probs = {fields[i]: m_probs[i] for i in range(num_fields)}
        self.u_probs = {fields[i]: u_probs[i] for i in range(num_fields)}
        
        # Calculate match weights
        self.match_weights = {}
        self.non_match_weights = {}
        
        for field in fields:
            self.match_weights[field] = {}
            self.non_match_weights[field] = {}
            
            for level in [0, 1, 2]:
                m_prob = self.m_probs[field][level]
                u_prob = self.u_probs[field][level]
                
                if u_prob > 0:
                    self.match_weights[field][level] = np.log2(m_prob / u_prob)
                else:
                    self.match_weights[field][level] = 10.0
        
        self.is_trained = True
        print("Parameter estimation completed")
    
    def calculate_match_score(self, record1: pd.Series, record2: pd.Series, 
                            fields: List[str]) -> float:
        """Calculate match score for a pair of records."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating match scores")
        
        comparison_vector = self.generate_comparison_vector(record1, record2, fields)
        
        total_score = 0.0
        for field, level in zip(fields, comparison_vector):
            total_score += self.match_weights[field][level]
        
        return total_score
    
    def calculate_match_probability(self, record1: pd.Series, record2: pd.Series, 
                                  fields: List[str]) -> float:
        """Calculate match probability for a pair of records."""
        match_score = self.calculate_match_score(record1, record2, fields)
        
        # Convert log-odds to probability
        odds = 2 ** match_score
        probability = odds / (1 + odds)
        
        return probability
    
    def classify_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      fields: List[str], upper_threshold: float = 0.8,
                      lower_threshold: float = 0.2) -> pd.DataFrame:
        """Classify record pairs into matches, non-matches, and possible matches."""
        if not self.is_trained:
            raise ValueError("Model must be trained before classifying pairs")
        
        results = []
        
        for i in range(len(df1)):
            for j in range(len(df2)):
                record1 = df1.iloc[i]
                record2 = df2.iloc[j]
                
                match_prob = self.calculate_match_probability(record1, record2, fields)
                match_score = self.calculate_match_score(record1, record2, fields)
                
                if match_prob >= upper_threshold:
                    classification = "Match"
                elif match_prob <= lower_threshold:
                    classification = "Non-Match"
                else:
                    classification = "Possible Match"
                
                results.append({
                    'record1_idx': i,
                    'record2_idx': j,
                    'match_probability': match_prob,
                    'match_score': match_score,
                    'classification': classification
                })
        
        return pd.DataFrame(results)
    
    def get_model_summary(self) -> Dict:
        """Get summary of model parameters."""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        return {
            "m_probabilities": self.m_probs,
            "u_probabilities": self.u_probs,
            "match_weights": self.match_weights,
            "prior_match_probability": self.prior_match_prob
        }


# Example usage with personal identification data
if __name__ == "__main__":
    # Create sample datasets with personal identification information
    np.random.seed(42)
    
    # Dataset 1
    df1 = pd.DataFrame({
        'name': ['John Michael Smith', 'Jane Elizabeth Doe', 'Robert James Johnson', 'Alice Marie Brown'],
        'dob': ['1990-05-15', '1985-12-03', '1978-08-22', '1992-03-10'],
        'phone': ['555-123-4567', '555-987-6543', '555-456-7890', '555-234-5678'],
        'address': ['123 Main St, New York, NY 10001', 
                   '456 Oak Ave, Boston, MA 02101',
                   '789 Pine Rd, Chicago, IL 60601',
                   '321 Elm Dr, Seattle, WA 98101'],
        'government_id': ['123-45-6789', '987-65-4321', '456-78-9012', '234-56-7890']
    })
    
    # Dataset 2 (with some matching records but variations)
    df2 = pd.DataFrame({
        'name': ['Jon M. Smith', 'Jane E. Doe', 'Bob Johnson', 'Alice Brown', 'Michael Wilson'],
        'dob': ['1990-05-15', '1985-12-03', '1978-08-23', '1992-03-10', '1980-07-14'],
        'phone': ['(555) 123-4567', '555.987.6543', '555-456-7891', '555-234-5678', '555-888-9999'],
        'address': ['123 Main Street, New York, NY 10001',
                   '456 Oak Avenue, Boston, MA 02101', 
                   '789 Pine Road, Chicago, IL 60601',
                   '321 Elm Drive, Seattle, WA 98101',
                   '999 Broadway, Denver, CO 80201'],
        'government_id': ['123-45-6789', '987-65-4321', '456-78-9013', '234-56-7890', '555-66-7777']
    })
    
    # Initialize model
    model = FelligiSunterModel()
    
    # Train model
    fields = ['name', 'dob', 'phone', 'address', 'government_id']
    model.estimate_parameters_em(df1, df2, fields)
    
    # Print model summary
    print("\nModel Summary:")
    summary = model.get_model_summary()
    for key, value in summary.items():
        if key != 'match_weights':  # Skip detailed weights for brevity
            print(f"{key}: {value}")
    
    # Test individual comparisons
    print("\nTesting individual comparison functions:")
    print(f"Name comparison: {model.name_comparison('John Michael Smith', 'Jon M. Smith')}")
    print(f"DOB comparison: {model.dob_comparison('1990-05-15', '1990-05-15')}")
    print(f"Phone comparison: {model.phone_comparison('555-123-4567', '(555) 123-4567')}")
    print(f"Address comparison: {model.address_comparison('123 Main St, New York, NY 10001', '123 Main Street, New York, NY 10001')}")
    print(f"Government ID comparison: {model.government_id_comparison('123-45-6789', '123-45-6789')}")
    
    # Classify pairs
    print("\nClassifying pairs...")
    results = model.classify_pairs(df1, df2, fields, upper_threshold=0.9, lower_threshold=0.1)
    
    # Display results
    print("\nClassification Results:")
    for _, row in results.iterrows():
        name1 = df1.iloc[row['record1_idx']]['name']
        name2 = df2.iloc[row['record2_idx']]['name']
        print(f"{name1} <-> {name2}: {row['classification']} "
              f"(prob: {row['match_probability']:.3f}, score: {row['match_score']:.3f})")
