# Entity-Management
In today's data-driven world, organizations increasingly rely on vast and diverse datasets sourced from internal systems, partners, and third-party providers. These datasets often contain multiple records that refer to the same real-world entities—such as individuals, businesses, or products—but with variations in formatting, spelling, missing information, or inconsistencies due to system differences. Entity Management is the discipline of identifying, linking, and maintaining a single, consistent representation of entities across disparate datasets. It plays a foundational role in data integration, customer 360 views, master data management (MDM), fraud detection, and compliance reporting.

At the heart of entity management is the task of data matching, which involves determining whether two or more records refer to the same entity. This task is complicated by noisy data, cultural naming conventions, nicknames, typographical errors, and structural discrepancies. To address these challenges, probabilistic matching models are often employed—offering flexibility and accuracy beyond rule-based systems.

One of the most well-established probabilistic models is the Fellegi-Sunter model, developed in the 1960s for record linkage in census data. It provides a mathematical foundation for deciding whether record pairs match, based on the likelihood of agreement across multiple attributes (such as name, date of birth, address, etc.).

 # Felligi-Sunter Model for Person Entity Record Linkage

A comprehensive Python implementation of the Felligi-Sunter probabilistic record linkage model, specifically optimized for person entity identification records containing name, date of birth, phone number, address, and government ID information.

## Overview

The Felligi-Sunter model is a probabilistic approach to record linkage that calculates match probabilities between record pairs based on agreement patterns across multiple fields. This implementation provides specialized comparison functions for common personal identification data types and uses an Expectation-Maximization (EM) algorithm for parameter estimation.

## Features

### Specialized Comparison Functions

- **Name Comparison**: Multi-level fuzzy matching using Jaro-Winkler similarity and component-based matching
- **Date of Birth Comparison**: Handles multiple date formats with tolerance for common data entry errors
- **Phone Number Comparison**: Normalizes phone formats and supports partial matching
- **Address Comparison**: Component-based matching with address normalization and abbreviation handling
- **Government ID Comparison**: Secure matching with support for partial ID comparison

### Advanced Capabilities

- **Multi-level Matching**: Each field returns 0 (no match), 1 (partial match), or 2 (strong match)
- **Automated Parameter Learning**: Uses EM algorithm to learn optimal match/non-match probabilities
- **Scalable Processing**: Handles large datasets with intelligent sampling
- **Privacy-Aware**: Includes partial matching options for sensitive government IDs

## Installation

```bash
pip install numpy pandas scipy
```

## Quick Start

```python
from felligi_sunter import FelligiSunterModel
import pandas as pd

# Load your datasets
df1 = pd.DataFrame({
    'name': ['John Michael Smith', 'Jane Elizabeth Doe'],
    'dob': ['1990-05-15', '1985-12-03'],
    'phone': ['555-123-4567', '555-987-6543'],
    'address': ['123 Main St, New York, NY 10001', '456 Oak Ave, Boston, MA 02101'],
    'government_id': ['123-45-6789', '987-65-4321']
})

df2 = pd.DataFrame({
    'name': ['Jon M. Smith', 'Jane E. Doe'],
    'dob': ['1990-05-15', '1985-12-03'],
    'phone': ['(555) 123-4567', '555.987.6543'],
    'address': ['123 Main Street, New York, NY 10001', '456 Oak Avenue, Boston, MA 02101'],
    'government_id': ['123-45-6789', '987-65-4321']
})

# Initialize and train model
model = FelligiSunterModel()
fields = ['name', 'dob', 'phone', 'address', 'government_id']
model.estimate_parameters_em(df1, df2, fields)

# Classify record pairs
results = model.classify_pairs(df1, df2, fields)
print(results)
```

## Detailed Usage

### 1. Model Initialization

```python
model = FelligiSunterModel()
```

The model comes pre-configured with specialized comparison functions for personal identification data.

### 2. Training the Model

```python
# Define fields to compare
fields = ['name', 'dob', 'phone', 'address', 'government_id']

# Train using EM algorithm
model.estimate_parameters_em(
    df1, df2, fields,
    max_iterations=100,
    tolerance=1e-6,
    max_pairs=50000  # Limit for large datasets
)
```

### 3. Individual Record Comparison

```python
# Calculate match probability for a specific pair
record1 = df1.iloc[0]
record2 = df2.iloc[0]

probability = model.calculate_match_probability(record1, record2, fields)
score = model.calculate_match_score(record1, record2, fields)

print(f"Match probability: {probability:.3f}")
print(f"Match score: {score:.3f}")
```

### 4. Batch Classification

```python
# Classify all pairs with custom thresholds
results = model.classify_pairs(
    df1, df2, fields,
    upper_threshold=0.8,  # Above this = Match
    lower_threshold=0.2   # Below this = Non-Match
)

# Filter for matches only
matches = results[results['classification'] == 'Match']
```

### 5. Model Analysis

```python
# Get model parameters
summary = model.get_model_summary()
print("M-probabilities:", summary['m_probabilities'])
print("U-probabilities:", summary['u_probabilities'])
```

## Comparison Function Details

### Name Comparison
- **Level 0**: No meaningful similarity
- **Level 1**: Partial match (e.g., similar components, moderate Jaro-Winkler score)
- **Level 2**: Strong match (exact match or high similarity with matching components)

**Examples**:
- "John Michael Smith" vs "Jon M. Smith" → Level 2
- "John Smith" vs "John Doe" → Level 1
- "John Smith" vs "Mary Johnson" → Level 0

### Date of Birth Comparison
- **Level 0**: No match or significantly different dates
- **Level 1**: Close match with potential data entry errors
- **Level 2**: Exact match

**Tolerance for**:
- Wrong day (±2 days in same month/year)
- Wrong month (±1 month in same year/day)
- Wrong year (±1 year in same month/day)

### Phone Number Comparison
- **Level 0**: No match
- **Level 1**: Partial match (same local number or missing area code)
- **Level 2**: Exact match after normalization

**Handles**:
- Different formatting: "(555) 123-4567" vs "555-123-4567"
- Country codes: "+1-555-123-4567" vs "555-123-4567"
- Extensions and additional digits

### Address Comparison
- **Level 0**: No significant component matches
- **Level 1**: Some component matches (50-79% of components)
- **Level 2**: Strong component matches (80%+ of components)

**Normalizes**:
- Street types: "Street" → "St", "Avenue" → "Ave"
- Apartment indicators: "Apartment" → "Apt"
- Case and spacing variations

### Government ID Comparison
- **Level 0**: No match
- **Level 1**: Partial match (e.g., last 4 digits of SSN)
- **Level 2**: Exact match after normalization

**Features**:
- Removes formatting (dashes, spaces)
- Case-insensitive matching
- Partial matching for privacy protection

## Performance Considerations

### Large Datasets
- The model automatically samples pairs when dealing with large datasets
- Adjust `max_pairs` parameter in `estimate_parameters_em()` based on memory constraints
- Consider implementing blocking strategies for very large datasets

### Memory Usage
- Memory usage scales with the number of unique comparison patterns
- Typical usage: ~1-10MB for datasets with 1,000-10,000 records
- For larger datasets, consider processing in batches

### Training Time
- Training time depends on dataset size and number of EM iterations
- Typical training: 1-30 seconds for datasets with 1,000-10,000 records
- EM algorithm typically converges in 10-50 iterations

## Model Parameters

### Thresholds
- **Upper Threshold (default: 0.8)**: Above this probability, pairs are classified as matches
- **Lower Threshold (default: 0.2)**: Below this probability, pairs are classified as non-matches
- **Prior Match Probability (default: 0.01)**: Expected proportion of true matches in the dataset

### EM Algorithm
- **Max Iterations (default: 100)**: Maximum number of EM iterations
- **Tolerance (default: 1e-6)**: Convergence tolerance for parameter changes
- **Max Pairs (default: 50,000)**: Maximum number of pairs to use for training

## Advanced Usage

### Custom Comparison Functions

```python
# Add custom comparison function
def custom_name_comparison(name1, name2):
    # Your custom logic here
    return 0  # or 1 or 2

model.comparison_functions['name'] = custom_name_comparison
```

### Handling Missing Data
- All comparison functions handle missing/null values gracefully
- Missing values automatically return comparison level 0 (no match)
- Consider data imputation strategies for better results

### Performance Tuning

```python
# For large datasets, reduce max_pairs and increase tolerance
model.estimate_parameters_em(
    df1, df2, fields,
    max_pairs=10000,    # Reduce for faster training
    tolerance=1e-4,     # Increase for faster convergence
    max_iterations=50   # Reduce if needed
)
```

## Output Format

### Classification Results
The `classify_pairs()` method returns a DataFrame with:
- `record1_idx`: Index of record in first dataset
- `record2_idx`: Index of record in second dataset
- `match_probability`: Probability that the pair is a match (0-1)
- `match_score`: Log-likelihood ratio score
- `classification`: "Match", "Non-Match", or "Possible Match"

### Model Summary
The `get_model_summary()` method returns:
- `m_probabilities`: Agreement probabilities given match
- `u_probabilities`: Agreement probabilities given non-match
- `match_weights`: Log-likelihood ratios for each comparison level
- `prior_match_probability`: Prior probability of a match

## Limitations

- **Computational Complexity**: O(n²) for comparing all pairs; consider blocking for large datasets
- **Parameter Sensitivity**: Results may vary with different threshold settings
- **Domain Specific**: Optimized for personal identification data; may need adaptation for other domains
- **Privacy Considerations**: Exercise caution when working with sensitive personal information

## Best Practices

1. **Data Preprocessing**: Clean and standardize data before training
2. **Threshold Tuning**: Adjust thresholds based on your precision/recall requirements
3. **Validation**: Use ground truth data when available to validate model performance
4. **Blocking**: Implement blocking strategies for large datasets to improve efficiency
5. **Privacy**: Consider data anonymization and secure processing practices

## License

This implementation is provided as-is for educational and research purposes under Apache 2.0 license. Please ensure compliance with applicable privacy laws and regulations when working with personal identification data.

## Contributing

Contributions are welcome! Please consider:
- Additional comparison functions for other data types
- Performance optimizations
- Blocking strategy implementations
- Evaluation metrics and validation tools

## References

- Fellegi, I. P., & Sunter, A. B. (1969). A theory for record linkage. Journal of the American Statistical Association, 64(328), 1183-1210.
- Christen, P. (2012). Data matching: concepts and techniques for record linkage, entity resolution, and duplicate detection. Springer Science & Business Media.
