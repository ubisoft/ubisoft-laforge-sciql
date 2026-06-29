import numpy as np
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Union
from collections import Counter
from typing import Iterable, List, Optional, Set

def majority_window(
    labels: Iterable[int],
    window: int,
    *,
    exclude: Optional[Set[int]] = None,
    tie_break: str = "keep",   # "prev" | "keep" | "min"
) -> List[int]:
    """
    Centered majority vote over a sliding window.
    At time t, votes over indices [t - window//2, t + window//2] (clamped).

    - Even or <1 windows are bumped up to the next odd >= 1.
    - 'exclude' lets you ignore labels (e.g., undetermined bin).
    - Ties resolved by:
        "prev": prefer previously emitted label if present in tie; else "min"
        "keep": prefer the current center label if present in tie; else "min"
        "min" : choose the smallest label id among the tie
    """
    labs = list(labels)
    n = len(labs)
    if n == 0:
        return []
    if window == 1:
        return labels

    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    ex = exclude or set()

    out: List[int] = []
    prev = None
    for t in range(n):
        start = max(0, t - half)
        end = min(n, t + half + 1)  # end exclusive
        segment = [x for x in labs[start:end] if x not in ex]

        if not segment:
            choice = labs[t]  # if everything excluded, keep center
        else:
            counts = Counter(segment)
            maxc = max(counts.values())
            tied = [k for k, v in counts.items() if v == maxc]
            if len(tied) == 1:
                choice = tied[0]
            else:
                if tie_break == "prev" and prev in tied:
                    choice = prev
                elif tie_break == "keep" and labs[t] in tied:
                    choice = labs[t]
                elif tie_break == "min":
                    choice = min(tied)
                else:
                    choice = min(tied)  # default fallback
        out.append(int(choice))
        prev = out[-1]
    return out

class LabelsDistribution:
    """Encapsulates utilities for computing marginal, joint, and conditional
    distributions over categorical label sequences. All previously standalone
    functions are now instance methods. The original function signatures are
    preserved except that they now accept *self* as the first argument. When a
    *labels* or *value_ranges* parameter is omitted (``None``), the method falls
    back to the data stored on the instance.
    """

    def __init__(self, labels: Dict[str, List[int]], value_ranges: Dict[str, List[int]] = None):
        self.labels = labels
        self.value_ranges = value_ranges

    # ---------------------------------------------------------------------
    # Distributions
    # ---------------------------------------------------------------------
    def compute_label_distributions(self,
                                    labels: Dict[str, List[int]] = None,
                                    value_ranges: Dict[str, List[int]] = None) -> Dict:
        """Wrapper around the original *compute_label_distributions* function.
        If *labels* or *value_ranges* are not supplied, the instance attributes
        are used instead.
        """
        labels = labels if labels is not None else self.labels
        value_ranges = value_ranges if value_ranges is not None else self.value_ranges

        assert labels, "Labels dictionary cannot be empty"

        # Verify all sequences have same length
        lengths = [len(seq) for seq in labels.values()]
        assert all(l == lengths[0] for l in lengths), "All label sequences must have same length"

        label_names = list(labels.keys())
        T = lengths[0]  # sequence length

        # Infer value ranges if not provided
        if value_ranges is None:
            value_ranges = {
                name: sorted(set(values))
                for name, values in labels.items()
            }

        # Validate that actual values are within specified ranges
        for name, values in labels.items():
            if name in value_ranges:
                actual_values = set(values)
                allowed_values = set(value_ranges[name])
                if not actual_values.issubset(allowed_values):
                    raise ValueError(f"Label '{name}' contains values {actual_values - allowed_values} "
                                     f"not in specified range {allowed_values}")

        # --- Marginal Distributions (with 0% for missing values) ---
        marginals = {}
        for name in label_names:
            value_counts = Counter(labels[name])
            marginals[name] = {
                v: 100 * value_counts.get(v, 0) / T
                for v in value_ranges[name]
            }

        # --- Joint Distributions (with 0% for missing combinations) ---
        joint_distributions = {}

        for r in range(2, len(label_names) + 1):
            for combo in itertools.combinations(label_names, r):
                combo_key = ' + '.join(combo)

                # Extract value tuples from the sequence
                value_tuples = []
                for t in range(T):
                    tuple_vals = tuple(labels[label][t] for label in combo)
                    value_tuples.append(tuple_vals)

                counts = Counter(value_tuples)

                # Build the full Cartesian product of possible values
                value_space = [value_ranges[label] for label in combo]
                all_possible = list(itertools.product(*value_space))

                joint_distributions[combo_key] = {
                    v: 100 * counts.get(v, 0) / T
                    for v in all_possible
                }

        return {
            'marginals': marginals,
            'joint_distributions': joint_distributions
        }

    # ---------------------------------------------------------------------
    # Conditional probability of a single event
    # ---------------------------------------------------------------------
    def compute_conditional_probability(self,
                                         labels: Dict[str, List[int]] = None,
                                         target_labels: List[str] = None,
                                         target_values: List[int] = None,
                                         condition_labels: List[str] = None,
                                         condition_values: List[int] = None,
                                         allow_overlap: bool = True) -> float:
        """Compute the conditional probability
        P(target_labels = target_values | condition_labels = condition_values).
        All parameters mirror the original standalone function. Any missing
        *labels* parameter defaults to the instance data.
        """
        labels = labels if labels is not None else self.labels
        assert labels, "Labels dictionary cannot be empty"
        assert len(target_labels) == len(target_values), "target_labels and target_values must have same length"
        assert len(condition_labels) == len(condition_values), "condition_labels and condition_values must have same length"

        # Check all labels exist
        missing_labels = (set(target_labels) | set(condition_labels)) - set(labels.keys())
        if missing_labels:
            raise ValueError(f"Labels not found in data: {missing_labels}")

        # Check for overlap between target and condition labels
        if not allow_overlap:
            overlap = set(target_labels) & set(condition_labels)
            if overlap:
                raise ValueError(f"Target and condition labels overlap: {overlap}. "
                                 f"This leads to tautological conditioning. "
                                 f"Set allow_overlap=True to allow this.")

        # Verify all sequences have same length
        lengths = [len(seq) for seq in labels.values()]
        assert all(l == lengths[0] for l in lengths), "All label sequences must have same length"

        T = lengths[0]  # Sequence length

        # Count matching time steps
        count_condition = 0
        count_joint = 0

        for t in range(T):
            # Check if condition is satisfied at time t
            cond_satisfied = all(
                labels[label][t] == value
                for label, value in zip(condition_labels, condition_values)
            )

            if cond_satisfied:
                count_condition += 1

                # Check if target is also satisfied at time t
                target_satisfied = all(
                    labels[label][t] == value
                    for label, value in zip(target_labels, target_values)
                )

                if target_satisfied:
                    count_joint += 1

        if count_condition == 0:
            return 0.0  # conditioning event never happened

        return count_joint / count_condition

    # ---------------------------------------------------------------------
    # Exhaustive conditional probabilities
    # ---------------------------------------------------------------------
    def compute_all_conditional_probabilities(self,
                                              labels: Dict[str, List[int]] = None,
                                              value_ranges: Dict[str, List[int]] = None,
                                              max_condition_size: int = 2,
                                              max_target_size: int = 2,
                                              min_condition_count: int = 1) -> Dict:
        """Compute *all* conditional probabilities up to the specified sizes.
        Mirrors the original function. Uses stored labels/value_ranges if not
        supplied.
        """
        labels = labels if labels is not None else self.labels
        value_ranges = value_ranges if value_ranges is not None else self.value_ranges

        assert labels, "Labels dictionary cannot be empty"

        # Setup
        lengths = [len(seq) for seq in labels.values()]
        assert all(l == lengths[0] for l in lengths), "All sequences must have same length"

        if value_ranges is None:
            value_ranges = {name: sorted(set(values)) for name, values in labels.items()}

        label_names = list(labels.keys())
        T = lengths[0]

        all_conditionals = {}

        # Generate all possible target combinations (1 to max_target_size labels)
        for target_size in range(1, min(max_target_size + 1, len(label_names) + 1)):
            for target_labels in itertools.combinations(label_names, target_size):
                target_value_space = [value_ranges[label] for label in target_labels]

                # All possible value combinations for this target set
                for target_values in itertools.product(*target_value_space):

                    # Generate all possible condition combinations
                    remaining_labels = [l for l in label_names if l not in target_labels]

                    for cond_size in range(1, min(max_condition_size + 1, len(remaining_labels) + 1)):
                        for condition_labels in itertools.combinations(remaining_labels, cond_size):
                            cond_value_space = [value_ranges[label] for label in condition_labels]

                            # All possible value combinations for this condition set
                            for condition_values in itertools.product(*cond_value_space):

                                # Count occurrences
                                count_condition = 0
                                count_joint = 0

                                for t in range(T):
                                    # Check condition
                                    cond_satisfied = all(
                                        labels[label][t] == value
                                        for label, value in zip(condition_labels, condition_values)
                                    )

                                    if cond_satisfied:
                                        count_condition += 1

                                        # Check target
                                        target_satisfied = all(
                                            labels[label][t] == value
                                            for label, value in zip(target_labels, target_values)
                                        )

                                        if target_satisfied:
                                            count_joint += 1

                                # Only include if condition occurs enough times
                                if count_condition >= min_condition_count:
                                    prob = count_joint / count_condition if count_condition > 0 else 0.0

                                    # Create readable key
                                    target_str = ', '.join(f"{l}={v}" for l, v in zip(target_labels, target_values))
                                    cond_str = ', '.join(f"{l}={v}" for l, v in zip(condition_labels, condition_values))
                                    key = f"P({target_str} | {cond_str})"

                                    all_conditionals[key] = {
                                        'probability': prob,
                                        'target_combo': (target_labels, target_values),
                                        'condition_combo': (condition_labels, condition_values),
                                        'condition_count': count_condition,
                                        'joint_count': count_joint
                                    }

        return all_conditionals

    # ---------------------------------------------------------------------
    # Full conditional table for specific sets
    # ---------------------------------------------------------------------
    def compute_conditional_probability_table(self,
                                              labels: Dict[str, List[int]] = None,
                                              target_labels: List[str] = None,
                                              condition_labels: List[str] = None,
                                              value_ranges: Dict[str, List[int]] = None) -> Dict:
        """Compute a full conditional probability table for given target and
        condition label lists.
        """
        labels = labels if labels is not None else self.labels
        value_ranges = value_ranges if value_ranges is not None else self.value_ranges

        if value_ranges is None:
            value_ranges = {name: sorted(set(values)) for name, values in labels.items()}

        # Check that target and condition don't overlap
        overlap = set(target_labels) & set(condition_labels)
        if overlap:
            raise ValueError(f"Target and condition labels overlap: {overlap}")

        T = len(next(iter(labels.values())))

        target_value_space = [value_ranges[label] for label in target_labels]
        cond_value_space = [value_ranges[label] for label in condition_labels]

        table = {}

        for condition_values in itertools.product(*cond_value_space):
            cond_key = tuple(zip(condition_labels, condition_values))

            # Count condition occurrences
            count_condition = 0
            target_counts = {}

            for t in range(T):
                # Check if condition is satisfied
                cond_satisfied = all(
                    labels[label][t] == value
                    for label, value in zip(condition_labels, condition_values)
                )

                if cond_satisfied:
                    count_condition += 1

                    # Record target values when condition is met
                    target_vals = tuple(labels[label][t] for label in target_labels)
                    target_counts[target_vals] = target_counts.get(target_vals, 0) + 1

            # Compute probabilities for all possible target combinations
            cond_str = ', '.join(f"{l}={v}" for l, v in cond_key)

            for target_values in itertools.product(*target_value_space):
                target_str = ', '.join(f"{l}={v}" for l, v in zip(target_labels, target_values))
                key = f"P({target_str} | {cond_str})"

                if count_condition == 0:
                    table[key] = 0.0
                else:
                    joint_count = target_counts.get(target_values, 0)
                    table[key] = joint_count / count_condition

        return table

    # ---------------------------------------------------------------------
    # Pretty printers
    # ---------------------------------------------------------------------
    def print_distributions(self, result: Dict, max_combinations: int = 10):
        """Pretty-print the distribution results. Identical to original."""
        print("\U0001F4CA Marginal Distributions:")
        for label, dist in result['marginals'].items():
            print(f"  {label}:")
            for val, perc in sorted(dist.items()):
                print(f"    {val}: {perc:.2f}%")

        print("\n\U0001F517 Joint Distributions:")
        for combo, dist in result['joint_distributions'].items():
            print(f"  {combo}:")
            # Sort by percentage (descending) then by values
            sorted_items = sorted(dist.items(), key=lambda x: (-x[1], x[0]))
            for i, (vals, perc) in enumerate(sorted_items):
                if i >= max_combinations:
                    remaining = len(sorted_items) - max_combinations
                    print(f"    ... and {remaining} more combinations")
                    break
                print(f"    {vals}: {perc:.2f}%")

    def print_conditional_probabilities(self,
                                        conditionals: Dict,
                                        min_probability: float = 0.0,
                                        max_results: int = 20):
        """Pretty-print conditional probabilities. Identical to original."""
        print("\U0001F3AF Conditional Probabilities:")

        # Filter and sort by probability
        filtered = {
            k: v for k, v in conditionals.items()
            if v['probability'] >= min_probability
        }

        sorted_items = sorted(
            filtered.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )

        for i, (key, info) in enumerate(sorted_items):
            if i >= max_results:
                remaining = len(sorted_items) - max_results
                print(f"    ... and {remaining} more probabilities")
                break

            prob = info['probability']
            cond_count = info['condition_count']
            joint_count = info['joint_count']

            print(f"  {key} = {prob:.4f} ({joint_count}/{cond_count})")


# -------------------------------------------------------------------------
# Self-test (mirrors original __main__ block)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Test data
    labels = {
        'label_a': [0, 1, 2, 3, 2, 0, 1],
        'label_b': [1, 1, 0, 0, 1, 1, 0],
        'label_c': [2, 1, 0, 1, 2, 0, 1]
    }

    # Specify value ranges (optional - will be inferred if not provided)
    value_ranges = {
        'label_a': [0, 1, 2, 3],
        'label_b': [0, 1],
        'label_c': [0, 1, 2]
    }

    ld = LabelsDistribution(labels, value_ranges)

    print("=== Testing Distribution Computation ===")
    result = ld.compute_label_distributions()
    ld.print_distributions(result, max_combinations=5)

    print("\n=== Testing Single Conditional Probability ===")

    # Valid conditional probability
    try:
        p1 = ld.compute_conditional_probability(
            target_labels=['label_a'],
            target_values=[0],
            condition_labels=['label_c'],
            condition_values=[2]
        )
        print(f"P(label_a=0 | label_c=2) = {p1:.4f}")
    except Exception as e:
        print(f"Error: {e}")

    # Multiple target and condition labels
    try:
        p2 = ld.compute_conditional_probability(
            target_labels=['label_a', 'label_b'],
            target_values=[0, 1],
            condition_labels=['label_c'],
            condition_values=[2]
        )
        print(f"P(label_a=0, label_b=1 | label_c=2) = {p2:.4f}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Testing ALL Conditional Probabilities ===")

    # Compute all possible conditional probabilities (limited scope for demo)
    all_conditionals = ld.compute_all_conditional_probabilities(
        max_condition_size=1,  # Only single-label conditions for demo
        max_target_size=1,     # Only single-label targets for demo
        min_condition_count=2  # Only include if condition occurs at least 2 times
    )

    print(f"Found {len(all_conditionals)} conditional probabilities:")
    ld.print_conditional_probabilities(all_conditionals, min_probability=0.1, max_results=10)

    print("\n=== Testing Conditional Probability Table ===")

    # Complete table for specific target/condition sets
    table = ld.compute_conditional_probability_table(
        target_labels=['label_a'],
        condition_labels=['label_b']
    )

    print("Complete table P(label_a | label_b):")
    for key, prob in sorted(table.items()):
        print(f"  {key} = {prob:.4f}")

    print("\n=== Testing Error Cases ===")

    # This should raise an error (overlap between target and condition)
    try:
        p3 = ld.compute_conditional_probability(
            target_labels=['label_a'],
            target_values=[2],
            condition_labels=['label_a'],
            condition_values=[2]
        )
        print(f"P(label_a=2 | label_a=2) = {p3:.4f}")
    except Exception as e:
        print(f"Expected error caught: {e}")

    # Same as above but with overlap allowed
    try:
        p4 = ld.compute_conditional_probability(
            target_labels=['label_a'],
            target_values=[2],
            condition_labels=[],
            condition_values=[],
            allow_overlap=True
        )
        print(f"P(label_a=2 | label_a=2) with overlap allowed = {p4:.4f}")
    except Exception as e:
        print(f"Error: {e}")
