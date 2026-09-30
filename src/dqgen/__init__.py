"""dqgen: LLM-assisted generation of data quality constraints.

Experiment codebase for the paper "LLM-Assisted Generation of Data Quality
Constraints from Schema and Documentation".
"""

__version__ = "0.1.0"

# Canonical column name used to align injected-error labels with Great
# Expectations' ``unexpected_index_list`` output. Every clean/corrupted
# dataframe carries this column; it is stable across row drops/duplication.
ROW_ID = "_row_id"
