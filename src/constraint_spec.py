"""
constraint_spec.py
------------------
Pydantic schema that the LLM must populate.
The instance_builder translates a valid ConstraintSpec into a solver-ready instance.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_BLOCKS   = {'34/7', '34/10'}
VALID_RIGS     = {'Rig-Alpha', 'Rig-Beta', 'Rig-Gamma'}
VALID_PRIORITY = {'makespan', 'deferred'}


class ConstraintSpec(BaseModel):
    # --- Subset filters ---
    n_wells: Optional[int] = Field(
        default=None, ge=2, le=20,
        description="Max number of wells to schedule (top-N by DST rate)"
    )
    min_flow_rate: Optional[int] = Field(
        default=None, ge=0,
        description="Minimum DST oil rate (bbl/day) – exclude wells below this"
    )
    sub_blocks: Optional[list[str]] = Field(
        default=None,
        description="Restrict to these block(s): '34/7' and/or '34/10'"
    )

    # --- Rig constraints ---
    n_rigs: Optional[int] = Field(
        default=None, ge=1, le=3,
        description="Number of rigs to use (1-3)"
    )
    rig_cluster_lock: Optional[dict[str, str]] = Field(
        default=None,
        description="Lock a rig to start in a specific block, e.g. {'Rig-Alpha': '34/7'}"
    )

    # --- Campaign deadline ---
    max_makespan_days: Optional[int] = Field(
        default=None, ge=100, le=1000,
        description="Hard upper bound on campaign length (days). "
                    "Only Pareto points within this limit are shown."
    )

    # --- Objective priority ---
    prioritize: Optional[str] = Field(
        default=None,
        description="'makespan' or 'deferred' – which objective to emphasise"
    )

    # --- Always required ---
    explanation: str = Field(
        description="One sentence explaining what this spec does in plain English"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator('sub_blocks')
    @classmethod
    def check_blocks(cls, v):
        if v is not None:
            bad = [b for b in v if b not in VALID_BLOCKS]
            if bad:
                raise ValueError(
                    "Unknown block(s): {}. Must be from {}.".format(bad, VALID_BLOCKS))
        return v

    @field_validator('rig_cluster_lock')
    @classmethod
    def check_rig_cluster(cls, v):
        if v is not None:
            for rig, block in v.items():
                if rig not in VALID_RIGS:
                    raise ValueError(
                        "Unknown rig '{}'. Must be one of {}.".format(rig, VALID_RIGS))
                if block not in VALID_BLOCKS:
                    raise ValueError(
                        "Unknown block '{}'. Must be one of {}.".format(block, VALID_BLOCKS))
        return v

    @field_validator('prioritize')
    @classmethod
    def check_priority(cls, v):
        if v is not None and v not in VALID_PRIORITY:
            raise ValueError(
                "prioritize must be 'makespan' or 'deferred', got '{}'".format(v))
        return v
