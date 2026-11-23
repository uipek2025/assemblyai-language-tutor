"""Test-wide defaults: never call real services, never spend tokens probing models."""
import os

os.environ.setdefault("LLM_STARTUP_PROBE", "0")
