from .core import (
    deepanalyze,
    deepanalyze_interceptor,
    deepanalyze_completer,
    _apply_polars_compat_shim,
    __version__
)
from . import cleaners
from . import privacy_knife
from . import dashboard
from . import statistical_engine
from . import storyteller
from . import feature_forge
from . import forecaster
from . import drift_sentinel
from . import schema_synthesizer
from . import synthetic_data
from . import turbo_compiler
from . import debate_router
from . import causal_engine
from . import enricher
from . import pipeline_compiler
from . import optimizer
from . import brain
from . import server
from . import mole_telemetry


def load_ipython_extension(ipython):
    """Called automatically by IPython when running %load_ext deepanalyze"""
    # 1. Apply Polars compatibility runtime shims
    _apply_polars_compat_shim()

    # 2. Register magic function
    ipython.register_magic_function(deepanalyze, magic_kind='line_cell', magic_name='deepanalyze')
    
    # 3. Register tab-completion hook
    ipython.set_hook("complete_command", deepanalyze_completer, re_key=r"%?deepanalyze")
    
    # 4. Register auto-pilot interceptor
    if deepanalyze_interceptor not in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.append(deepanalyze_interceptor)
    
    # 5. Register continuous Live Host Memory HUD
    try:
        if hasattr(ipython, "events") and hasattr(ipython.events, "register"):
            # Avoid duplicate registrations
            if mole_telemetry.post_run_cell_memory_hud not in ipython.events.callbacks.get("post_run_cell", []):
                ipython.events.register("post_run_cell", mole_telemetry.post_run_cell_memory_hud)
    except Exception:
        pass

    print(f"✅ DeepAnalyze Engine (v{__version__} - Universal Adapter) loaded successfully!")
    print("   🦔 Live Host RAM HUD: Active after each cell (toggle with `%deepanalyze --mem-hud`)")
    print("   Tab completion active. Type %deepanalyze --status to check backend.")

def unload_ipython_extension(ipython):
    if deepanalyze_interceptor in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.remove(deepanalyze_interceptor)
    try:
        if hasattr(ipython, "events") and hasattr(ipython.events, "unregister"):
            ipython.events.unregister("post_run_cell", mole_telemetry.post_run_cell_memory_hud)
    except Exception:
        pass