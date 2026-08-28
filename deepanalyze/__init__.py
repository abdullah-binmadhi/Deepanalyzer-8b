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
    
    print(f"✅ DeepAnalyze Engine (v{__version__} - Universal Adapter) loaded successfully!")
    print("   Tab completion active. Type %deepanalyze --status to check backend.")

def unload_ipython_extension(ipython):
    if deepanalyze_interceptor in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.remove(deepanalyze_interceptor)