from .core import deepanalyze, deepanalyze_interceptor

def load_ipython_extension(ipython):
    """Called automatically by IPython when running %load_ext deepanalyze"""
    ipython.register_magic_function(deepanalyze, magic_kind='line_cell', magic_name='deepanalyze')
    
    if deepanalyze_interceptor not in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.append(deepanalyze_interceptor)
    
    print("✅ DeepAnalyze Engine (v2.0 - Universal Adapter) loaded successfully!")
    print("   Type %deepanalyze --status to check backend.")

def unload_ipython_extension(ipython):
    if deepanalyze_interceptor in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.remove(deepanalyze_interceptor)