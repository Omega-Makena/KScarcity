import streamlit as st
import functools
import types

def log_args_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open("streamlit_widget_log.txt", "a") as f:
            f.write(f"Calling: {func.__name__}\n")
            f.write(f"  args: {args}\n")
            f.write(f"  kwargs: {kwargs}\n")
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            with open("streamlit_widget_log.txt", "a") as f:
                f.write(f"  ERROR CAUGHT: {e}\n")
            raise
    return wrapper

# Wrap common widgets
st.selectbox = log_args_wrapper(st.selectbox)
st.multiselect = log_args_wrapper(st.multiselect)
st.radio = log_args_wrapper(st.radio)
st.button = log_args_wrapper(st.button)
st.slider = log_args_wrapper(st.slider)
st.markdown = log_args_wrapper(st.markdown)

# Also wrap caching
if hasattr(st, "cache_data"):
    original_cache = st.cache_data
    def new_cache(*c_args, **c_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def inner_wrapper(*args, **kwargs):
                with open("streamlit_widget_log.txt", "a") as f:
                    f.write(f"Calling cached func: {func.__name__}\n")
                    f.write(f"  args: {args}\n")
                    f.write(f"  kwargs: {kwargs}\n")
                try:
                    return func(*args, **kwargs)
                except TypeError as e:
                    with open("streamlit_widget_log.txt", "a") as f:
                        f.write(f"  CACHE ERROR CAUGHT: {e}\n")
                    raise
            # We don't actually cache here to ensure it runs
            return inner_wrapper
        return decorator
    st.cache_data = new_cache

with open("streamlit_widget_log.txt", "w") as f:
    f.write("--- Start Log ---\n")

try:
    from kshiked.ui.institution.executive_simulator import render_executive_simulator
    render_executive_simulator()
except Exception as e:
    import traceback
    with open("streamlit_widget_log.txt", "a") as f:
        f.write(f"FATAL ERROR:\n{traceback.format_exc()}\n")
    print("Logged to streamlit_widget_log.txt")
