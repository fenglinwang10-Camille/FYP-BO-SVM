"""
display_utils.py

Small optional helpers for showing results directly inside Jupyter.
Terminal runs still work because these helpers silently do nothing when
IPython display tools are not available.
"""


def display_dataframe(dataframe):
    """Display a pandas DataFrame in Jupyter if possible."""
    try:
        from IPython.display import display

        display(dataframe)
    except Exception:
        pass


def display_image(image_path):
    """Display a saved image in Jupyter if possible."""
    try:
        from IPython.display import Image, display

        display(Image(filename=str(image_path)))
    except Exception:
        pass
