from typing import Callable, Any
import pandas as pd

def is_function(value: any) -> bool:
    """
    Processes the input, which can be either a constant (int or float) or a function returning an int or float.
    
    Args:
        value: A number or a function that returns a number.

    Returns:
        The evaluated number.
    """
    if (callable(value)):
        return True
    else:
        return False





    #--------------------------------------------



if __name__ == "__main__":
    # This will only run if you execute utils.py directly
    print(is_function(2))
