# ------------------------------------------------------------------------------
# A module showing the style conventions for class and public methods.
#
# For types, the absolute path does not need to be specified for 
# classes defined in the project, unless there are several classes with the 
# same name.
#
# Please try to limit line length to 80 characters and write pytest 
# tests for deterministic functions. They should be placed in the test directory.
# ------------------------------------------------------------------------------

class ExampleClass:
    r"""Class description.

    Args:
        example_arg (type): description of __init__ argument
    """
    def __init__(self, example_arg):
        self.example_arg = example_arg

def example_method(arg_1, arg_2):
    r"""Description.

    Args:
        arg_1 (type): description
        arg_2 (Seq[type]): description

    Returns (type): description

    Examples:
        >>> example_method(1, 2) --> 2
    """
    return arg_1*arg_2
