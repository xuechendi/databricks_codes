"""
Test utilities shared between the lookup-client and the core client
"""


def mock_context_manager(mock_context_mgr, resource):
    """
    Modify the mock context manager to return the given resource when it's __enter__() protocol method is invoked.
    """
    mock_context_mgr.__enter__.return_value = resource
