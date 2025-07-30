"""A tool for handling sensitive data."""

def handle_sensitive_data(data: str) -> dict:
    """Handles sensitive data.

    Args:
        data (str): The sensitive data to handle.

    Returns:
        dict: A dictionary with the status of the operation.
    """
    if "sensitive" in data:
        return {"status": "error", "message": "This data is too sensitive to handle."}
    return {"status": "success", "data": f"Successfully handled data: {data}"}
