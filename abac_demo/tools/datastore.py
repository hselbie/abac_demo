import json
from google.adk.tools import ToolContext

def get_datastore_content(datastore_name: str, access_level: str, tool_context: ToolContext) -> dict:
    """Fetches content from the specified datastore based on the user's access level.

    Args:
        datastore_name (str): The name of the datastore to access (e.g., 'marketing', 'sales').
        access_level (str): The user's access level ('employee' or 'manager').
        tool_context (ToolContext): The tool context.

    Returns:
        dict: The datastore content or an error message.
    """
    try:
        file_path = f'abac_demo/data/{datastore_name}_data.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if access_level == 'manager':
            return {"status": "success", "data": data['full']}
        elif access_level == 'employee':
            return {"status": "success", "data": data['limited']}
        else:
            return {"status": "error", "message": "Invalid access level."}

    except FileNotFoundError:
        return {"status": "error", "message": f"Datastore '{datastore_name}' not found."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
