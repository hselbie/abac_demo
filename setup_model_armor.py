
import os
from google.api_core import exceptions
from google.cloud import modelarmor_v1

# --- Configuration ---
# These values are used to create the template.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "zinc-forge-302418")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
TEMPLATE_ID = "ma-all-low" # The template ID your agent uses


def create_model_armor_template():
    """
    Connects to the Model Armor API and creates a predefined template.

    This function defines the configuration for the 'ma-all-low' template,
    which includes a comprehensive set of filters for prompt and response
    sanitization at a low confidence threshold (i.e., high sensitivity).

    You only need to run this script once per project.
    """
    print(f"Attempting to create template '{TEMPLATE_ID}' in project '{PROJECT_ID}'...")

    try:
        # Instantiate the Model Armor client
        # Using a synchronous client here for a simple, one-off script.
        client = modelarmor_v1.ModelArmorClient(
            transport="rest",
            client_options={"api_endpoint": "modelarmor.us-central1.rep.googleapis.com"}
        )

        # Define the full structure of the template
        template_config = {
            "filter_config": {
                "rai_settings": {
                    "rai_filters": [
                        {"filter_type": "HATE_SPEECH", "confidence_level": "LOW_AND_ABOVE"},
                        {"filter_type": "SEXUALLY_EXPLICIT", "confidence_level": "LOW_AND_ABOVE"},
                        {"filter_type": "HARASSMENT", "confidence_level": "LOW_AND_ABOVE"},
                        {"filter_type": "DANGEROUS", "confidence_level": "LOW_AND_ABOVE"},
                    ]
                },
                "pi_and_jailbreak_filter_settings": {
                    "filter_enforcement": "ENABLED",
                    "confidence_level": "LOW_AND_ABOVE",
                },
                "malicious_uri_filter_settings": {"filter_enforcement": "ENABLED"},
                "sdp_settings": {"basic_config": {"filter_enforcement": "ENABLED"}},
            },
            "template_metadata": {
                "log_template_operations": True,
                "log_sanitize_operations": True,
            },
        }

        # Construct the request
        request = modelarmor_v1.CreateTemplateRequest(
            parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
            template_id=TEMPLATE_ID,
            template=template_config,
        )

        # Make the API call to create the template
        response = client.create_template(request=request)

        print("---" * 10)
        print(f"✅ Successfully created template '{TEMPLATE_ID}'.")
        print("You can now run your agent.")
        print("---" * 10)
        print("Template details:")
        print(response)

    except exceptions.AlreadyExists:
        print("---" * 10)
        print(f"✅ Template '{TEMPLATE_ID}' already exists in project '{PROJECT_ID}'.")
        print("No action needed. You can run your agent.")
        print("---" * 10)
    except exceptions.PermissionDenied as e:
        print("---" * 10)
        print("❌ ERROR: Permission Denied.")
        print("Please ensure you have the 'Model Armor Admin' (roles/modelarmor.admin) role")
        print(f"in the '{PROJECT_ID}' project and that the Model Armor API is enabled.")
        print(f"Full error: {e}")
        print("---" * 10)
    except Exception as e:
        print("---" * 10)
        print(f"❌ An unexpected error occurred: {e}")
        print("---" * 10)

if __name__ == "__main__":
    create_model_armor_template()
