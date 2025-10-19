import os
import yaml
from dotenv import load_dotenv

def try_cast(value: str):
    """Try to cast env values to int or float when appropriate."""
    if value is None:
        return None
    value = value.strip()
    # Try int
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Return as string otherwise
    return value

def replace_placeholders(obj):
    """Recursively replace ${VAR} placeholders in both keys and values."""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Replace in key
            if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
                env_key = key[2:-1]
                key = os.getenv(env_key, key)
            elif isinstance(key, str):
                # Replace inside key if partial pattern exists (e.g. "prefix-${VAR}-suffix")
                matches = [part for part in key.split("${") if "}" in part]
                for match in matches:
                    var_name = match.split("}")[0]
                    env_value = os.getenv(var_name)
                    if env_value:
                        key = key.replace("${" + var_name + "}", env_value)
            # Replace recursively in values
            new_dict[key] = replace_placeholders(value)
        return new_dict

    elif isinstance(obj, list):
        return [replace_placeholders(i) for i in obj]

    elif isinstance(obj, str):
        # Full substitution
        if obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            val = os.getenv(env_var)
            if val is None:
                raise ValueError(f"Missing environment variable: {env_var}")
            return try_cast(val)
        # Partial substitution (e.g., "some-${VAR}-value")
        matches = [part for part in obj.split("${") if "}" in part]
        for match in matches:
            var_name = match.split("}")[0]
            env_value = os.getenv(var_name)
            if env_value:
                obj = obj.replace("${" + var_name + "}", env_value)
        return obj

    else:
        return obj


def load_and_replace(yaml_path: str, output_path: str):
    """Load .env and replace placeholders in YAML."""
    load_dotenv()
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    updated_data = replace_placeholders(data)
    with open(output_path, "w") as f:
        yaml.dump(updated_data, f, sort_keys=False)
    print(f"✅ Generated: {output_path}")

if __name__ == "__main__":
    configs = [
        ("config/config1.example.yaml", "config/config1.yaml"),
        ("config/config2.example.yaml", "config/config2.yaml"),
    ]
    for src, dst in configs:
        load_and_replace(src, dst)
