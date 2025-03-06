# Run this to find where CrewAI is installed and examine its structure
import crewai
import inspect
import os

# Get the file location of the crewai module
crewai_location = inspect.getfile(crewai)
crewai_package_dir = os.path.dirname(crewai_location)

print(f"CrewAI is installed at: {crewai_package_dir}")

# Let's see what files might handle LLM integration
print("\nListing key files in the package:")
for root, dirs, files in os.walk(crewai_package_dir):
    for file in files:
        if file.endswith('.py') and ('llm' in file.lower() or 'language' in file.lower()):
            rel_path = os.path.relpath(os.path.join(root, file), crewai_package_dir)
            print(f" - {rel_path}")

# To examine a specific file's content (after identifying relevant files):
def examine_file(filename):
    full_path = os.path.join(crewai_package_dir, filename)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()
        print(f"\nContents of {filename}:")
        print(content)
    else:
        print(f"File not found: {filename}")

# After running the above, you can examine specific files like:
# examine_file('llms.py')  # or whatever file is discovered to be relevant

import crewai
import os

crewai_package_dir = "/Volumes/Samsung/MO/crewai/text_processor_api/text_processor_api_env/lib/python3.10/site-packages/crewai"

# First, let's examine the main LLM module
llm_path = os.path.join(crewai_package_dir, "llm.py")
with open(llm_path, 'r') as f:
    llm_content = f.read()
print("Contents of llm.py:")
print("=" * 50)
print(llm_content)

# Next, let's look at the LLM utilities module
llm_utils_path = os.path.join(crewai_package_dir, "utilities/llm_utils.py")
with open(llm_utils_path, 'r') as f:
    llm_utils_content = f.read()
print("\nContents of utilities/llm_utils.py:")
print("=" * 50)
print(llm_utils_content)