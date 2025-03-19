import os
import sys
import shutil
import time
import json
from typing import List, Dict, Any
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

def process_file(file_path: str, target_dir: str, batch_items: List[Dict[str, Any]], file_mapping: Dict[str, str]):
    """
    Processes the given file by:
    - Copying C/C++ files to the target directory.
    - Preparing non-C/C++ files for batch translation to C++.
    """
    # Get the file extension and base name
    _, file_extension = os.path.splitext(file_path)
    base_name = os.path.basename(file_path)

    # Copy C/C++ files directly
    if file_extension.lower() in ['.c', '.cpp', '.h', '.hpp', '.cc', '.cxx', '.hxx']:
        output_path = os.path.join(target_dir, base_name)
        shutil.copy(file_path, output_path)
        print(f"File '{file_path}' copied to '{output_path}'.")
        return

    # Read content for non-C/C++ files
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary file '{file_path}'.")
        return

    # Create a unique ID for this file
    file_id = f"file_{len(batch_items)}"

    # Add to batch items
    batch_items.append({
        "id": file_id,
        "messages": [
            {"role": "user", "content": f"\n\n```{file_extension[1:]}\n{content}\n```\nCan you translate this into a single C++ file? Assume all the necessary functions and classes are already implemented and can be included with `#include \"types.hpp\"`. Don't define code that does not have a direct equivalent in the original code. At the end don't explain what you did."}
        ]
    })

    # print("Messages:", batch_items[-1]["messages"])

    # Store mapping from file_id to output path
    output_filename = f"{os.path.splitext(base_name)[0]}.cpp"
    output_path = os.path.join(target_dir, output_filename)
    file_mapping[file_id] = output_path

    print(f"Added '{file_path}' to batch for conversion to '{output_path}'.")

def create_batch_request(batch_items: List[Dict[str, Any]], client):
    """
    Creates a batch request using the Anthropic API.
    Returns the batch ID.
    """
    print(f"Creating batch request with {len(batch_items)} items...")

    requests = []
    for item in batch_items:
        requests.append(
            anthropic.types.messages.batch_create_params.Request(
                custom_id=item["id"],
                params=anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                    #model="claude-3-5-haiku-20241022",
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=30000,
                    messages=item["messages"]
                )
            )
        )

    batch = client.messages.batches.create(requests=requests)
    return batch.id

def wait_for_batch_completion(batch_id: str, client, max_wait_time: int = 600, check_interval: int = 10):
    """
    Waits for the batch to complete processing.
    Returns True if the batch completed, False if it timed out.
    """
    print(f"Waiting for batch {batch_id} to complete...")
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        batch_status = client.messages.batches.retrieve(batch_id)

        if batch_status.processing_status != "in_progress":
            print(f"Batch {batch_id} processing completed (status: {batch_status.processing_status}).")
            return True

        print(f"Batch status: {batch_status.processing_status}... waiting {check_interval} seconds.")
        time.sleep(check_interval)

    print(f"Batch processing timed out after {max_wait_time} seconds.")
    return False

def process_batch_results(batch_id: str, client, file_mapping: Dict[str, str]):
    """
    Processes the results from the batch request and saves the converted files.
    """
    print(f"Processing results for batch {batch_id}...")

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        file_path = file_mapping.get(custom_id)

        if not file_path:
            print(f"Warning: Could not find mapping for item {custom_id}")
            continue

        match result.result.type:
            case "succeeded":
                print(f"Success for {custom_id} - saving to {file_path}")

                # Extract the response content
                response_content = result.result.message.content[0].text

                # Extract code blocks
                cpp_code = extract_code_blocks(response_content)
                if not cpp_code:
                    # If no code blocks found, use the entire response
                    cpp_code = response_content

                # Save the converted code
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cpp_code)

            case "errored":
                print(f"Error for {custom_id}: {result.result.error.type}")
                print(f"Full: {result}")
                print(f"Error message: {result.result.error.error.message}")

            case "expired":
                print(f"Request expired for {custom_id}")

def extract_code_blocks(text: str) -> str:
    """Extract C++ code from markdown code blocks."""
    import re

    print("Full text:", text)

    # Look for ```cpp or ``` blocks
    cpp_blocks = re.findall(r'```(?:cpp)?\n(.*?)```', text, re.DOTALL)

    if cpp_blocks:
        # Join all code blocks if multiple are found
        return '\n\n'.join(cpp_blocks)
    return ""

def main():
    """
    Main function that processes all files in the source directory using batch processing.
    """
    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_directory> <target_directory>")
        sys.exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]

    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Create target directory
    try:
        os.makedirs(target_dir, exist_ok=False)
    except Exception as e:
        print(f"Error creating target directory: {e}")
        sys.exit(1)

    # Initialize Anthropic client
    try:
        client = anthropic.Client()
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}")
        print("Make sure your ANTHROPIC_API_KEY environment variable is set.")
        sys.exit(1)

    # Prepare batch items
    batch_items = []
    file_mapping = {}

    # Process files and collect them for batch processing
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path, target_dir, batch_items, file_mapping)

    # If no files to convert, exit
    if not batch_items:
        print("No files to convert. All files were either C/C++ or binary files.")
        return

    # Create batch request
    try:
        batch_id = create_batch_request(batch_items, client)
        print(f"Batch created with ID: {batch_id}")
    except Exception as e:
        print(f"Error creating batch request: {e}")
        sys.exit(1)

    # Wait for batch processing to complete
    batch_completed = wait_for_batch_completion(batch_id, client)
    if not batch_completed:
        print("Batch processing did not complete in the allocated time.")
        sys.exit(1)

    # Process and save results
    process_batch_results(batch_id, client, file_mapping)
    print("Batch processing completed successfully.")

if __name__ == "__main__":
    main()
