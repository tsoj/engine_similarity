#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

def remove_comments(content: str) -> str:
    """
    Removes all comments from the provided C++ source code string.
    """
    # Remove single-line comments (// ...)
    content = re.sub(r"//.*", "", content)
    # Remove multi-line comments (/* ... */)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return content

def apply_clang_format(src_file: Path, target_file: Path) -> None:
    """
    Use clang-format to format the source file and write to target file.
    """
    try:
        # Read the source file content
        with open(src_file, "r") as f:
            original_content = f.read()

        # Remove comments from the content
        uncommented_content = remove_comments(original_content)

        # Define custom style: LLVM with no column limit
        custom_style = "{BasedOnStyle: LLVM, ColumnLimit: 160}"

        # Run clang-format on the uncommented content with the custom style
        result = subprocess.run(
            ["clang-format", "-style", custom_style],
            input=uncommented_content,
            capture_output=True,
            text=True,
            check=True
        )
        formatted_code = result.stdout
    except subprocess.CalledProcessError as err:
        print(f"Error formatting {src_file}: {err.stderr}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Exception while formatting {src_file}: {e}", file=sys.stderr)
        return

    # Write formatted code to the target file
    try:
        with open(target_file, "w") as f:
            f.write(formatted_code)
        print(f"Formatted '{src_file.name}' into '{target_file}'")
    except Exception as e:
        print(f"Error writing file {target_file}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Apply clang-format to C++ source files in a directory after removing comments."
    )
    parser.add_argument(
        "src_dir",
        help="Path to the source directory containing C++ files."
    )
    parser.add_argument(
        "target_dir",
        help="Path to the target directory where formatted files will be saved."
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    target_dir = Path(args.target_dir)

    # Ensure the source directory exists and is a directory.
    if not src_dir.is_dir():
        print(f"Error: Source directory '{src_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Create the target directory if it does not exist.
    target_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each file in the source directory.
    for src_file in src_dir.iterdir():
        if src_file.is_file() and src_file.suffix in {".cpp", ".c", ".cc", ".cxx", ".h", ".hpp", ".hxx"}:
            target_file = target_dir / src_file.name
            apply_clang_format(src_file, target_file)

if __name__ == "__main__":
    main()
