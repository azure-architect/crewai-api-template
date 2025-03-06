

import argparse
import importlib
import inspect
import os
import pkgutil
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


def get_installed_package_location(package_name: str) -> Optional[str]:
    """Find the installation location of a pip package."""
    try:
        package = importlib.import_module(package_name)
        package_path = os.path.dirname(inspect.getfile(package))
        return package_path
    except (ImportError, AttributeError) as e:
        print(f"Error: Package '{package_name}' not found or not properly installed. {e}")
        return None


def list_package_modules(package_name: str, package_path: str) -> List[str]:
    """List all the modules within a package."""
    modules = []
    
    try:
        package = importlib.import_module(package_name)
        
        # First, add the base package
        modules.append(package_name)
        
        # Then find all submodules recursively
        for _, name, is_pkg in pkgutil.walk_packages([package_path], f"{package_name}."):
            modules.append(name)
            
    except (ImportError, AttributeError) as e:
        print(f"Error exploring package modules: {e}")
    
    return modules


def get_package_files(package_path: str, max_files: int = 20, focus_modules: List[str] = None) -> List[Tuple[str, str]]:
    """
    Get key Python files from the package directory.
    Returns a list of (file_path, relative_path) tuples.
    
    Args:
        package_path: Path to the package directory
        max_files: Maximum number of files to return
        focus_modules: Optional list of specific modules to prioritize (e.g., ['llm', 'utilities/llm_utils'])
    """
    python_files = []
    excluded_dirs = {'__pycache__', 'tests', 'examples', 'docs', 'test'}
    
    # Get the parent directory to calculate relative paths
    package_parent = os.path.dirname(package_path)
    
    # First, handle specific focus modules if provided
    if focus_modules and len(focus_modules) > 0:
        focused_files = []
        for module in focus_modules:
            # Try direct file match
            file_path = os.path.join(package_path, f"{module}.py")
            if os.path.isfile(file_path):
                rel_path = os.path.relpath(file_path, package_parent)
                focused_files.append((file_path, rel_path))
            
            # Try as directory with __init__.py
            dir_path = os.path.join(package_path, module)
            init_path = os.path.join(dir_path, "__init__.py")
            if os.path.isfile(init_path):
                rel_path = os.path.relpath(init_path, package_parent)
                focused_files.append((init_path, rel_path))
            
            # Try with path separators
            if '/' in module or '\\' in module:
                file_path = os.path.join(package_path, module.replace('/', os.sep).replace('\\', os.sep) + '.py')
                if os.path.isfile(file_path):
                    rel_path = os.path.relpath(file_path, package_parent)
                    focused_files.append((file_path, rel_path))
        
        # If we found specific files, return them first
        python_files.extend(focused_files)
        
        # If we have enough files from focused modules, return early
        if len(python_files) >= max_files:
            return python_files[:max_files]
    
    # Walk through the package directory for remaining files
    for root, dirs, files in os.walk(package_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        # Add Python files
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, package_parent)
                
                # Skip if we already have this file from focus_modules
                if any(existing_path == file_path for existing_path, _ in python_files):
                    continue
                
                python_files.append((file_path, rel_path))
                
                # Respect max_files limit
                if len(python_files) >= max_files:
                    return python_files
    
    return python_files


def read_file_content(file_path: str) -> Optional[str]:
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def extract_key_classes_functions(content: str) -> Dict[str, List[str]]:
    """
    Extract class and function definitions from Python content.
    Returns a dictionary with 'classes' and 'functions' keys.
    """
    lines = content.split('\n')
    classes = []
    functions = []
    current_block = []
    in_class = False
    in_function = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check for class definition
        if stripped.startswith('class ') and not in_class and not in_function:
            in_class = True
            current_block = [line]
        
        # Check for function definition outside of a class
        elif stripped.startswith('def ') and not in_class and not in_function:
            in_function = True
            current_block = [line]
        
        # Add line to current block
        elif (in_class or in_function) and current_block:
            current_block.append(line)
            
            # Check for end of block (unindented line)
            if (i < len(lines) - 1 and 
                lines[i+1].strip() and 
                not lines[i+1].startswith(' ') and 
                not lines[i+1].startswith('\t')):
                
                block_content = '\n'.join(current_block)
                if in_class:
                    classes.append(block_content)
                    in_class = False
                else:
                    functions.append(block_content)
                    in_function = False
                current_block = []
    
    # Handle last block
    if current_block:
        block_content = '\n'.join(current_block)
        if in_class:
            classes.append(block_content)
        elif in_function:
            functions.append(block_content)
            
    return {
        'classes': classes,
        'functions': functions
    }


def analyze_imports(files_content: Dict[str, str]) -> Set[str]:
    """Analyze imports across all files to identify dependencies."""
    imports = set()
    
    for content in files_content.values():
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Extract the main package being imported
                parts = stripped.split()
                if stripped.startswith('import '):
                    pkg = parts[1].split('.')[0]
                else:  # from ... import ...
                    pkg = parts[1].split('.')[0]
                
                # Exclude standard library modules
                if not is_stdlib_module(pkg):
                    imports.add(pkg)
    
    return imports


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is part of the Python standard library."""
    stdlib_modules = sys.stdlib_module_names
    return module_name in stdlib_modules


def generate_report(
    package_name: str, 
    package_path: str, 
    files_data: List[Tuple[str, str, str, Dict[str, List[str]]]], 
    dependencies: Set[str]
) -> str:
    """
    Generate a comprehensive report about the package.
    
    For CrewAI and Gemini integration specifically, this will focus on:
    - How LLM class works with different model providers
    - Environment variable configuration
    - Model name formatting requirements
    """
    report = []
    
    # Package information
    report.append(f"# Package Analysis: {package_name}")
    report.append(f"\nInstallation path: {package_path}")
    
    # Dependencies
    report.append("\n## External Dependencies")
    if dependencies:
        report.append("\nThis package depends on:")
        for dep in sorted(dependencies):
            report.append(f"- {dep}")
    else:
        report.append("\nNo external dependencies found.")
    
    # Files analysis
    report.append("\n## Key Files Analysis")
    
    for file_path, rel_path, _, code_blocks in files_data:
        report.append(f"\n### {rel_path}")
        
        # Classes
        if code_blocks['classes']:
            report.append("\n#### Classes:")
            for i, class_def in enumerate(code_blocks['classes']):
                # Extract just the class definition line
                class_header = class_def.split('\n')[0]
                report.append(f"{i+1}. `{class_header}`")
        
        # Functions
        if code_blocks['functions']:
            report.append("\n#### Functions:")
            for i, func_def in enumerate(code_blocks['functions']):
                # Extract just the function definition line
                func_header = func_def.split('\n')[0]
                report.append(f"{i+1}. `{func_header}`")
    
    # Detailed code section
    report.append("\n## Detailed Source Code")
    
    for file_path, rel_path, content, _ in files_data:
        report.append(f"\n### {rel_path}")
        report.append("```python")
        report.append(content)
        report.append("```")
    
    return '\n'.join(report)


def explore_package(package_name: str, max_files: int = 20, output_file: Optional[str] = None, focus_modules: List[str] = None) -> str:
    """
    Main function to explore a package and generate a report.
    
    Args:
        package_name: Name of the pip package to analyze
        max_files: Maximum number of files to analyze (default: 20)
        output_file: Optional file path to save the report
        focus_modules: Optional list of specific modules to focus on
        
    Returns:
        A string containing the analysis report
    """
    # Find package location
    package_path = get_installed_package_location(package_name)
    if not package_path:
        return f"Error: Could not find package '{package_name}'"
    
    print(f"Analyzing package: {package_name}")
    print(f"Package location: {package_path}")
    
    # Get package files with optional focus on specific modules
    files = get_package_files(package_path, max_files, focus_modules)
    
    if not files:
        return f"Error: Could not find Python files in package '{package_name}'"
    
    print(f"Found {len(files)} Python files")
    
    # Read file contents
    files_data = []
    files_content = {}
    
    for file_path, rel_path in files:
        content = read_file_content(file_path)
        if content:
            code_blocks = extract_key_classes_functions(content)
            files_data.append((file_path, rel_path, content, code_blocks))
            files_content[rel_path] = content
    
    # Analyze dependencies
    dependencies = analyze_imports(files_content)
    
    # Generate report
    report = generate_report(package_name, package_path, files_data, dependencies)
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    return report


def main():
    """Parse command-line arguments and run the package explorer."""
    parser = argparse.ArgumentParser(description="Analyze a pip package and extract its structure and source code.")
    parser.add_argument("--package", "-p", required=True, help="Name of the pip package to analyze")
    parser.add_argument("--max_files", "-m", type=int, default=20, help="Maximum number of files to analyze")
    parser.add_argument("--output", "-o", help="Output file path (if not specified, prints to stdout)")
    parser.add_argument("--focus", "-f", nargs='+', help="Focus on specific modules (e.g., 'llm' 'utilities/llm_utils')")
    parser.add_argument("--search", "-s", help="Search term in files (case insensitive)")
    
    args = parser.parse_args()
    
    # For the specific case of CrewAI + Gemini integration
    if args.package == "crewai" and not args.focus:
        print("CrewAI detected - focusing on LLM integration modules by default")
        focus_modules = ["llm", "utilities/llm_utils", "utilities/exceptions/context_window_exceeding_exception", "cli/constants"]
    else:
        focus_modules = args.focus
    
    report = explore_package(args.package, args.max_files, args.output, focus_modules)
    
    # If search term provided, add highlighting to the report
    if args.search and args.search.strip():
        search_term = args.search.lower()
        lines = report.split('\n')
        highlighted_lines = []
        
        for line in lines:
            if search_term in line.lower():
                # Add highlighting for terminal output (not saved to file)
                highlighted_lines.append(">>> " + line)
            else:
                highlighted_lines.append(line)
        
        report = '\n'.join(highlighted_lines)
    
    if not args.output:
        print("\n" + "=" * 80)
        print(report)


if __name__ == "__main__":
    main()