import ast
import os


def find_set_verbose_assignments(file_path):
    """
    Finds all assignments of dheera_ai.set_verbose = True in a given Python file.
    Returns a list of tuples (line_number, assignment_text).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Error parsing file {file_path}: {e}")
        return []

    assignments = []
    content_lines = content.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Check if this is an assignment to dheera_ai.set_verbose
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    # Check if it's dheera_ai.set_verbose
                    if (isinstance(target.value, ast.Name) and 
                        target.value.id == "dheera_ai" and 
                        target.attr == "set_verbose"):
                        
                        # Check if the value being assigned is True
                        if (isinstance(node.value, ast.Constant) and 
                            node.value.value is True):
                            line_num = node.lineno
                            line_text = content_lines[line_num - 1].strip() if line_num <= len(content_lines) else ""
                            assignments.append((line_num, line_text))
                        elif (isinstance(node.value, ast.NameConstant) and 
                              node.value.value is True):  # For older Python versions
                            line_num = node.lineno
                            line_text = content_lines[line_num - 1].strip() if line_num <= len(content_lines) else ""
                            assignments.append((line_num, line_text))

    return assignments


def scan_dheera_ai_files(base_dir):
    """
    Scans all Python files in the dheera_ai directory for set_verbose assignments.
    Returns a dictionary mapping file paths to lists of assignments.
    """
    violations = {}
    dheera_ai_dirs = [
        "dheera_ai",
        "enterprise"
    ]

    for dheera_ai_dir in dheera_ai_dirs:
        dir_path = os.path.join(base_dir, dheera_ai_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        print(f"Scanning directory: {dir_path}")
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, base_dir)
                    
                    assignments = find_set_verbose_assignments(file_path)
                    if assignments:
                        violations[relative_path] = assignments

    return violations


def test_no_hardcoded_set_verbose():
    """
    Pytest-compatible test function that ensures no hardcoded dheera_ai.set_verbose = True assignments exist.
    """
    base_dir = "./"  # Adjust path as needed for your setup
    
    violations = scan_dheera_ai_files(base_dir)
    
    if violations:
        violation_details = []
        total_violations = 0
        for file_path, assignments in violations.items():
            for line_num, line_text in assignments:
                violation_details.append(f"{file_path}:{line_num} -> {line_text}")
                total_violations += 1
        
        error_msg = (
            f"Found {total_violations} prohibited dheera_ai.set_verbose = True assignments:\n"
            + "\n".join(violation_details) + 
            "\n\nREASON: dheera_ai.set_verbose = True should not be hardcoded in production code. "
            "Instead, use environment variables or configuration files to control verbosity."
        )
        
        raise AssertionError(error_msg)


def main():
    """
    Main function that scans for dheera_ai.set_verbose = True assignments and fails if any are found.
    """
    base_dir = "./"  # Adjust path as needed for your setup
    
    print("Scanning for dheera_ai.set_verbose = True assignments...")
    violations = scan_dheera_ai_files(base_dir)
    
    if violations:
        print("\n❌ FOUND PROHIBITED dheera_ai.set_verbose = True ASSIGNMENTS:")
        print("=" * 60)
        
        total_violations = 0
        for file_path, assignments in violations.items():
            print(f"\nFile: {file_path}")
            for line_num, line_text in assignments:
                print(f"  Line {line_num}: {line_text}")
                total_violations += 1
        
        print(f"\n📊 Total violations found: {total_violations}")
        print("\n🚫 REASON: dheera_ai.set_verbose = True should not be hardcoded in production code.")
        print("   Instead, use environment variables or configuration files to control verbosity.")
        print("   Example alternatives:")
        print("   - Use DHEERA_AI_LOG=DEBUG environment variable")
        print("   - Use dheera_ai.set_verbose = os.getenv('DHEERA_AI_VERBOSE', 'false').lower() == 'true'")
        print("   - Use configuration-based verbosity settings")
        
        raise Exception(
            f"Found {total_violations} prohibited dheera_ai.set_verbose = True assignments. "
            "Remove these hardcoded verbosity settings and use configuration-based approaches instead."
        )
    else:
        print("✅ No prohibited dheera_ai.set_verbose = True assignments found.")


if __name__ == "__main__":
    main() 