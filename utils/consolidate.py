import os

def consolidate_chapters_to_complete_text(chapters_dir, output_file="text/complete_text.md"):
    """
    Consolidates all markdown files in the 'chapters' folder into a single file called 'complete_text.md'.

    Args:
        chapters_dir (str): Path to the 'chapters' directory.
        output_file (str): Name of the output file. Defaults to 'complete_text.md'.
    """


    # Ensure the output file is in the same directory as the chapters
    output_path = os.path.join(chapters_dir, output_file)

    # List all markdown files in the chapters directory
    markdown_files = [f for f in os.listdir(chapters_dir) if f.endswith('.md')]
    # Sort the files to ensure they are in the correct order
    markdown_files.sort()
    # Initialize an empty list to hold the contents of each file
    complete_text = []

    # Read each markdown file and append its content to the complete_text list
    for filename in markdown_files:
        with open(os.path.join(chapters_dir, filename), 'r', encoding='utf-8') as file:
            complete_text.append(file.read())

    # Write the consolidated text to the output file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write("\n\n".join(complete_text))

    print(f"Consolidated text written to {output_path}")


if __name__ == "__main__":
    # Define the path to the chapters directory
    chapters_directory = "./data/chapters"

    # Call the function to consolidate chapters
    consolidate_chapters_to_complete_text(chapters_directory)
    # You can specify a different output file name if needed