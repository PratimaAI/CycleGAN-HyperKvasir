import os

def update_gitignore():
    # Define the size threshold (in bytes)
    size_threshold = 90 * 1024 * 1024  # 80 MB

    # Traverse through the repository directory
    large_files = [f for f in os.listdir('.') if os.path.isfile(f) and os.path.getsize(f) > size_threshold]

    # Update .gitignore with large files
    with open('.gitignore', 'a') as gitignore:
        for file in large_files:
            gitignore.write(file + '\n')

if __name__ == "__main__":
    update_gitignore()
