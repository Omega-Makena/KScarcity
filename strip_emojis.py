import os
import emoji

def remove_emoji(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    clean_content = emoji.replace_emoji(content, replace='')
    
    # Clean up double spaces caused by emoji removal
    clean_content = clean_content.replace('  ', ' ')
    
    if content != clean_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clean_content)
        print(f"Removed emojis from {file_path}")

if __name__ == '__main__':
    ui_dir = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\ui"
    for root, dirs, files in os.walk(ui_dir):
        for file in files:
            if file.endswith('.py'):
                remove_emoji(os.path.join(root, file))
