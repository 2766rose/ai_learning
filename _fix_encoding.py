import os

def fix_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        raw.decode('utf-8')
        return False
    except UnicodeDecodeError:
        pass

    for enc in ['gb18030', 'gbk', 'gb2312', 'latin-1']:
        try:
            text = raw.decode(enc)
            if '\ufffd' not in text and '?' not in text[:200]:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(text)
                print(f"[FIXED] ({enc}): {file_path}")
                return True
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue

    print(f"[FAILED]: {file_path}")
    return False

count = 0
for root, dirs, files in os.walk(r'D:\ai_learning\src'):
    for fname in files:
        if fname.endswith('.py'):
            fpath = os.path.join(root, fname)
            if fix_encoding(fpath):
                count += 1

print(f"\nTotal fixed: {count} files")
