import os
import re
import time
from deep_translator import GoogleTranslator

# 支持的图片文件扩展名
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg']

# 指定图片目录和Markdown目录
IMAGE_DIRECTORY = './images/'        # 替换为您的图片目录路径
MARKDOWN_DIRECTORY = './'  # 替换为您的Markdown目录路径

# 检查文件名是否包含中文字符
def contains_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

# 使用deep_translator翻译中文文本
def translate_text(text):
    translator = GoogleTranslator(source='zh-CN', target='en')
    try:
        translated_text = translator.translate(text)
        # 去除特殊字符，只保留字母、数字和下划线
        translated_text = re.sub(r'\W+', '_', translated_text)
        return translated_text.lower()
    except Exception as e:
        print(f"翻译出错：{e}")
        return text

# 获取新的文件名，避免重复
def get_new_filename(original_name, extension, existing_names):
    base_name = original_name
    new_name = base_name
    counter = 1
    while new_name + extension in existing_names:
        new_name = f"{base_name}_{counter}"
        counter += 1
    return new_name + extension

# 主函数
def main():
    # 记录已重命名的文件映射：原始文件路径 -> 新文件路径
    filename_mapping = {}

    # 第一步：遍历图片目录，找到中文命名的图片文件并重命名
    for root, dirs, files in os.walk(IMAGE_DIRECTORY):
        existing_files = set(files)
        for filename in files:
            name, extension = os.path.splitext(filename)
            if extension.lower() in IMAGE_EXTENSIONS and contains_chinese(name):
                # 翻译文件名
                translated_name = translate_text(name)
                # 获取新的文件名，确保不重复
                new_filename = get_new_filename(translated_name, extension, existing_files)
                # 重命名文件
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"重命名：{old_file_path} -> {new_file_path}")
                # 更新映射关系（使用相对路径）
                relative_old_path = os.path.relpath(old_file_path, start=MARKDOWN_DIRECTORY)
                relative_new_path = os.path.relpath(new_file_path, start=MARKDOWN_DIRECTORY)
                filename_mapping[relative_old_path] = relative_new_path
                # 更新现有文件列表
                existing_files.remove(filename)
                existing_files.add(new_filename)
                # 避免过快处理
                time.sleep(0.5)

    # 第二步：更新Markdown文件中的图片引用
    md_extensions = ['.md', '.markdown']
    for root, dirs, files in os.walk(MARKDOWN_DIRECTORY):
        for filename in files:
            name, extension = os.path.splitext(filename)
            if extension.lower() in md_extensions:
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 替换图片文件名
                updated = False
                for original_path, new_path in filename_mapping.items():
                    if original_path in content:
                        content = content.replace(original_path, new_path)
                        updated = True
                if updated:
                    # 将更新后的内容写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"已更新Markdown文件：{file_path}")

if __name__ == "__main__":
    main()