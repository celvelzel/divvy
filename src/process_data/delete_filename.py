import os

# 指定要处理的文件夹路径
folder_path = 'C:\\Users\\celcelcel\\Desktop\\test\\批量改格式_20240402140005'

# 遍历文件夹中的所有文件和子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 检查文件名是否包含"_转自CSV"
        if "_转自CSV" in file:
            # 获取原始文件名（不包括路径）
            original_filename = os.path.basename(file)

            # 分离出文件名和扩展名
            filename, extension = os.path.splitext(original_filename)

            # 删除"_转自CSV"部分并重新组合文件名
            new_filename = filename.replace("_转自CSV", "") + extension

            # 获取完整的旧文件路径和新的文件路径
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)

print("已完成文件名的批量修改。")
