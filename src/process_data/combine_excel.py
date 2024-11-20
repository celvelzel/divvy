import pandas as pd
import os
import glob


def combine_csv_files(input_folder, output_file):
    """
    读取指定文件夹中的所有 CSV 文件，并将它们重新组合在一起，确保第一列不重复。

    :param input_folder: 输入文件夹路径，包含要组合的 CSV 文件
    :param output_file: 输出文件路径，组合后的 CSV 文件
    """
    # 获取文件夹中所有的 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified folder.")

    # 读取第一个文件以获取第一列
    first_file = pd.read_csv(csv_files[0])
    first_column_name = first_file.columns[0]

    # 初始化一个空的 DataFrame 来存储组合后的数据
    combined_df = pd.DataFrame(columns=first_file.columns)

    # 逐个读取并合并 CSV 文件
    for file in csv_files:
        df = pd.read_csv(file)
        # 只保留第一列和未见过的其他列
        new_columns = [col for col in df.columns if col not in combined_df.columns or col == first_column_name]
        combined_df = pd.concat([combined_df, df[new_columns]], ignore_index=True)

    # 去除重复的第一列
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # 保存组合后的 DataFrame 到新的 CSV 文件
    combined_df.to_csv(output_file, index=False)
    print(f'Saved combined file to {output_file}')


if __name__ == '__main__':
    # 输入文件夹路径
    input_folder = 'path/to/your/output/folder'

    # 输出文件路径
    output_file = 'path/to/your/combined_output.csv'

    # 调用函数
    combine_csv_files(input_folder, output_file)