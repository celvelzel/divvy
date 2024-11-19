import pandas as pd
import os


def split_dataframe_by_columns(input_file, output_folder, num_tables):
    """
    读取一个 CSV 文件，根据列将其等分为多个表格，并将每个表格重命名后输出到指定的文件夹中。
    每个子文件保留原文件的第一列，并且用原文件名加数字的方法来命名新生成的文件。

    :param input_file: 输入的 CSV 文件路径
    :param output_folder: 输出文件夹路径
    :param num_tables: 分割成的表格数量
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 获取原文件名（不包括扩展名）
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 计算每张表的列数（不包括第一列）
    num_columns = df.shape[1] - 1
    columns_per_table = num_columns // num_tables
    if num_columns % num_tables != 0:
        columns_per_table += 1

    # 第一列的名称
    first_column_name = df.columns[0]

    # 分割 DataFrame 并保存为多个 CSV 文件
    start_column = 1
    for i in range(num_tables):
        end_column = start_column + columns_per_table
        if end_column > num_columns + 1:
            end_column = num_columns + 1

        # 保留第一列
        columns_to_keep = [first_column_name] + list(df.columns[start_column:end_column])
        table = df[columns_to_keep]

        # 生成输出文件名
        output_file = os.path.join(output_folder, f'{base_filename}_{i + 1}.csv')

        # 保存表格到 CSV 文件
        table.to_csv(output_file, index=False)
        print(f'Saved {output_file}')

        # 更新起始列索引
        start_column = end_column


if __name__ == '__main__':
    # 输入文件路径
    input_file = '../../data/aggregated_trip_counts/aggregated_trip_counts_docked.csv'

    # 输出文件夹路径
    output_folder = '../../output/aggregated_trip_counts'

    # 分割成的表格数量
    num_tables = 8

    # 调用函数
    split_dataframe_by_columns(input_file, output_folder, num_tables)