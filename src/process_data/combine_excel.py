import pandas as pd
import os
import glob


def combine_csv_files(input_folder, output_file):
    """
    合并指定文件夹中的所有 CSV 文件，首列是相同的日期列，保留首列并将其他列合并到一个文件中。

    :param input_folder: 输入文件夹路径，包含要合并的 CSV 文件
    :param output_file: 输出文件路径，合并后的 CSV 文件
    """
    # 获取文件夹中所有的 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified folder.")

    # 初始化一个空的 DataFrame 来存储合并后的数据
    combined_df = None

    # 逐个读取并合并 CSV 文件
    for file in csv_files:
        df = pd.read_csv(file)

        # 过滤掉列名中包含 "Upper Bound" 和 "Lower Bound" 的列
        columns_to_keep = [col for col in df.columns if "Upper Bound" not in col and "Lower Bound" not in col]
        df = df[columns_to_keep]

        # 如果是第一个文件，直接将其作为合并的基础
        if combined_df is None:
            combined_df = df
        else:
            # 确保首列（日期列）相同
            if not combined_df.iloc[:, 0].equals(df.iloc[:, 0]):
                raise ValueError("The first column (date column) of the files do not match.")

            # 将当前文件的其他列合并到 combined_df 中
            # for col in df.columns[1:]:
            #     # 如果列名重复，则重命名
            #     if col in combined_df.columns:
            #         new_col_name = f"{col}_{os.path.basename(file).split('.')[0]}"
            #         combined_df[new_col_name] = df[col]
            #     else:
            #         combined_df[col] = df[col]

            # 将当前文件的其他列合并到 combined_df 中
            # 使用 pd.concat 一次性合并所有列
            combined_df = pd.concat([combined_df, df.iloc[:, 1:]], axis=1)

    # 保存合并后的 DataFrame 到新的 CSV 文件
    combined_df.to_csv(output_file, index=False)
    print(f'Saved combined file to {output_file}')


if __name__ == '__main__':
    # 输入文件夹路径
    input_folder = '../../output/prediction_result/docked'

    # 输出文件路径
    output_file = '../../output/final_result/final_result_docked.csv'

    # 调用函数
    combine_csv_files(input_folder, output_file)