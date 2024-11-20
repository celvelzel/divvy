import pandas as pd
import os
import glob


def combine_csv_files(input_folder, output_file):
    """
    ��ȡָ���ļ����е����� CSV �ļ��������������������һ��ȷ����һ�в��ظ���

    :param input_folder: �����ļ���·��������Ҫ��ϵ� CSV �ļ�
    :param output_file: ����ļ�·������Ϻ�� CSV �ļ�
    """
    # ��ȡ�ļ��������е� CSV �ļ�
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified folder.")

    # ��ȡ��һ���ļ��Ի�ȡ��һ��
    first_file = pd.read_csv(csv_files[0])
    first_column_name = first_file.columns[0]

    # ��ʼ��һ���յ� DataFrame ���洢��Ϻ������
    combined_df = pd.DataFrame(columns=first_file.columns)

    # �����ȡ���ϲ� CSV �ļ�
    for file in csv_files:
        df = pd.read_csv(file)
        # ֻ������һ�к�δ������������
        new_columns = [col for col in df.columns if col not in combined_df.columns or col == first_column_name]
        combined_df = pd.concat([combined_df, df[new_columns]], ignore_index=True)

    # ȥ���ظ��ĵ�һ��
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # ������Ϻ�� DataFrame ���µ� CSV �ļ�
    combined_df.to_csv(output_file, index=False)
    print(f'Saved combined file to {output_file}')


if __name__ == '__main__':
    # �����ļ���·��
    input_folder = 'path/to/your/output/folder'

    # ����ļ�·��
    output_file = 'path/to/your/combined_output.csv'

    # ���ú���
    combine_csv_files(input_folder, output_file)