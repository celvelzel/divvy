import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import re
from sklearn.preprocessing import MinMaxScaler
import logging

# ������־��¼
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ����·��
# data_path = 'data/dataset/grid_poi_counts.csv'
# ģ�ͱ���·��
model_path = '../../model'
# ģ���ļ���
model_file_name = 'rfc_model.pkl'
# ���Լ�����
test_size_ratio = 0.3
# �����0�г��������
zero_trip_ratio = 0
# �Ƿ����ʱ����Ϊ��������
add_time_feature = True
# ѵ�����Ͳ��Լ��Ļ�������
divide_date = '2020-12-28'

# �г����ݵ�Ŀ¼
trip_file_path = '../../data/finished_trips/training'
# �������ݵ�Ŀ¼
test_file_path = '../../data/finished_trips/test'
# POI���ݵ�·��
poi_file_path = '../../data/dataset/grid_poi_counts.csv'

# �������Ŀ¼
try:
    os.makedirs(model_path)
    logging.info('Ŀ¼�����ɹ�')
except FileExistsError:
    logging.info(f'Ŀ¼{model_path}�Ѿ�����')


def add_time_features(trip_file, trips):
    """
    ���ļ�������ȡ���ڣ������ʱ��������

    ����:
    trip_file (str): �ļ���
    trips (DataFrame): �г�����

    ����:
    DataFrame: �����ʱ���������г�����
    """
    try:
        date_str = trip_file.split('_')[-1].replace('.csv', '')
        matcher = re.compile(r'\d{4}-\d{2}-\d{2}')
        date_str = matcher.search(date_str).group()
        date = pd.to_datetime(date_str)

        trips['month'] = date.month  # �·�
        trips['day_of_year'] = date.dayofyear  # һ���еĵڼ���
        return trips
    except Exception as e:
        logging.error(f"�����ļ� {trip_file} ʱ��������: {e}")
        return trips


def is_exceed_date(trip_file):
    """
    ����:
    trip_file (str): �ļ���

    �жϸ��г��ļ��Ƿ񳬹��������ڣ�
    �������Ϊ���Լ������δ����Ϊѵ����
    """
    try:
        date_str = trip_file.split('_')[-1].replace('.csv', '')
        matcher = re.compile(r'\d{4}-\d{2}-\d{2}')
        date_str = matcher.search(date_str).group()
        date = pd.to_datetime(date_str)

        if date > pd.to_datetime(divide_date):
            return True
    except Exception as e:
        logging.error(f"�����ļ� {trip_file} ʱ��������: {e}")
        return False


def load_and_process_data():
    """
    ���غʹ������ݡ�

    ����:
    DataFrame: ����������
    """
    # ��ȡpoi����
    try:
        poi_data = pd.read_csv(poi_file_path)
        poi_data.drop(columns=['geometry'], inplace=True, errors='ignore')
    except Exception as e:
        logging.error(f"��ȡPOI����ʱ��������: {e}")
        return None

    # ��ÿ���г��ļ�ƴ��
    training_set = []
    testing_set = []
    trips_file_dict = []
    for trip_file in os.listdir(trip_file_path):
        if trip_file.endswith('.csv'):
            csv_path = os.path.join(trip_file_path, trip_file)
            # δ�����ָ����ڵ�Ϊѵ����
            if not is_exceed_date(trip_file):
                try:

                    trips_file_dict.append({'file_path': csv_path, 'category': 'train'})
                except Exception as e:
                    logging.error(f"��ȡ�ļ� {trip_file} ʱ��������: {e}")
            else:
                trips_file_dict.append({'file_path': csv_path, 'category': 'test'})

    # �����ֵ�
    for item in trips_file_dict:
        csv_path = item['file_path']
        trips = pd.read_csv(csv_path)
        # ���ʱ������
        if add_time_feature:
            trips = add_time_features(csv_path, trips)
        trips.drop(columns=['geometry'], inplace=True, errors='ignore')
        if item['category'] == 'test':
            testing_set.append(trips)
        else:
            training_set.append(trips)

    if not training_set:
        logging.error("û���ҵ������������ļ�")
        return None

    # ƴ����������
    training_set = pd.concat(training_set, ignore_index=True)
    testing_set = pd.concat(testing_set, ignore_index=True)
    logging.info('�г�����ƴ�����')

    # �ϲ��г����ݺ�POI����
    dataset = []
    merged_training_data = pd.merge(training_set, poi_data, on='grid_id', how='left')
    dataset.append({'category': 'train', 'data': merged_training_data})
    merged_testing_data = pd.merge(testing_set, poi_data, on='grid_id', how='left')
    dataset.append({'category': 'test', 'data': merged_testing_data})
    logging.info('�г����ݺ�POI���ݺϲ����')

    # ����ÿ�������ƽ���г���
    # avg_trip_count = merged_data.groupby('grid_id')['trip_count'].mean().reset_index()
    # avg_trip_count.columns = ['grid_id', 'avg_trip_count']
    # merged_data = pd.merge(merged_data, avg_trip_count, on='grid_id', how='left')
    # logging.info('ƽ���г�����ƴ�����')

    return dataset


def preprocess_data(data):
    """
    Ԥ�������ݡ�

    ����:
    data (DataFrame): ԭʼ����

    ����:
    DataFrame: Ԥ����������
    """
    # ɸѡ�г̲�Ϊ0������
    non_zero_trip_grid = data[data['trip_count'] > 0]

    # ɸѡ�г�Ϊ0������
    zero_trip_grid = data[data['trip_count'] == 0]

    # �����ȡһ��������0�г���������
    num_zero_samples = int(len(non_zero_trip_grid) * zero_trip_ratio)
    sampled_zero_trips = zero_trip_grid.sample(n=num_zero_samples, random_state=0)

    # �ϲ�����
    combined_data = pd.concat([non_zero_trip_grid, sampled_zero_trips])

    # ɾ������Ҫ����
    combined_data.drop(columns=['grid_id'], inplace=True, errors='ignore')

    # ��ȡ����������Ŀ�����
    # x = combined_data.drop(['trip_count', 'avg_trip_count'], axis=1)  # ��������
    x = combined_data.drop(['trip_count'], axis=1)  # ��������
    y = combined_data['trip_count']  # Ŀ�����

    # �������ƽ��ֵ
    logging.info('���ݼ�trip_countƽ��ֵ:')
    logging.info(y.mean())

    return x, y


def train_and_evaluate_model(x, y):
    """
    ѵ��������ģ�͡�

    ����:
    x (DataFrame): ��������
    y (Series): Ŀ�����
    """
    # ����ѵ�����Ͳ��Լ�
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_ratio, random_state=0)

    # ��һ��
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # ѵ��ģ��
    model = RandomForestRegressor()  # ʹ��Ĭ�����ɭ�ֻع���
    model.fit(x_train_scaled, y_train)  # ʹ��ѵ�����ݼ�ѵ�����ɭ��ģ��

    # ����ģ��
    y_pred = model.predict(x_test_scaled)
    importances = model.feature_importances_  # ����������Ҫ��
    importances_percent = importances * 100

    # ��ӡ������Ҫ��
    feature_names = x.columns.tolist()
    logging.info("������Ҫ�� (%):")
    for feature, importance in zip(feature_names, importances_percent):
        logging.info(f"{feature}: {importance:.2f}%")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"������� (MSE): {mse:.4f}")
    logging.info(f"��������� (RMSE): {rmse:.4f}")
    logging.info(f"ƽ��������� (MAE): {mae:.4f}")
    logging.info(f"����ϵ�� (R?): {r2:.4f}")

    # ����ѵ���õ�ģ��
    final_path = os.path.join(model_path, model_file_name)
    joblib.dump(model, final_path)
    logging.info("ģ���ѱ���")


def main():
    """
    ��������ִ���������̡�
    """
    # ���غʹ�������
    raw_data = load_and_process_data()
    if raw_data is None:
        logging.error("���ݼ���ʧ��")
        return

    # Ԥ��������
    x, y = preprocess_data(raw_data)

    # ѵ��������ģ��
    train_and_evaluate_model(x, y)

    print("ѵ������")


if __name__ == "__main__":
    main()
