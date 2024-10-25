import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start = datetime.now()

# 1. 生成模拟数据
try:
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    dates = pd.date_range(start='2013-01-01', periods=572, freq='W')
    n_weeks = 572
    trend = np.linspace(100, 200, n_weeks)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = np.random.normal(0, 5, n_weeks)
    total_trips = trend + seasonality + noise

    # 创建DataFrame
    data = pd.DataFrame({'Date': dates, 'TotalTrips': total_trips})
    data.set_index('Date', inplace=True)

    logging.info("Data generation complete.")
    print(data.head())
except Exception as e:
    logging.error(f"Error generating data: {e}")
    raise

# 2. 数据预处理
try:
    # 检查是否有缺失值
    if data.isnull().sum().sum() > 0:
        logging.warning("Interpolating missing values.")
        data = data.interpolate()

    # 数据归一化
    scaler = MinMaxScaler()
    total_trips_scaled = scaler.fit_transform(data['TotalTrips'].values.reshape(-1, 1))


    # 创建训练数据集
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
            Y.append(data[i + time_step])
        return np.array(X), np.array(Y)


    time_step = 12  # 使用过去12周的数据预测下一周
    X, y = create_dataset(total_trips_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

except Exception as e:
    logging.error(f"Error in data preprocessing: {e}")
    raise


# 3. 构建 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, n_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_size, hidden_dim)  # 输入嵌入层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_embedding(x)  # 进行输入嵌入
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x


# 4. 训练模型
try:
    model = TransformerModel(input_size=1)  # 输入维度为1
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.Tensor(X)
    y_train_tensor = torch.Tensor(y)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

except Exception as e:
    logging.error(f"Error training Transformer model: {e}")
    raise

# 5. 预测未来52周的数据
try:
    model.eval()
    with torch.no_grad():
        # 使用最后的输入数据进行预测
        last_input = X_train_tensor[-1].unsqueeze(0)  # 取最后一个输入并增加一个维度
        predictions = []

        for _ in range(52):  # 预测未来52周
            next_pred = model(last_input)
            predictions.append(next_pred.item())
            # 更新输入，移除最旧的输入并添加新的预测
            last_input = torch.cat((last_input[:, 1:, :], next_pred.unsqueeze(0)), dim=1)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # 创建预测结果的 DataFrame
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(weeks=1), periods=52, freq='W')
    forecast_df = pd.DataFrame({
        'Forecast': predictions.flatten(),
    }, index=future_dates)

    # 查看预测结果
    logging.info("Forecasting complete.")
    print(forecast_df.head())
except Exception as e:
    logging.error(f"Error in forecasting: {e}")
    raise

finished = datetime.now()
print("Total time taken: ", finished - start)

# 6. 绘制预测结果
try:
    plt.figure(figsize=(12, 6))

    # 绘制原始数据
    plt.plot(data.index, data['TotalTrips'], label='Actual Total Trips')

    # 绘制预测结果
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecasted Total Trips', linestyle='--', color='red')

    plt.xlabel('Date')
    plt.ylabel('Total Trips')
    plt.title('Weekly Total Trips Forecast using Transformer')
    plt.legend()
    plt.show()
except Exception as e:
    logging.error(f"Error in plotting forecast results: {e}")
    raise
