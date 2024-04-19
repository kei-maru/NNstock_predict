import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

# 数据加载与探索
def load_and_explore_data(path):
    data = pd.read_csv(path)
    print(data.head())
    data.describe()
    data['Date'] = pd.to_datetime(data['Date'])
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'])
    # 设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # 设置日期间隔，每100天显示一次
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=5))
    plt.title('Close Price over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.show()
    return data

# 数据预处理
def preprocess_data(data):
    # 检查数据中是否有缺失值
    print(data.isnull().sum())

    # 填充缺失值
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # 将特征和目标分开，并对特征进行缩放
    features = data[['Open', 'High', 'Low', 'Volume']]
    dates = data['Date']
    target = data['Close']
    scaler = MinMaxScaler()  # 使用 StandardScaler 替代 MinMaxScaler
    scaled_features = scaler.fit_transform(features)

    # 按时间顺序拆分数据
    train_size = int(len(scaled_features) * 0.8)
    X_train = scaled_features[:train_size]
    X_test = scaled_features[train_size:]
    y_train = target[:train_size]
    y_test = target[train_size:]
    dates_train = dates[:train_size]
    dates_test = dates[train_size:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test, scaler


# 模型定义
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)
# 更简单的模型定义
class SimpleStockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleStockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 使用更少的神经元
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # 只使用一个隐藏层和输出层

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


# 模型训练
def train_model(model, X_train, y_train,name, num_epochs=2000, lr=0.001):
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) # 添加 L2 正则化
    early_stopping_counter = 0
    min_loss = np.inf
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        #scheduler.step()  # 更新学习率
        if loss.item() < min_loss:
            min_loss = loss.item()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter > 100:  # 如果100个epoch损失没有改善，则停止
                print("Early stopping triggered.")
                break
    torch.save(model.state_dict(), name)

# 模型评估

def evaluate_model_extended(modle_name,X_test, y_test, dates_test, input_dim):
    if (modle_name=='modle_complex'):
        model = StockPredictor(input_dim)
        model.load_state_dict(torch.load('modle_complex'))
    else:
        model = SimpleStockPredictor(input_dim)
        model.load_state_dict(torch.load('modle_simple'))
    model.eval()  # 将模型设置为评估模式

    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)
    criterion = nn.MSELoss()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        mse = criterion(predictions, y_test_tensor)
        dates_test = list(dates_test.sort_values())
        print(f'MSE: {mse.item():.4f}')

        mae = mean_absolute_error(y_test_tensor.numpy(), predictions.numpy())
        print(f'MAE: {mae:.4f}')

        rmse = mean_squared_error(y_test_tensor.numpy(), predictions.numpy(), squared=False)
        print(f'RMSE: {rmse:.4f}')

        # MAPE 需要避免分母为零
        mape = np.mean(np.abs((y_test_tensor.numpy() - predictions.numpy()) / y_test_tensor.numpy())) * 100
        print(f'MAPE: {mape:.4f}%')

        r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())
        print(f'R^2 Score: {r2:.4f}')

        # 可视化预测结果与实际结果
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test, y_test_tensor.numpy(), label='Actual Value', marker='.', linestyle='-', color='blue')
        plt.plot(dates_test, predictions.numpy(), label='Predicted Value', marker='.', linestyle='--', color='red')

        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

# 主函数
def main(path):
    data = load_and_explore_data(path)
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler = preprocess_data(data)
    input_dim = X_train.shape[1]  # 获取特征数量作为input_dim
    # 训练原始模型
    print("Training the original model with more neurons...")
    original_model = StockPredictor(input_dim=input_dim)
    simple_model = SimpleStockPredictor(input_dim=input_dim)
    save_root_simple = 'modle_simple'
    save_root_complex = 'modle_complex'
    #train_model(simple_model, X_train, y_train, save_root_simple)
    #train_model(original_model, X_train, y_train, save_root_complex)

    evaluate_model_extended(save_root_simple,X_test, y_test, dates_test,input_dim)
    evaluate_model_extended(save_root_complex, X_test, y_test, dates_test, input_dim)

# 运行主函数
main('stocks_price.csv')
#main('upload_DJIA_table.csv')


# https://www.heywhale.com/mw/dataset/634d1afda40ed1671d65423b/file