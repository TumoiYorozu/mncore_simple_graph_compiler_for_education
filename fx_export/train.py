#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from typing import Dict, Tuple, Optional, Any

# 設定定数
ERROR_THRESHOLD: float = 1e-5
DEFAULT_BATCH_SIZE: int = 256
# テンソルを別パラメータとして渡すか埋め込むかを決定するための閾値
TENSOR_EMBED_THRESHOLD: int = 100

# PADDED_MNIST: 入力を1024次元(32x32)、出力を16次元に拡張する実験的機能
# 環境変数から設定可能
PADDED_MNIST: bool = os.environ.get('PADDED_MNIST', '1') != '0'


class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 16)  # 28x28 = 784
        self.fc2 = nn.Linear(16, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # type: ignore


class SimpleNN1024(nn.Module):
    """PADDED_MNIST用: 入力1024次元(32x32)、出力16次元のモデル"""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 16)  # 32x32 = 1024
        # self.fex = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)     # 出力を16次元に拡張
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fex(x))
        return self.fc2(x)  # type: ignore


# train_step用のグローバルなmodel、optimizer、criterion
model: Optional[nn.Module] = None  # SimpleNNまたはSimpleNN1024
optimizer: Optional[torch.optim.SGD] = None
criterion: Optional[nn.CrossEntropyLoss] = None

def init_globals() -> None:
    global model, optimizer, criterion
    if model is None:
        if PADDED_MNIST:
            model = SimpleNN1024()
        else:
            model = SimpleNN()
        # パラメータが勾配を持つように設定
        for param in model.parameters():
            param.requires_grad_(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()


# init_globals() をコメントアウト - PADDED_MNISTの値に応じて必要時に初期化される
# init_globals()

def train_step(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """訓練とエクスポートの両方で使用される訓練ステップ関数"""
    # 必要に応じてグローバル変数を初期化
    if model is None:
        init_globals()
    
    x = inputs["x"]
    t = inputs["t"]
    assert optimizer is not None, "optimizer must be initialized"
    assert model is not None, "model must be initialized"
    assert criterion is not None, "criterion must be initialized"
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, t)
    loss.backward()
    optimizer.step()
    return {"loss": loss, "output": output}

def get_example_function_and_inputs(example_name: str, batch_size: int = 256) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """
    例の名前から関数とダミー入力を取得する共通関数
    
    Args:
        example_name: 'train_step', 'inference', または 'custom'
        batch_size: バッチサイズ
    
    Returns:
        (func, dummy_inputs) - 関数とダミー入力の辞書
    """
    if example_name == 'train_step':
        # train_stepを使用する前にグローバル変数を初期化
        init_globals()
        if PADDED_MNIST:
            # 32x32の入力、ラベルは0-9のまま（16クラスではなく10クラスのまま）
            dummy_inputs = {
                "x": torch.randn(batch_size, 1, 32, 32),
                "t": torch.randint(0, 10, (batch_size,))
            }
        else:
            dummy_inputs = {
                "x": torch.randn(batch_size, 1, 28, 28),
                "t": torch.randint(0, 10, (batch_size,))
            }
        return train_step, dummy_inputs
    
    elif example_name == 'inference':
        init_globals()
        assert model is not None, "model must be initialized"
        model.eval()
        
        def inference(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            x = inputs["x"]
            with torch.no_grad():
                output = model(x)
                probs = torch.nn.functional.softmax(output, dim=1)
            return {"output": output, "probs": probs}
        
        if PADDED_MNIST:
            dummy_inputs = {
                "x": torch.randn(batch_size, 1, 32, 32)
            }
        else:
            dummy_inputs = {
                "x": torch.randn(batch_size, 1, 28, 28)
            }
        return inference, dummy_inputs
    
    elif example_name == 'custom':
        def custom_computation(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            a = inputs["a"]
            b = inputs["b"]
            c = torch.matmul(a, b)
            d = torch.nn.functional.relu(c)
            e = d.sum(dim=-1)
            return {"matmul_result": c, "relu_result": d, "sum_result": e}
        
        dummy_inputs = {
            "a": torch.randn(256, 16),
            "b": torch.randn(16, 16)
        }
        return custom_computation, dummy_inputs
    
    else:
        raise ValueError(f"Unknown example: {example_name}")

def get_mnist_transform():
    """MNISTデータセット用の変換を取得（PADDED_MNISTの設定に応じて）"""
    if PADDED_MNIST:
        # 28x28 → 32x32にパディング（上下左右に2ピクセルずつ）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Pad(2)  # 上下左右に2ピクセルずつパディング
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform

def train(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss: float = 0.0
    for data, target in train_loader:
        result = train_step({"x": data, "t": target})
        total_loss += result["loss"].item()
    return total_loss / len(train_loader)

def evaluate(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    correct: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)  # type: ignore
    return accuracy, 0.0  # 精度と dummy loss を返す

def train_model(batch_size=DEFAULT_BATCH_SIZE, test_batch_size=1000):
    """MNISTモデルを訓練する簡単な関数"""
    
    # シングルスレッドで実行するための環境変数を設定
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # PyTorchのスレッド数も1に設定
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    # データの準備
    transform = get_mnist_transform()
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    
    # グローバル変数を初期化
    init_globals()
    
    # 学習ループ
    for epoch in range(100):
        loss = train(model, train_loader, optimizer, criterion)
        acc, _ = evaluate(model, test_loader)
        print(f'Epoch {epoch+1:2d}: Loss: {loss:.4f}, Accuracy: {acc:.2f}%', flush=True)
    
    return model
