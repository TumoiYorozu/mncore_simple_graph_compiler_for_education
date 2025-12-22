# 検証とテストのためのMakefile
.PHONY: all test_all quick test_et test_inference test_custom test_loss test_mncpp \
	test_train test_traincpp test_unit test_unit_mncpp mn_unittest \
	mypy clean help

# デフォルトターゲット
all: test_all

# すべてのテストを実行
test_all: test_et test_inference test_custom test_loss test_mncpp test_train test_traincpp test_unit mn_unittest test_unit_mncpp mypy
	@echo "======================================"
	@echo "✅ すべてのテストが完了しました"
	@echo "======================================"

# クイックテスト（最速、推奨）
quick: test_et test_inference test_custom test_loss test_mncpp
	@echo "======================================"
	@echo "✅ クイックテスト完了"
	@echo "======================================"

# train_step のエクスポート・コンパイル・検証
test_et:
	@echo "======================================"
	@echo "Testing train_step (default)..."
	@echo "======================================"
	./haribote_graph_compiler.py et

# inference のエクスポート・コンパイル・検証
test_inference:
	@echo "======================================"
	@echo "Testing inference..."
	@echo "======================================"
	./haribote_graph_compiler.py et --example inference

# custom のエクスポート・コンパイル・検証
test_custom:
	@echo "======================================"
	@echo "Testing custom..."
	@echo "======================================"
	./haribote_graph_compiler.py et --example custom

# loss を無視したテスト
test_loss:
	@echo "======================================"
	@echo "Testing with --ignore=loss..."
	@echo "======================================"
	./haribote_graph_compiler.py et --ignore=loss

# mncpp でのエクスポート・コンパイル・検証
test_mncpp:
	@echo "======================================"
	@echo "Testing train_step with mncpp..."
	@echo "======================================"
	./haribote_graph_compiler.py et --mncpp --ignore=loss

# PyTorchでの訓練動作確認（2エポック）
test_train:
	@echo "======================================"
	@echo "Testing PyTorch training (2 epochs)..."
	@echo "======================================"
	@timeout 12 ./haribote_graph_compiler.py train 2>&1 | head -n2
	@echo "Expected: Epoch 1 ~85%, Epoch 2 ~88%"

# C++での訓練動作確認
test_traincpp:
	@echo "======================================"
	@echo "Testing C++ training..."
	@echo "======================================"
	@timeout 15 ./haribote_graph_compiler.py traincpp 2>&1 | head -n29
	@echo "Expected: Epoch 1 ~85%, Epoch 2 ~88%"

# ユニットテストの生成と実行
test_unit:
	@echo "======================================"
	@echo "Building and running unit tests..."
	@echo "======================================"
	./haribote_graph_compiler.py e
	./haribote_graph_compiler.py build_unit_tests /tmp/train_step --extra
	./haribote_graph_compiler.py test unit_tests/train_step/*
	
# MN-Core用ユニットテストの生成
build_mn_unittest:
	@echo "======================================"
	@echo "Building MN-Core unit tests..."
	@echo "======================================"
	./haribote_graph_compiler.py e --ignore=loss
	./haribote_graph_compiler.py build_unit_tests /tmp/train_step --mntest --extra

# MN-Core用ユニットテストの生成と実行
mn_unittest:
	@echo "======================================"
	@echo "Building MN-Core unit tests..."
	@echo "======================================"
	./haribote_graph_compiler.py e --ignore=loss
	./haribote_graph_compiler.py build_unit_tests /tmp/train_step --mntest --extra
	./haribote_graph_compiler.py test unit_tests/train_step/*

# MN-Core C++用ユニットテスト
test_unit_mncpp:
	@echo "======================================"
	@echo "Building and running mncpp unit tests..."
	@echo "======================================"
	./haribote_graph_compiler.py e
	./haribote_graph_compiler.py build_unit_tests /tmp/train_step --extra --mncpp
	./haribote_graph_compiler.py test unit_tests/train_step/*

# mypy型チェック
mypy:
	@echo "======================================"
	@echo "Running mypy type checking..."
	@echo "======================================"
	mypy *.py fx_export/

# 生成されたファイルのクリーンアップ
clean:
	@echo "======================================"
	@echo "Cleaning up generated files..."
	@echo "======================================"
	@rm -rf /tmp/train_step /tmp/inference /tmp/custom /tmp/export
	@rm -rf unit_tests
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.so" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

# ヘルプ
help:
	@echo "======================================"
	@echo "使用可能なターゲット:"
	@echo "======================================"
	@echo "  make test_all       - すべてのテストとmypyを実行"
	@echo "  make quick          - クイックテスト（基本的な5つのetテスト）"
	@echo "  make test_et        - train_stepのテスト"
	@echo "  make test_inference - inferenceのテスト"
	@echo "  make test_custom    - customのテスト"
	@echo "  make test_loss      - --ignore=lossのテスト"
	@echo "  make test_mncpp     - mncppでのテスト"
	@echo "  make test_train     - PyTorchでの訓練確認"
	@echo "  make test_traincpp  - C++での訓練確認"
	@echo "  make test_unit      - ユニットテスト生成と実行"
	@echo "  make test_unit_mncpp - mncpp用ユニットテスト"
	@echo "  make mn_unittest   - MN-Core用ユニットテスト生成と実行"
	@echo "  make mypy           - 型チェック"
	@echo "  make clean          - 生成ファイルの削除"
	@echo "  make help           - このヘルプを表示"