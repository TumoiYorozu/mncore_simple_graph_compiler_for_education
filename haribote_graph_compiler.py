#!/usr/bin/env python3

import torch
import argparse
import shutil
import os
import sys

# モジュール化されたインポート
from fx_export.train import (
    get_example_function_and_inputs, train_model, DEFAULT_BATCH_SIZE
)
from fx_export.export import export_function
from fx_export.compile import generate_code_from_onnx
from fx_export.test import check_cpp_outputs, run_multiple_tests, parse_cpp_output_info
from fx_export.build_unit_tests import build_unit_tests
from fx_export.mncore_transform import transform_to_mncore
from fx_export.traincpp import train_model_cpp


def main():
    parser = argparse.ArgumentParser(description='MNIST訓練とエクスポートツール')
    
    # グローバル引数
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size (デフォルト: {DEFAULT_BATCH_SIZE})')
    
    # 異なるモード用のサブパーサーを作成
    subparsers = parser.add_subparsers(dest='mode', help='実行モード')
    
    # ショートカットコマンドを追加  
    e_parser = subparsers.add_parser('e', help='exportのショートカット')
    e_parser.add_argument('output_dir', nargs='?', default=None,
                          help='出力ディレクトリ (省略時: /tmp/{example}/)')
    e_parser.add_argument('--example', choices=['train_step', 'inference', 'custom'],
                          default='train_step',
                          help='エクスポートする関数の例 (デフォルト: train_step)')
    e_parser.add_argument('--ignore', nargs='*', default=[],
                          help='エクスポートから除外する出力名 (例: --ignore loss)')
    e_parser.add_argument('--no-decompose', action='store_true',
                          help='複雑なオペレーターの分解を無効化')
    e_parser.add_argument('--mncpp', action='store_true',
                          help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    
    c_parser = subparsers.add_parser('c', help='compileのショートカット')
    c_parser.add_argument('onnx_file', help='ONNXファイルまたはディレクトリ')
    c_parser.add_argument('--mncpp', action='store_true',
                         help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    
    ec_parser = subparsers.add_parser('et', help='export + compile + checkのショートカット')
    ec_parser.add_argument('--example', choices=['train_step', 'inference', 'custom'],
                           default='train_step',
                           help='エクスポートする関数の例 (デフォルト: train_step)')
    ec_parser.add_argument('--ignore', nargs='*', default=[],
                           help='エクスポートから除外する出力名 (例: --ignore loss)')
    ec_parser.add_argument('--no-decompose', action='store_true',
                           help='複雑なオペレーターの分解を無効化')
    ec_parser.add_argument('--mncpp', action='store_true',
                           help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    
    # 訓練モード
    train_parser = subparsers.add_parser('train', help='モデルを訓練')
    train_parser.add_argument('--test-batch-size', type=int, default=1000,
                             help='Test batch size (デフォルト: 1000)')
    
    # C++訓練モード
    train_cpp_parser = subparsers.add_parser('train_cpp', help='C++でコンパイルしたtrain_stepを使ってモデルを訓練')
    train_cpp_parser.add_argument('--batch-size', type=int, default=256,
                                  help='Batch size (デフォルト: 256)')
    train_cpp_parser.add_argument('--test-batch-size', type=int, default=1000,
                                  help='Test batch size (デフォルト: 1000)')
    
    # C++訓練モード（lossなし）
    train_cpp_noloss_parser = subparsers.add_parser('train_cpp_noloss', help='C++でコンパイルしたtrain_step（lossなし）を使ってモデルを訓練')
    train_cpp_noloss_parser.add_argument('--batch-size', type=int, default=256,
                                         help='Batch size (デフォルト: 256)')
    train_cpp_noloss_parser.add_argument('--test-batch-size', type=int, default=1000,
                                         help='Test batch size (デフォルト: 1000)')
    
    # エミュレータ訓練モード
    train_emu_parser = subparsers.add_parser('train_emu', help='エミュレータでVSMを実行してモデルを訓練')
    train_emu_parser.add_argument('--batch-size', type=int, default=256,
                                       help='Batch size (デフォルト: 256)')
    train_emu_parser.add_argument('--test-batch-size', type=int, default=1000,
                                       help='Test batch size (デフォルト: 1000)')
    
    # MN-Core2実機訓練モード
    train_mncore2_parser = subparsers.add_parser('train_mncore2', 
                                                     help='MN-Core2実機でVSMを実行してモデルを訓練')
    train_mncore2_parser.add_argument('--batch-size', type=int, default=256,
                                          help='Batch size (デフォルト: 256)')
    train_mncore2_parser.add_argument('--test-batch-size', type=int, default=1000,
                                          help='Test batch size (デフォルト: 1000)')
    
    # エクスポートモード
    export_parser = subparsers.add_parser('export', help='PyTorch関数をONNXにエクスポート')
    export_parser.add_argument('output_dir', nargs='?', default=None,
                               help='出力ディレクトリ (省略時: /tmp/{example}/)')
    export_parser.add_argument('--example', choices=['train_step', 'inference', 'custom'],
                               default='train_step',
                               help='エクスポートする関数の例 (デフォルト: train_step)')
    export_parser.add_argument('--ignore', nargs='*', default=[],
                               help='エクスポートから除外する出力名 (例: --ignore loss)')
    export_parser.add_argument('--no-decompose', action='store_true',
                               help='複雑なオペレーターの分解を無効化')
    export_parser.add_argument('--mncpp', action='store_true',
                               help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    
    # コンパイルモード
    compile_parser = subparsers.add_parser('compile', help='ONNXからC++/Pythonコードを生成')
    compile_parser.add_argument('onnx_file', help='ONNXファイルのパス')
    compile_parser.add_argument('--mncpp', action='store_true',
                               help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    
    # unit test 生成モード
    unit_test_parser = subparsers.add_parser('build_unit_tests', help='ONNXグラフを分解してテストケースを生成')
    unit_test_parser.add_argument('onnx_dir', 
                                    help='エクスポート済みディレクトリ（model.onnxとnpyファイルを含む）')
    unit_test_parser.add_argument('--output-dir', default='./unit_tests',
                                    help='ユニットテストの出力先ディレクトリ（デフォルト: ./unit_tests）')
    unit_test_parser.add_argument('--extra', action='store_true',
                                    help='累積テスト（z_nodes_*）も生成する')
    unit_test_parser.add_argument('--mncpp', action='store_true',
                                    help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    unit_test_parser.add_argument('--mntest', action='store_true',
                                    help='MN-Coreアセンブリ用のテスト作成モード（--mncppを自動有効化）')
    
    # テストモード
    test_parser = subparsers.add_parser('test', help='ディレクトリ内のコードをテスト')
    test_parser.add_argument('test_paths', nargs='+', 
                            help='テストディレクトリ (例: /tmp/train_step または unit_tests/train_step/Gemm_256x784_16x784)')
    test_parser.add_argument('--mncpp', action='store_true',
                            help='MN-Core用のDL/ULノードを含むmn_model.onnxを使用')
    test_parser.add_argument('-f', '--force-continue', action='store_true',
                            help='generate_vsmでNotImplementedErrorが発生しても続行し、失敗としてカウント')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return

    if args.mode == 'train':
        train_model(args.batch_size, args.test_batch_size)
    elif args.mode == 'train_cpp':
        train_model_cpp(args.batch_size, args.test_batch_size)
    elif args.mode == 'train_cpp_noloss':
        train_model_cpp(args.batch_size, args.test_batch_size, ignore_loss=True)
    elif args.mode == 'train_emu':
        train_model_cpp(args.batch_size, args.test_batch_size, ignore_loss=True, backend="emu")
    elif args.mode == 'train_mncore2':
        train_model_cpp(args.batch_size, args.test_batch_size, ignore_loss=True, backend="mncore2")
    elif args.mode in ['export', 'e', 'et']:
        # 出力ディレクトリを決定（共通処理）
        output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else f"/tmp/{args.example}"
        
        # 共通関数を使用して関数とダミー入力を取得
        func, dummy_inputs = get_example_function_and_inputs(args.example, args.batch_size)
        
        # エクスポートを実行
        decompose = not (hasattr(args, 'no_decompose') and args.no_decompose)
        output_dir, inputs_list, outputs_list = export_function(func, dummy_inputs, output_dir,
                                    ignore_outputs=args.ignore if hasattr(args, 'ignore') else [],
                                    decompose=decompose)
        
        # MN-Core変換も実行
        onnx_path = os.path.join(output_dir, "model.onnx")
        mncore_path = os.path.join(output_dir, "mn_model.onnx")
        transform_to_mncore(onnx_path, mncore_path)
        
        # ecモードの場合はコンパイルとチェックも実行（すべてのexampleで共通）
        if args.mode == 'et':
            # --mncppオプションに応じて使用するONNXファイルを選択
            if hasattr(args, 'mncpp') and args.mncpp:
                onnx_path = os.path.join(output_dir, "mn_model.onnx")
            else:
                onnx_path = os.path.join(output_dir, "model.onnx")
            mn_onnx_path = os.path.join(output_dir, "mn_model.onnx")
            python_path, cpp_path = generate_code_from_onnx(onnx_path, output_dir)
            # python_path, cpp_path = generate_code_from_onnx(onnx_path, output_dir, mn_onnx_path)
            
            print("\n=== C++コード検証 ===")
            check_cpp_outputs(cpp_path, outputs_list, inputs_list)
    
    elif args.mode in ['compile', 'c']:
        onnx_path = args.onnx_file
        
        if not os.path.exists(onnx_path):
            print(f"エラー: ONNXファイルが見つかりません: {onnx_path}")
            sys.exit(1)
        
        if not onnx_path.endswith('.onnx'):
            print(f"エラー: ONNXファイル（.onnx）を指定してください: {onnx_path}")
            sys.exit(1)
        
        output_dir = os.path.dirname(onnx_path)
        mn_onnx_path = onnx_path if os.path.basename(onnx_path) == "mn_model.onnx" else None
        generate_code_from_onnx(onnx_path, output_dir, mn_onnx_path)
        print(f"\nコンパイル完了: {onnx_path}")
        
    elif args.mode == 'build_unit_tests':
        # unit test を生成
        output_dir = args.output_dir
        
        # デフォルトディレクトリまたは/tmp以下の場合はクリーンアップ
        if (output_dir == './unit_tests' or output_dir.startswith('/tmp/')) and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"既存ディレクトリをクリーンアップしました: {output_dir}")
        
        # --mntestが有効なら--mncppも自動有効化
        if hasattr(args, 'mntest') and args.mntest:
            args.mncpp = True
        
        # --mncppオプションに応じてONNXファイル名を選択
        onnx_filename = "mn_model.onnx" if hasattr(args, 'mncpp') and args.mncpp else "model.onnx"
        
        # mntestオプションを渡す
        mntest = hasattr(args, 'mntest') and args.mntest
        build_unit_tests(args.onnx_dir, output_dir, extra=args.extra, onnx_filename=onnx_filename, mntest=mntest)
    
    elif args.mode == 'test':
        # --mncppオプションに応じてONNXファイル名を選択
        onnx_filename = "mn_model.onnx" if hasattr(args, 'mncpp') and args.mncpp else "model.onnx"
        force_continue = hasattr(args, 'force_continue') and args.force_continue
        run_multiple_tests(args.test_paths, onnx_filename=onnx_filename, force_continue=force_continue)


if __name__ == '__main__':
    main()
