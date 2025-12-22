#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import ctypes
import tempfile
import onnx
import shutil
import subprocess
import platform
from typing import Dict, List, Any, Tuple, Optional

# メインのtrainモジュールから定数と関数をimport
from .train import ERROR_THRESHOLD
from .compile import compile_cpp_library, parse_cpp_output_info, generate_code_from_onnx, generate_vsm_from_onnx

# ============= Test関連の関数 =============

def verify_results(torch_results: Dict[str, torch.Tensor], generated_results: Dict[str, torch.Tensor], 
                  language: str = "Python", 
                  model_ref: Optional[nn.Module] = None, model_gen: Optional[nn.Module] = None, 
                  param_after: Optional[Dict[str, torch.Tensor]] = None,
                  exit_on_error: bool = True) -> bool:
    """
    PyTorchの結果と生成されたコードの結果を比較する統一検証関数。
    
    以下を処理可能:
    - 標準的な結果検証 (loss, output, gradients)
    - 勾配+パラメータ更新の組み合わせ検証
    - モデル間のパラメータ更新検証
    - 推論モード (outputとprobsのみ)
    
    Args:
        torch_results: PyTorchからの参照結果
        generated_results: 生成されたコードからの結果
        language: 表示用の言語/コンテキスト文字列
        model_ref: 参照モデル (パラメータ更新検証用)
        model_gen: 生成/更新されたモデル (パラメータ更新検証用)
        param_after: 更新されたパラメータ (トレースベース検証用)
        exit_on_error: エラー時にsys.exitするかどうか (デフォルト: True)
    
    Returns:
        bool: 検証が成功した場合True、失敗した場合False
    """
    print(f"\n=== {language}検証 ===")
    
    # 推論モードかどうかをチェック
    is_inference = 'loss' not in torch_results and 'probs' in torch_results
    
    # トレースされたコードからの更新されたパラメータを処理
    if not is_inference and 'updated_params' in generated_results and generated_results['updated_params']:
        # 更新されたパラメータをtorch_resultsとgenerated_resultsの両方の勾配に追加
        if 'gradients' not in torch_results:
            torch_results['gradients'] = {}  # type: ignore[assignment]
        if 'gradients' not in generated_results:
            generated_results['gradients'] = {}  # type: ignore[assignment]
            
        # torch_resultsもupdated_paramsを持っている場合、それらを使用
        if 'updated_params' in torch_results and torch_results['updated_params']:
            for name, param in generated_results['updated_params'].items():  # type: ignore[attr-defined]
                if name in torch_results['updated_params']:
                    # 勾配と区別するためにupdated_プレフィックスを付けて追加
                    generated_results['gradients'][f'updated_{name}'] = param  # type: ignore[index]
                    torch_results['gradients'][f'updated_{name}'] = torch_results['updated_params'][name]  # type: ignore[index]
        elif param_after:
            # param_afterが提供された場合のレガシーパス
            for name, param in generated_results['updated_params'].items():  # type: ignore[attr-defined]
                if name in param_after:
                    # 勾配と区別するためにupdated_プレフィックスを付けて追加
                    generated_results['gradients'][f'updated_{name}'] = param  # type: ignore[index]
                    torch_results['gradients'][f'updated_{name}'] = param_after[name]  # type: ignore[index]
    
    # 検証モードを決定
    is_param_update_only = torch_results is None
    
    # 検証するすべての項目を収集
    verification_items = []
    
    if not is_param_update_only:
        # まず、標準的な訓練/推論結果のように見えるかチェック
        has_loss = 'loss' in torch_results
        has_probs = 'probs' in torch_results
        
        if is_inference or (has_probs and not has_loss):
            # 推論モード - outputとprobsのみを検証
            if 'output' in torch_results and 'output' in generated_results:
                verification_items.append(('output', torch_results['output'], generated_results['output']))
            if 'probs' in torch_results and 'probs' in generated_results:
                verification_items.append(('probs', torch_results['probs'], generated_results['probs']))
        elif has_loss:
            # 訓練モード - すべての一致するキーを検証
            for key in torch_results.keys():
                if key in generated_results and key not in ['gradients', 'updated_params']:
                    verification_items.append((key, torch_results[key], generated_results[key]))
            
            # gradients辞書内のものも検証（もし存在すれば）
            for name in torch_results.get('gradients', {}):
                if name in generated_results.get('gradients', {}):
                    verification_items.append((f'grad_{name}', torch_results['gradients'][name], 
                                             generated_results['gradients'][name]))
        else:
            # 汎用モード - 一致するすべてのキーを検証
            # これは任意の出力を持つモデルフリー関数を処理
            for key in torch_results.keys():
                if key in generated_results and key not in ['gradients', 'updated_params']:
                    verification_items.append((key, torch_results[key], generated_results[key]))
    
    # model_refが提供されている場合、パラメータ更新検証を追加
    if model_ref is not None and model_gen is not None:
        for name, param_ref in model_ref.named_parameters():
            # モデルからパラメータを取得
            param_gen = dict(model_gen.named_parameters())[name]
            verification_items.append((name, param_ref, param_gen))
    
    # フォーマット用に最長の名前と形状を検索
    all_names = [item[0] for item in verification_items]
    max_name_len = max(len(name) for name in all_names) if all_names else 20
    
    # 最大の形状文字列長を見つけるためにすべての形状を収集
    all_shapes = []
    for name, ref_val, gen_val in verification_items:
        all_shapes.append(str(tuple(ref_val.shape)))
    max_shape_len = max(len(s) for s in all_shapes) if all_shapes else 15
    
    # ヘッダーを出力
    print(f"{'Var':<{max_name_len}} |"
          f"{'Error':^9} | {'Ref Min':^9} | "
          f"{'Ref Max':^9} | "
          f"{'Ref Avg':^9} |"
          f"{'Shape':^{max_shape_len}}| v")
    print("-" * (max_name_len + max_shape_len + 52))
    
    # 比較が失敗したかどうかを追跡
    has_errors = False
    
    # 各項目を検証
    for name, ref_val, gen_val in verification_items:
        # テンソル比較
        error = (ref_val - gen_val).abs().max().item()
        
        ref_min = ref_val.min().item()
        ref_max = ref_val.max().item()
        # 平均計算のために整数テンソルはfloatに変換する
        if ref_val.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
            ref_avg = ref_val.float().mean().item()
        else:
            ref_avg = ref_val.mean().item()
        shape_str = str(tuple(ref_val.shape))
        # 累積演算やlog演算にはわずかに高いしきい値を使用
        # Sum演算は浮動小数点誤差を蓄積する可能性がある
        # Log演算も数値精度の問題で誤差が大きくなることがある
        if name.endswith('_sum') or 'sum' in name:
            threshold = ERROR_THRESHOLD * 10
        elif 'log' in name.lower():
            threshold = ERROR_THRESHOLD * 10  # Logも高めのしきい値を使用
        else:
            threshold = ERROR_THRESHOLD
        match_str = "✓" if error < threshold else "✗"
        
        if error >= threshold:
            has_errors = True
            
        print(f"{name:<{max_name_len}} |{error:9.2e} | "
                f"{ref_min:9.2e} | {ref_max:9.2e} | {ref_avg:9.2e} |"
                f"{shape_str:<{max_shape_len}}| {match_str}")
    # 一致しない値がある場合はエラーで終了
    if has_errors:
        error_type = "Parameter update" if is_param_update_only else language
        print(f"\nERROR: {error_type}検証失敗 - いくつかの値が誤差しきい値({ERROR_THRESHOLD})を超えています")
        if exit_on_error:
            sys.exit(1)
        return False
    
    return True


def run_simple_cpp_code(lib_path: str, all_inputs_list: List[Tuple[str, torch.Tensor]], cpp_code: str, onnx_model=None) -> Dict[str, torch.Tensor]:
    """シンプルなC++コードをctypes経由で実行する
    
    Args:
        lib_path: 共有ライブラリのパス
        all_inputs_list: [(name, tensor), ...] の順序付きリスト
        cpp_code: C++コード文字列（出力情報を解析するため）
        onnx_model: ONNXモデル（出力のデータ型を判定するため）
    """
    
    # C++コードから出力情報を解析
    output_info = parse_cpp_output_info(cpp_code)
    
    # 共有ライブラリを読み込む
    lib = ctypes.CDLL(lib_path)
    
    # 関数を取得
    try:
        func = getattr(lib, "forward_backward")
    except AttributeError as e:
        raise AttributeError("C++ライブラリにforward_backward関数が見つかりませんでした") from e
    
    # リストなので順序は既に決まっている
    # print(f"入力順序: {[name for name, _ in all_inputs_list]}")
    # print(f"検出された出力: {[info['name'] for info in output_info]}")
    
    # 入力と出力に基づいて引数の型を準備
    argtypes = []
    
    # 入力ポインタを追加（リストの順序に従う）
    for name, tensor in all_inputs_list:
        # テンソルの型を確認して適切な型を使用
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            argtypes.append(ctypes.POINTER(ctypes.c_int))
        else:
            argtypes.append(ctypes.POINTER(ctypes.c_float))  # type: ignore
    
    # 出力ポインタを追加
    for info in output_info:
        output_name = info['name']
        # ONNXモデルで実際の出力データ型を確認
        output_is_int = False
        if onnx_model:
            for out in onnx_model.graph.output:
                if out.name == output_name and out.type.HasField('tensor_type'):
                    dtype = out.type.tensor_type.elem_type
                    if dtype == 6 or dtype == 7:  # ONNXのINT32/INT64
                        output_is_int = True
                        break
        
        if output_is_int:
            argtypes.append(ctypes.POINTER(ctypes.c_int))
        else:
            argtypes.append(ctypes.POINTER(ctypes.c_float))  # type: ignore
    
    func.argtypes = argtypes
    func.restype = None
    
    # 入力配列を準備
    input_arrays = {}
    args = []
    
    for name, tensor in all_inputs_list:
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            array = tensor.detach().cpu().numpy().astype(np.int32)
            input_arrays[name] = array
            args.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        else:
            array = tensor.detach().cpu().numpy().astype(np.float32)
            input_arrays[name] = array
            args.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))  # type: ignore
    
    # 出力配列を準備（リストの順序に従う）
    output_arrays = {}
    
    for info in output_info:
        name = info['name']
        # ONNXモデルから出力データ型を決定
        output_is_int = False
        if onnx_model:
            for out in onnx_model.graph.output:
                if out.name == name and out.type.HasField('tensor_type'):
                    dtype = out.type.tensor_type.elem_type
                    if dtype == 6 or dtype == 7:  # ONNXのINT32/INT64
                        output_is_int = True
                        break
        
        if output_is_int:
            output_dtype: Any = np.int32
            ctypes_ptr_type = ctypes.POINTER(ctypes.c_int)
        else:
            output_dtype = np.float32
            ctypes_ptr_type = ctypes.POINTER(ctypes.c_float)  # type: ignore
        
        if info['is_scalar']:
            # スカラーの場合
            array = np.zeros(1, dtype=output_dtype)
        elif info['shape']:
            # 形状が既知の場合
            array = np.zeros(info['shape'], dtype=output_dtype)
        else:
            # 形状が不明な場合、入力から推測を試みる
            # これは最後の手段 - 通常は形状情報が含まれているはず
            raise ValueError(f"出力 {name} の形状を決定できません")
        
        output_arrays[name] = array
        args.append(array.ctypes.data_as(ctypes_ptr_type))
    
    # 関数を呼び出す
    # print(f"C++関数を呼び出し中。引数の数: {len(args)}")
    # print(f"入力引数の数: {len(all_inputs_list)}, 出力引数の数: {len(output_info)}")
    try:
        func(*args)
        # print("C++関数の呼び出しが成功しました")
    except Exception as e:
        print(f"C++関数の呼び出し中にエラー: {e}")
        raise
    
    # 結果をtorchテンソルに変換（辞書形式で返す）
    results = {}
    for info in output_info:
        name = info['name']
        array = output_arrays[name]
        
        if info['is_scalar']:
            # スカラーの場合
            results[name] = torch.tensor(array[0])
        else:
            # テンソルの場合
            results[name] = torch.from_numpy(array.copy())
    
    return results

def check_cpp_outputs(cpp_file, expected_outputs_list, all_inputs_list, exit_on_error=True, onnx_model=None):
    """C++の出力をPython関数の出力と比較する
    
    Args:
        cpp_file: テストするC++ファイルのパス
        expected_outputs_list: [(name, tensor), ...] の順序付きリスト
        all_inputs_list: [(name, tensor), ...] の順序付きリスト
        exit_on_error: エラー時にsys.exitするかどうか (デフォルト: True)
        onnx_model: ONNXモデル（出力のデータ型を判定するため）
    
    Returns:
        bool: 検証が成功した場合True、失敗した場合False
    """
    
    # C++コードを読み込む
    with open(cpp_file, 'r') as f:
        cpp_code = f.read()
    
    # C++出力用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"C++出力directory: {tmpdir}")
        
        # C++コードをコンパイルして実行
        print(f"\n=== Compiling and running {cpp_file} ===")
        
        # 共有ライブラリにコンパイル
        lib_path = compile_cpp_library(cpp_code, os.path.join(tmpdir, "check_cpp"))
        
        # C++コードを実行
        generated_results = run_simple_cpp_code(lib_path, all_inputs_list, cpp_code, onnx_model=onnx_model)
        
        # 結果を比較
        # リストを辞書に変換して検証
        expected_outputs_dict = dict(expected_outputs_list)
        # print(f"Python結果のキー: {list(expected_outputs_dict.keys())}")
        # print(f"C++結果のkey: {list(generated_results.keys())}")
        
        # 検証
        # print("\n=== C++ check検証 ===")
        success = verify_results(expected_outputs_dict, generated_results, "C++ check", exit_on_error=exit_on_error)
        return success




# ==================== Autogradベースのエクスポート関数 ====================
# 以下のコードは3_fx_export_graph_autograd.pyから移植されました

def run_multiple_tests(test_paths, onnx_filename="model.onnx", force_continue=False):
    """複数のunit testを実行してサマリーを表示
    
    Args:
        test_paths: テストディレクトリパスのリスト
        onnx_filename: 使用するONNXファイル名（デフォルト: model.onnx）
        force_continue: generate_vsmでNotImplementedErrorが発生しても続行するかどうか
    """
    
    results = []
    
    # 単一テストかどうかを判定（引数が1個だけの場合）
    single_test = len(test_paths) == 1
    
    for test_path in test_paths:
        # パスが正しい形式かチェック
        if not os.path.exists(test_path):
            print(f"エラー: テストディレクトリが見つかりません: {test_path}")
            results.append((test_path, False))
            continue
        
        # ファイルの場合はスキップ
        if os.path.isfile(test_path):
            print(f"スキップ: {test_path} はファイルです（ディレクトリを指定してください）")
            continue
        
        print(f"\n{'='*60}")
        print(f"=== テスト実行: {test_path} ===")
        success = run_single_test_impl(test_path, onnx_filename, single_test, force_continue)
        results.append((test_path, success))
    
    # サマリーを表示
    print(f"\n{'='*60}")
    print("=== テストサマリー ===")
    passed = sum(1 for _, success in results if success)
    failed = sum(1 for _, success in results if not success)
    total = len(results)
    
    # 失敗したテストを先頭に表示
    failed_tests = [(path, success) for path, success in results if not success]
    passed_tests = [(path, success) for path, success in results if success]
    
    # 失敗したテストを先に表示
    for test_path, _ in failed_tests:
        print(f"✗ FAIL: {test_path}")
    
    # 成功したテストを表示
    for test_path, _ in passed_tests:
        print(f"✓ PASS: {test_path}")
    
    print(f"\n合計: {passed}/{total} テスト成功")
    if failed > 0:
        print(f"失敗: {failed} テスト")
        sys.exit(1)

def run_mncore_test(test_dir, onnx_filename="model.onnx", single_test=False, force_continue=False):
    """MN-Core用のテストを実行
    
    Args:
        test_dir: テストディレクトリのパス
        onnx_filename: 使用するONNXファイル名
        single_test: 単一テストの実行かどうか（testコマンドの引数が1個の場合True）
        force_continue: generate_vsmでNotImplementedErrorが発生しても続行するかどうか
    
    Returns:
        bool: テストが成功した場合True
    """
    print(f"\n=== MN-Core用テスト: {test_dir} ===")
    
    # ONNXモデルのパス
    model_path = os.path.join(test_dir, onnx_filename)
    if not os.path.exists(model_path):
        print(f"エラー: ONNXモデルが見つかりません: {model_path}")
        return False
    
    # VSMファイルを探す
    vsm_files = [f for f in os.listdir(test_dir) if f.endswith('.vsm')]
    
    # テスト対象のVSMファイルを選択（offseted.vsmを優先）
    test_vsm_files = []
    if 'offseted.vsm' in vsm_files:
        test_vsm_files.append('offseted.vsm')
    # シンボリックリンク以外のVSMファイルも対象に
    for vsm_file in vsm_files:
        vsm_path = os.path.join(test_dir, vsm_file)
        if not os.path.islink(vsm_path) and vsm_file != 'offseted.vsm':
            test_vsm_files.append(vsm_file)
    
    if not test_vsm_files:
        print(f"警告: テスト可能なVSMファイルが見つかりません: {test_dir}")
        return False
    
    # VSMコードを生成
    print(f"VSMコード生成中: {model_path}")
    
    # テスト名を取得（パスから）
    test_name = os.path.basename(test_dir)
    
    # 固定ディレクトリに保存
    base_test_dir = "/tmp/mncore_vsm_test"
    os.makedirs(base_test_dir, exist_ok=True)
    
    # テスト用ディレクトリを作成
    temp_dir = os.path.join(base_test_dir, test_name)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # generate_vsm_from_onnxを使ってVSMコードを生成
    # force_continueが有効な場合はNotImplementedErrorをキャッチ
    if force_continue:
        try:
            generated_vsm = generate_vsm_from_onnx(model_path)
        except NotImplementedError as e:
            print(f"エラー: generate_vsmがNotImplementedErrorを返しました: {e}")
            print("[-f オプションにより続行し、このテストは失敗としてカウントします]")
            return False
    else:
        # force_continueが無効な場合は通常通りエラーを伝播
        generated_vsm = generate_vsm_from_onnx(model_path)
    
    generated_vsm_path = os.path.join(temp_dir, "generated.vsm")
    
    with open(generated_vsm_path, 'w') as f:
        f.write(generated_vsm)
    
    print(f"生成されたVSM: {generated_vsm_path}")
    # testコマンドの引数が1個だけの場合、生成されたVSMを表示（コメントと空行を除く）
    if single_test:
        try:
            with open(generated_vsm_path, 'r') as f:
                non_comment_lines = []
                for line in f:
                    stripped = line.strip()
                    # コメント行（#で始まる）と空行を除外
                    if stripped and not stripped.startswith('#'):
                        non_comment_lines.append(line.rstrip())
                
                # 10行以下なら全て表示、そうでなければ先頭4行と末尾4行を表示
                if len(non_comment_lines) <= 10:
                    print(f"\n生成されたVSM（全{len(non_comment_lines)}行、コメントと空行を除く）:")
                    for line in non_comment_lines:
                        print(f"  {line}")
                else:
                    print(f"\n生成されたVSM（先頭4行と末尾4行、全{len(non_comment_lines)}行中）:")
                    # 先頭4行
                    for line in non_comment_lines[:4]:
                        print(f"  {line}")
                    print("  ...")
                    # 末尾4行
                    for line in non_comment_lines[-4:]:
                        print(f"  {line}")
                
                if not non_comment_lines:
                    print("\n  （コメント以外の行が見つかりませんでした）")
        except Exception as e:
            print(f"\n  VSMファイルの読み込みエラー: {e}")
    
    # 各VSMファイルに対してテストを実行
    all_tests_passed = True
    
    for vsm_file in test_vsm_files:
        testcase_vsm_path = os.path.join(test_dir, vsm_file)
        
        # offseted.vsmの場合は通常テストと--dirtyテストの両方を実行
        if vsm_file == 'offseted.vsm':
            # 通常テスト
            result1 = run_judge(testcase_vsm_path, generated_vsm_path, dirty=False)
            
            if result1:
                # 通常テストが成功した場合のみ--dirtyテストを実行
                result2 = run_judge(testcase_vsm_path, generated_vsm_path, dirty=True)
                
                if result2:
                    print(f"✓ {vsm_file}: テスト成功")
                else:
                    print(f"✗ {vsm_file}: --dirtyテストが失敗")
                    all_tests_passed = False
            else:
                print(f"✗ {vsm_file}: テストが失敗")
                all_tests_passed = False
        else:
            # 通常テストのみ
            print(f"\nテスト実行: {vsm_file}")
            result = run_judge(testcase_vsm_path, generated_vsm_path, dirty=False)
            
            if result:
                print(f"✓ {vsm_file}: テスト成功")
            else:
                print(f"✗ {vsm_file}: テスト失敗")
                all_tests_passed = False
    
    
    return all_tests_passed

def run_judge(testcase_vsm, generated_vsm, dirty=False):
    """judge.pyを実行してテストを判定
    
    Args:
        testcase_vsm: テストケースのVSMファイルパス
        generated_vsm: 生成されたVSMファイルパス
        dirty: --dirtyオプションを使用するかどうか
    
    Returns:
        bool: テストが成功した場合True
    """
    
    # OSを判定してjudgeのパスを設定
    if platform.system() == "Darwin":  # macOS用
        # macOSではjudge/judgeスクリプトを使用（Docker使用）
        judge_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "judge", "judge"
        )
        cmd = ["sh", judge_path, testcase_vsm, generated_vsm]
    else:
        # Linux等ではjudge.pyを直接使用
        judge_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "judge", "judge-py", "judge.py"
        )
        cmd = ["python3", judge_path, testcase_vsm, generated_vsm]
    
    if dirty:
        cmd.append("--dirty")
    
    try:
        # judge.pyを実行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 出力から"ACCEPTED!!"を含む行を探す
        for line in result.stdout.split('\n'):
            if line.startswith("ACCEPTED!!"):
                if dirty:
                    print(f"  {line}")
                return True
        
        # 失敗した場合
        if result.returncode != 0:
            # 再現用コマンドを表示
            cmd_str = ' '.join(cmd)
            print(f"  失敗: 再現コマンド: {cmd_str}")
            print()
            print(f"  テストケースファイル: {testcase_vsm}")
            print(f"  生成された VSM: {generated_vsm}")
            
            # エラー出力があれば最初の5行を表示
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')[:5]
                for line in error_lines:
                    print(f"  エラー: {line}")
        
        return False
        
    except subprocess.TimeoutExpired:
        print(f"  タイムアウト: テストの実行が30秒を超えました")
        return False
    except Exception as e:
        print(f"  エラー: judge.pyの実行に失敗: {e}")
        return False

def run_single_test(test_path, onnx_filename="model.onnx"):
    """単一のtestを実行（コンパイルも行う）
    
    Args:
        test_path: テストディレクトリのパス
        onnx_filename: 使用するONNXファイル名（デフォルト: model.onnx）
    """
    # 単一テストとして実行
    return run_single_test_impl(test_path, onnx_filename, single_test=True)

def run_single_test_impl(test_path, onnx_filename="model.onnx", single_test=False, force_continue=False):
    """単一のtestを実行（コンパイルも行う）- 内部実装
    
    Args:
        test_path: テストディレクトリのパス
        onnx_filename: 使用するONNXファイル名（デフォルト: model.onnx）
        single_test: 単一テストの実行かどうか
        force_continue: generate_vsmでNotImplementedErrorが発生しても続行するかどうか
    """
    
    # test_pathをそのまま使用
    test_dir = test_path
    
    if not os.path.exists(test_dir):
        print(f"エラー: テストディレクトリが見つかりません: {test_dir}")
        return False
    
    # MN-Core用テストかどうかを判定（.vsmファイルの存在チェック）
    vsm_files = [f for f in os.listdir(test_dir) if f.endswith('.vsm')]
    npy_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]
    
    if vsm_files and not npy_files:
        # MN-Core用テストの実行
        return run_mncore_test(test_dir, onnx_filename, single_test=single_test, force_continue=force_continue)
    
    # ONNXモデルをロード
    model_path = os.path.join(test_dir, onnx_filename)
    if not os.path.exists(model_path):
        print(f"エラー: ONNXモデルが見つかりません: {model_path}")
        return False
    
    # 使用するONNXファイルを明示的に表示
    print(f"使用するONNXファイル: {model_path}")
    
    # コンパイル（C++/Pythonコード生成）
    print("コンパイル中...")
    python_path, cpp_path = generate_code_from_onnx(model_path, test_dir)
    
    model = onnx.load(model_path)
    
    # 入力データをロード
    inputs = {}
    i = 0
    while True:
        input_path = os.path.join(test_dir, f"input_{i}.npy")
        if not os.path.exists(input_path):
            break
        data = np.load(input_path)
        # ONNX入力名を取得（仮定: 順序は保持される）
        if i < len(model.graph.input):
            input_name = model.graph.input[i].name
            inputs[input_name] = data
        i += 1
    
    # 期待される出力をロード
    expected_outputs = []
    i = 0
    while True:
        output_path = os.path.join(test_dir, f"output_{i}.npy")
        if not os.path.exists(output_path):
            break
        expected_outputs.append(np.load(output_path))
        i += 1
    
    # C++コードを生成して実行
    # test_pathから名前部分を抽出してtemp_dir名を作成
    test_name = test_path.replace('/', '_').replace('\\', '_')
    temp_dir = f"/tmp/test_{test_name}"
    
    # 既存のディレクトリがある場合は中身を削除
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # ONNXからコード生成
    print("ONNXからコード生成...")
    generate_code_from_onnx(model_path, temp_dir)
    
    # C++コードを検証
    cpp_path = os.path.join(temp_dir, "forward_backward.cpp")
    
    # torch形式の入力に変換
    torch_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    
    # C++コードを検証
    print("C++コードを検証...")
    
    # 入力と出力を[(name, tensor), ...]形式に変換
    all_inputs_list = [(name, tensor) for name, tensor in torch_inputs.items()]
    
    # MN-Coreモデルの場合、期待される出力を適切にマッピング
    if 'mn_model' in onnx_filename:
        # MN-Coreモデルの出力名を取得
        output_names = [out.name for out in model.graph.output]
        expected_outputs_list = []
        
        # 各MN-Core出力名に対して、対応する期待値を見つける
        for i, mn_name in enumerate(output_names):
            if i < len(expected_outputs):
                expected_outputs_list.append((mn_name, torch.from_numpy(expected_outputs[i])))
    else:
        # 通常のモデル
        expected_outputs_list = [(model.graph.output[i].name, torch.from_numpy(expected_outputs[i])) 
                                for i in range(len(expected_outputs))]
    
    # check_cpp_outputsを使用して検証（きれいなフォーマット出力を得るため）
    # testコマンドではexit_on_error=Falseでエラー時も処理を継続
    try:
        success = check_cpp_outputs(cpp_path, expected_outputs_list, all_inputs_list, exit_on_error=False, onnx_model=model)
        if success:
            print(f"✓ テスト成功: {test_path}")
            return True
        else:
            print(f"✗ テスト失敗: {test_path}")
            return False
    except Exception as e:
        # その他の予期しないエラー
        print(f"エラー: {e}")
        print(f"✗ テスト失敗: {test_path}")
        return False
