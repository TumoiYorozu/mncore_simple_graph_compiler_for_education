#!/bin/bash

# PID付きのユニークな一時ファイル名を作成
TEMP_FILE="/tmp/judge_$(date +%Y%m%d_%H%M%S)_$$.zip"

# ファイルをダウンロード
echo "$TEMP_FILE にダウンロード中..."
curl -L -o "$TEMP_FILE" "https://mncore-challenge.preferred.jp/judge-mnist.zip"

# ダウンロードの成功確認
if [ ! -f "$TEMP_FILE" ]; then
    echo "エラー: judge.zip のダウンロードに失敗しました"
    exit 1
fi

# 既存のjudgeディレクトリを削除
rm -rf ./judge

# zipファイルを現在のディレクトリに展開（zipには judge/ ディレクトリが含まれている）
echo "現在のディレクトリに展開中..."
if unzip -o "$TEMP_FILE"; then
    echo "展開成功"
    
    # 一時ファイルを削除
    echo "一時ファイル $TEMP_FILE を削除中..."
    rm "$TEMP_FILE"

    # mac 用パッチ: テスト実行時に生成される VSM は /tmp 以下に置かれるが、
    # judge/judge の docker コンテナはカレントディレクトリしかマウントしないため、
    # コンテナから /tmp も参照できるようにマウントを追加する
    if ! grep -q 'src=/tmp,dst=/tmp' judge/judge; then
        sed -i.bak 's|^docker run |docker run --mount type=bind,src=/tmp,dst=/tmp |' judge/judge
        rm -f judge/judge.bak
        chmod +x judge/judge
    fi
    if grep -q 'src=/tmp,dst=/tmp' judge/judge; then
        echo "judge/judge に /tmp マウントパッチを適用しました"
    else
        echo "警告: judge/judge の docker run 行が見つからず、パッチを適用できませんでした（mac ではユニットテストが失敗する可能性があります）"
    fi

    echo "セットアップ完了！"
else
    echo "エラー: judge.zip の展開に失敗しました"
    rm "$TEMP_FILE"
    exit 1
fi

# 動作確認
if [[ "$OSTYPE" == "darwin"* ]]; then
    # 初回は docker の準備が行われる
    judge/judge judge/example/hello_world/testcase.vsm judge/example/hello_world/example.vsm 
else
    python3 judge/judge-py/judge.py judge/example/hello_world/testcase.vsm judge/example/hello_world/example.vsm 
fi
