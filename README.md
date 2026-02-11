# MN-Core Simple Graph Compiler for Education

『MN-Coreグラフコンパイラを自作して MNIST を学習させよう！』の学習用グラフコンパイラのリポジトリです。

MN-Core2 用のアセンブリを出力し、MN-Core2 エミュレータを使って多層パーセプトロン（MLP）を用いた MNIST 分類器を動作させるところまで体験できます。

詳しくは以下のブログ記事、スライド、MN-Core Challenge の特設問題セットをご覧ください。
- [ブログ](https://tech.preferred.jp/ja/blog/mn-core2_graphcompiler_scratch/)
- [スライド](https://speakerdeck.com/pfn/202512_mncore-graph-compiler-mnist)
- [MN-Core Challenge の特設問題セット](https://mncore-challenge.preferred.jp/mnist/)


## 対応環境
- Ubuntu 22.04 (WSL 可)
- mac（Docker 必須）

## 必要環境
- Python 3.10+

## セットアップ
```bash
pip install -r requirements.txt
bash setup_judge.sh
```
