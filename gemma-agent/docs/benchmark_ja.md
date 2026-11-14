# ベンチマーク: 低スペック・CPU環境における EmbeddingGemma スクリプトの検証

本ドキュメントは、低スペックなCPU環境PCにおいて測定された `apps/` 配下の推論スクリプトの実行時間およびメモリ使用量をまとめたレポートです（モデルがローカルキャッシュに保存された状態での測定）。

## 1. テスト環境

| 項目 | 値 |
|---|---|
| CPU | Intel N100 (4コア, 4スレッド, AVX2対応, AVX-512非対応) |
| RAM | 3.7 GiB (+ 1.0 GiB スワップ) |
| GPU | なし |
| OS | Ubuntu 24.04.3 LTS on WSL2 (カーネル 6.6.87.2-microsoft-standard-WSL2) |
| Python | 3.12.3 |
| torch | 2.15.0 (CPU) |
| transformers | 5.22.2 |
| sentence-transformers | 6.2.0 |
| litert-lm-api | 0.19.1 |
| qdrant-edge-py | 0.9.0 |
| edgeparse | 0.3.2 |
| huggingface_hub | 2.4.0 |

## 2. 測定方法

- すべてのスクリプトは、`gemma-agent/` からプロジェクト仮想環境 (`.venv`) を使用して `uv run python ../apps/<スクリプト名>.py` で実行しました。
- **合計時間** および **ピークメモリ**（最大常駐セットサイズ: `MaxRSS`）は `/usr/bin/time` で測定しました。
- **ステップごとの所要時間** は、スクリプトの標準出力の各行にタイムスタンプを付与することで算出しました（`python -u`、バッファリングなし出力）。
- モデルやアセットはローカルキャッシュおよび `./assets/` に保存済みの状態（通常運用の定常状態）で測定しています。
- WSL2 上での実行時間は OS のページキャッシュやバックグラウンド負荷によって変動するため、約 ±10〜20秒程度の差は測定誤差の範囲内として解釈してください。

## 3. サマリー

| スクリプト | 合計時間 | ピークメモリ | 備考 |
|---|---:|---:|---|
| `inference_embeddinggemma_with_sentence_transformers.py` | 69.8 s | 2.18 GiB | テキスト専用モデル |
| `multimodal_embeddinggemma_with_sentence_transformers.py` | 454.0 s | 2.77 GiB | 画像＋テキストモデル |
| `fine_tuning_embeddinggemma_with_sentence_transformers.py` | 50.1 s | 2.30 GiB | ファインチューニング済みテキストモデル（推論のみ） |
| `fine_tuning_multimodal_embeddinggemma_with_sentence_transformers.py` | 121.2 s | 2.89 GiB | ファインチューニング済みマルチモーダルモデル（推論のみ） |
| `RAG_with_EmbeddingGemma.py` | 94.3 s | 2.18 GiB | Sentence Transformers + Gemma 4 E2B (LiteRT-LM)、質問4件 |
| `EmbeddingGemma2_LiteRT_RAG.py` | 51.3 s | 1.46 GiB | LiteRT-LM のみ、質問2件、インデックス再利用 |

> `fine_tuning_*` スクリプトは学習を行うものではなく、すでにファインチューニングされたモデル（`shogoorg/my-embeddinggemma-2-finetuned`、`shogoorg/my-multimodal-embeddinggemma-2-finetuned`）を使用した推論を実行します。ローカルキャッシュがない初回実行時のみ、モデルのダウンロード時間が追加されます。

## 4. 詳細結果

### 4.1 `inference_embeddinggemma_with_sentence_transformers.py`

テキスト専用の EmbeddingGemma 2（ビジョンおよびオーディオエンコーダーは無効化、パラメータ数: 271,002,624、float32）。

| ステップ | 所要時間 |
|---|---:|
| Python 起動 + モデル読み込み | 58.6 s |
| 3単語の埋め込みベクトル生成 | 3.3 s |
| 3組の文ペアの類似度計算（プロンプトなし） | 0.7 s |
| 3組の文ペアの類似度計算 (`STS` プロンプトあり) | 0.6 s |
| インタプリタ終了処理（最後の出力後） | 6.4 s |
| **合計** | **69.8 s** |

- モデルの読み込みが完了した後は、短い文ペアのエンコード処理は **約0.2秒** しかかかりません。
- **モデルの読み込みが合計時間の 70〜85% を占めています。**

### 4.2 `multimodal_embeddinggemma_with_sentence_transformers.py`

ビジョンエンコーダーを有効化した EmbeddingGemma 2（オーディオエンコーダーは無効化、パラメータ数: 438,760,448、float32）。各類似度計算ステップでは、クエリ画像1枚と候補4件（画像1件＋テキスト3件）をエンコードします。

| ステップ | 所要時間 |
|---|---:|
| Python 起動 + モデル読み込み | 41.7 s |
| クロスモーダル類似度計算（デフォルトの画像トークン予算） | 73.4 s |
| クロスモーダル類似度計算 (`max_soft_tokens=1120`) | 319.1 s |
| インタプリタ終了処理 | 19.8 s |
| **合計** | **454.0 s** |

- **CPU 環境では画像エンコードが最大のボトルネックになります。** デフォルトのトークン予算でも画像2枚で約1分を要します。
- 最大予算の `max_soft_tokens=1120` を指定した場合、同ステップに **5〜6分（約5倍遅い）** かかります。

### 4.3 `fine_tuning_embeddinggemma_with_sentence_transformers.py`

ファインチューニング済みテキストモデル、クエリ1件およびドキュメント1件。

| ステップ | 所要時間 |
|---|---:|
| Python 起動 + モデル読み込み + エンコード | 44.6 s |
| インタプリタ終了処理 | 5.5 s |
| **合計** | **50.1 s** |

- 類似度: `0.8493`。
- 処理時間のほとんどをモデルの読み込みが占めています。

### 4.4 `fine_tuning_multimodal_embeddinggemma_with_sentence_transformers.py`

ファインチューニング済みマルチモーダルモデル、テキストクエリ1件および画像3枚。

| ステップ | 所要時間 |
|---|---:|
| Python 起動 + モデル読み込み + エンコード | 107.3 s |
| インタプリタ終了処理 | 13.9 s |
| **合計** | **121.2 s** |

- 類似度: `[0.7091, 0.6116, 0.6654]`。
- 本スクリプトは `config_kwargs` を指定せずに読み込むため、オーディオエンコーダーも読み込まれます。そのため、**全スクリプト中最大のピークメモリ（2.89 GiB）** を記録し、3.7 GiB の制限に近い値となります。

### 4.5 `RAG_with_EmbeddingGemma.py`

Sentence Transformers（テキスト専用、float32）による2段階検索（カテゴリ → ドキュメント）と、LiteRT-LM CPU バックエンドの Gemma 4 E2B による回答生成。2つのモデルは同時にメモリ上に保持されません。

**ステップ別内訳**（合計 約107秒）

| ステップ | 所要時間 |
|---|---:|
| Python 起動 + EmbeddingGemma 読み込み | 56.9 s |
| 埋め込みモデル破棄 + Gemma 4 E2B 読み込み | 5.5 s |

| 質問 | 検索 | 生成 | 合計 |
|---|---:|---:|---:|
| How many days of annual paid leave do I get? | 8.9 s | 9.6 s | 18.5 s |
| How do I reset my password? | 5.0 s | 6.8 s | 11.8 s |
| What are the reimbursable expense guidelines for business travel? | 1.9 s | 7.1 s | 9.0 s |
| Can I receive personal packages at the office? | 3.2 s | 2.2 s | 5.4 s |
| **平均** | **4.8 s** | **6.4 s** | **11.2 s** |

- 初回検索はウォームアップ処理のため若干時間がかかります。
- ピークメモリ: **2.18 GiB**。

### 4.6 `EmbeddingGemma2_LiteRT_RAG.py`

Gemma 4 テクニカルレポート（45チャンク）を対象とした RAG。埋め込みと生成の両方に LiteRT-LM CPU バックエンドを使用し、ベクトルストアとして Qdrant Edge を利用。

**ステップ別内訳**（`db_gemma/` の既存インデックスを再利用した場合）

| ステップ | 所要時間 |
|---|---:|
| Python 起動 + モデルパス解決 | 1.1 s |
| 埋め込みエンジン起動 + 既存インデックスオープン | 2.8 s |
| クエリ2件の埋め込みと検索 | 1.9 s (各 約0.9 s) |
| 埋め込みエンジン破棄 + Gemma 4 E2B 読み込み | 1.9 s |

| 質問 | 生成 |
|---|---:|
| What architecture is Gemma 4 based on? | 18.9 s |
| What context length does Gemma 4 support? | 12.0 s |

- 合計実行時間: **51.3 s**、ピークメモリ: **1.46 GiB**。
- 回答はストリーミング出力されるため、上記生成時間が経過する前から最初の単語が表示されます。

## 5. 主な知見

1. **Sentence Transformers (PyTorch) のモデル読み込みが最大の固定コスト（38〜59秒）**。テキスト専用スクリプトの場合、実際の推論は10秒未満で終わります。一方、LiteRT-LM は約1〜3秒でモデルをロードできます。
2. **`EmbeddingGemma2_LiteRT_RAG.py` が最も高速かつ省メモリなパイプライン（51秒、ピーク 1.46 GiB）**。PDF検索とLLM生成の両方を含みながら、全体の中で最も軽量です。
3. **CPU での画像エンコードは処理負荷が高い**。デフォルトトークン数で画像2枚約1分、`max_soft_tokens=1120` では5〜6分を要します。低スペックPCではトークン予算を小さく設定することが推奨されます。
4. **メモリが最も逼迫するリソース**。マルチモーダルスクリプトは 3.7 GiB の環境で 2.75〜2.89 GiB に達します。未使用のエンコーダーを無効化（`config_kwargs={"audio_config": None}` 等）することでメモリ使用量を大幅に削減できます。
5. **RAG の質問あたりレイテンシは約5〜20秒**で、その大半は Gemma 4 E2B による回答生成です。
6. **インタプリタ終了処理に5〜20秒**を要します（大きなテンソルの解放処理によるもので、合計時間に含まれています）。

## 6. 低スペックPC向けの推奨事項

- RAG 用途には、埋め込み・生成ともに LiteRT-LM を使用する **`EmbeddingGemma2_LiteRT_RAG.py`** の構成が最適です。
- 複数クエリを連続して実行する場合は、対話型ループ等でモデルをメモリ上に常駐させ、読み込みコストを1回で済ませる運用を推奨します。
- マルチモーダル処理では、`max_soft_tokens` を小さめに設定し、不要なオーディオエンコーダーは無効化してください。
- スワップアウトを防ぐため、マルチモーダルスクリプト実行前は他の重いアプリケーションを終了させておくことを推奨します。
- RAG の生成時間を短縮したい場合は、`RETRIEVAL_LIMIT` や `CHUNK_SIZE` を小さく設定してください。
