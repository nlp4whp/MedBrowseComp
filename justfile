
# 解码`final50.csv`数据 > just convert
covert:
    uv run python data/encode_decode.py cell-decode-combo data/final50_cell_combo_shift3_b64.csv --shift 3

# 评估`final50.csv`数据 > just eval
eval:
    uv run python process_NCT_predictions.py data/final50.csv --task filled50 --use_tools \
        --output results50.csv \
        --model qwen-plus \
        --use_judge --judge_model qwen-plus
