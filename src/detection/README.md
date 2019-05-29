### Detection

Put `datatset,backbone` to the corresponding position in `mmdetetcion`

To test the detector and save the results.

`python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE>`

For example, to test the original frames and save the result as results_ori.pkl

`python tools/test.py configs/R50_FPN_DCN_test_fby.py --gpus 8 --out results_ori.pkl`

Convert the results to the corresponding format. See `utils/pkl2json.py`
