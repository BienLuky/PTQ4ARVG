
# VAR Code


### Pretrained Model
You need to download the pretrained models from the *[official VAR website](https://github.com/bytedance/1d-tokenizer)*, including VAE and VAR weights, and place them in `/PATH/TO/YOU/`.


### Example: VAR-d16
```bash
cd VAR
```
0. Sample FP Images (Optional: for evalution)
```bash
python ./sample_var.py --model var_d16 --ckpt_dir /PATH/TO/YOU/
```
1. Obtain Original Calibration
```bash
python ./sample_cali.py --model var_d16 --ckpt_dir /PATH/TO/YOU/
```
2. Get DGC Calibration
```bash
python ./sample_cali_dgc.py --model var_d16 --ckpt_dir /PATH/TO/YOU/
```
3. Quantize Model And Sample Quant Images
```bash
python ./sample_quant.py --model var_d16 --ckpt_dir /PATH/TO/YOU/
```

### implement
We use the global variable `globalvar` to access the calibration, so certain parts of the code need to be uncommented, as described in `sample_cali.py`. During quantization, we assign token-wise quantizers `TokenAwareUniformQuantizer` to layers affected by token-level variation. For layers with outliers, we first calculate scaling factors using GPS and then absorb them into the network weights to avoid additional computation during inference.


### Results

  <div align=center>
    <img src="../assets/var.png" width="100%" />
  </div>





