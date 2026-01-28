
# RAR Code


### Pretrained Model
You need to download the pretrained models from the *[official RAR website](https://github.com/bytedance/1d-tokenizer)*, including MaskGIT-VQGAN and RAR weights, and place them in `/PATH/TO/YOU/`.


### Example: RAR-B
```bash
cd RAR
```
0. Sample FP Images (Optional: for evalution)
```bash
python ./sample_rar.py --model rar_b --ckpt_dir /PATH/TO/YOU/
```
1. Obtain Original Calibration
```bash
python ./sample_cali.py --model rar_b --ckpt_dir /PATH/TO/YOU/
```
2. Get DGC Calibration
```bash
python ./sample_cali_dgc.py --model rar_b --ckpt_dir /PATH/TO/YOU/
```
3. Quantize Model And Sample Quant Images
```bash
python ./sample_quant.py --model rar_b --ckpt_dir /PATH/TO/YOU/
```

### implement
We use the global variable `globalvar` to access the calibration, so certain parts of the code need to be uncommented, as described in `sample_cali.py`. During quantization, we assign token-wise quantizers `TokenAwareUniformQuantizer` to layers affected by token-level variation. For layers with outliers, we first calculate scaling factors using GPS and then absorb them into the network weights to avoid additional computation during inference.


### Results

  <div align=center>
    <img src="../assets/rar.png" width="100%" />
  </div>

  **NOTE: Quarot experiments are excluded from RAR-XXL results as the model does not meet Quarot's requirements.**




