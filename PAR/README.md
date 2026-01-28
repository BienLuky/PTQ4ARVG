
# PAR Code


### Pretrained Model
You need to download the pretrained models from the *[official PAR website](https://github.com/YuqingWang1029/PAR)*, including VQ-VAE and PAR weights, and place them in `/PATH/TO/YOU/`.


### Example: PAR-XL-4x
```bash
cd PAR
```
0. Sample FP Images (Optional: for evalution)
```bash
python ./autoregressive/sample/sample_c2i_ddp_sample.py --model PAR-XL-4x --ckpt_dir /PATH/TO/YOU/
```
1. Obtain Original Calibration
```bash
python ./autoregressive/sample/sample_c2i_ddp_cali.py --model PAR-XL-4x  --ckpt_dir /PATH/TO/YOU/
```
2. Get DGC Calibration
```bash
python ./autoregressive/sample/sample_c2i_ddp_cali_dgc.py --model PAR-XL-4x  --ckpt_dir /PATH/TO/YOU/
```
3. Quantize Model And Sample Quant Images
```bash
python ./autoregressive/sample/sample_c2i_ddp_quant.py --model PAR-XL-4x  --ckpt_dir /PATH/TO/YOU/
```

### implement
We use the global variable `globalvar` to access the calibration, so certain parts of the code need to be uncommented, as described in `sample_c2i_ddp_cali.py`. During quantization, we assign token-wise quantizers `TokenAwareUniformQuantizer` to layers affected by token-level variation. For layers with outliers, we first calculate scaling factors using GPS and then absorb them into the network weights to avoid additional computation during inference.


### Results

  <div align=center>
    <img src="../assets/par.png" width="100%" />
  </div>

  **NOTE: Quarot experiments are excluded from PAR results as the models do not meet Quarot's requirements.**




