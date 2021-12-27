# TtT
code for our ACL2021 paper "Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction"

The pretrained BERT model can be downloaded from: https://drive.google.com/file/d/1gX9YYcGpR44BsUgoJDtbWqV-4PsaWpq1/view?usp=sharing

Training:
```
./train.sh
```

Tips to reproduce the results:
- More epochs: more than 200;
- Larger batchsize on GPUs with large memory such as V100.
