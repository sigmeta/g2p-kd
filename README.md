# g2p-kd
Token-Level Ensemble Distillation for Grapheme-to-Phoneme Conversion

Accepted by Interspeech2019.

The code is modified from fairseq, added with knowledge distillation criterion.
You can use it by setting the criterion

```
--criterion word_knowledge_distillation
```

You can check the file fairseq.md for more information.

# Citation:
```
@article{sun2019token,
  title={Token-Level Ensemble Distillation for Grapheme-to-Phoneme Conversion},
  author={Sun, Hao and Tan, Xu and Gan, Jun-Wei and Liu, Hongzhi and Zhao, Sheng and Qin, Tao and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:1904.03446},
  year={2019}
}
```