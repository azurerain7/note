# Bilinear Pooling Evaluation
-Tensorflow Slim
-GoogLeNet v1
  - Bilinear: *add one fc256 before softmax, dropout(keep prob)=0.4*
  
  -food-101
  >  Top 1/5     | Bilinear | Input Resolution
  > ------------ | -------- | -------------
  > 81.90/95.60  | []       | 224 X 224 *(base)* 
  > 82.68/96.03  | [x]      | 224 X 224 
  > 86.65/97.28  | []       | 448 X 448 
  > 87.50/97.55  | [x]      | 448 X 448
  > 85.40/97.06  | []       | 896 X 896 *(btch:16)*

  -fgvc-aircraft
  >  Top 1/5     | Bilinear | Input Resolution
  > ------------ | -------- | -------------
  > 76.10/95.08  | []       | 224 X 224 *(base)* 
  > 79.56/95.56  | [x]      | 224 X 224 
  > 82.79/97.21  | []       | 448 X 448 
  > 87.50/97.45  | [x]      | 448 X 448
  > 83.93/97.72  | []       | 896 X 896
