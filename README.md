# Bilinear Pooling Evaluation
- Tensorflow Slim
- GoogLeNet v1
  - Bilinear: *add one fc256 before softmax, dropout(keep prob)=0.4*
  
  - food-101
  >  Top 1/5     | Bilinear | Input Resolution
  > ------------ | -------- | -------------
  > 81.90/95.60  | N        | 224 X 224 *(base)* 
  > 82.68/96.03  | Y        | 224 X 224 
  > 86.65/97.28  | N        | 448 X 448 
  > 87.50/97.55  | Y        | 448 X 448
  > 85.40/97.06  | N        | 896 X 896 *(btch:16)*

  - fgvc-aircraft
  >  Top 1/5     | Bilinear | Input Resolution
  > ------------ | -------- | -------------
  > 76.10/95.08  | N        | 224 X 224 *(base)* 
  > 79.56/95.56  | Y        | 224 X 224 
  > 82.79/97.21  | N        | 448 X 448 
  > 87.50/97.45  | Y        | 448 X 448
  > 83.93/97.72  | N        | 896 X 896
