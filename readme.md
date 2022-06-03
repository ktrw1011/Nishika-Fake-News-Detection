# Nishika Fake News detection 2nd Place Solution
Competition Site: https://www.nishika.com/competitions/27/summary

## CV
- StratifiedKFold (n_split=5)  
  - src/_make_fold.ipynb

## Models
| EXP | Model                         | CV       | Weight | 
| --- | ----------------------------- | -------- | ------ | 
| 005 | xlm-roberta-large             | 0.996562 | 0.45   | 
| 009 | cl-tohoku/bert-large-japanese | 0.991801 | 0.1    | 
| 018 | microsoft/mdeberta-v3-base    | 0.996826 | 0.4725 | 

## Ensemble
- src/sub_emsamble_pp.ipynb
- Weighted Average(Nelder-Mead)

### PostProcess
They are all labeled isFake=1 in the training set
- 47によると
- \(、以下同
- ^.?Cによると
- ^.?C、.+によると

CV/LB：0.99814/0.996296