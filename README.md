# best score file

## submit file
```
final_result.csv
``` 

## model file
```
b0_0/best_score_fold0_010.pth
b0_1/best_score_fold0_010.pth
b1/best_score_fold0_010.pth
b2/best_score_fold0_010.pth
b3/best_score_fold0_010.pth
b4/best_score_fold0_010.pth
```
<hr>

# 1. Running train code
```
python inference-train.py
```
위 명령어 실행시 아래와 같은 경로에 모델 파일 생성
-  output file (model with log)
    - tf_efficientnet_b0_ns_0.7/
    - tf_efficientnet_b0_ns_0.8/
    - tf_efficientnet_b1_ns_0.8/
    - tf_efficientnet_b2_ns_0.8/
    - tf_efficientnet_b3_ns_0.8/
    - tf_efficientnet_b4_ns_0.8/
    

<hr>
# 2. Running inference code

 - inference-train.py 코드 작성
```
 python inference-train.py
```


 - inference-test1.py 코드 작성
```
 python inference-test1.py
```
 output : 최고점을 받은 모델과 비슷한 성능을 보이는 모델 (inference_result1.csv)




 - inference-test2.py 코드 작성
```
 python inference-test2.py
```
 output : 최고점을 받은 모델 (inference_result2.csv)


