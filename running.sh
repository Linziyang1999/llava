launch --type v100-32g -- python3 -m llava.serve.test \
    --model-path ./checkpoints/UIllava-mistral-insconv-v6-compare \
    --image-path ./adversarial-00000-of-00001