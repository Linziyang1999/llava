launch --type v100-32g -- python3 -m llava.serve.cli \
    --model-path ./checkpoints/UIllava-mistral-insconv-v6 \
    --image-file "images/2.jpg" \
   