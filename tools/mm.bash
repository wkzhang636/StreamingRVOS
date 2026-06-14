accelerate launch --num_processes 1 --main_process_port 12380 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/