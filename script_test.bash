test="python turngpt/test.py --batch_size 16 --load_from_checkpoint --pretrained_model_name_or_path runs/TurnGPT/TurnGPT_1j6b07j2/epoch=8_val_loss=1.2753.ckpt --accelerator gpu"

# datasets:
# default is to use all
# 
# "curiosity_dialogs",
# "daily_dialog",
# "multi_woz_v22",
# "meta_woz",
# "taskmaster1",
# "taskmaster2",
# "taskmaster3",
# Example:
# $test --datasets daily_dialog meta_woz
$test --datasets daily_dialog --overwrite true
#$test --datasets curiosity_dialogs multi_woz_v22 meta_woz taskmaster1 taskmaster2 taskmaster3
