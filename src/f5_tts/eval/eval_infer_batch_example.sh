#!/bin/bash

# e.g. F5-TTS, 16 NFE
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -t "seedtts_test_zh" -nfe 16
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -t "seedtts_test_en" -nfe 16
accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_v1_Base" -t "ls_pc_test_clean" -nfe 16 -p data/LibriSpeech/test-clean


# e.g. evaluate F5-TTS 16 NFE result on Seed-TTS test-zh
python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l zh --gen_wav_dir results/F5TTS_v1_Base_1250000/seedtts_test_zh/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 --gpu_nums 8
python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l zh --gen_wav_dir results/F5TTS_v1_Base_1250000/seedtts_test_zh/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 --gpu_nums 8
python src/f5_tts/eval/eval_utmos.py --audio_dir results/F5TTS_v1_Base_1250000/seedtts_test_zh/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0

# etc.
