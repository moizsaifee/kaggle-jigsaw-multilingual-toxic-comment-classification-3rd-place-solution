python3 encode_text.py --path '../../Input/Ujjwal/Data'

python3 pretrain_xlm.py --path '../../Input/Ujjwal/Data' --mode version1
python3 pretrain_xlm.py --path '../../Input/Ujjwal/Data' --mode version2
python3 pretrain_xlm.py --path '../../Input/Ujjwal/Data' --mode version3

python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version1 --fold 1 --pseudo 0 --out 'simpl_v1_1' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version1 --fold 4 --pseudo 0 --out 'simpl_v1_4' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version1 --fold 0 --pseudo 1 --out 'pslbl_v1_0' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version2 --fold 8 --pseudo 0 --out 'simpl_v2_8' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version2 --fold 5 --pseudo 0 --out 'simpl_v2_5' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version2 --fold 7 --pseudo 1 --out 'pslbl_v2_7' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version3 --fold 6 --pseudo 0 --out 'simpl_v3_6' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version3 --fold 3 --pseudo 0 --out 'simpl_v3_3' 
python3 finetune_xlm.py --path '../../Input/Ujjwal/Data' --mode version3 --fold 8 --pseudo 1 --out 'pslbl_v3_8'
