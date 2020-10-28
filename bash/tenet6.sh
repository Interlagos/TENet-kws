python -m main --dataset_path ../dataset_v1 --arch TENet6Model \
--kernel_list 3,5,7,9 --save_folder save/MTENet6Model

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet6Model --kernel_list 3,5,7,9 \
--checkpoint_path save/MTENet6Model/TENet6Model-30000

python -m tenet_fusion --arch TENet6Model --kernel_list 3,5,7,9 --save_folder save/TENet6Model/ \
--checkpoint_path save/MTENet6Model/TENet6Model-30000

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet6Model \
--checkpoint_path save/TENet6Model/TENet6Model-30000
