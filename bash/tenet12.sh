python -m main --dataset_path ../dataset_v1 --arch TENet12Model \
--kernel_list 3,5,7,9 --save_folder save/MTENet12Model

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet12Model --kernel_list 3,5,7,9 \
--checkpoint_path save/MTENet12Model/TENet12Model-30000

python -m tenet_fusion --arch TENet12Model --kernel_list 3,5,7,9 --save_folder save/TENet12Model/ \
--checkpoint_path save/MTENet12Model/TENet12Model-30000

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet12Model \
--checkpoint_path save/TENet12Model/TENet12Model-30000
