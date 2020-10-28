python -m main --dataset_path ../dataset_v1 --arch TENet6NarrowModel \
--kernel_list 3,5,7,9 --save_folder save/MTENet6NarrowModel

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet6NarrowModel --kernel_list 3,5,7,9 \
--checkpoint_path save/MTENet6NarrowModel/TENet6NarrowModel-30000

python -m tenet_fusion --arch TENet6NarrowModel --kernel_list 3,5,7,9 --save_folder save/TENet6NarrowModel/ \
--checkpoint_path save/MTENet6NarrowModel/TENet6NarrowModel-30000

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet6NarrowModel \
--checkpoint_path save/TENet6NarrowModel/TENet6NarrowModel-30000
