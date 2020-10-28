python -m main --dataset_path ../dataset_v1 --arch TENet12NarrowModel \
--kernel_list 3,5,7,9 --save_folder save/MTENet12NarrowModel

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet12NarrowModel --kernel_list 3,5,7,9 \
--checkpoint_path save/MTENet12NarrowModel/TENet12NarrowModel-30000

python -m tenet_fusion --arch TENet12NarrowModel --kernel_list 3,5,7,9 --save_folder save/TENet12NarrowModel/ \
--checkpoint_path save/MTENet12NarrowModel/TENet12NarrowModel-30000

python -m main --mod eval --dataset_path ../dataset_v1 --dataset_name test \
--arch TENet12NarrowModel \
--checkpoint_path save/TENet12NarrowModel/TENet12NarrowModel-30000
