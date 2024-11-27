# Documentation for Custom Configuration Files

This document provides step-by-step instructions for setting up custom configuration files for your algorithm.

## Steps to Create a Custom Configuration File

1. **Copy an Existing Configuration File**  
   Duplicate an existing configuration file for the algorithm you want to test.

2. **Import the Dataset File**  
   Import the corresponding dataset file from the `../_base_/datasets/` directory, and comment out the used file in `_base_ = []`.

3. **Modify the `train_dataloader` Configuration**  
   Update the `train_dataloader` variable as follows:

   ```python
   train_dataloader = dict(
       batch_size=int,  # Retain original configuration
       num_workers=int,  # Retain original configuration
       drop_last=bool,  # Retain original configuration
       persistent_workers=bool,  # Retain original configuration
       sampler=dict(),  # Retain original configuration
       collate_fn=dict(),  # Retain original configuration
       dataset=dict(
           type='CustomDataset',  # Change to 'CustomDataset'
           data_root='path/to/your/data',
           ann_file='',  # Specify only for supervised tasks
           data_prefix='',  # Use for supervised tasks; otherwise, specify in data_root
           pipeline=train_pipeline,
           with_label=False  # Set for unsupervised tasks
       )
   )
   ```

## Important Notes

### 1. Setting `with_label`

- If `with_label=True`, the model requires the following folder structure:

  ```plaintext
  data_prefix/
      ├── class_x
      │   ├── xxx.png
      │   ├── xxy.png
      │   └── ...
      │       └── xxz.png
      └── class_y
          ├── 123.png
          ├── nsdf3.png
          └── ...
          └── asd932_.png
  ```

- If `with_label=False`, the model processes **all** images directly from the directory structure below:

  ```plaintext
  data_prefix/
      ├── folder_1
      │   ├── xxx.png
      │   ├── xxy.png
      │   └── ...
      ├── 123.png
      ├── nsdf3.png
      └── ...
  ```

### 2. Unsupervised vs. Supervised Tasks

- For **supervised tasks**, ensure the `ann_file` and `data_prefix` are correctly configured.
- For **unsupervised tasks**, set `with_label=False` and organize data according to the directory structure mentioned above.