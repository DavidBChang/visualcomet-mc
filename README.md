# visualcomet-mc

## Get Started
<p>
  Generate the multiple choice data from the visual comet annotations located in the `visualcomet` source directory and specify the file that stores the multiple choice data located in `data` destination directory. </p>
  <p>
  The visualcomet data should be in the following folder structure:
  
</p>
  <p>
  Run the following command:
  </p>
  
  <p>
    `python create_data.py --data-src-dir '/path/to/visualcomet_annotations/' --data-dest-dir './data/dataset_filename.json' `
</p>
  <p>
  For example, I run this command to generate the multiple choice training data:
    `python create_data.py --data-src-dir '../visualcomet/train_annots.json' --data-dest-dir './data/train.json' `
</p>



