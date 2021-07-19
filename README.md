# visualcomet-mc

## Get Started
<p>
  Generate the multiple choice data from the visual comet annotations located in the source directory, visualcomet, and specify the file that stores the multiple choice data located in destination directory, data. </p>
  <p>
  The visualcomet data should be in the following folder structure:
  </p>
  
  
```
visualcomet/
|-- features/
|-- vcr1images/
|   |-- VERSION.txt
|   |-- movie name, like movieclips_A_Fistful_of_Dollars
|   |   |-- image files, like Sv_GcxkmW4Y@29.jpg
|   |   |-- metadata files, like Sv_GcxkmW4Y@29.json
|-- train_annots.json
|-- val_annots.json
|-- test_annots.json
```
  

  <p>
  Run the following command:
  </p>
  
  
    python create_data.py --data-src-dir '/path/to/visualcomet_annotations/' --data-dest-dir './data/dataset_filename.json'

  <p>
  For example, I run this command to generate the multiple choice training data:
  </p>
    
    python create_data.py --data-src-dir '../visualcomet/train_annots.json' --data-dest-dir './data/train.json'




