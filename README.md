# tdt4305-phase2

Setup:
- Clone the repo
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

Usage:
- `python3 classify.py`
- Arguments:
  - `-training` or `-t`
    - Takes the path to the training set as a parameter. Should be a `.tsv`-file. The `place_name`- and `tweet_text`-fields should have index 4 and 10 respectively.
  - `-input` or `-i`
    - Takes the path to the input file as a parameter. Should be a text file with a single line of text.
  - `-output` or `-o`
    - Takes the desired path of the output file as a parameter.
  - `-sample` or `-s` 
    - Flag for sampling the dataset.
  - `-pretty` or `-p`
    - Flag for printing extra information during classification. Shows probabilites of each place analysed and more.

