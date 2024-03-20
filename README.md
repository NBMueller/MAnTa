# MAnTa

**M**anual **An**notation of **Ta**pestri DNA-seq data

---

## Setup
### Installation
**Requirements**
- Python 3.X
- Python packages (listed in requirements.txt)

**Optional:** create and source conda environment
```bash
conda create --name manta python=3
conda activate manta
```

**Install packages** from requirements.txt file, e.g., via pip:
```bash
python -m pip install -r requirements.txt
```

### Data
Required files per sample stored in **one** input folder:
- `{PREFIX}.barcode.cell.distribution.merged.tsv` file from `dna/results/tsv/` directory
- `{PREFIX}.filtered_variants.csv` file from [`mosaic_preprocessing.py`](https://github.com/cbg-ethz/demoTape/blob/main/workflow/scripts/mosaic_preprocessing.py))

Additionally required somewhere else:
- Annotated Tapestri panel in bed format ([annotation tool](https://github.com/vladsavelyev/bed_annotation))

## Running
**Run the app**
```bash
python run_app.py -i <DATA_DIR> -p <PANEL_file>
```
**Open webrowser** and go to: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)