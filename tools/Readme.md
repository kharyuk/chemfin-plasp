# Medicinal plants extracts: an LC-MS dataset

## Installation

**Anaconda** python distribution (with **python 2.7**) is highly recommended to use with
provided tools:

https://www.continuum.io/downloads

To compress and restore data it is necessary to install additional packages:

- **numpy**
```shell
conda install numpy
```
or
```shell
pip install numpy
```
- **pylzma**
```shell
pip install git+https://github.com/fancycode/pylzma.git
```

## Quick start

Open a terminal, change directory to database folder. Enter the following command:
```shell
python decompress.py
```
As a result, new folder with reconstructed mzXML files will be created.

## Compression and reconstruction

In this article experimental data is given as a set of compressed binary files with
.dat extensions. Utilities for restoring or compressing are placed
in *mzxml.py* file.


Following commands are used for reconstruction of all mzXML files from specified directory:

```python
import tools

dirname = './mzxml_data_compressed'
tools.multipleBin2MzXML(dirname)

```
Single mzXML file may be reconstructed by the following commands:

```python
import tools

fileName = './mzxml_data_compressed/mzxml_file.dat'
saveFileName = './restored_mzxml_file'
tools.saveMzXML(fileName, savefname=saveFileName)

```

Compression can be done with functions *multipleMzXML2Bin()* for all data in specified
directory: 
```python
import tools

dirname = './mzxml_data'
tools.multipleMzXML2Bin(dirname)

```
and *saveMzXML()* for a single file:
```python
import tools

fileName = './mzxml_data/mzxml_file.dat'
saveFileName = './comressed_mzxml_file'
tools.saveMzXML(fileName, savefname=saveFileName)

```

## Supplementary information

Supplementary information is given in two formats: as CSV table and as sqlite3
database. *Tools* directory contains file *db_convert.py* with python scripts
for conversion between both formats.

```python
import tools

filenameDB = 'db_in.sqlite'
filenameCSV = 'table_in.csv'

# convert sqlite3 database into CSV table
db2csv(filenameDB, filenameCSV='table_out')
# convert CSV table into sqlite3 database
csv2db(filenameCSV, filenameDB='db_out')
```
