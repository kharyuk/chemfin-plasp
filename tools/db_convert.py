# -*- coding: utf-8 -*-

# ##############################################################################
#
# We provide source code for easy conversion between csv file and sqlite
# database which contains description of each data file. It is designed
# for Python 2.7, and if user uses Python 3.x, it is required to make
# certain correctives.
#
################################################################################

import sqlite3
import csv

_COLUMNS = {
    'Filename': 'Filename',
    'Treatment': 'Additional treatment',
    'Species name': 'Species name',
    'Volume': 'Injection volume (ul)',
    'Equipment': 'Equipment',
    'Origin': 'Origin',
    'Year': 'Harvest year'
}


def csv2db(filenameCSV, filenameDB='db'):
    '''
    Convert CSV file into sqlite3 database.
    '''
    
    if not filenameDB.endswith('.sqlite'):
        filenameDB += '.sqlite'
    
    conn = sqlite3.connect(filenameDB)
    cur = conn.cursor()
    # create table with harvest years
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Harvest_year (
        id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        year INTEGER UNIQUE
    )''')
    # create table with species
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Species (
        id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        name    TEXT UNIQUE
    )''')
    # create table with manufacturer
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Origin (
        id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        name  TEXT UNIQUE
    )''')
    # create table with equipment
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Equipment (
        id  INTEGER NOT NULL PRIMARY KEY 
            AUTOINCREMENT UNIQUE,
        name TEXT UNIQUE
    )''')
    # create table with injection volumes
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Injection_volume (
        id  INTEGER NOT NULL PRIMARY KEY 
            AUTOINCREMENT UNIQUE,
        volume INTEGER UNIQUE
    )''')
    # create table with sample instances
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Sample (
        id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        filename TEXT,
        additional_treatment TEXT,
        Equipment_id INTEGER,
        Year_id INTEGER,
        Species_id INTEGER,
        Volume_id INTEGER,
        Origin_id INTEGER,
        UNIQUE(filename, Equipment_id, Year_id, Species_id, Volume_id, Origin_id,
        additional_treatment) ON CONFLICT REPLACE
    )''')
    # fulfill tables
    with open(filenameCSV, 'r') as csvfile:
        data = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in data:
            cur.execute('''INSERT OR IGNORE INTO Harvest_year (year)
            VALUES (?)''', (row[_COLUMNS['Year']],))
            cur.execute('SELECT id FROM Harvest_year WHERE year = ?',
                        (row[_COLUMNS['Year']],)) 
            Year_id = cur.fetchone()[0]
            
            cur.execute('''INSERT OR IGNORE INTO Species (name)
            VALUES (?)''', (row[_COLUMNS['Species name']],))
            cur.execute('SELECT id FROM Species WHERE name = ?',
                        (row[_COLUMNS['Species name']],)) 
            Species_id = cur.fetchone()[0]
            
            cur.execute('''INSERT OR IGNORE INTO Origin (name)
            VALUES (?)''', (row[_COLUMNS['Origin']],))
            cur.execute('SELECT id FROM Origin WHERE name = ?',
                        (row[_COLUMNS['Origin']],)) 
            Origin_id = cur.fetchone()[0]
            
            cur.execute('''INSERT OR IGNORE INTO Equipment (name)
            VALUES (?)''', (row[_COLUMNS['Equipment']],))
            cur.execute('SELECT id FROM Equipment WHERE name = ?',
                        (row[_COLUMNS['Equipment']],)) 
            Equipment_id = cur.fetchone()[0]
           
            cur.execute('''INSERT OR IGNORE INTO Injection_volume (volume)
            VALUES (?)''', (row[_COLUMNS['Volume']],))
            cur.execute('SELECT id FROM Injection_volume WHERE volume = ?',
                        (row[_COLUMNS['Volume']],)) 
            Volume_id = cur.fetchone()[0]
            
            cur.execute('''INSERT OR IGNORE INTO Sample (filename, Equipment_id,
            Year_id, Species_id, Volume_id, Origin_id, additional_treatment)
            VALUES (?,?,?,?,?,?,?)''',
                (row[_COLUMNS['Filename']], Equipment_id, Year_id,
                Species_id, Volume_id, Origin_id, row[_COLUMNS['Treatment']])
            )
                        
        conn.commit()
            
def db2csv(filenameDB, filenameCSV='table'):
    '''
    Convert sqlite3 database into CSV file
    '''
    
    if not filenameCSV.endswith('.csv'):
        filenameCSV += '.csv'
    conn = sqlite3.connect(filenameDB)
    cur = conn.cursor()
    data = cur.execute('''

    SELECT Sample.filename, Sample.additional_treatment, Species.name,
           Injection_volume.volume, Origin.name,
           Equipment.name, Harvest_year.year
    FROM Sample
        JOIN Species
            ON Species.id = Sample.Species_id
        JOIN Injection_volume
            ON Injection_volume.id = Sample.Volume_id
        JOIN Origin
            ON Origin.id = Sample.Origin_id
        JOIN Equipment
            ON Equipment.id = Sample.Equipment_id
        JOIN Harvest_year
            ON Harvest_year.id = Sample.year_id            
            
    ''')

    with open(filenameCSV, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        header = [_COLUMNS['Filename'], _COLUMNS['Treatment'],
                  _COLUMNS['Species name'], _COLUMNS['Volume'],
                  _COLUMNS['Origin'], _COLUMNS['Equipment'],
                  _COLUMNS['Year']]
        writer.writerow(header)
        writer.writerows(data)
        
if __name__ == '__main__':
    filenameCSV = 'table.csv'
    filenameCSV2 = 'table2'
    csv2db(filenameCSV, filenameDB='db')
    db2csv(filenameDB='db.sqlite', filenameCSV='table2')
    
    
    
    
    
    
    
    
    
    
    
